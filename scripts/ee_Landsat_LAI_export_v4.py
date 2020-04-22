#!/usr/bin/env python3
import argparse
import datetime
import json
import logging
import os
import pprint

import ee
from google.cloud import datastore

import openet.lai
import openet.lai.landsat
import openet.core.utils as utils

TOOL_NAME = 'ee_Landsat_LAI_export'
# TOOL_NAME = os.path.basename(__file__)
TOOL_VERSION = 'v4'


def main(start_dt, end_dt, wrs2_tiles, overwrite_flag=False,
         gee_key_file=None):
    """Export Landsat LAI images

    Parameters
    ----------
    start_dt : datetime, optional
        Override the start date in the INI file
        (the default is None which will use the INI start date).
    end_dt : datetime, optional
        Override the (inclusive) end date in the INI file
        (the default is None which will use the INI end date).
    wrs2_tiles : str, None, optional
        List of WRS2 tiles to process.
    overwrite_flag : bool, optional
        If True, overwrite existing files (the default is False).
    gee_key_file : str, None, optional
        Earth Engine service account JSON key file (the default is None).

    Returns
    -------
    None

    """
    logging.info('\nExport Landsat LAI images')

    start_date = start_dt.strftime('%Y-%m-%d')
    end_dt = end_dt + datetime.timedelta(days=1)
    end_date = end_dt.strftime('%Y-%m-%d')

    # CM - Hard coding input parameters for now
    coll_id = 'projects/earthengine-legacy/assets/projects/openet/lai/landsat/scene'
    # coll_id = 'projects/openet/lai/landsat/scene'
    cloud_cover_max = 70
    log_tasks = True
    clip_ocean_flag = True
    scale_factor = 10000
    output_type = 'uint16'
    model_name = 'LAI'
    export_id_name = '_v' + openet.lai.__version__.replace('.', 'p')

    # Central Valley WRS2 tiles
    # wrs2_tiles = [
    #     'p044r033', 'p044r032', 'p044r034',  # openet
    #     'p043r034', 'p043r033', 'p043r035',  # openet-dri
    #     'p042r035', 'p042r034', 'p042r036',  # openet-api
    #     'p041r035', 'p041r036',
    #     'p045r032', 'p045r033',
    # ]

    wrs2_tile_fmt = 'p{:03d}r{:03d}'

    logging.debug('User tiles: {}'.format(wrs2_tiles))
    wrs2_tiles = [y.strip() for x in wrs2_tiles for y in x.split(',')]
    wrs2_tiles = [x.lower() for x in wrs2_tiles if x]
    logging.info('WRS2 tiles: {}'.format(', '.join(wrs2_tiles)))

    logging.info('\nInitializing Earth Engine')
    if gee_key_file:
        logging.info('  Using service account key file: {}'.format(gee_key_file))
        # The "EE_ACCOUNT"  doesn't seem to be used if the key file is valid
        ee.Initialize(ee.ServiceAccountCredentials('test', key_file=gee_key_file),
                      use_cloud_api=True)
    else:
        ee.Initialize(use_cloud_api=True)


    # TODO: set datastore key file as a parameter?
    datastore_key_file = 'openet-dri-datastore.json'
    if log_tasks and not os.path.isfile(datastore_key_file):
        logging.info('Task logging disabled, datastore key does not exist')
        log_tasks = False
        # input('ENTER')
    if log_tasks:
        logging.info('\nInitializing task datastore client')
        try:
            datastore_client = datastore.Client.from_service_account_json(
                datastore_key_file)
        except Exception as e:
            logging.error('{}'.format(e))
            return False


    # # Get current running tasks
    # tasks = utils.get_ee_tasks()
    # if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
    #     logging.debug('  Tasks: {}'.format(len(tasks)))
    #     input('ENTER')


    # if not ee.data.getInfo(coll_id.rsplit('/', 1)[0]):
    #     logging.debug('\nExport folder does not exist and will be built'
    #                   '\n  {}'.format(coll_id.rsplit('/', 1)[0]))
    #     input('Press ENTER to continue')
    #     ee.data.createAsset({'type': 'FOLDER'}, coll_id.rsplit('/', 1)[0])
    if not ee.data.getInfo(coll_id):
        logging.info('\nExport collection does not exist and will be built'
                     '\n  {}'.format(coll_id))
        input('Press ENTER to continue')
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, coll_id)


    for wrs2_tile in wrs2_tiles:
        logging.debug('\nWRS2: {}'.format(wrs2_tile))
        path, row = int(wrs2_tile[1:4]), int(wrs2_tile[5:8])

        # Get list of existing images for the target tile
        logging.debug('  Getting existing asset ID list')
        asset_coll = ee.ImageCollection(coll_id) \
            .filterDate(start_date, end_date) \
            .filterMetadata('wrs2_tile', 'equals',
                            wrs2_tile_fmt.format(path, row))
        asset_props = {f'{coll_id}/{x["properties"]["system:index"]}':
                           x['properties']
                       for x in asset_coll.getInfo()['features']}
        # asset_props = {x['id']: x['properties']
        #                for x in assets_info['features']}

        # Get a list of the available Landsat asset IDs
        logging.debug('  Getting source image ID list')
        input_asset_id_list = getLandsatSR(start_date, end_date, path, row,
                                           cloud_cover_max) \
            .sort('system:time_start') \
            .aggregate_array('system:id') \
            .getInfo()

        # Process each Landsat image separately
        for image_id in input_asset_id_list:
            scene_id = image_id.split('/')[-1]
            asset_id = f'{coll_id}/{scene_id.lower()}'
            logging.info(image_id)
            logging.debug('  {}'.format(image_id))
            logging.debug('  {}'.format(asset_id))

            export_id = '{}_{}'.format(
                model_name.lower(), image_id.lower().replace('/', '_'))
            export_id += export_id_name
            logging.debug('  {}'.format(export_id))

            if overwrite_flag:
                # if export_id in tasks.keys():
                #     logging.info('  Task already submitted, cancelling')
                #         ee.data.cancelTask(tasks[export_id]['id'])
                # This is intentionally not an "elif" so that a task can be
                # cancelled and an existing image/file/asset can be removed
                if asset_props and asset_id in asset_props.keys():
                    logging.info('  Asset already exists, removing')
                    ee.data.deleteAsset(asset_id)
            else:
                # if export_id in tasks.keys():
                #     logging.info('  Task already submitted, skipping')
                #     continue
                if asset_props and asset_id in asset_props.keys():
                    logging.info('  Asset already exists, skipping')
                    continue

            # TODO: Module should handle the band renaming and scaling
            # Copied from PTJPL Image.from_landsat_c1_sr()
            input_img = ee.Image(image_id)
            input_bands = ee.Dictionary({
                'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
                'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
                'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'pixel_qa']})
            output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir',
                            'pixel_qa']
            spacecraft_id = ee.String(input_img.get('SATELLITE'))
            prep_img = input_img \
                .select(input_bands.get(spacecraft_id), output_bands) \
                .set({'system:index': input_img.get('system:index'),
                      'system:time_start': input_img.get('system:time_start'),
                      'system:id': input_img.get('system:id'),
                      'SATELLITE': spacecraft_id,
                     })
            # CM - Don't unscale the images yet
            # The current implementation is expecting raw unscaled images
            #     .multiply([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.1, 1]) \

            # Sensor must currently be a python string because it is used to build
            #   a feature collection ID in the lai code
            sensor = scene_id.split('_')[0].upper()

            # Get the projection as a server side object for production to avoid
            #   an extra getInfo call
            image_proj = input_img.select(['B4']).projection()
            image_crs = image_proj.crs()
            image_transform = ee.List(ee.Dictionary(
                ee.Algorithms.Describe(image_proj)).get('transform'))

            # Compute LAI image
            output_img = openet.lai.landsat.getLAIImage(prep_img, sensor, nonveg=1)

            if scale_factor > 1:
                if output_type == 'int16':
                    output_img = output_img.multiply(scale_factor).round()\
                        .clamp(-32768, 32767).int16()
                elif output_type == 'uint16':
                    output_img = output_img.multiply(scale_factor).round()\
                        .clamp(0, 65536).uint16()
                else:
                    raise ValueError('unsupported combination of output_type '
                                     'and scale_factor')

            if clip_ocean_flag:
                output_img = output_img \
                    .updateMask(ee.Image('projects/openet/ocean_mask'))

            # Copied properties from scene export tool
            # Many of these are so the tasks can be tracked in the datastore
            input_info = input_img.getInfo()
            properties = {
                'coll_id': coll_id,
                'date_ingested': datetime.datetime.today().strftime('%Y-%m-%d'),
                'image_id': image_id,
                'landsat_lai_version': openet.lai.__version__,
                'model_name': model_name,
                'model_version': openet.lai.__version__,
                'scale_factor': 1.0 / scale_factor,
                'scene_id': scene_id,
                'tool_name': TOOL_NAME,
                'tool_version': TOOL_VERSION,
                'wrs2_tile': 'p{}r{}'.format(
                    scene_id.split('_')[1][:3], scene_id.split('_')[1][3:]),
                'CLOUD_COVER': input_info['properties']['CLOUD_COVER'],
                'CLOUD_COVER_LAND': input_info['properties']['CLOUD_COVER_LAND'],
                'system:time_start': input_info['properties']['system:time_start'],
            }
            output_img = output_img.set(properties)

            # CM - Copy all the properties from the source image
            # output_img = ee.Image(output_img.copyProperties(input_img))

            # Start export
            task = ee.batch.Export.image.toAsset(
                image=output_img,
                description=export_id,
                assetId=asset_id,
                crs=image_crs,
                crsTransform=image_transform,
            )
            task.start()
            logging.debug('  {}'.format(task.id))


            # Write the export task info the openet-dri project datastore
            if log_tasks:
                logging.debug('  Writing datastore entity')
                try:
                    task_obj = datastore.Entity(key=datastore_client.key(
                        'Task', task.status()['id']),
                        exclude_from_indexes=['properties'])
                    for k, v in task.status().items():
                        task_obj[k] = v
                    # task_obj['date'] = datetime.datetime.today() \
                    #     .strftime('%Y-%m-%d')
                    task_obj['index'] = properties.pop('wrs2_tile')
                    # task_obj['wrs2_tile'] = properties.pop('wrs2_tile')
                    task_obj['model_name'] = properties.pop('model_name')
                    # task_obj['model_version'] = properties.pop('model_version')
                    task_obj['runtime'] = 0
                    task_obj['start_timestamp_ms'] = 0
                    task_obj['tool_name'] = properties.pop('tool_name')
                    task_obj['properties'] = json.dumps(properties)
                    datastore_client.put(task_obj)
                except Exception as e:
                    # CGM - The message/handling will probably need to be updated
                    #   We may want separate try/excepts on the create and the put
                    logging.warning('\nDatastore entity was not written')
                    logging.warning('{}\n'.format(e))

            # input('ENTER')


# TODO: Test if DisALEXI Collection class could be used instead
def getLandsatSR(start_date, end_date, path, row, cloud_cover_max=70):
    """
    Get Landsat SR image collection
    """
    # CM - Note, v2 export script filtered on 'CLOUD_COVER' not 'CLOUD_COVER_LAND'
    # CM - None, v2 export script did not filter on 'DATA_TYPE'
    Landsat8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
        .filterDate(start_date, end_date) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_cover_max) \

    Landsat7_sr = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')  \
        .filterDate(start_date, end_date) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_cover_max) \

    Landsat5_sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR') \
        .filterDate(start_date, end_date) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER_LAND', 'less_than', cloud_cover_max) \

    Landsat_sr_coll = Landsat8_sr.merge(Landsat5_sr).merge(Landsat7_sr)

    return Landsat_sr_coll


# def renameLandsat(image):
#     """
#     Function that renames Landsat bands
#     """
#     sensor = ee.String(image.get('SATELLITE'))
#     from_list = ee.Algorithms.If(
#         sensor.compareTo('LANDSAT_8'),
#         ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
#         ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'])
#     to_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']
#
#     return image.select(from_list, to_list)


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Export Landsat LAI images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     '-i', '--ini', type=utils.arg_valid_file,
    #     help='Input file', metavar='FILE')
    parser.add_argument(
        '-s', '--start', type=utils.arg_valid_date, metavar='DATE', default=None,
        help='Start date (format YYYY-MM-DD)')
    parser.add_argument(
        '-e', '--end', type=utils.arg_valid_date, metavar='DATE', default=None,
        help='End date (format YYYY-MM-DD)')
    parser.add_argument(
        '--key', type=utils.arg_valid_file, metavar='FILE',
        help='Earth Engine service account JSON key file')
    parser.add_argument(
        '--tiles', default='', nargs='+',
        help='Comma/space separated list of tiles to process')
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='Force overwrite of existing files')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    # parser.add_argument(
    #     '--delay', default=0, type=float,
    #     help='Delay (in seconds) between each export tasks')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = arg_parse()
    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)

    main(start_dt=args.start, end_dt=args.end, wrs2_tiles=args.tiles,
         overwrite_flag=args.overwrite, gee_key_file=args.key,
         )
