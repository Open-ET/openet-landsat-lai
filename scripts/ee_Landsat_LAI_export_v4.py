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
import openet.lai.utils as utils

TOOL_NAME = 'ee_Landsat_LAI_export'
# TOOL_NAME = os.path.basename(__file__)
TOOL_VERSION = 'v4'


def main(start_dt, end_dt, overwrite_flag=False, gee_key_file=None):
    """Export Landsat LAI images

    Parameters
    ----------
    start_dt : datetime, optional
        Override the start date in the INI file
        (the default is None which will use the INI start date).
    end_dt : datetime, optional
        Override the (inclusive) end date in the INI file
        (the default is None which will use the INI end date).
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
    # wrs2_tile_list = ['p044r034']  # openet-api-gee
    # wrs2_tile_list = ['p043r035']  # openet-dri-gee
    wrs2_tile_list = ['p042r036']   # openet-gee
    # wrs2_tile_list = [
    #     'p044r032', 'p043r033', 'p042r034',
    #     'p044r033', 'p043r034', 'p042r035', 'p044r034', 'p043r035', 'p042r036',
    #     'p041r035', 'p041r036',
    #     'p045r032', 'p045r033',
    # ]
    version = openet.lai.__version__.replace('.', 'p')
    coll_id = 'projects/openet/lai/landsat/scene'
    # gee_key_file = None
    # gee_key_file = '/Users/mortonc/Projects/keys/openet-api-gee.json'
    # gee_key_file = '/Users/mortonc/Projects/keys/openet-dri-gee.json'
    gee_key_file = '/Users/mortonc/Projects/keys/openet-gee.json'
    cloud_cover_max = 70
    log_tasks = True
    clip_ocean_flag = True
    scale_factor = 10000
    output_type = 'uint16'
    model_name = 'LAI'


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


    if not ee.data.getInfo(coll_id):
        logging.info('\nExport collection does not exist and will be built'
                     '\n  {}'.format(coll_id))
        input('Press ENTER to continue')
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, coll_id)


    for wrs2_tile in wrs2_tile_list:
        path, row = int(wrs2_tile[1:4]), int(wrs2_tile[5:8])

        # Get a list of the available Landsat asset IDs
        input_asset_id_list = getLandsatSR(start_date, end_date, path, row,
                                           cloud_cover_max) \
            .sort('system:time_start') \
            .aggregate_array('system:id') \
            .getInfo()

        # Process each Landsat image separately
        for input_asset_id in input_asset_id_list:
            landsat_id = input_asset_id.split('/')[-1]
            output_asset_id = f'{coll_id}/{landsat_id.lower()}'
            logging.info(landsat_id)
            logging.debug('  {}'.format(input_asset_id))
            logging.debug('  {}'.format(output_asset_id))

            if ee.data.getInfo(output_asset_id):
                if overwrite_flag:
                    logging.info('  asset already exists - overwriting')
                    ee.data.deleteAsset(output_asset_id)
                else:
                    logging.info('  asset already exists - skipping')
                    continue

            # TODO: Module should handle the band renaming and scaling
            # Copied from PTJPL Image.from_landsat_c1_sr()
            landsat_img = ee.Image(input_asset_id)
            input_bands = ee.Dictionary({
                'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
                'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
                'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'pixel_qa']})
            output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir',
                            'pixel_qa']
            spacecraft_id = ee.String(landsat_img.get('SATELLITE'))
            prep_img = landsat_img \
                .select(input_bands.get(spacecraft_id), output_bands) \
                .set({'system:index': landsat_img.get('system:index'),
                      'system:time_start': landsat_img.get('system:time_start'),
                      'system:id': landsat_img.get('system:id'),
                      'SATELLITE': spacecraft_id,
                     })
            # CM - Don't unscale the images yet
            # The current implementation is expecting raw unscaled images
            #     .multiply([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.1, 1]) \

            # Sensor must currently be a python string because it is used to build
            #   a feature collection ID in the lai code
            sensor = landsat_id.split('_')[0].upper()

            # Get the projection as a server side object for production to avoid
            #   an extra getInfo call
            image_proj = landsat_img.select(['B4']).projection()
            image_crs = image_proj.crs()
            image_transform = ee.List(ee.Dictionary(
                ee.Algorithms.Describe(image_proj)).get('transform'))

            # Compute LAI image
            output_img = openet.lai.landsat.getLAIImage(prep_img, sensor, nonveg=1)

            # Copied properties from scene export tool
            # Many of these are so the tasks can be tracked in the datastore
            properties = {
                'coll_id': coll_id,
                'date_ingested': datetime.datetime.today().strftime('%Y-%m-%d'),
                'image_id': input_asset_id,
                'lai_version': openet.lai.__version__,
                'model_name': 'LAI',
                'model_version': openet.lai.__version__,
                'scale_factor': 1.0 / scale_factor,
                'scene_id': landsat_id,
                'tool_name': TOOL_NAME,
                'tool_version': TOOL_VERSION,
                'wrs2_tile': 'p{}r{}'.format(
                    landsat_id.split('_')[1][:3], landsat_id.split('_')[1][3:]),
            }
            output_img = output_img.set(properties)

            if scale_factor > 1:
                if output_type == 'int16':
                    output_img = output_img.multiply(scale_factor).round()\
                        .clamp(-32768, 32767).int16()
                elif output_type == 'uint16':
                    output_img = output_img.multiply(scale_factor).round()\
                        .clamp(0, 65536).uint16()

            if clip_ocean_flag:
                output_img = output_img \
                    .updateMask(ee.Image('projects/openet/ocean_mask'))

            # Start export
            task = ee.batch.Export.image.toAsset(
                image=output_img,
                description='{}_LAI_{}'.format(landsat_id, version),
                assetId=output_asset_id,
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

    main(start_dt=args.start, end_dt=args.end,
         overwrite_flag=args.overwrite, gee_key_file=args.key,
         )
