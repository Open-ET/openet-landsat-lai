#!/usr/bin/env python3
import argparse
import datetime
import logging
import pprint

import ee

import openet.lai
import openet.lai.landsat
import openet.lai.utils as utils


def main(start_dt=None, end_dt=None, overwrite_flag=False, gee_key_file=None):
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

    # CM - Hard coding input parameters for now
    path = 44
    row = 33
    version_str = openet.lai.__version__.replace('.', 'p')
    # version_str = openet.lai.__version__.replace('.', 'p') + '_smile_new'
    # version_str = openet.lai.__version__.replace('.', 'p') + '_smile_old'
    # version_str = openet.lai.__version__.replace('.', 'p') + '_nonsmile_new'
    # version_str = openet.lai.__version__.replace('.', 'p') + '_nonsmile_old'
    coll_id = f'projects/openet/lai/landsat/v{version_str}'

    # start_date = '2017-07-01'
    # end_date = '2017-08-01'
    start_date = '2017-01-01'
    end_date = '2018-01-01'
    # gee_key_file = None
    gee_key_file = '/Users/mortonc/Projects/keys/openet-api-gee.json'
    # gee_key_file = '/Users/mortonc/Projects/keys/openet-dri-gee.json'
    # gee_key_file = '/Users/mortonc/Projects/keys/openet-gee.json'
    cloud_cover_max = 70


    logging.info('\nInitializing Earth Engine')
    if gee_key_file:
        logging.info('  Using service account key file: {}'.format(gee_key_file))
        # The "EE_ACCOUNT"  doesn't seem to be used if the key file is valid
        ee.Initialize(ee.ServiceAccountCredentials('test', key_file=gee_key_file),
                      use_cloud_api=True)
    else:
        ee.Initialize(use_cloud_api=True)

    if not ee.data.getInfo(coll_id):
        logging.info('\nExport collection does not exist and will be built'
                     '\n  {}'.format(coll_id))
        input('Press ENTER to continue')
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, coll_id)

    # Get a list of the available Landsat asset IDs
    input_asset_id_list = getLandsatSR(start_date, end_date, path, row,
                                       cloud_cover_max) \
        .sort('system:time_start') \
        .aggregate_array('system:id') \
        .getInfo()
    # pprint.pprint(input_asset_id_list)
    # input('ENTER')

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
        # sensor_dict = {'LANDSAT_5': 'LT05', 'LANDSAT_7': 'LE07', 'LANDSAT_8': 'LC08'}
        # sensor = landsat_img.get('SATELLITE').getInfo()
        # sensor = sensor_dict[sensor]

        # Get the projection as a client side object (for testing and debug)
        # image_proj = landsat_img.select([0]).projection().getInfo()

        # Get the projection as a server side object for production to avoid
        #   an extra getInfo call
        image_proj = landsat_img.select(['B4']).projection()
        image_crs = image_proj.crs()
        image_transform = ee.List(ee.Dictionary(
            ee.Algorithms.Describe(image_proj)).get('transform'))


        # TODO: Long term we may want to save as scaled integers to save space
        # if scale_factor > 1:
        #     if output_type == 'int16':
        #         output_img = output_img.multiply(scale_factor).round()\
        #             .clamp(-32768, 32767).int16()
        #     elif output_type == 'uint16':
        #         output_img = output_img.multiply(scale_factor).round()\
        #             .clamp(0, 65536).uint16()

        # TODO: Long term we may want to clip ocean areas similar to scene export
        # if clip_ocean_flag:
        #     output_img = output_img\
        #         .updateMask(ee.Image('projects/openet/ocean_mask'))

        # Compute LAI image
        output_img = openet.lai.landsat.getLAIImage(prep_img, sensor, nonveg=1) \
            .set({
                'date_ingested': datetime.datetime.today().strftime('%Y-%m-%d'),
                'lai_version': openet.lai.__version__,
                # CM - Properties in scene export that may eventually be useful
                # 'coll_id': coll_id,
                # 'image_id': input_asset_id,
                # 'scene_id': landsat_id,
                # 'scale_factor': 1.0 / scale_factor,
            })
        #     .rename(['LAI']) \

        # CM - Copy all the properties from the source image
        output_img = ee.Image(output_img.copyProperties(landsat_img))

        # Start export
        task = ee.batch.Export.image.toAsset(
            image=output_img,
            description='{}_LAI_{}'.format(landsat_id, version_str),
            assetId=output_asset_id,
            crs=image_crs,
            crsTransform=image_transform,
            # crs=image_proj['crs'],
            # crsTransform=str(image_proj['transform']),
        )
        task.start()
        logging.debug('  {}'.format(task.id))


# TODO: Test is DisALEXI Collection class could be used instead
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
