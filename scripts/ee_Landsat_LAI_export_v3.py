#!/usr/bin/env python3
import argparse
import datetime
import logging
import pprint

import ee

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

    # Hard code input parameters for now
    path = 44
    row = 33
    version = 'v3'
    coll_id = 'projects/openet/lai/landsat/v3/'
    start_date = '2017-07-01'
    end_date = '2017-08-01'
    # start_date = '2017-01-01'
    # end_date = '2018-01-01'
    gee_key_file = '/Users/mortonc/Projects/keys/openet-api-gee.json'


    logging.info('\nInitializing Earth Engine')
    if gee_key_file:
        logging.info('  Using service account key file: {}'.format(gee_key_file))
        # The "EE_ACCOUNT"  doesn't seem to be used if the key file is valid
        ee.Initialize(ee.ServiceAccountCredentials('test', key_file=gee_key_file),
                      use_cloud_api=True)
    else:
        ee.Initialize(use_cloud_api=True)

    # Get a list of the available Landsat asset IDs
    input_asset_id_list = getLandsatSR(start_date, end_date, path, row) \
        .sort('system:time_start') \
        .aggregate_array('system:id') \
        .getInfo()

    # Process each Landsat image separately
    for input_asset_id in input_asset_id_list:
        landsat_id = input_asset_id.split('/')[-1]
        output_asset_id = coll_id + landsat_id.lower()
        logging.info(landsat_id)
        logging.debug('  {}'.format(input_asset_id))
        logging.debug('  {}'.format(output_asset_id))

        if ee.data.getInfo(output_asset_id):
            if overwrite_flag:
                print('  asset already exists - overwriting')
                ee.data.deleteAsset(output_asset_id)
            else:
                print('  asset already exists - skipping')
                continue

        # Copied from PTJPL Image.from_landsat_c1_sr()
        landsat_img = ee.Image(input_asset_id)
        input_bands = ee.Dictionary({
            'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
            'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'B6', 'pixel_qa'],
            'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'pixel_qa']})
        output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'tir', 'pixel_qa']
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

        # Apply the cloud mask to the image to mimic what is happening in the
        # version 2 export script where the Landsat collections are masked
        prep_img = maskLST(prep_img)

        # Sensor must currently be a python string because it is used to build
        #   a feature collection ID in the lai code
        sensor = landsat_id.split('_')[0].upper()
        # sensor_dict = {'LANDSAT_5': 'LT05', 'LANDSAT_7': 'LE07', 'LANDSAT_8': 'LC08'}
        # sensor = landsat_img.get('SATELLITE').getInfo()
        # sensor = sensor_dict[sensor]

        # Get the projection as a client side object (for testing and debug)
        image_proj = landsat_img.select([0]).projection().getInfo()

        # Get the projection as a server side object for production to avoid
        #   an extra getInfo call
        # image_proj = landsat_img.select([0]).projection()
        # image_crs = image_proj.crs()
        # image_transform = ee.List(ee.Dictionary(
        #     ee.Algorithms.Describe(image_proj)).get('transform'))

        # Compute LAI image
        lai_img = openet.lai.landsat.getLAIImage(prep_img, sensor, nonveg=1) \
            .set({
                'date_ingested': datetime.datetime.today().strftime('%Y-%m-%d'),
                'lai_version': openet.lai.__version__,
            })
        #     .rename(['LAI']) \

        # Start export
        task = ee.batch.Export.image.toAsset(
            image=lai_img,
            description='{}_LAI_{}'.format(landsat_id, version),
            assetId=output_asset_id,
            crs=image_proj['crs'],
            crsTransform=str(image_proj['transform']),
        )
        task.start()
        logging.debug('  {}'.format(task.id))


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


def getQABits(image, start, end, newName):
    """
     Function that returns an image containing just the specified QA bits.
    """
    # Compute the bits we need to extract.
    pattern = 0
    for i in range(start, end + 1):
         pattern = pattern + 2**i

    # Return a single band image of the extracted QA bits, giving the band
    # a new name.
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)


def maskLST(image):
    """
    Function that masks a Landsat image based on the QA band
    """
    pixelQA = image.select('pixel_qa')
    cloud = getQABits(pixelQA, 1, 1, 'clear')
    return image.updateMask(cloud.eq(1))


def getLandsatSR(start, end, path, row):
    """
    Get Landsat image collection
    """
    # Landsat 8
    Landsat8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER', 'less_than', 70)
        # .filterMetadata('DATA_TYPE', 'equals', 'L1TP')
        # .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'],
        #         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \

    # Landsat 7
    Landsat7_sr = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')  \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER', 'less_than', 70)
        # .filterMetadata('DATA_TYPE', 'equals', 'L1TP')
        # .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
        #         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \

    # Landsat 5
    Landsat5_sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals', path) \
        .filterMetadata('WRS_ROW', 'equals', row) \
        .filterMetadata('CLOUD_COVER', 'less_than', 70)
        # .filterMetadata('DATA_TYPE', 'equals', 'L1TP')
        # .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
        #         ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \

    Landsat_sr_coll = Landsat8_sr.merge(Landsat5_sr).merge(Landsat7_sr)

    return Landsat_sr_coll


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
