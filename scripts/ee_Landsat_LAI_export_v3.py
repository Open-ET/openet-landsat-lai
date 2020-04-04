#!/usr/bin/env python3
import datetime
import pprint
import sys

import ee

import openet.lai

ee.Initialize()


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


def setDate(image):
    """
    Function that adds a "date" property to an image in format "YYYYmmdd"
    """

    eeDate = ee.Date(image.get('system:time_start'))
    date = eeDate.format('YYYYMMdd')
    return image.set('date', date)


def getLandsat(start, end, path, row):
    """
    Get Landsat image collection
    """
    # Landsat 8
    Landsat8_sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals',path) \
        .filterMetadata('WRS_ROW', 'equals',row) \
        .filterMetadata('CLOUD_COVER', 'less_than',70) \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'],
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \
        .map(maskLST)

    # Landsat 7
    Landsat7_sr = ee.ImageCollection('LANDSAT/LE07/C01/T1_SR')  \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals',path) \
        .filterMetadata('WRS_ROW', 'equals',row) \
        .filterMetadata('CLOUD_COVER', 'less_than',70) \
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \
        .map(maskLST)

    # Landsat 5
    Landsat5_sr = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR') \
        .filterDate(start, end) \
        .filterMetadata('WRS_PATH', 'equals',path) \
        .filterMetadata('WRS_ROW', 'equals',row) \
        .filterMetadata('CLOUD_COVER', 'less_than',70) \
        .select(['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
                ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']) \
        .map(maskLST)

    Landsat_sr_coll = Landsat8_sr.merge(Landsat5_sr).merge(Landsat7_sr).map(setDate)

    return Landsat_sr_coll


def main(argv):

    # # Set path row
    # path = int(argv[1])
    # row = int(argv[2])
    # start_year = int(argv[3])
    # end_year = int(argv[4])
    # version = int(argv[5])
    #
    # pathrow = str(path).zfill(3)+str(row).zfill(3)
    # assetDir = 'projects/disalexi/example_data/LAI/LAI_'+pathrow+'_v'+str(version)+'/'
    #
    # start = str(start_year)+'-01-01'
    # end = str(end_year+1)+'-01-01'

    # Set path row
    path = 44
    row = 33
    version = 'v3'
    coll_id = 'users/cgmorton/lai/landsat/v3/'
    start = '2017-07-01'
    end = '2017-08-01'

    # Get Landsat collection
    landsat_coll = getLandsat(start, end, path, row)
    landsat_coll = landsat_coll.sort('system:time_start')

    image_id_list = landsat_coll.aggregate_array('system:id').getInfo()
    pprint.pprint(image_id_list)
    input('ENTER')

    for image_id in image_id_list:
        print(image_id)
        landsat_id = image_id.split('/')
        landsat_img = ee.Image(image_id)

        sensor = landsat_id.split('/')[-1].split('_')[0].upper()
        # sensor_dict = {'LANDSAT_5': 'LT05', 'LANDSAT_7': 'LE07', 'LANDSAT_8': 'LC08'}
        # sensor = landsat_img.get('SATELLITE').getInfo()
        # sensor = sensor_dict[sensor]

        image_proj = landsat_img.select([0]).projection().getInfo()

        # Compute LAI image
        lai_img = openet.lai.landsat.getLAIImage(landsat_img, sensor, nonveg=1)

        # Set properties
        properties = {
            'date_ingested': datetime.datetime.today().strftime('%Y-%m-%d'),
            'lai_version': openet.lai.__version__,
        }
        lai_img = lai_img.set(properties)

        # Start export
        task = ee.batch.Export.image.toAsset(
            image=lai_img,
            description='{}_LAI_{}'.format(landsat_id, version),
            assetId=coll_id + landsat_id.lower(),
            crs=image_proj['crs'],
            crsTransform=str(image_proj['transform']),
        )
        task.start()
        print('  '.format(task.id))


if __name__ == "__main__":
    main(sys.argv)
