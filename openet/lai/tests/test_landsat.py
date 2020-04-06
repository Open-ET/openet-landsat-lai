import datetime
import logging
import pprint

import ee
import pytest

import openet.lai.landsat
import openet.lai.utils as utils

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

LANDSAT8_IMAGE_ID = 'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716'
TEST_POINT = (-121.5265, 38.7399)


def test_ee_init():
    assert ee.Number(1).getInfo() == 1


def test_setDate():
    # Set 'date' property with EE format 'YYYYMMdd'
    input_date = '2017-07-16'
    input_img = ee.Image.constant(2) \
        .set({'system:time_start': ee.Date(input_date).millis()})
    output = openet.lai.landsat.setDate(input_img).get('date').getInfo()
    assert output == input_date.replace('-', '')


@pytest.mark.parametrize(
    'image_id',
    [
        ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716'],
        ['LANDSAT/LE07/C01/T1_SR/LE07_044033_20170724'],
        # ['LANDSAT/LT05/C01/T1_SR/LT05_044033_201107XX'],
        # # This will fail for Landsat TOA images
        # ['LANDSAT/LC08/C01/T1_TOA/LC08_044033_20170716'],
        # ['LANDSAT/LE07/C01/T1_TOA/LE07_044033_20170724'],
        # ['LANDSAT/LT05/C01/T1_TOA/LT05_044033_201107XX'],
    ]
)
def test_renameLandsat(image_id):
    output = openet.lai.landsat.renameLandsat(ee.Image(image_id))\
        .bandNames().getInfo()
    assert set(output) == {'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa'}


@pytest.mark.parametrize(
    "expected, img_value, start, end, newName",
    [
        # CM - Because of how the getQABits function is written and the
        #   structure of the pixel_qa band we don't need to check all of these
        [1, '0000000000000010', 1, 1, 'Clear'],       # Clear
        # [1, '0000000000000100', 2, 2, 'Water'],       # Water
        # [1, '0000000000001000', 3, 3, 'Shadow'],      # Cloud Shadow
        # [1, '0000000000010000', 4, 4, 'Snow'],        # Snow
        # [1, '0000000000100000', 5, 5, 'Cloud'],       # Cloud
        [1, '0000000001100000', 6, 7, 'Confidence'],  # Cloud Confidence
        [2, '0000000010100000', 6, 7, 'Confidence'],
        [3, '0000000011100000', 6, 7, 'Confidence'],
    ]
)
def test_getQABits(expected, img_value, start, end, newName):
    input_img = ee.Image.constant(int(img_value, 2))
    output = utils.constant_image_value(
        openet.lai.landsat.getQABits(input_img, start, end, newName))
    assert output[newName] == expected


@pytest.mark.parametrize(
    "pixel_qa",
    [
        '0000000101000100',  # 324 - Water
    ]
)
def test_maskLST_masked_constant_values(pixel_qa):
    input_img = ee.Image([
        ee.Image.constant(0.6).rename(['nir']),
        ee.Image.constant(int(pixel_qa, 2)).int().rename(['pixel_qa']),
    ])
    output = utils.constant_image_value(openet.lai.landsat.maskLST(input_img))
    assert output['nir'] is None


@pytest.mark.parametrize(
    "pixel_qa",
    [
        '0000000000000010',  # 66 - Clear
        '0000000001000010',  # 66 - Clear
    ]
)
def test_maskLST_nonmasked_constant_values(pixel_qa):
    input_img = ee.Image([
        ee.Image.constant(0.6).rename(['nir']),
        ee.Image.constant(int(pixel_qa, 2)).int().rename(['pixel_qa']),
    ])
    output = utils.constant_image_value(openet.lai.landsat.maskLST(input_img))
    assert output['nir'] is not None


@pytest.mark.parametrize(
    "image_id, xy",
    [
        [LANDSAT8_IMAGE_ID, [-121.1445, 38.7205]],  # Folsom Lake
        ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.15610, 38.87292]],  # Water
        ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.14846, 38.87185]],  # Cloud
        ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.17361, 38.86911]],  # Snow
        ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.14546, 38.86904]],  # Shadow
    ]
)
def test_maskLST_masked_point_values(image_id, xy):
    # Check that all non-clear pixels are masked
    input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
    output = utils.point_image_value(openet.lai.landsat.maskLST(input_img), xy=xy)
    pprint.pprint(utils.point_image_value(input_img, xy=xy))
    assert output['nir'] is None


@pytest.mark.parametrize("image_id, xy", [[LANDSAT8_IMAGE_ID, TEST_POINT]])
def test_maskLST_nonmasked_point_values(image_id, xy):
    input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
    output = utils.point_image_value(openet.lai.landsat.maskLST(input_img), xy=xy)
    assert output['nir'] > 0


def test_getVIs_bands():
    # Check that the expected bands are added to the output image
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    input_bands = input_img.bandNames().getInfo()
    output = openet.lai.landsat.getVIs(input_img).bandNames().getInfo()
    assert set(output) == set(input_bands) | {'SR', 'NDVI', 'EVI', 'NDWI'}


@pytest.mark.parametrize(
    "blue, red, nir, swir1, sr, ndvi, evi, ndwi",
    [
        # Raw scaled (0-10000) SR values
        [1000, 2000, 8000, 3000, 4.0, 0.6, 0.6666, 0.4545],
        # Unscaled (0-1) SR values
        # [0.1, 0.2, 0.8, 0.3, 0.6, 0.6, 0.6, 0.6],
    ]
)
def test_getVIs_constant_values(blue, red, nir, swir1, sr, ndvi, evi, ndwi,
                                tol=0.01):
    # Check that the VI calculations are valid using constant images
    input_img = ee.Image.constant([blue, red, nir, swir1])\
        .rename(['blue', 'red', 'nir', 'swir1'])
    output = utils.constant_image_value(
        openet.lai.landsat.getVIs(input_img))
    pprint.pprint(output)
    # evi = (2.5 * (nir - red)) / (nir + 6 * red - 7.5 * blue + 1)
    assert abs(output['SR'] - sr) <= tol
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['EVI'] - evi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol


@pytest.mark.parametrize(
    "image_id, xy, sr, ndvi, ndwi, evi",
    [
        [LANDSAT8_IMAGE_ID, TEST_POINT, 14.9227, 0.8744, 0.5043, 0.5301],
    ]
)
def test_getVIs_point_values(image_id, xy, sr, ndvi, evi, ndwi, tol=0.0001):
    # Check that the VI calculations are valid at specific points
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    pprint.pprint(utils.point_image_value(input_img, xy=xy))
    output = utils.point_image_value(openet.lai.landsat.getVIs(input_img), xy=xy)
    assert abs(output['SR'] - sr) <= tol
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['EVI'] - evi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol


def test_getTrainImg_bands():
    input_bands = {'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa'}
    vi_bands = {'SR', 'NDVI', 'NDWI', 'EVI'}
    training_bands = {'nlcd', 'lon', 'lat', 'path', 'row', 'sun_zenith',
                      'sun_azimuth', 'biome2'}
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    output_bands = openet.lai.landsat.getTrainImg(input_img).bandNames().getInfo()
    assert (input_bands | vi_bands | training_bands) == set(output_bands)


# def test_getTrainImg_bands():
#     input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
#     output = openet.lai.landsat.getTrainImg(input_img).getInfo()

    # # nlcd2011 = ee.Image('USGS/NLCD/NLCD2011')
    #
    # # NLCD processing
    # year = ee.Date(image.get('system:time_start')).get('year')
    # nlcd_all = ee.Image('USGS/NLCD/NLCD2001').select(['landcover'], ['landcover_2001']) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2004').select(['landcover'], ['landcover_2004'])) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2006').select(['landcover'], ['landcover_2006'])) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2008').select(['landcover'], ['landcover_2008'])) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2011').select(['landcover'], ['landcover_2011'])) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2013').select(['landcover'], ['landcover_2013'])) \
    #     .addBands(ee.Image('USGS/NLCD/NLCD2016').select(['landcover'], ['landcover_2016']))
    #
    # nlcd_dict = {
    #     '1997':0, '1998':0, '1999':0, '2000':0, '2001':0, '2002':0,
    #     '2003':1, '2004':1, '2005':1,
    #     '2006':2, '2007':2,
    #     '2008':3, '2009':3,
    #     '2010':4, '2011':4, '2012':4, '2013':5,
    #     '2014':5,
    #     '2015':6, '2016':6, '2017':6, '2018':6, '2019':6, '2020':6,
    # }
    # nlcd_dict = ee.Dictionary(nlcd_dict)
    #
    # # CGM - This nlcd image is not being used after this, should it be?
    # # Get NLCD for corresponding year
    # nlcd = nlcd_all.select([nlcd_dict.get(ee.Number(year).format('%d'))])
    #
    # # add bands
    # image = maskLST(image)
    # image = getVIs(image)
    # mask_prev = image.select([0]).mask()
    #
    # # TODO: Try applying these values to a mask image instead of clipping
    # # add other bands
    # image = image.addBands(nlcd.select([0], ['nlcd']).clip(image.geometry()))
    # image = image.addBands(
    #     ee.Image.pixelLonLat().select(['longitude', 'latitude'], ['lon', 'lat']).clip(image.geometry()))
    # # image = image.addBands(ee.Image.constant(ft.get('year')).select([0], ['year']).clip(image.geometry()))
    # image = image.addBands(
    #     ee.Image.constant(ee.Number(image.get('WRS_PATH'))).select([0], ['path'])).clip(image.geometry())
    # image = image.addBands(
    #     ee.Image.constant(ee.Number(image.get('WRS_ROW'))).select([0], ['row'])).clip(image.geometry())
    # image = image.addBands(
    #     ee.Image.constant(ee.Number(image.get('SOLAR_ZENITH_ANGLE'))).select([0], ['sun_zenith'])).clip(image.geometry())
    # image = image.addBands(
    #     ee.Image.constant(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE'))).select([0], ['sun_azimuth'])).clip(image.geometry())
    # # image = image.addBands(ee.Image.constant(ee.Number(ft.get('doy'))).select([0], ['DOY'])).clip(image.geometry())
    #
    # # add biome band
    # fromList = [21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    # toList = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8]
    # image = image.addBands(image.select('nlcd').remap(fromList, toList).rename('biome2')) \
    #     .updateMask(mask_prev)
    #
    # return image


# def test_trainRF():
#     output = openet.lai.landsat.trainRF(samples, features, classProperty).getInfo()
#     assert output
#
#     # rfRegressor = ee.Classifier.randomForest(numberOfTrees=100,
#     #                                          minLeafPopulation=20,
#     #                                          variablesPerSplit=8) \
#     #                             .setOutputMode('REGRESSION') \
#     #                             .train(features=samples,
#     #                                    classProperty='MCD_LAI',
#     #                                    inputProperties=features)
#     #
#     # return rfRegressor
#
#
# def test_getRFModel():
#     output = openet.lai.landsat.getRFModel(sensor, biome, threshold).getInfo()
#     assert output
#
#     # assetDir = 'users/yanghui/OpenET/LAI_US/train_samples/'
#     # # change training sample
#     # filename = 'LAI_train_samples_' + sensor + '_v10_1_labeled'
#     # train = ee.FeatureCollection(assetDir + filename)
#     #
#     # # filter sat_flag
#     # train = train.filterMetadata('sat_flag', 'equals', 'correct')
#     #
#     # # get train sample by biome
#     # train = ee.Algorithms.If(ee.Number(biome).eq(0), train,
#     #                          train.filterMetadata('biome2', 'equals', biome))
#     #
#     # # percentage of saturated samples to use
#     # train = ee.FeatureCollection(train)
#     #
#     # """
#     # train = train.filterMetadata('mcd_qa', 'equals',1) \
#     #              .filterMetadata('random', 'less_than',threshold) \
#     #              .merge(train.filterMetadata('mcd_qa', 'equals',0))
#   	# """
#     #
#     # # train
#     # features = ['red', 'green', 'nir', 'swir1', 'lat', 'lon', 'NDVI', 'NDWI',
#     #             'sun_zenith', 'sun_azimuth']
#     # rf = trainRF(train, features, 'MCD_LAI')
#     # return rf
#
#
# def test_getLAIforBiome(image, biome, rf_models):
#     output = openet.lai.landsat.getLAIforBiome(image, biome, rf_models).getInfo()
#     assert output
#
#     # biome_band = image.select('biome2')
#     # model = rf_models.get(ee.String(biome))
#     # lai = image.updateMask(biome_band.select('biome2').eq(ee.Image.constant(ee.Number.parse(biome)))) \
#     #            .classify(model, 'LAI')
#     # return lai
#
#
# def test_getLAI():
#     output = openet.lai.landsat.getLAI(image, rf_models).getInfo()
#     assert output
#
#     # # Add necessary bands to image
#     # image = getTrainImg(image)
#     # mask_prev = image.select([0]).mask()
#     #
#     # # Apply regressor for each biome
#     # biomes = rf_models.keys()
#     # lai_coll = ee.List(biomes).map(lambda b: getLAIforBiome(image, b, rf_models))
#     # lai_coll = ee.ImageCollection(lai_coll)
#     #
#     # # combine lai of all biomes
#     # lai_img = ee.Image(lai_coll.mean().copyProperties(image)).updateMask(mask_prev) \
#     #     .set('system:time_start',image.get('system:time_start'))
#     #
#     # return lai_img
#
#
# def test_trainModels():
#     output = openet.lai.landsat.trainModels(sensor, nonveg).getInfo()
#     assert output
#
#     # biomes = [1, 2, 3, 4, 5, 6, 7, 8]
#     # biomes_str = ['1', '2', '3', '4', '5', '6', '7', '8']
#     #
#     # if nonveg:
#     #     biomes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     #     biomes_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
#     #
#     # # Get models for each biome
#     # rf_models = ee.List(biomes).map(lambda biome: getRFModel(sensor, biome, 1))
#     # rf_models = ee.Dictionary.fromLists(biomes_str, rf_models)
#     #
#     # return rf_models
#
#
# def test_getLAIImage():
#     output = openet.lai.landsat.getLAIImage(image, sensor, nonveg=1).getInfo()
#     assert output
#
#     # # train random forest models
#     # # image = renameLandsat(image)
#     # rf_models = trainModels(sensor, nonveg)
#     # laiImg = getLAI(image, rf_models)
#     # return ee.Image(laiImg).clip(image.geometry())
