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


@pytest.mark.parametrize(
    'image_id',
    [
        [LANDSAT8_IMAGE_ID],
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


def test_getVIs_bands():
    # Check that the expected bands are added to the output image
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    input_bands = input_img.bandNames().getInfo()
    output = openet.lai.landsat.getVIs(input_img).bandNames().getInfo()
    assert set(output) == set(input_bands) | {'NDVI', 'NDWI'}
    # CM - SR and EVI are not used in the classification, commenting out for now
    # assert set(output) == set(input_bands) | {'SR', 'NDVI', 'EVI', 'NDWI'}


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
    output = utils.constant_image_value(openet.lai.landsat.getVIs(input_img))
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol

    # CM - SR and EVI are not used in the classification, commenting out for now
    # assert abs(output['EVI'] - evi) <= tol
    # assert abs(output['SR'] - sr) <= tol


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
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol

    # CM - SR and EVI are not used in the classification, commenting out for now
    # assert abs(output['EVI'] - evi) <= tol
    # assert abs(output['SR'] - sr) <= tol


def test_getTrainImg_bands():
    input_bands = {'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa'}
    # Both the VI and training bands get added in getTrainImg
    vi_bands = {'NDVI', 'NDWI'}
    training_bands = {'biome2', 'lon', 'lat', 'sun_zenith', 'sun_azimuth', 'mask'}
    # CM - SR and EVI are not used in the classification, commenting out for now
    # vi_bands = {'SR', 'NDVI', 'NDWI', 'EVI'}
    # training_bands = {'nlcd', 'lon', 'lat', 'path', 'row', 'sun_zenith',
    #                   'sun_azimuth', 'biome2'}
    target_bands = input_bands | vi_bands | training_bands
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    output_bands = openet.lai.landsat.getTrainImg(input_img).bandNames().getInfo()
    assert target_bands == set(list(output_bands))


@pytest.mark.parametrize(
    "date, nlcd_band",
    [
        # CM - We don't really need to test all of these
        ['2003-01-01', 'landcover_2004'],
        ['2007-01-01', 'landcover_2006'],
        ['2008-01-01', 'landcover_2008'],
        ['2012-01-01', 'landcover_2011'],
        # Check if the transition at the new year is handled
        ['2014-12-31', 'landcover_2013'],
        ['2015-01-01', 'landcover_2016'],
        # Check the supported start/end years
        ['1997-01-01', 'landcover_2001'],
        ['2020-01-01', 'landcover_2016'],
        # # What should happen for years outside the supported range
        # # Currently this will raise a EEException
        # # (about the dictionary not having the correct key)
        # ['1996-01-01', 'landcover_1997'],
        # ['2021-01-01', 'landcover_2016'],
    ]
)
def test_getTrainImg_nlcd_year(date, nlcd_band):
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID)) \
        .set({'system:time_start': ee.Date(date).millis()})
    output = openet.lai.landsat.getTrainImg(input_img).get('nlcd_year').getInfo()
    assert output == nlcd_band


def test_getTrainImg_property_values():
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    output = utils.point_image_value(
        openet.lai.landsat.getTrainImg(input_img), xy=TEST_POINT)
    assert abs(output['lon'] - TEST_POINT[0]) <= 0.0001
    assert abs(output['lat'] - TEST_POINT[1]) <= 0.0001

    # CM - Hardcoding solar angles for now to avoid extra getInfo calls
    assert output['sun_azimuth'] == 127.089134
    assert output['sun_zenith'] == 25.720642
    # assert output['sun_azimuth'] == input_img.get('SOLAR_AZIMUTH_ANGLE').getInfo()
    # assert output['sun_zenith'] == input_img.get('SOLAR_ZENITH_ANGLE').getInfo()

    # CM - path and row are not used in the classification, commenting out for now
    # scene_id = LANDSAT8_IMAGE_ID.split('/')[-1]
    # assert output['path'] == int(scene_id.split('_')[1][0:3])
    # assert output['row'] == int(scene_id.split('_')[1][3:6])


@pytest.mark.parametrize(
    "image_id, xy, nlcd, biome2",
    [
        [LANDSAT8_IMAGE_ID, TEST_POINT, 81, 6],
        # CM - These are getting set to None right now
        [LANDSAT8_IMAGE_ID, [-121.1445, 38.7205], 11, 0],   # Folsom Lake
        ['LANDSAT/LC08/C01/T1_SR/LC08_042034_20170718',
         [-118.51162, 36.55814], 12, 0],
    ]
)
def test_getTrainImg_biome_point_values(image_id, xy, nlcd, biome2):
    input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
    output = utils.point_image_value(
        openet.lai.landsat.getTrainImg(input_img), xy=xy)
    assert output['biome2'] == biome2


# DEADBEEF - This test is only needed if NLCD 11 and 12 aren't mapped to biome 0
# @pytest.mark.parametrize(
#     "image_id, xy, nlcd, biome2",
#     [
#         [LANDSAT8_IMAGE_ID, [-121.1445, 38.7205], 11, 0],   # Folsom Lake
#         ['LANDSAT/LC08/C01/T1_SR/LC08_042034_20170718',
#          [-118.51162, 36.55814], 12, 0],
#     ]
# )
# def test_getTrainImg_biome_nodata(image_id, xy, nlcd, biome2):
#     input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
#     output = utils.point_image_value(
#         openet.lai.landsat.getTrainImg(input_img), xy=xy)
#     assert output['biome2'] is None


def test_trainRF():
    # CM - Get examples of "samples":
    # sensor = 'LC08'
    # assetDir = 'users/yanghui/OpenET/LAI_US/train_samples/'
    # filename = 'LAI_train_samples_' + sensor + '_v10_1_labeled'
    # train = ee.FeatureCollection(assetDir + filename)
    # pprint.pprint(train.first().getInfo())
    # # train = train.filterMetadata('sat_flag', 'equals', 'correct')
    # # train = ee.Algorithms.If(ee.Number(biome).eq(0), train,
    # #                          train.filterMetadata('biome2', 'equals', biome))
    # # samples = ee.FeatureCollection(train)

    # CM - What needs to be in samples to test if trainRF is working?
    samples = ee.FeatureCollection([
        ee.Feature(None, {'MCD_LAI': 1.0}),
    ])
    features = ['red', 'green', 'nir', 'swir1', 'lat', 'lon', 'NDVI', 'NDWI',
                'sun_zenith', 'sun_azimuth']
    output = openet.lai.landsat.trainRF(
        samples, features, classProperty='MCD_LAI').getInfo()
    assert output


# CM - How do we test if the classifier is correct?
#   Currently it is only testing if something is returned
# def test_getRFModel(sensor='LC08', biome=0):
#     output = openet.lai.landsat.getRFModel(sensor, biome).getInfo()
#     assert output


@pytest.mark.parametrize(
    "sensor, biome",
    [
        ['LC08', 0],
        ['LE07', 0],
        ['LT05', 0],
    ]
)
def test_getRFModel_sensor(sensor, biome):
    # For now just test that something is returned for each sensor option
    output = openet.lai.landsat.getRFModel(sensor, biome).getInfo()
    assert output


# CM - How do we test if the biome parameter is working?
# def test_getRFModel_biome(sensor, biome):
#     output = openet.lai.landsat.getRFModel(sensor, biome).getInfo()
#     assert output


# CM - Invalid sensor values will/should raise an EE exception
#   Landsat 4 is not currently supported
def test_getRFModel_sensor_exception(sensor='LT04', biome=0):
    with pytest.raises(ee.ee_exception.EEException) as e_info:
        openet.lai.landsat.getRFModel(sensor, biome).getInfo()


# TODO: Test that here is a separate classfier for each biome
#   A test on the sensor probably isn't needed since that is passed through
def test_trainModels():
    output = openet.lai.landsat.trainModels(sensor='LE07', nonveg=True)
    # pprint.pprint(output)
    assert type(output) is dict
    assert set(output.keys()) == set(range(0, 9))
    assert output[1].getInfo()


def test_trainModels_nonveg_true():
    output = openet.lai.landsat.trainModels(sensor='LC08', nonveg=True)
    assert 0 in output.keys()


def test_trainModels_nonveg_false():
    output = openet.lai.landsat.trainModels(sensor='LC08', nonveg=False)
    assert 0 not in output.keys()


@pytest.mark.parametrize(
    "image_id, xy, biome, expected",
    [
        # CM - Biomes are string values (from dictionary keys)
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 0, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 1, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 2, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 3, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 4, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 5, None],
        [LANDSAT8_IMAGE_ID, TEST_POINT, 6, 4.3203],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 7, None],
        # [LANDSAT8_IMAGE_ID, TEST_POINT, 8, None],
    ]
)
def test_getLAIforBiome_point_values(image_id, xy, biome, expected, tol=0.0001):
    input_img = openet.lai.landsat.getTrainImg(
        openet.lai.landsat.renameLandsat(ee.Image(image_id)))
    sensor = LANDSAT8_IMAGE_ID.split('/')[-1][:4]
    rf_model = openet.lai.landsat.trainModels(sensor, nonveg=True)[biome]
    output = utils.point_image_value(
        openet.lai.landsat.getLAIforBiome(input_img, biome, rf_model), xy=xy)
    assert abs(output['LAI'] - expected) <= tol


# def test_getLAI():
#     input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
#     sensor = LANDSAT8_IMAGE_ID.split('/')[-1][:4]
#     rf_models = openet.lai.landsat.trainModels(sensor, nonveg=True)
#     output = openet.lai.landsat.getLAI(input_img, rf_models).getInfo()
#     pprint.pprint(output)
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


def test_getLAIImage_band_name():
    input_img = openet.lai.landsat.renameLandsat(ee.Image(LANDSAT8_IMAGE_ID))
    output = openet.lai.landsat.getLAIImage(input_img, 'LC08', nonveg=1)\
        .bandNames().getInfo()
    assert set(output) == {'LAI'}


def test_getLAIImage_point_values(image_id=LANDSAT8_IMAGE_ID, xy=TEST_POINT,
                                  expected=4.3203, tol=0.0001):
    input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
    sensor = image_id.split('/')[-1][:4]
    output = utils.point_image_value(
        openet.lai.landsat.getLAIImage(input_img, sensor, nonveg=1), xy=xy)
    assert abs(output['LAI'] - expected) <= tol


# DEADBEEF - setDate function is commented out in landsat.py
# def test_setDate():
#     # Set 'date' property with EE format 'YYYYMMdd'
#     input_date = '2017-07-16'
#     input_img = ee.Image.constant(2) \
#         .set({'system:time_start': ee.Date(input_date).millis()})
#     output = openet.lai.landsat.setDate(input_img).get('date').getInfo()
#     assert output == input_date.replace('-', '')


# DEADBEEF - getQABits function is commented out in landsat.py
# @pytest.mark.parametrize(
#     "expected, img_value, start, end, newName",
#     [
#         # CM - Because of how the getQABits function is written and the
#         #   structure of the pixel_qa band we don't need to check all of these
#         [1, '0000000000000010', 1, 1, 'Clear'],       # Clear
#         # [1, '0000000000000100', 2, 2, 'Water'],       # Water
#         # [1, '0000000000001000', 3, 3, 'Shadow'],      # Cloud Shadow
#         # [1, '0000000000010000', 4, 4, 'Snow'],        # Snow
#         # [1, '0000000000100000', 5, 5, 'Cloud'],       # Cloud
#         [1, '0000000001100000', 6, 7, 'Confidence'],  # Cloud Confidence
#         [2, '0000000010100000', 6, 7, 'Confidence'],
#         [3, '0000000011100000', 6, 7, 'Confidence'],
#     ]
# )
# def test_getQABits(expected, img_value, start, end, newName):
#     input_img = ee.Image.constant(int(img_value, 2))
#     output = utils.constant_image_value(
#         openet.lai.landsat.getQABits(input_img, start, end, newName))


# DEADBEEF - maskLST function is commented out in landsat.py
# @pytest.mark.parametrize(
#     "pixel_qa",
#     [
#         '0000000101000100',  # 324 - Water
#     ]
# )
# def test_maskLST_masked_constant_values(pixel_qa):
#     input_img = ee.Image([
#         ee.Image.constant(0.6).rename(['nir']),
#         ee.Image.constant(int(pixel_qa, 2)).int().rename(['pixel_qa']),
#     ])
#     output = utils.constant_image_value(openet.lai.landsat.maskLST(input_img))
#     assert output['nir'] is None
#
#
# @pytest.mark.parametrize(
#     "pixel_qa",
#     [
#         '0000000000000010',  # 66 - Clear
#         '0000000001000010',  # 66 - Clear
#     ]
# )
# def test_maskLST_nonmasked_constant_values(pixel_qa):
#     input_img = ee.Image([
#         ee.Image.constant(0.6).rename(['nir']),
#         ee.Image.constant(int(pixel_qa, 2)).int().rename(['pixel_qa']),
#     ])
#     output = utils.constant_image_value(openet.lai.landsat.maskLST(input_img))
#     assert output['nir'] is not None
#
#
# @pytest.mark.parametrize(
#     "image_id, xy",
#     [
#         [LANDSAT8_IMAGE_ID, [-121.1445, 38.7205]],  # Folsom Lake
#         ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.15610, 38.87292]],  # Water
#         ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.14846, 38.87185]],  # Cloud
#         ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.17361, 38.86911]],  # Snow
#         ['LANDSAT/LE07/C01/T1_SR/LE07_043033_20060719', [-120.14546, 38.86904]],  # Shadow
#     ]
# )
# def test_maskLST_masked_point_values(image_id, xy):
#     # Check that all non-clear pixels are masked
#     input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
#     output = utils.point_image_value(openet.lai.landsat.maskLST(input_img), xy=xy)
#     pprint.pprint(utils.point_image_value(input_img, xy=xy))
#     assert output['nir'] is None
#
#
# @pytest.mark.parametrize("image_id, xy", [[LANDSAT8_IMAGE_ID, TEST_POINT]])
# def test_maskLST_nonmasked_point_values(image_id, xy):
#     input_img = openet.lai.landsat.renameLandsat(ee.Image(image_id))
#     output = utils.point_image_value(openet.lai.landsat.maskLST(input_img), xy=xy)
#     assert output['nir'] > 0
