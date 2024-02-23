import logging

import ee
import pytest

import openet.lai
import openet.core.utils as utils

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

TEST_IMAGE_ID = 'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716'
TEST_SENSOR = 'LC08'
TEST_POINT = (-121.5265, 38.7399)
DEFAULT_BANDS = ['green', 'red', 'nir', 'swir1', 'qa']
DEFAULT_VALUES = [0.1, 0.1, 0.3, 0.1, 1]


def test_ee_init():
    assert ee.Number(1).getInfo() == 1


def test_add_vi_bands_bandnames():
    # Check that the expected bands are added to the output image
    input_img = openet.lai.Landsat(image_id=TEST_IMAGE_ID).image
    output = utils.get_info(openet.lai.model.add_vi_bands(input_img).bandNames())
    assert set(output) == set(DEFAULT_BANDS) | {'NDVI', 'NDWI'}


@pytest.mark.parametrize(
    "green, red, nir, swir1, ndvi, ndwi, evi, sr",
    [
        # Raw scaled (0-10000) SR values
        [1000, 2000, 8000, 3000, 0.6, 0.4545, 4.0, 0.6666],
        # Unscaled (0-1) SR values
        # [0.1, 0.2, 0.8, 0.3, 0.6, 0.4545, 4.0, 0.6666],
    ]
)
def test_add_vi_bands_constant_values(green, red, nir, swir1, ndvi, ndwi, evi, sr, tol=0.01):
    # Check that the VI calculations are valid using constant images
    input_img = ee.Image.constant([green, red, nir, swir1])\
        .rename(['green', 'red', 'nir', 'swir1'])
    output = utils.constant_image_value(openet.lai.model.add_vi_bands(input_img))
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol


@pytest.mark.parametrize(
    "image_id, xy, ndvi, ndwi, evi, sr",
    [
        [
            'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
            TEST_POINT, 0.8472, 0.4851, 0.5301, 14.9227
        ],
        [
            # Folsom Lake
            'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
            [-121.1445, 38.7205], -1.3085, 1.9329, 0, 0
        ],
        # [
        #     'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716',
        #     TEST_POINT, 0.8744, 0.5043, 0.5301, 14.9227
        # ],
        # [
        #     # Folsom Lake
        #     'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716',
        #     [-121.1445, 38.7205], -0.5294, 0.4328, 0, 0
        # ],
    ]
)
def test_add_vi_bands_point_values(image_id, xy, ndvi, ndwi, evi, sr, tol=0.0001):
    # Check that the VI calculations are valid at specific points
    output = utils.point_image_value(
        openet.lai.model.add_vi_bands(openet.lai.Landsat(image_id=image_id).image), xy=xy)
    assert abs(output['NDVI'] - ndvi) <= tol
    assert abs(output['NDWI'] - ndwi) <= tol


def test_get_train_img_bandnames():
    # Both the VI and training bands get added in getTrainImg
    vi_bands = {'NDVI', 'NDWI'}
    training_bands = {'biome2', 'lon', 'lat', 'sun_zenith', 'sun_azimuth', 'mask'}
    target_bands = set(DEFAULT_BANDS) | vi_bands | training_bands
    input_img = openet.lai.Landsat(image_id=TEST_IMAGE_ID).image
    output_bands = utils.get_info(openet.lai.model.get_train_img(input_img).bandNames())
    assert target_bands == set(list(output_bands))


@pytest.mark.parametrize(
    "date, nlcd_band",
    [
        # CM - We don't really need to test all of these
        ['2003-01-01', '2004'],
        ['2007-01-01', '2006'],
        ['2008-01-01', '2008'],
        ['2012-01-01', '2011'],
        # Check if the transition at the new year is handled
        ['2014-12-31', '2013'],
        ['2015-01-01', '2016'],
        # Check the supported start/end years
        ['2017-01-01', '2016'],
        ['2018-01-01', '2019'],
        ['2019-01-01', '2019'],
        ['2020-01-01', '2019'],
        ['2021-01-01', '2021'],
        ['2022-01-01', '2021'],
        ['2023-01-01', '2021'],
        # # What should happen for years outside the supported range
        # # Currently this will raise a EEException
        # # (about the dictionary not having the correct key)
        pytest.param('1998-01-01', '2001', marks=pytest.mark.xfail),
        # pytest.param('2023-01-01', '2019', marks=pytest.mark.xfail),
    ]
)
def test_get_train_img_nlcd_year(date, nlcd_band):
    input_img = openet.lai.Landsat(image_id=TEST_IMAGE_ID).image\
        .set({'system:time_start': ee.Date(date).millis()})
    output = utils.get_info(openet.lai.model.get_train_img(input_img).get('nlcd_year'))
    assert output == nlcd_band


@pytest.mark.parametrize(
    "image_id, xy, azimuth, zenith",
    [
        [TEST_IMAGE_ID, TEST_POINT, 127.089134, 25.720642],
    ]
)
def test_get_train_img_property_values(image_id, xy, azimuth, zenith):
    input_img = openet.lai.Landsat(image_id=image_id).image
    output = utils.point_image_value(openet.lai.model.get_train_img(input_img), xy=xy)
    assert abs(output['lon'] - xy[0]) <= 0.0001
    assert abs(output['lat'] - xy[1]) <= 0.0001
    assert abs(output['sun_azimuth'] - azimuth) <= 0.0001
    assert abs(output['sun_zenith'] - zenith) <= 0.0001
    # assert output['sun_azimuth'] == utils.get_info(input_img.get('SOLAR_AZIMUTH_ANGLE'))
    # assert output['sun_zenith'] == utils.get_info(input_img.get('SOLAR_ZENITH_ANGLE'))


@pytest.mark.parametrize(
    "image_id, xy, nlcd, biome2",
    [
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', TEST_POINT, 81, 6],
        # Folsom Lake
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], 11, 0],
        ['LANDSAT/LC08/C02/T1_L2/LC08_042034_20170718', [-118.51162, 36.55814], 12, 0],
    ]
)
def test_get_train_img_biome_point_values(image_id, xy, nlcd, biome2):
    output = utils.point_image_value(
        openet.lai.model.get_train_img(openet.lai.Landsat(image_id=image_id).image), xy=xy)
    assert output['biome2'] == biome2


# DEADBEEF - This test is only needed if NLCD 11 and 12 aren't mapped to biome 0
# @pytest.mark.parametrize(
#     "image_id, xy, nlcd, biome2",
#     [
#         ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], 11, 0],
#         ['LANDSAT/LC08/C02/T1_L2/LC08_042034_20170718', [-118.51162, 36.55814], 12, 0],
#     ]
# )
# def test_get_train_img_biome_nodata(image_id, xy, nlcd, biome2):
#     input_img = openet.lai.Landsat(image_id=image_id)
#     output = utils.point_image_value(openet.lai.model.get_train_img(input_img), xy=xy)
#     assert output['biome2'] is None


# CM - How do we test if the classifier is correct?
#   Currently it is only testing if something is returned
# def test_get_rf_model(sensor='LC08', biome=0):
#     output = utils.get_info(openet.lai.model.get_rf_model(sensor, biome))
#     assert output


@pytest.mark.parametrize(
    "sensor, biome",
    [
        ['LT05', 0],
        ['LE07', 0],
        ['LC08', 0],
        ['LC09', 0],
    ]
)
def test_getRFModel_sensor(sensor, biome):
    # For now just test that something is returned for each sensor option
    output = utils.get_info(openet.lai.model.get_rf_model(sensor, biome))
    assert output


# CM - How do we test if the biome parameter is working?
# def test_get_rf_model_biome(sensor, biome):
#     output = utils.get_info(openet.lai.model.get_rf_model(sensor, biome))
#     assert output


# CM - How should we test if an unsupported sensor value is passed
#   There are no Landsat 4 features in the collection
#   We could try writing the feature collection size as a property?
# def test_get_rf_model_sensor_unsupported(sensor='LT04', biome=0):
#     output = utils.get_info(openet.lai.model.get_rf_model(sensor, biome))
#     pprint.pprint(output)
#     assert False


@pytest.mark.parametrize(
    "image_id, xy, biome, expected",
    [
        # Test values for LAI_train_sample_unsat_v10_1_final, numberOfTrees=100,
        #   minLeafPopulation=50, variablesPerSplit=5
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', TEST_POINT, 6, 3.63011],              # NLCD 82
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.52650, 38.73990], 6, 3.63011],  # NLCD 82
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.14450, 38.72050], 0, 0.78379],  # NLCD 11 (Folsom Lake)
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-120.81146, 38.82813], 1, 1.27416],  # NLCD 41
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-120.77515, 38.81689], 2, 3.29797],  # NLCD 42
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-120.76897, 38.82505], 3, 3.09790],  # NLCD 43
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-120.79558, 38.81790], 4, 1.92601],  # NLCD 52
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.42478, 38.73954], 5, 0.46389],  # NLCD 71
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.43285, 38.73834], 5, 0.42424],  # NLCD 81
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.25980, 38.89904], 7, 2.72870],  # NLCD 90
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-120.63588, 38.90885], 8, 2.80465],  # NLCD 95
        # # Test values for LAI_train_sample_unsat_v10_1_final, numberOfTrees=100,
        # #   minLeafPopulation=50, variablesPerSplit=5
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', TEST_POINT, 6, 3.644629],              # NLCD 82
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-121.52650, 38.73990], 6, 3.644629],  # NLCD 82
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-121.14450, 38.72050], 0, 0.769142],  # NLCD 11 (Folsom Lake)
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-120.81146, 38.82813], 1, 1.241524],  # NLCD 41
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-120.77515, 38.81689], 2, 3.320077],  # NLCD 42
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-120.76897, 38.82505], 3, 3.102253],  # NLCD 43
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-120.79558, 38.81790], 4, 1.895867],  # NLCD 52
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-121.42478, 38.73954], 5, 0.466429],  # NLCD 71
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-121.43285, 38.73834], 5, 0.426706],  # NLCD 81
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-121.25980, 38.89904], 7, 2.751934],  # NLCD 90
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', [-120.63588, 38.90885], 8, 2.774822],  # NLCD 95
    ]
)
def test_get_lai_for_biome_point_values(image_id, xy, biome, expected, tol=0.0001):
    training_img = openet.lai.model.get_train_img(openet.lai.Landsat(image_id=image_id).image)
    sensor = image_id.split('/')[-1][:4]
    rf_model = openet.lai.model.get_rf_model(sensor, biome)
    output = utils.point_image_value(
        openet.lai.model.get_lai_for_biome(training_img, biome, rf_model), xy=xy)
    assert abs(output['LAI'] - expected) <= tol


def test_get_lai_image_band_name():
    input_img = openet.lai.Landsat(image_id=TEST_IMAGE_ID).image
    output = utils.get_info(
        openet.lai.model.get_lai_image(input_img, TEST_SENSOR, nonveg=1).bandNames())
    assert set(output) == {'LAI', 'QA'}


@pytest.mark.parametrize(
    "image_id, xy, expected",
    [
        # Test values for LAI_train_sample_unsat_v10_1_final, numberOfTrees=100,
        #   minLeafPopulation=50, variablesPerSplit=5
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', TEST_POINT, 3.630114],
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', TEST_POINT, 3.644629],
        # Water - Should it be 0 or masked?
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], 0],
        # ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], None],
    ]
)
def test_getLAIImage_point_values(image_id, xy, expected, tol=0.0001):
    input_img = openet.lai.Landsat(image_id=image_id).image
    output_img = openet.lai.model.get_lai_image(input_img, sensor=image_id.split('/')[1], nonveg=1)
    output = utils.point_image_value(output_img, xy=xy)
    if expected is None:
        assert output['LAI'] is None
    else:
        assert abs(output['LAI'] - expected) <= tol


# CGM - Until functions are moved into Model class, test Model after functions
def test_Model_init():
    input_img = ee.Image.constant(DEFAULT_VALUES).rename(DEFAULT_BANDS)
    image_obj = openet.lai.Model(image=input_img, sensor='LC08')
    assert set(utils.get_info(image_obj.image.bandNames())) == set(DEFAULT_BANDS)
    assert image_obj.sensor == 'LC08'


def test_Model_image_type_exception():
    with pytest.raises(Exception):
        openet.lai.Model(image=TEST_IMAGE_ID, sensor='DEADBEEF')


def test_Model_sensor_exception():
    with pytest.raises(Exception):
        input_img = ee.Image.constant(DEFAULT_VALUES).rename(DEFAULT_BANDS)
        openet.lai.Model(image=input_img, sensor='DEADBEEF')


@pytest.mark.parametrize(
    "image_id, xy, expected",
    [
        # Test values for LAI_train_sample_unsat_v10_1_final, numberOfTrees=100,
        #   minLeafPopulation=50, variablesPerSplit=5
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', TEST_POINT, 3.63011],
        # ['LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716', TEST_POINT, 3.644629],
        # Water - Should it be 0 or masked?
        ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], 0],
        # ['LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716', [-121.1445, 38.7205], None],
    ]
)
def test_Model_lai_point_values(image_id, xy, expected, tol=0.0001):
    input_img = openet.lai.Landsat(image_id=image_id).image
    output_img = openet.lai.Model(input_img, sensor=image_id.split('/')[1]).lai(nonveg=1)
    output = utils.point_image_value(output_img, xy=xy)
    if expected is None:
        assert output['LAI'] is None
    else:
        assert abs(output['LAI'] - expected) <= tol


# TODO: Add a test for nonveg=0
