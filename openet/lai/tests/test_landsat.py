import logging

import ee
import pytest

import openet.lai
import openet.core.utils as utils

logging.basicConfig(level=logging.DEBUG, format='%(message)s')

TEST_IMAGE_ID = 'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716'
TEST_SENSOR = 'LC08'
TEST_POINT = (-121.5265, 38.7399)
DEFAULT_BANDS = ['green', 'red', 'nir', 'swir1', 'pixel_qa']


def test_ee_init():
    assert ee.Number(1).getInfo() == 1


@pytest.mark.parametrize(
    'image_id',
    [
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
        # 'LANDSAT/LE07/C02/T1_L2/LE07_044033_20170724',
        # 'LANDSAT/LT05/C02/T1_L2/LT05_044033_20110716',
        # 'LANDSAT/LT04/C02/T1_L2/LT04_044033_19830812',
    ]
)
def test_Landsat_C02_SR_band_names(image_id):
    output = openet.lai.Landsat_C02_SR(image_id).image.bandNames().getInfo()
    assert set(output) == set(DEFAULT_BANDS)


def test_Landsat_C02_SR_image_properties():
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716'
    output = openet.lai.Landsat_C02_SR(image_id).image.getInfo()
    assert output['properties']['system:time_start']
    assert output['properties']['SOLAR_ZENITH_ANGLE']
    assert output['properties']['SOLAR_AZIMUTH_ANGLE']


# CGM - The C02 SR images are being scaled to match the C01 SR
def test_Landsat_C02_SR_scaling():
    image_id = 'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716'
    output = utils.point_image_value(
        openet.lai.Landsat_C02_SR(image_id).image, xy=TEST_POINT)
    assert output['nir'] > 1000


# CGM - sensor is not currently being set as a class property
# def test_Landsat_C02_SR_sensor():
#     sensor = openet.lai.Landsat_C02_SR(TEST_IMAGE_ID).sensor
#     assert sensor == TEST_IMAGE_ID.split('/')[1]


@pytest.mark.parametrize(
    'image_id',
    [
        'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716',
        'LANDSAT/LE07/C01/T1_SR/LE07_044033_20170724',
        'LANDSAT/LT05/C01/T1_SR/LT05_044033_20110716',
        # Landsat 4 is not currently supported
        # 'LANDSAT/LT04/C01/T1_SR/LT04_044033_19830812',
    ]
)
def test_Landsat_C01_SR_band_names(image_id):
    output = openet.lai.Landsat_C01_SR(image_id).image.bandNames().getInfo()
    assert set(output) == set(DEFAULT_BANDS)


def test_Landsat_C01_SR_image_properties():
    image_id = 'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716'
    output = openet.lai.Landsat_C01_SR(image_id).image.getInfo()
    assert output['properties']['system:time_start']
    assert output['properties']['SOLAR_ZENITH_ANGLE']
    assert output['properties']['SOLAR_AZIMUTH_ANGLE']


def test_Landsat_C01_SR_scaling():
    image_id = 'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716'
    output = utils.point_image_value(
        openet.lai.Landsat_C01_SR(image_id).image, xy=TEST_POINT)
    assert output['nir'] > 1000


# CGM - sensor is not currently being set as a class property
# def test_Landsat_C01_SR_sensor():
#     sensor = openet.lai.Landsat_C01_SR(TEST_IMAGE_ID).sensor
#     assert sensor == TEST_IMAGE_ID.split('/')[1]


@pytest.mark.parametrize(
    'image_id',
    [
        'LANDSAT/LC08/C01/T1_TOA/LC08_044033_20170716',
        'LANDSAT/LE07/C01/T1_TOA/LE07_044033_20170724',
        'LANDSAT/LT05/C01/T1_TOA/LT05_044033_20110716',
        # Landsat 4 is not currently supported
        # 'LANDSAT/LT04/C01/T1_TOA/LT04_044033_19830812',
    ]
)
def test_Landsat_C01_TOA_band_names(image_id):
    output = openet.lai.Landsat_C01_TOA(image_id).image.bandNames().getInfo()
    assert set(output) == set(DEFAULT_BANDS)


def test_Landsat_C01_TOA_image_properties():
    image_id = 'LANDSAT/LC08/C01/T1_TOA/LC08_044033_20170716'
    output = openet.lai.Landsat_C01_TOA(image_id).image.getInfo()
    assert output['properties']['system:time_start']
    assert output['properties']['SOLAR_ZENITH_ANGLE']
    assert output['properties']['SOLAR_AZIMUTH_ANGLE']


# CGM - The C01 TOA images are being scaled to match the C01 SR
def test_Landsat_C01_TOA_scaling():
    image_id='LANDSAT/LC08/C01/T1_TOA/LC08_044033_20170716'
    output = utils.point_image_value(
        openet.lai.Landsat_C01_TOA(image_id).image, xy=TEST_POINT)
    assert output['nir'] > 1000


# CGM - sensor is not currently being set as a class property
# def test_Landsat_C01_TOA_sensor():
#     sensor = openet.lai.Landsat_C01_TOA(TEST_IMAGE_ID).sensor
#     assert sensor == TEST_IMAGE_ID.split('/')[1]


@pytest.mark.parametrize(
    'image_id',
    [
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
        'LANDSAT/LC08/C01/T1_SR/LC08_044033_20170716',
        'LANDSAT/LC08/C01/T1_TOA/LC08_044033_20170716',
    ]
)
def test_Landsat_band_names(image_id):
    output = openet.lai.Landsat(image_id).image.bandNames().getInfo()
    assert set(output) == set(DEFAULT_BANDS)


# CGM - sensor is not currently being set as a class property
# def test_Landsat_sensor(image_id='LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716'):
#     assert openet.lai.Landsat(image_id).image.sensor == image_id.split('/')[1]
