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


def test_ee_init():
    assert ee.Number(1).getInfo() == 1


def test_Landsat_C02_L2_band_names():
    output = utils.get_info(openet.lai.Landsat_C02_L2(TEST_IMAGE_ID).image.bandNames())
    assert set(output) == set(DEFAULT_BANDS)


@pytest.mark.parametrize(
    'image_id',
    [
        TEST_IMAGE_ID,
        # 'LANDSAT/LT04/C02/T1_L2/LT04_044033_19830812',
        'LANDSAT/LT05/C02/T1_L2/LT05_044033_20110716',
        'LANDSAT/LE07/C02/T1_L2/LE07_044033_20170708',
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
        'LANDSAT/LC09/C02/T1_L2/LC09_044033_20220127',
        'LC08_044033_20170716',
        'lc08_044033_20170716',
    ]
)
def test_Landsat_C02_L2_image_id_formats(image_id):
    output = utils.get_info(openet.lai.Landsat_C02_L2(image_id).image.bandNames())
    assert set(output) == set(DEFAULT_BANDS)


def test_Landsat_C02_L2_image_properties():
    output = utils.get_info(openet.lai.Landsat_C02_L2(TEST_IMAGE_ID).image)
    assert output['properties']['system:time_start']
    assert abs(output['properties']['SOLAR_ZENITH_ANGLE'] - 25.7206) <= 0.0001
    assert abs(output['properties']['SOLAR_AZIMUTH_ANGLE'] - 127.0891) <= 0.0001


# CGM - The C02 SR images are being scaled to match the C01 SR
def test_Landsat_C02_L2_scaling():
    output = utils.point_image_value(
        openet.lai.Landsat_C02_L2(TEST_IMAGE_ID).image, xy=TEST_POINT)
    assert output['nir'] > 1000


# CGM - sensor is not currently being set as a class property
# def test_Landsat_C02_L2_sensor():
#     sensor = openet.lai.Landsat_C02_L2(TEST_IMAGE_ID).sensor
#     assert sensor == TEST_IMAGE_ID.split('/')[1]


@pytest.mark.parametrize(
    'image_id',
    [
        TEST_IMAGE_ID,
        'LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716',
        'LC08_044033_20170716',
        'lc08_044033_20170716',
    ]
)
def test_Landsat_image_id_formats(image_id):
    output = utils.get_info(openet.lai.Landsat(image_id).image.bandNames())
    assert set(output) == set(DEFAULT_BANDS)


# CGM - sensor is not currently being set as a class property
# def test_Landsat_sensor(image_id='LANDSAT/LC08/C02/T1_L2/LC08_044033_20170716'):
#     assert openet.lai.Landsat(image_id).image.sensor == image_id.split('/')[1]
