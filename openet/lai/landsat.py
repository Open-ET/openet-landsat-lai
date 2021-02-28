import re

import ee

from .model import Model


class Landsat(object):
    # CGM - Using the __new__ to return is discouraged and is probably not
    #   great Python but it was the only way I could find to make the general
    #   Landsat class directly callable like the collection specific ones
    # def __init__(self):
    #     """"""
    #     pass

    def __new__(cls, image_id):
        if type(image_id) is not str:
            raise ValueError('unsupported input type')
        elif re.match('LANDSAT/L[TEC]0[4578]/C02/T1_L2', image_id):
            return Landsat_C02_SR(image_id)
        elif re.match('LANDSAT/L[TEC]0[4578]/C01/T1_SR', image_id):
            return Landsat_C01_SR(image_id)
        elif re.match('LANDSAT/L[TEC]0[4578]/C01/T1_TOA', image_id):
            return Landsat_C01_TOA(image_id)
        else:
            raise ValueError('unsupported image_id')


class Landsat_C02_SR(Model):
    def __init__(self, image_id):
        """"""
        # TODO: Support input being an ee.Image
        # For now assume input is always an image ID
        if type(image_id) is not str:
            raise ValueError('unsupported input type')
        elif (image_id.startswith('LANDSAT/') and
                not re.match('LANDSAT/L[TEC]0[4578]/C02/T1_L2', image_id)):
            raise ValueError('unsupported collection ID')
        raw_image = ee.Image(image_id)

        # CGM - Testing out not setting any self. parameters and passing inputs
        #   to the super().__init__() call instead

        # CGM - Not sure if we need any of these properties
        # self.id = self.raw_image.get('system:id')
        # self.index = self.raw_image.get('system:index')
        # self.time_start = self.raw_image.get('system:time_start')

        # It might be safer to do this with a regex
        sensor = image_id.split('/')[1]

        spacecraft_id = ee.String(raw_image.get('SPACECRAFT_ID'))

        input_bands = ee.Dictionary({
            'LANDSAT_4': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                          'QA_PIXEL'],
            'LANDSAT_5': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                          'QA_PIXEL'],
            'LANDSAT_7': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7',
                          'QA_PIXEL'],
            'LANDSAT_8': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                          'QA_PIXEL'],
        })
        output_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c2_sr_cloud_mask(
        #     raw_image, **cloudmask_args)

        input_image = raw_image \
            .select(input_bands.get(spacecraft_id), output_bands)\
            .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275,
                       0.0000275, 0.0000275, 1])\
            .add([-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 1])

        input_image = input_image\
            .divide([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1])

        input_image = input_image\
            .set({'system:time_start': raw_image.get('system:time_start'),
                  'system:index': raw_image.get('system:index'),
                  # Convert elevation to zenith
                  'SOLAR_ZENITH_ANGLE':
                        ee.Number(raw_image.get('SUN_ELEVATION')).multiply(-1).add(90),
                  'SOLAR_AZIMUTH_ANGLE': raw_image.get('SUN_AZIMUTH'),
                  })

        # CGM - super could be called without the init if we set input_image and
        #   spacecraft_id as properties of self
        super().__init__(input_image, sensor)
        # super()


class Landsat_C01_SR(Model):
    def __init__(self, image_id):
        """"""
        if type(image_id) is not str:
            raise ValueError('unsupported input type')
        elif (image_id.startswith('LANDSAT/') and
                not re.match('LANDSAT/L[TEC]0[4578]/C01/T1_SR', image_id)):
            raise ValueError('unsupported collection ID')
        raw_image = ee.Image(image_id)

        sensor = image_id.split('/')[1]

        # The SATELLITE property in this collection is equivalent to SPACECRAFT_ID
        spacecraft_id = ee.String(raw_image.get('SATELLITE'))

        input_bands = ee.Dictionary({
            'LANDSAT_4': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
            'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
            'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
            'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa']})
        output_bands = [
            'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c1_sr_cloud_mask(
        #     raw_image, **cloudmask_args)

        input_image = raw_image \
            .select(input_bands.get(spacecraft_id), output_bands)
        # CM - Don't unscale the images yet
        #   The current implementation is expecting raw unscaled images
        #     .multiply([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1])\
        #     .set({'SPACECRAFT_ID': spacecraft_id})
        #     .updateMask(cloud_mask)

        input_image = input_image\
            .set({'system:time_start': raw_image.get('system:time_start'),
                  'system:index': raw_image.get('system:index'),
                  'SOLAR_ZENITH_ANGLE': raw_image.get('SOLAR_ZENITH_ANGLE'),
                  'SOLAR_AZIMUTH_ANGLE': raw_image.get('SOLAR_AZIMUTH_ANGLE'),
                  })

        super().__init__(input_image, sensor)


class Landsat_C01_TOA(Model):
    def __init__(self, image_id):
        """"""
        # TODO: Support input being an ee.Image
        # For now assume input is always an image ID
        if type(image_id) is not str:
            raise ValueError('unsupported input type')
        elif (image_id.startswith('LANDSAT/') and
                not re.match('LANDSAT/L[TEC]0[4578]/C01/T1_TOA', image_id)):
            raise ValueError('unsupported collection ID')
        raw_image = ee.Image(image_id)

        # It might be safer to do this with a regex
        sensor = image_id.split('/')[1]

        # Use the SPACECRAFT_ID property identify each Landsat type
        spacecraft_id = ee.String(raw_image.get('SPACECRAFT_ID'))

        input_bands = ee.Dictionary({
            'LANDSAT_4': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'BQA'],
            'LANDSAT_5': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'BQA'],
            'LANDSAT_7': ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'BQA'],
            'LANDSAT_8': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'BQA']})
        output_bands = [
            'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c1_toa_cloud_mask(
        #     raw_image, **cloudmask_args)

        input_image = raw_image \
            .select(input_bands.get(spacecraft_id), output_bands)

        # DEADBEEF - Until code is updated, scale image to match a C01 1 SR image
        input_image = input_image\
            .divide([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 1])

        input_image = input_image\
            .set({'system:time_start': raw_image.get('system:time_start'),
                  'system:index': raw_image.get('system:index'),
                  # Convert elevation to zenith
                  'SOLAR_ZENITH_ANGLE':
                        ee.Number(raw_image.get('SUN_ELEVATION')).multiply(-1).add(90),
                  'SOLAR_AZIMUTH_ANGLE': raw_image.get('SUN_AZIMUTH'),
                  })

        super().__init__(input_image, sensor)
