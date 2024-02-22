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
        elif re.match('LANDSAT/L[TEC]0[45789]/C02/T1_L2', image_id):
            return Landsat_C02_L2(image_id)
        elif re.match('LANDSAT/L[TEC]0[4578]/C01/T1_SR', image_id):
            return Landsat_C01_SR(image_id)
        elif re.match('LANDSAT/L[TEC]0[4578]/C01/T1_TOA', image_id):
            return Landsat_C01_TOA(image_id)
        else:
            raise ValueError('unsupported image_id')


class Landsat_C02_L2(Model):
    def __init__(self, image_id):
        """"""
        # TODO: Support input being an ee.Image
        # For now assume input is always an image ID
        if type(image_id) is not str:
            raise ValueError('unsupported input type')
        elif (image_id.startswith('LANDSAT/') and
                not re.match('LANDSAT/L[TEC]0[45789]/C02/T1_L2', image_id)):
            raise ValueError('unsupported collection ID')
        raw_image = ee.Image(image_id)

        # CGM - Testing out not setting any self. parameters and passing inputs
        #   to the super().__init__() call instead

        # It might be safer to do this with a regex
        sensor = image_id.split('/')[1]

        spacecraft_id = ee.String(raw_image.get('SPACECRAFT_ID'))

        input_bands = ee.Dictionary({
            'LANDSAT_4': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_5': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_7': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_8': ['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL'],
            'LANDSAT_9': ['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL'],
        })
        output_bands = ['green', 'red', 'nir', 'swir1', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c2_sr_cloud_mask(
        #     raw_image, **cloudmask_args)

        # The reflectance values are intentionally being scaled by 10000 to
        #   match the Collection 1 SR scaling.
        # The elevation angle is being converted to a zenith angle
        input_image = (
            raw_image
            .select(input_bands.get(spacecraft_id), output_bands)
            .multiply([0.0000275, 0.0000275, 0.0000275, 0.0000275, 1])
            .add([-0.2, -0.2, -0.2, -0.2, 1])
            .divide([0.0001, 0.0001, 0.0001, 0.0001, 1])
            .set({'system:time_start': raw_image.get('system:time_start'),
                  'system:index': raw_image.get('system:index'),
                  'SOLAR_ZENITH_ANGLE': ee.Number(raw_image.get('SUN_ELEVATION')).multiply(-1).add(90),
                  'SOLAR_AZIMUTH_ANGLE': raw_image.get('SUN_AZIMUTH'),
            })
        )

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
            'LANDSAT_4': ['B2', 'B3', 'B4', 'B5', 'pixel_qa'],
            'LANDSAT_5': ['B2', 'B3', 'B4', 'B5', 'pixel_qa'],
            'LANDSAT_7': ['B2', 'B3', 'B4', 'B5', 'pixel_qa'],
            'LANDSAT_8': ['B3', 'B4', 'B5', 'B6', 'pixel_qa']})
        output_bands = ['green', 'red', 'nir', 'swir1', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c1_sr_cloud_mask(
        #     raw_image, **cloudmask_args)

        input_image = raw_image \
            .select(input_bands.get(spacecraft_id), output_bands)
        # CM - Don't unscale the images yet
        #   The current implementation is expecting raw unscaled images
        #     .multiply([0.0001, 0.0001, 0.0001, 0.0001, 1])\

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
            'LANDSAT_4': ['B2', 'B3', 'B4', 'B5', 'BQA'],
            'LANDSAT_5': ['B2', 'B3', 'B4', 'B5', 'BQA'],
            'LANDSAT_7': ['B2', 'B3', 'B4', 'B5', 'BQA'],
            'LANDSAT_8': ['B3', 'B4', 'B5', 'B6', 'BQA']})
        output_bands = ['green', 'red', 'nir', 'swir1', 'pixel_qa']

        # # Cloud mask function must be passed with raw/unnamed image
        # cloud_mask = openet.core.common.landsat_c1_toa_cloud_mask(
        #     raw_image, **cloudmask_args)

        # The reflectance values are intentionally being scaled by 10000 to
        #   match the Collection 1 SR scaling.
        # The elevation angle is being converted to a zenith angle
        input_image = raw_image \
            .select(input_bands.get(spacecraft_id), output_bands)\
            .divide([0.0001, 0.0001, 0.0001, 0.0001, 1])\
            .set({'system:time_start': raw_image.get('system:time_start'),
                  'system:index': raw_image.get('system:index'),
                  'SOLAR_ZENITH_ANGLE':
                        ee.Number(raw_image.get('SUN_ELEVATION')).multiply(-1).add(90),
                  'SOLAR_AZIMUTH_ANGLE': raw_image.get('SUN_AZIMUTH'),
                  })

        super().__init__(input_image, sensor)
