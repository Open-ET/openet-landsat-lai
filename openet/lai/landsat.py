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
        elif re.match('^LANDSAT/L[TEC]0[45789]/C02/T1_L2/\\w+', image_id):
            return Landsat_C02_L2(image_id)
        elif re.match('^L[TEC]0[45789]_\\d{6}_\\d{8}', image_id, re.IGNORECASE):
            print(f'{image_id[:4].upper()}/{image_id.upper()}')
            return Landsat_C02_L2(f'LANDSAT/{image_id[:4].upper()}/C02/T1_L2/{image_id.upper()}')
        else:
            raise ValueError('unsupported image_id')


class Landsat_C02_L2(Model):
    def __init__(self, image_id):
        """"""
        # TODO: Support input being an ee.String (or ee.Image)
        # For now assume input is always an image ID
        if type(image_id) is not str:
            raise ValueError('unsupported input type')

        # Match either the full image ID or just the scene ID
        image_id_match = re.match(
            '(?P<COLL_ID>LANDSAT/L[TEC]0[45789]/C02/T1_L2/)?'
            '(?P<SENSOR>L[TEC]0[45789])_\\d{6}_\\d{8}',
            image_id, re.IGNORECASE
        )
        if not image_id_match:
            raise ValueError('unsupported image ID')

        # Get the sensor type from the image ID
        sensor = image_id_match.group('SENSOR').upper()

        # Build a full image ID if the input was only the scene ID
        if ('COLL_ID' not in image_id_match.groupdict().keys() or
                not image_id_match.group('COLL_ID')):
            image_id = f'LANDSAT/{sensor}/C02/T1_L2/{image_id.upper()}'

        raw_image = ee.Image(image_id)
        spacecraft_id = ee.String(raw_image.get('SPACECRAFT_ID'))
        input_bands = ee.Dictionary({
            'LANDSAT_4': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_5': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_7': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'QA_PIXEL'],
            'LANDSAT_8': ['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL'],
            'LANDSAT_9': ['SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'QA_PIXEL'],
        })
        output_bands = ['green', 'red', 'nir', 'swir1', 'qa']

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
            .set({
                'system:time_start': raw_image.get('system:time_start'),
                'system:index': raw_image.get('system:index'),
                'SOLAR_AZIMUTH_ANGLE': ee.Number(raw_image.get('SUN_AZIMUTH')),
                'SOLAR_ZENITH_ANGLE': ee.Number(raw_image.get('SUN_ELEVATION'))
                    .multiply(-1).add(90),
            })
        )

        # CGM - super could be called without the init if we set input_image and
        #   spacecraft_id as properties of self
        super().__init__(input_image, sensor)
        # super()
