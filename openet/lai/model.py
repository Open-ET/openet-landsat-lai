import ee


class Model:
    def __init__(self, image, sensor):
        """

        Parameters
        ----------
        image : ee.Image
            Prepped input image.  Must have the following bands and properties.
            Reflectance values must be scaled from 0-10000, not 0-1.
            Bands: 'green', 'red', 'nir', 'swir1', 'qa'
            Properties: 'system:time_start', 'SOLAR_ZENITH_ANGLE', 'SOLAR_AZIMUTH_ANGLE'
            Note, the 'qa' band is only used to set the nodata mask.
        sensor : {'LT05', 'LE07', 'LC08', 'LC09'}
            Note, LC09 will currently use the LC08 training collections.

        Notes
        -----
        image must have the following properties set:
            system:time_start, SOLAR_ZENITH_ANGLE, SOLAR_AZIMUTH_ANGLE

        """
        if type(image) is not ee.Image:
            raise ValueError(f'unsupported input_img type: {type(image)}')
        if sensor not in ['LT05', 'LE07', 'LC08', 'LC09']:
            raise ValueError(f'unsupported sensor: {sensor}')

        self.image = image
        self.sensor = sensor

        # Use Landsat 8 training values for Landsat 9
        if self.sensor == 'LC09':
            self.sensor = 'LC08'

    def lai(self, nonveg=True):
        """Wrapper to the get_lai_image function"""
        return get_lai_image(self.image, self.sensor, nonveg)


# TODO: Move into Model class
def get_lai_image(image, sensor, nonveg):
    """Main Algorithm to compute LAI for a Landsat image

    Parameters
    ----------
    image : ee.Image
        Prepped input image.  Must have the following bands and properties.
        Bands: 'green', 'red', 'nir', 'swir1', 'qa'
        Properties: 'system:time_start', 'SOLAR_ZENITH_ANGLE', 'SOLAR_AZIMUTH_ANGLE'
        Note, the 'qa' band is only used to set the nodata mask.
    sensor : {'LT05', 'LE07', 'LC08'}
    nonveg : bool
        True if want to compute LAI for non-vegetation pixels

    Returns
    -------
    ee.Image

    """
    train_img = get_train_img(image)

    # Start with an image of all zeros
    lai_img = train_img.select(['mask'], ['LAI']).multiply(0).double()

    if nonveg:
        biomes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    else:
        biomes = [1, 2, 3, 4, 5, 6, 7, 8]

    # Apply LAI for each biome
    for biome in biomes:
        lai_img = lai_img.where(
            train_img.select('biome2').eq(biome),
            get_lai_for_biome(train_img, biome, get_rf_model(sensor, biome))
        )

    # Set water LAI to zero
    # Note, the reflectance values are scaled 0-10000, not 0-1
    # TODO: This should probably be in a separate function
    # TODO: Check what water_mask the other models are using
    water_mask = (
        train_img.select('NDVI').lt(0)
        # .And(train_img.select('NDWI').gt(0)
        .And(train_img.select('nir').lt(1000))
    )

    # CGM - Should we mask out water pixels instead of setting them to zero?
    lai_img = lai_img.where(water_mask, 0)
    # lai_img = lai_img.updateMask(water_mask.eq(0))

    # Add QA band
    qa_img = get_lai_qa(train_img, sensor, lai_img)
    lai_img = lai_img.addBands(qa_img.byte())

    return lai_img.set({
        'system:index': image.get('system:index'),
        'system:time_start': image.get('system:time_start'),
        # 'SOLAR_AZIMUTH_ANGLE': image.get('SOLAR_AZIMUTH_ANGLE'),
        # 'SOLAR_ZENITH_ANGLE': image.get('SOLAR_ZENITH_ANGLE'),
    })


def get_lai_qa(landsat, sensor, lai):
    """

    Parameters
    ----------
    landsat : ee.Image
        Landsat image (with 'biome2' band)
    sensor : {'LT05', 'LE07', 'LC08'}
    lai : ee.Image
        Computed lai image

    Notes
    -----
    QA is coded in a byte-size band occupying the least significant 3 bits
      Bit 0: Input
          0: Input within range
          1: Input out-of-range
      Bit 1: Output (LAI)
          0: LAI within range (0-8)
          1: LAI out-of-range
      Bit 2: Biome
          0: Vegetation (from NLCD scheme)
          1: Non-vegetation (from NLCD scheme)

    """

    # Maximum for surface reflectance; minimum is always 0
    red_max = 5100
    green_max = 5100
    nir_max = 7100
    swir1_max = 7100
    lai_max = 8

    # Get pre-coded convex hull
    data_id = 'projects/openet/lai/training/LAI_train_convex_hull_by_sensor_v10_1'
    # data_id = 'projects/openet/assets/lai/training/LAI_train_convex_hull_by_sensor_v10_1'
    hull_array = (
        ee.FeatureCollection(data_id)
        .filterMetadata('sensor', 'equals', sensor)
        .sort('index')
        .aggregate_array('in_hull')
    )
    hull_array_reshape = ee.Array(hull_array).reshape([10, 10, 10, 10])

    # Rescale landsat image
    image_scaled = (
        landsat.select(['red', 'green', 'nir', 'swir1'])
        .divide([red_max, green_max, nir_max, swir1_max])
        .multiply(10).floor().toInt()
    )
    # image_scaled = landsat.select('red').divide(red_max).multiply(10).floor().toInt() \
    #     .addBands(landsat.select('green').divide(green_max).multiply(10).floor().toInt()) \
    #     .addBands(landsat.select('nir').divide(nir_max).multiply(10).floor().toInt()) \
    #     .addBands(landsat.select('swir1').divide(swir1_max).multiply(10).floor().toInt())

    # Get an out-of-range mask
    range_mask = (
        landsat.select('red').gte(0)
        .And(landsat.select('red').lt(red_max))
        .And(landsat.select('green').gte(0))
        .And(landsat.select('green').lt(green_max))
        .And(landsat.select('nir').gte(0))
        .And(landsat.select('nir').lt(nir_max))
        .And(landsat.select('swir1').gte(0))
        .And(landsat.select('swir1').lt(swir1_max))
    )

    # Apply convex hull and get QA Band
    hull_image = (
        image_scaled.select('red').multiply(0).add(ee.Image(hull_array_reshape))
        .updateMask(range_mask)
    )

    in_mask = hull_image.arrayGet(image_scaled.updateMask(range_mask))

    in_mask = in_mask.unmask(0).updateMask(landsat.select('red').mask()).Not().int()

    # Check output range
    out_mask = (
        lai.gte(0).And(lai.lte(lai_max)).updateMask(landsat.select('red').mask()).Not().int()
    )

    # Indicate non-vegetation biome
    biome_mask = landsat.select('biome2').eq(0).int()

    # Combine
    qa_band = in_mask.bitwiseOr(out_mask.leftShift(1)).bitwiseOr(biome_mask.leftShift(2)).toByte()

    return qa_band.rename('QA')


def get_rf_model(sensor, biome):
    """Wrapper function to train RF model given biome and sensor

    Parameters
    ----------
    sensor: str, ee.String
    biome: int, ee.Number

    """

    training_coll_id = 'projects/openet/assets/lai/training/LAI_train_sample_unsat_v10_1_final'
    # training_coll_id = 'projects/earthengine-legacy/assets/' \
    #                    'projects/openet/lai/training/LAI_train_sample_unsat_v10_1_final'
    training_coll = (
        ee.FeatureCollection(training_coll_id).filterMetadata('sensor', 'equals', sensor)
    )

    # Get train sample by biome
    if biome > 0:
        training_coll = training_coll.filterMetadata('biome2', 'equals', biome)

    inputProperties = [
        'red', 'green', 'nir', 'swir1', 'lat', 'lon', 'NDVI', 'NDWI', 'sun_zenith', 'sun_azimuth'
    ]

    return (
        ee.Classifier.smileRandomForest(
            numberOfTrees=100, minLeafPopulation=50, variablesPerSplit=5)
        .setOutputMode('REGRESSION')
        .train(features=training_coll, classProperty='MCD_LAI', inputProperties=inputProperties)
    )


def get_lai_for_biome(train_img, biome, rf_model):
    """Compute LAI for an input Landsat image and Random Forest models

    Parameters
    ----------
    train_img : ee.Image
        Must have training bands added
    biome : int
    rf_model : ee.Classifier

    Returns
    -------
    ee.Image

    """
    return (
        train_img
        .updateMask(train_img.select('biome2').eq(ee.Number(biome)))
        .classify(rf_model, 'LAI')
    )


def get_train_img(image):
    """Function that takes a Landsat image and prepare feature bands

    Parameters
    ----------
    image : ee.Image

    Returns
    -------
    ee.Image

    """
    nlcd_img_dict = ee.Dictionary({
        '2021': ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021'),
        '2020': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019'),
        '2019': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019'),
        '2018': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2019'),
        '2017': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016'),
        '2016': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016'),
        '2015': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2016'),
        '2014': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2013'),
        '2013': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2013'),
        '2012': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2011'),
        '2011': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2011'),
        '2010': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2011'),
        '2009': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2008'),
        '2008': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2008'),
        '2007': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2006'),
        '2006': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2006'),
        '2005': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2004'),
        '2004': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2004'),
        '2003': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2004'),
        '2002': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2001'),
        '2001': ee.Image('USGS/NLCD_RELEASES/2019_REL/NLCD/2001'),
    })

    # Hardcode min/max for now, but they could be computed from the dictionary
    nlcd_year_min = 2001
    nlcd_year_max = 2021
    # nlcd_year_min = ee.List(nlcd_img_dict.keys()).reduce(ee.Reducer.min())
    # nlcd_year_max = ee.List(nlcd_img_dict.keys()).reduce(ee.Reducer.max())
    image_year = (
        ee.Number(ee.Date(image.get('system:time_start')).get('year'))
        .max(nlcd_year_min).min(nlcd_year_max).format('%d')
    )

    nlcd_img = ee.Image(nlcd_img_dict.get(image_year)).select(['landcover'])
    nlcd_year = nlcd_img.get('system:index')

    # CM - Add the NLCD year as a property to track which year was used
    #   This probably isn't needed long term but it is useful for testing
    image = image.set({'nlcd_year': nlcd_year})

    # Add the vegetation indices as additional bands
    image = add_vi_bands(image)

    # Map NLCD codes to biomes
    nlcd_biom_remap = {
        11: 0, 12: 0, 21: 0, 22: 0, 23: 0, 24: 0, 31: 0,
        41: 1,
        42: 2,
        43: 3,
        52: 4,
        71: 5, 81: 5,
        82: 6,
        90: 7,
        95: 8,
    }
    biom_img = nlcd_img.remap(*zip(*nlcd_biom_remap.items()))

    # Add other bands

    # Map all bands to mask image to avoid clip or updateMask calls
    mask_img = image.select(['qa'], ['mask']).multiply(0)
    image = image.addBands([
        mask_img.add(biom_img).rename('biome2'),
        mask_img.add(ee.Image.pixelLonLat().select(['longitude'])).rename(['lon']),
        mask_img.add(ee.Image.pixelLonLat().select(['latitude'])).rename(['lat']),
        mask_img.float().add(ee.Number(image.get('SOLAR_ZENITH_ANGLE'))).rename(['sun_zenith']),
        mask_img.float().add(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE'))).rename(['sun_azimuth']),
        mask_img.add(1),
    ])

    # # Test adding all bands directly and the calling updateMask to clip
    # mask_img = image.select(['qa'], ['mask']).multiply(0)
    # image = (
    #     image.addBands(biom_img.rename('biome2'))
    #     .addBands(ee.Image.pixelLonLat().select(['longitude']).rename(['lon']))
    #     .addBands(ee.Image.pixelLonLat().select(['latitude']).rename(['lat']))
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
    #               .rename(['sun_zenith']))
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
    #               .rename(['sun_azimuth']))
    #     .addBands(mask_img.add(1))
    #     .updateMask(mask_img.add(1))
    # )

    return image


def add_vi_bands(image):
    """Compute VIs for an Landsat image

    Parameters
    ----------
    image : ee.Image

    Returns
    -------
    ee.Image

    """
    ndvi_img = image.expression('float((b("nir") - b("red"))) / (b("nir") + b("red"))')
    ndwi_img = image.expression('float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))')

    return image.addBands([ndvi_img.rename('NDVI'), ndwi_img.rename('NDWI')])
