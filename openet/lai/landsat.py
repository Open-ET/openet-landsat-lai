import ee


# TODO: Define LandsatSR and LandsatTOA (and maybe Landsat) classes


def getLAIImage(image, sensor, nonveg):
    """
    Main Algorithm to compute LAI for a Landsat image
    Args:
        image: ee.Image
        sensor: str {'LT05', 'LE07', 'LC08'} (cannot be an EE object)
        nonveg: True if want to compute LAI for non-vegetation pixels
    """
    # Add necessary bands to image
    # image = renameLandsat(image)
    train_img = getTrainImg(image)

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
            getLAIforBiome(train_img, biome, getRFModel(sensor, biome)))

    # Set water LAI to zero
    # TODO: This should probably be in a separate function
    # TODO: Check what water_mask the other models are using (PTJPL?)
    water_mask = train_img.select('NDVI').lt(0) \
        .And(train_img.select('nir').lt(1000))
    # water_mask = train_img.select('NDVI').lt(0) \
    #     .And(train_img.select('NDWI').gt(0))
    lai_img = lai_img.where(water_mask, 0)
    qa = getLAIQA(train_img,sensor,lai_img)

    lai_img = lai_img.addBands(qa.byte())

    # CM - copyProperties drops the type
    return ee.Image(lai_img.copyProperties(image)) \
        .set('system:time_start', image.get('system:time_start'))


def getLAIQA(landsat, sensor, lai):
    """
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
      
      args: landsat - Landsat image (with 'biome2' band)
            sensor - "LT05"/"LE07"/"LC08"
            lai - computed lai image
    """

    # maximum for surface reflectance; minimum is always 0
    red_max = 5100
    green_max = 5100
    nir_max = 7100
    swir1_max = 7100
    lai_max = 8
  
    # information from the Landsat image
    # crs = landsat.select('red').projection().crs()
    # transform = getAffineTransform(landsat.select('red'))

    # Get pre-coded convex hull
    data = ee.FeatureCollection('projects/openet/lai/trianing/LAI_train_convex_hull_by_sensor_v10_1')

    subset = data.filterMetadata('sensor','equals',sensor)
    subset = subset.sort('index')
    hull_array = subset.aggregate_array('in_hull')
    hull_array_reshape = ee.Array(hull_array).reshape([10,10,10,10])

    # rescale landsat image
    image_scaled = landsat.select('red').divide(red_max).multiply(10).floor().toInt() \
        .addBands(landsat.select('green').divide(green_max).multiply(10).floor().toInt()) \
        .addBands(landsat.select('nir').divide(nir_max).multiply(10).floor().toInt()) \
        .addBands(landsat.select('swir1').divide(swir1_max).multiply(10).floor().toInt())

    # get an out-of-range mask
    range_mask = landsat.select('red').gte(0) \
        .And(landsat.select('red').lt(red_max)) \
        .And(landsat.select('green').gte(0)) \
        .And(landsat.select('green').lt(green_max)) \
        .And(landsat.select('nir').gte(0)) \
        .And(landsat.select('nir').lt(nir_max)) \
        .And(landsat.select('swir1').gte(0)) \
        .And(landsat.select('swir1').lt(swir1_max))

    # apply convel hull and get QA Band
    hull_image = image_scaled.select('red').multiply(0).add(ee.Image(hull_array_reshape)) \
        .updateMask(range_mask)

    in_mask = hull_image \
        .arrayGet(image_scaled.select(['red','green','nir','swir1']).updateMask(range_mask))

    in_mask = in_mask.unmask(0).updateMask(landsat.select('red').mask()).Not().int()

    # check output range
    out_mask = lai.gte(0).And(lai.lte(lai_max)).updateMask(landsat.select('red').mask()).Not().int()

    # indicate non-vegetation biome
    biome_mask = landsat.select('biome2').eq(0).int()

    # combine
    qa_band = in_mask.bitwiseOr(out_mask.leftShift(1)).bitwiseOr(biome_mask.leftShift(2)).toByte()

    return qa_band.rename('QA')


def getRFModel(sensor, biome):
    """
    Wrapper function to train RF model given biome and sensor
    Args:
        sensor: str, ee.String
        biome: int, ee.Number
    """

    # CM - The "projects/earthengine-legacy/assets/" probably isn't needed
    training_coll_id = 'projects/earthengine-legacy/assets/' \
                       'projects/openet/lai/training/' \
                       'LAI_train_sample_all_v10_1_final'
    training_coll = ee.FeatureCollection(training_coll_id) \
        .filterMetadata('sensor', 'equals', sensor)

    # DEADBEEF
    # training_coll_id = 'users/yanghui/OpenET/LAI_US/train_samples/' \
    #                    'LAI_train_samples_' + sensor + '_v10_1_labeled'
    # training_coll = ee.FeatureCollection(training_coll_id) \
    #     .filterMetadata('sat_flag', 'equals', 'correct')

    # Get train sample by biome
    if biome > 0:
        training_coll = training_coll.filterMetadata('biome2', 'equals', biome)

    inputProperties = ['red', 'green', 'nir', 'swir1', 'lat', 'lon',
                       'NDVI', 'NDWI', 'sun_zenith', 'sun_azimuth']

    return ee.Classifier.smileRandomForest(numberOfTrees=100,
                                           minLeafPopulation=50,
                                           variablesPerSplit=5) \
                        .setOutputMode('REGRESSION') \
                        .train(features=training_coll,
                               classProperty='MCD_LAI',
                               inputProperties=inputProperties)


# TODO: Change name of "image" to something that indicates it has training bands
def getLAIforBiome(image, biome, rf_model):
    """
    Function that computes LAI for an input Landsat image and Random Forest models
    Args:
        image: ee.Image, must have training bands added
        biome: int
        rf_model: ee.Classifier
    """
    biom_lai = image\
        .updateMask(image.select('biome2').eq(ee.Number(biome))) \
        .classify(rf_model, 'LAI')
    return biom_lai


def getTrainImg(image):
    """
    Function that takes an Landsat image and prepare feature bands
    """

    # Get NLCD for corresponding year
    # CM - I couldn't decide if it was better to use ints or strings
    nlcd_dict = {
        '2001': ['1997', '1998', '1999', '2000', '2001', '2002'],
        '2004': ['2003', '2004', '2005'],
        '2006': ['2006', '2007'],
        '2008': ['2008', '2009'],
        '2011': ['2010', '2011', '2012'],
        '2013': ['2013', '2014'],
        '2016': ['2015', '2016', '2017', '2018', '2019', '2020'],
    }
    nlcd_dict = ee.Dictionary({
        src_year: tgt_year
        for tgt_year, src_years in nlcd_dict.items()
        for src_year in src_years})
    nlcd_year = nlcd_dict.get(
        ee.Date(image.get('system:time_start')).get('year').format('%d'))
    nlcd_img = ee.ImageCollection('USGS/NLCD') \
        .filter(ee.Filter.eq('system:index', ee.String('NLCD').cat(nlcd_year))) \
        .first()

    # CM - Add the NLCD year as a property to track which year was used
    #   This probably isn't needed long term but it is useful for testing
    image = image.set({'nlcd_year': nlcd_year})

    # Add the vegetation indices as additional bands
    image = getVIs(image)

    # Map NLCD codes to biomes
    # CM - Added NLCD codes 11 and 12
    # CM - Switched from lists to a dictionary to improve readability
    nlcd_biom_remap = {
        11: 0, 12: 0,
        21: 0, 22: 0, 23: 0, 24: 0, 31: 0,
        41: 1, 42: 2, 43: 3, 52: 4,
        71: 5, 81: 5, 82: 6, 90: 7, 95: 8,
    }
    # fromList = [21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    # toList = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8]
    biom_img = nlcd_img.remap(*zip(*nlcd_biom_remap.items()) )
    # biom_img = nlcd_img.remap(
    #     list(nlcd_biom_remap.keys()), list(nlcd_biom_remap.values()))

    # Add other bands

    # CM - Map all bands to mask image to avoid clip or updateMask calls
    mask_img = image.select(['pixel_qa'], ['mask']).multiply(0)
    image = image.addBands(mask_img.add(biom_img).rename('biome2')) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['longitude']))
                    .rename(['lon'])) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['latitude']))
                    .rename(['lat'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
                    .rename(['sun_zenith'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
                    .rename(['sun_azimuth'])) \
        .addBands(mask_img.add(1))

    # # CM - Test adding all bands directly and the calling updateMask to clip
    # mask_img = image.select(['pixel_qa'], ['mask']).multiply(0)
    # image = image.addBands(biom_img.rename('biome2')) \
    #     .addBands(ee.Image.pixelLonLat().select(['longitude']).rename(['lon'])) \
    #     .addBands(ee.Image.pixelLonLat().select(['latitude']).rename(['lat'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
    #               .rename(['sun_zenith'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
    #               .rename(['sun_azimuth'])) \
    #     .addBands(mask_img.add(1)) \
    #     .updateMask(mask_img.add(1))

    return image


def getVIs(image):
    """
    Compute VIs for an Landsat image
    """
    ndvi_img = image.expression(
        'float((b("nir") - b("red"))) / (b("nir") + b("red"))')
    ndwi_img = image.expression(
        'float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))')

    return image.addBands(ndvi_img.select([0], ['NDVI'])) \
                .addBands(ndwi_img.select([0], ['NDWI']))


def renameLandsat(image):
    """
    Function that renames Landsat bands
    """
    sensor = ee.String(image.get('SATELLITE'))
    from_list = ee.Algorithms.If(
        sensor.compareTo('LANDSAT_8'),
        ['B1', 'B2', 'B3', 'B4', 'B5', 'B7', 'pixel_qa'],
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'pixel_qa'])
    to_list = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'pixel_qa']

    return image.select(from_list, to_list)


# DEADBEEF - Function is not used anymore
# def maskLST(image):
#     """
#     Function that masks a Landsat image based on the QA band
#     """
#     pixelQA = image.select('pixel_qa')
#     cloud = getQABits(pixelQA, 1, 1, 'clear')
#     # CM - Testing .eq(1) won't work for multibit values like cloud confidence
#     #   This should probably be .gte(1)
#     return image.updateMask(cloud.eq(1))


# DEADBEEF - Function is not used anymore
# def getQABits(image, start, end, newName):
#     """
#     Function that returns an image containing just the specified QA bits.
#     """
#     # Compute the bits we need to extract.
#     pattern = 0
#     for i in range(start, end + 1):
#          pattern = pattern + 2 ** i
#
#     # Return a single band image of the extracted QA bits, giving the band
#     # a new name.
#     return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)
