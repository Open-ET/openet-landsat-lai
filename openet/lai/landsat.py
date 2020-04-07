import ee


def getLAIImage(image, sensor, nonveg):
    """
    Main Algorithm to computer LAI for a Landsat image
    Args:
        image:
        sensor: needs to be specified as a String ('LT05', 'LE07', 'LC08')
        nonveg: True if want to compute LAI for non-vegetation pixels
    """
    # image = renameLandsat(image)
    rf_models = trainModels(sensor, nonveg)
    laiImg = getLAI(image, rf_models)
    # TODO: Test removing final clip call
    return ee.Image(laiImg).clip(image.geometry())


def trainModels(sensor, nonveg):
    """
    Function that trains biome-specific RF models for a specific sensor
    Args:
        sensor: String {'LT05', 'LE07', 'LC08'}
        nonveg: whether LAI is computed for non-vegetation pixels
    """

    biomes = [1, 2, 3, 4, 5, 6, 7, 8]
    biomes_str = ['1', '2', '3', '4', '5', '6', '7', '8']

    if nonveg:
        biomes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        biomes_str = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

    # Get models for each biome
    # CM - The threshold value of 1 isn't used in getRFModel()
    rf_models = ee.List(biomes).map(lambda biome: getRFModel(sensor, biome, 1))
    rf_models = ee.Dictionary.fromLists(biomes_str, rf_models)

    return rf_models


# CM - The "threshold" is not used in the function?
def getRFModel(sensor, biome, threshold=1):
    """
    Wrapper function to train RF model given biome and sensor
    "sensor" needs to be a "String"; it cannot be an EE computed object
    """

    assetDir = 'users/yanghui/OpenET/LAI_US/train_samples/'
    # change training sample
    filename = 'LAI_train_samples_' + sensor + '_v10_1_labeled'
    train = ee.FeatureCollection(assetDir + filename)

    # filter sat_flag
    train = train.filterMetadata('sat_flag', 'equals', 'correct')

    # get train sample by biome
    train = ee.Algorithms.If(ee.Number(biome).eq(0), train,
                             train.filterMetadata('biome2', 'equals', biome))

    # percentage of saturated samples to use
    train = ee.FeatureCollection(train)

    """
    train = train.filterMetadata('mcd_qa', 'equals',1) \
                 .filterMetadata('random', 'less_than',threshold) \
                 .merge(train.filterMetadata('mcd_qa', 'equals',0))
  	"""

    # train
    features = ['red', 'green', 'nir', 'swir1', 'lat', 'lon', 'NDVI', 'NDWI',
                'sun_zenith', 'sun_azimuth']
    rf = trainRF(train, features, classProperty='MCD_LAI')
    return rf


def trainRF(samples, features, classProperty='MCD_LAI'):
    """
    Function that trains a Random Forest regressor
    """
    rfRegressor = ee.Classifier.randomForest(numberOfTrees=100,
                                             minLeafPopulation=20,
                                             variablesPerSplit=8) \
                                .setOutputMode('REGRESSION') \
                                .train(features=samples,
                                       classProperty=classProperty,
                                       inputProperties=features)

    return rfRegressor


def getLAI(image, rf_models):
    """
    Function that computes LAI for an input Landsat image and Random Forest models
    """

    # Add necessary bands to image
    image = getTrainImg(image)

    # TODO: Add a standard mask band to the image in getTrainImg
    mask_prev = image.select([0]).mask()

    # Apply regressor for each biome
    biomes = rf_models.keys()
    lai_coll = ee.List(biomes).map(lambda b: getLAIforBiome(image, b, rf_models))
    lai_coll = ee.ImageCollection(lai_coll)

    # Combine LAI of all biomes
    lai_img = ee.Image(lai_coll.mean().copyProperties(image)) \
        .updateMask(mask_prev) \
        .set('system:time_start',image.get('system:time_start'))

    return lai_img


# TODO: Change "image" to something else that indicates it has training bands
#   Maybe "trained_img"?
def getLAIforBiome(image, biome, rf_models):
    """
    Function that computes LAI for an input Landsat image and Random Forest models
    Args:
        image: must have training bands added
        biome: string
        rf_models:
    """
    biome_band = image.select('biome2')
    model = rf_models.get(ee.String(biome))
    lai = image.updateMask(biome_band.select('biome2').eq(ee.Image.constant(ee.Number.parse(biome)))) \
               .classify(model, 'LAI')
    # TODO: Test if the comparison can be make just to the number
    #   (instead of the ee.Image.constant())
    # lai = image
    #     .updateMask(biome_band.select('biome2').eq(ee.Number.parse(biome))) \
    #     .classify(model, 'LAI')
    return lai


def getTrainImg(image):
    """
    Function that takes an Landsat image and prepare feature bands
    """

    # TODO: Change so that the target year image is selected directly
    #   instead of selecting from a merged image of all the years
    # Get NLCD for corresponding year
    year = ee.Date(image.get('system:time_start')).get('year')
    nlcd_all = ee.Image('USGS/NLCD/NLCD2001').select(['landcover'], ['landcover_2001']) \
        .addBands(ee.Image('USGS/NLCD/NLCD2004').select(['landcover'], ['landcover_2004'])) \
        .addBands(ee.Image('USGS/NLCD/NLCD2006').select(['landcover'], ['landcover_2006'])) \
        .addBands(ee.Image('USGS/NLCD/NLCD2008').select(['landcover'], ['landcover_2008'])) \
        .addBands(ee.Image('USGS/NLCD/NLCD2011').select(['landcover'], ['landcover_2011'])) \
        .addBands(ee.Image('USGS/NLCD/NLCD2013').select(['landcover'], ['landcover_2013'])) \
        .addBands(ee.Image('USGS/NLCD/NLCD2016').select(['landcover'], ['landcover_2016']))
    nlcd_dict = {
        '1997':0, '1998':0, '1999':0, '2000':0, '2001':0, '2002':0, 
        '2003':1, '2004':1, '2005':1,
        '2006':2, '2007':2, 
        '2008':3, '2009':3, 
        '2010':4, '2011':4, '2012':4,
        '2013':5, '2014':5,
        '2015':6, '2016':6, '2017':6, '2018':6, '2019':6, '2020':6,
    }
    nlcd_dict = ee.Dictionary(nlcd_dict)
    nlcd_img = nlcd_all.select([nlcd_dict.get(ee.Number(year).format('%d'))])

    # CM - Not applying fmask (cloud/snow/water/etc)
    # image = maskLST(image)

    # Add the vegetation indices as additional bands
    image = getVIs(image)

    # Map NLCD codes to biomes
    # CM - Added NLCD codes 11 and 12
    # CM - Switched from lists to a dictionary to improve readability
    biom_remap = {
        11: 0, 12: 0,
        21: 0, 22: 0, 23: 0, 24: 0, 31: 0,
        41: 1, 42: 2, 43: 3, 52: 4,
        71: 5, 81: 5, 82: 6, 90: 7, 95: 8,
    }
    # fromList = [21, 22, 23, 24, 31, 41, 42, 43, 52, 71, 81, 82, 90, 95]
    # toList = [0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 6, 7, 8]
    biom_img = nlcd_img.remap(*zip(*biom_remap.items()) )
    # biom_img = nlcd_img.remap(list(biom_remap.keys()), list(biom_remap.values()))

    # Add other bands

    # CM - Map all bands to mask image to avoid clip or updateMask calls
    mask_img = image.select(['pixel_qa'], ['mask']).multiply(0)
    image = image.addBands(mask_img.add(nlcd_img).rename(['nlcd'])) \
        .addBands(mask_img.add(biom_img).rename('biome2')) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['longitude']))
                    .rename(['lon'])) \
        .addBands(mask_img.add(ee.Image.pixelLonLat().select(['latitude']))
                    .rename(['lat'])) \
        .addBands(mask_img.add(ee.Number(image.get('WRS_PATH'))).rename(['path'])) \
        .addBands(mask_img.add(ee.Number(image.get('WRS_ROW'))).rename(['row'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
                    .rename(['sun_zenith'])) \
        .addBands(mask_img.float().add(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
                    .rename(['sun_azimuth']))
        # .addBands(ee.Image.constant(ft.get('year')).rename(['year'])))
        # .addBands(ee.Image.constant(ee.Number(ft.get('doy'))).rename(['DOY'])))

    # # CM - Test adding all bands directly and the calling updateMask to clip
    # image = image.addBands(nlcd_img.rename(['nlcd'])) \
    #     .addBands(ee.Image.pixelLonLat().select(['longitude']).rename(['lon'])) \
    #     .addBands(ee.Image.pixelLonLat().select(['latitude']).rename(['lat'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('WRS_PATH'))).rename(['path'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('WRS_ROW'))).rename(['row'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_ZENITH_ANGLE')))
    #               .rename(['sun_zenith'])) \
    #     .addBands(ee.Image.constant(ee.Number(image.get('SOLAR_AZIMUTH_ANGLE')))
    #               .rename(['sun_azimuth'])) \
    #     .addBands(biom_img.rename('biome2')) \
    #     .updateMask(mask_img.add(1))
    #     # .addBands(ee.Image.constant(ft.get('year')).rename(['year'])))
    #     # .addBands(ee.Image.constant(ee.Number(ft.get('doy'))).rename(['DOY'])))

    # CM - Add the NLCD band name as a property to see which year was selected
    #   This probably isn't needed long term but it is useful for testing
    image = image.set({'nlcd_year': nlcd_img.select([0]).bandNames().get(0)})

    return image


def getVIs(image):
    """
    Compute VIs for an Landsat image
    """
    SR = image.expression('float(b("nir")) / b("red")')
    NDVI = image.expression('float((b("nir") - b("red"))) / (b("nir") + b("red"))')
    NDWI = image.expression(
        'float((b("nir") - b("swir1"))) / (b("nir") + b("swir1"))')

    # CM - This equation is only correct for the raw scaled (0-10000) SR images
    EVI = image.expression(
        '2.5 * float((b("nir") - b("red"))) / '
        '(b("nir") + 6*b("red") - 7.5*float(b("blue")) + 10000)')
    # CM - This equation should be used for the unscaled (0-1) SR images
    # EVI = image.expression(
    #     '2.5 * float((b("nir") - b("red"))) / '
    #     '(b("nir") + 6*b("red") - 7.5 * float(b("blue")) + 1.0)')

    # CM - These equations may need to be modified if not using the raw scaled SR
    # GCI = image.expression('float(b("nir")) / b("green") - 1')
    # EVI2 = image.expression(
    #     '2.5 * float((b("nir") - b("red"))) / '
    #     '(b("nir") + 2.4 * float(b("red")) + 10000)')
    # OSAVI = image.expression(
    #     '1.16 * float(b("nir") - b("red")) / (b("nir") + b("red") + 1600)')
    # NDWI2 = image.expression(
    #     'float((b("nir") - b("swir2"))) / (b("nir") + b("swir2"))')
    # MSR = image.expression('float(b("nir")) / b("swir1")')
    # MTVI2 = image.expression(
    #     '1.5 * (1.2 * float(b("nir") - b("green")) - 2.5 * float(b("red") - b("green"))) / '
    #     'sqrt((2*b("nir")+10000)*(2*b("nir")+10000) - (6*b("nir") - 5*sqrt(float(b("nir"))))-5000)')

    return image.addBands(SR.select([0], ['SR'])) \
                .addBands(NDVI.select([0], ['NDVI'])) \
                .addBands(NDWI.select([0], ['NDWI'])) \
                .addBands(EVI.select([0], ['EVI']))
                # .addBands(GCI.select([0], ['GCI']))
                # .addBands(EVI2.select([0], ['EVI2']))
                # .addBands(OSAVI.select([0], ['OSAVI']))
                # .addBands(NDWI2.select([0], ['NDWI2']))
                # .addBands(MSR.select([0], ['MSR']))
                # .addBands(MTVI2.select([0], ['MTVI2']))


def maskLST(image):
    """
    Function that masks a Landsat image based on the QA band
    """
    pixelQA = image.select('pixel_qa')
    cloud = getQABits(pixelQA, 1, 1, 'clear')
    # CM - Testing .eq(1) won't work for multibit values like cloud confidence
    #   This should probably be .gte(1)
    return image.updateMask(cloud.eq(1))


def getQABits(image, start, end, newName):
    """
    Function that returns an image containing just the specified QA bits.
    """
    # Compute the bits we need to extract.
    pattern = 0
    for i in range(start, end + 1):
         pattern = pattern + 2 ** i

    # Return a single band image of the extracted QA bits, giving the band
    # a new name.
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)


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


def setDate(image):
    """
    Function that adds a "date" property to an image in format "YYYYmmdd"
    """
    image_date = ee.Date(image.get('system:time_start'))
    return image.set('date', image_date.format('YYYYMMdd'))
