import argparse
import datetime
import logging
import os
import time

import ee


def arg_valid_file(file_path):
    """Argparse specific function for testing if file exists

    Convert relative paths to absolute paths
    """
    if os.path.isfile(os.path.abspath(os.path.realpath(file_path))):
        return os.path.abspath(os.path.realpath(file_path))
        # return file_path
    else:
        raise argparse.ArgumentTypeError('{} does not exist'.format(file_path))


def arg_valid_date(input_date):
    """Check that a date string is ISO format (YYYY-MM-DD)

    Parameters
    ----------
    input_date : string

    Returns
    -------
    datetime

    """
    try:
        return datetime.datetime.strptime(input_date, '%Y-%m-%d')
    except ValueError:
        msg = f'Not a valid date: "{input_date}".'
        raise argparse.ArgumentTypeError(msg)


def getinfo(ee_obj, n=4):
    """Make an exponential back off getInfo call on an Earth Engine object"""
    output = None
    for i in range(1, n):
        try:
            output = ee_obj.getInfo()
        except ee.ee_exception.EEException as e:
            if 'Earth Engine memory capacity exceeded' in str(e):
                logging.info('    Resending query ({}/10)'.format(i))
                logging.debug('    {}'.format(e))
                time.sleep(i ** 2)
            else:
                raise e

        if output:
            break

    # output = ee_obj.getInfo()
    return output


# TODO: Import from openet.core.utils instead of defining here
def constant_image_value(image, crs='EPSG:32613', scale=1):
    """Extract the output value from a calculation done with constant images"""
    return getinfo(ee.Image(image).reduceRegion(
        reducer=ee.Reducer.first(), scale=scale,
        geometry=ee.Geometry.Rectangle([0, 0, 10, 10], crs, False)))


def point_image_value(image, xy, scale=1):
    """Extract the output value from a calculation at a point"""
    return getinfo(ee.Image(image).reduceRegion(
        reducer=ee.Reducer.first(), geometry=ee.Geometry.Point(xy),
        scale=scale))
