import logging
import pprint

import ee
# import pytest

import openet.lai

logging.basicConfig(level=logging.DEBUG, format='%(message)s')


def test_ee_init():
    assert ee.Number(1).getInfo() == 1
