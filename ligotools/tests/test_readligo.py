from ligotools import readligo

def test_ligo_default():
    assert toy() == 1

def test_toy_0():
    assert toy(0) == 1

def test_toy_1():
    assert toy(1) == 2