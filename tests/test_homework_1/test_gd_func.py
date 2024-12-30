import pytest as pt

from homework_1.gd_func import *


def test_temp_ratio():
    assert round(temp_ratio(0.6, 1.4), 3) == 1.072
    assert round(temp_ratio(1.72, 1.24), 3) == 1.355


def test_dens_ratio():
    assert round(dens_ration(0.6, 1.4), 3) == 1.19
    assert round(dens_ration(1.72, 1.24), 3) == 3.546


def test_press_ratio():
    assert round(press_ratio(0.6, 1.4), 3) == 1.276
    assert round(press_ratio(1.72, 1.24), 3) == 4.805


def test_square_ratio():
    assert square_ratio(1, 1.4) == 1
    assert round(square_ratio(0.5, 1.4), 3) == 1.34
    assert round(square_ratio(2, 1.2), 3) == 1.884
    with pt.raises(ZeroDivisionError):
        square_ratio(0, 1.4)


def test_lambda2mach():
    assert lambda2mach(0, 1.4) == 0
    assert lambda2mach(1, 1.4) == 1
    assert round(lambda2mach(0.4, 1.24), 3) == 0.381
    assert round(lambda2mach(1.5, 1.4), 3) == 1.732


def test_mach2lambda():
    assert mach2lambda(0, 1.4) == 0
    assert mach2lambda(1, 1.4) == 1
    assert round(mach2lambda(0.381, 1.24), 3) == 0.4
    assert round(mach2lambda(1.732, 1.4), 3) == 1.5


def test_sonic():
    assert round(sonic(1.4, 286, 293), 1) == 342.5
