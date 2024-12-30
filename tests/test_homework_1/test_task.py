import os
import pytest as pt

from homework_1 import Task


test_path_csv = os.path.join(
    os.path.dirname(__file__),
    "testdata",
    "variants.csv"
)
testdata_csv = [
    (
        test_path_csv,
        0,
        (0, 5.4e6, 2705, 325, 1.24, 0.6, 6, 3.5*0.6, 45,9, 0.8)
    ),
    (
        test_path_csv,
        3,
        (3, 4.4e6, 3050, 350, 1.27, 0.6, 6, 2.25*0.6, 48, 10, 0.75)
    )
]

test_path_yaml = os.path.join(
    os.path.dirname(__file__),
    "testdata",
    "variants.yml"
)
testdata_yaml = [
    (
        test_path_yaml,
        0,
        (0, 5.4e6, 2705, 325, 1.24, 0.6, 6, 3.5*0.6, 45,9, 0.8)
    ),
    (
        test_path_yaml,
        3,
        (3, 4.4e6, 3050, 350, 1.27, 0.6, 6, 2.25*0.6, 48, 10, 0.75)
    )
]

testdata = testdata_csv + testdata_yaml

wrong_variant = -777


@pt.mark.parametrize("path, variant, expected", testdata_csv)
def test_from_csv(path, variant, expected):
    task = Task.from_csv(path, variant)
    assert pt.approx(task.as_tuple()) == expected


def test_invalid_variant_csv():
    with pt.raises(ValueError):
        Task.from_csv(test_path_csv, wrong_variant)


@pt.mark.parametrize("path, variant, expected", testdata_yaml)
def test_from_yaml(path, variant, expected):
    task = Task.from_yaml(path, variant)
    assert pt.approx(task.as_tuple()) == expected


def test_invalid_variant_yaml():
    with pt.raises(KeyError):
        Task.from_yaml(test_path_yaml, wrong_variant)


@pt.mark.parametrize("path, variant, expected", testdata)
def test_from_file(path, variant, expected):
    task = Task.from_file(path, variant)
    assert pt.approx(task.as_tuple()) == expected


def test_invalid_variant():
    with pt.raises(ValueError):
        Task.from_file(test_path_csv, wrong_variant)
    with pt.raises(KeyError):
        Task.from_file(test_path_yaml, wrong_variant)
