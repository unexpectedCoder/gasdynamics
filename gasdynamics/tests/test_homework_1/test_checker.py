import os

import gasdynamics.config as config
from gasdynamics.homework_1 import checker


DIR_NAME = os.path.dirname(__file__)
SETTINGS_DIR = os.path.join(
    "gasdynamics", "homework_1"
)
config.load(SETTINGS_DIR)


def test_all_right():
    checked = checker.check(DIR_NAME)
    assert checker.all_right(checked[0])
    assert not checker.all_right(checked[1])


def test_check():
    checked = checker.check(DIR_NAME)

    correct = [
        v
        for k, sub_dict in checked[0].items()
        for v in sub_dict.values()
        if k not in ["Информация", "Проверка"]
    ]
    incorrect = [
        v
        for k, sub_dict in checked[1].items()
        for v in sub_dict.values()
        if k not in ["Информация", "Проверка"]
    ]

    assert all(correct)
    assert not all(incorrect)


def test_save_checked():
    checked = checker.check(DIR_NAME)
    checker.save_checked(checked, DIR_NAME)
