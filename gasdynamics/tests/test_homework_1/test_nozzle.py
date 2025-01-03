import numpy as np
import os
import pytest as pt

from gasdynamics.homework_1 import Nozzle
from gasdynamics.homework_1 import Task


path = os.path.join(
    os.path.dirname(__file__),
    "testdata",
    "variants.yml"
)
task = Task.from_file(path, variant=0)
noz = Nozzle.from_task(task)

testdata = [
    (0, 3.5*0.6),
    (0.75, 0.6),
    (3.5, 1.47)
]


def test_geometry():
    assert noz.d_chamber == task.d_chamber
    assert noz.d_critic == task.d_critic
    assert noz.area_ratio == task.area_ratio
    assert noz.alpha == task.alpha
    assert noz.beta == task.beta
    assert round(noz.d_out, 2) == 1.47
    assert round(noz.x_critic, 2) == 0.75
    assert round(noz.x_out, 2) == 3.5


@pt.mark.parametrize("x, expected", testdata)
def test_diameter_at(x, expected):
    print(noz.diameter_at(x))
    assert np.round(noz.diameter_at(x), 2) == expected
