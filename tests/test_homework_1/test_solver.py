import numpy as np
import os

from homework_1 import solver
from homework_1 import Nozzle
from homework_1 import Task


test_path = os.path.join(
    os.path.dirname(__file__),
    "testdata",
    "test_variant.yml"
)
test_variant = 0
task = Task.from_file(test_path, test_variant)
noz = Nozzle.from_task(task)
p_a = 1e5


def test_solver():
    sol = solver.solve(task, noz, p_a)
    assert np.round(sol.mass_consumption, 1) == 1068.5
    assert np.round(sol.speed_out) == 2172
    assert np.round(sol.pressure_out * 1e-5, 3) == 1.225
    assert np.round(sol.thrust * 1e-6, 3) == 2.359
    assert np.round(sol.thrust_specific) == 2208
    assert np.round(sol.thrust_space * 1e-6, 2) == 2.53
    assert np.round(sol.thrust_specific_space) == 2367
    assert np.round(sol.ideal_speed) == 3809


def test_adapt_nozzle():
    adapted = solver.adapt_nozzle(task, noz, p_a)
    assert round(adapted.mach_out, 2) == 3.11
    assert round(adapted.d_out, 3) == 1.581
    assert round(adapted.x_out, 3) == 3.847
    assert round(adapted.dx, 3) == 0.352
    assert round(adapted.dx_percents, 1) == 10.1
