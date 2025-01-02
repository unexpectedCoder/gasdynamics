import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


class InitialDistributionVariant(Enum):
    """Возможные варианты функции q(x) начального распределения.
    
    Варианты
    --------
    PARABOLIC
        Парабола.
    STEP
        Ступенчатая функция.
    LINEAR
        Линейная функция.
    """
    PARABOLIC = 0
    STEP = 1
    LINEAR = 2


IDV = InitialDistributionVariant


def get_initial_distrib(idv: IDV):
    """Получить начальное распределение q(x).

    Параметры
    ---------
    v
        Вариант функции начального распределения.

    Возвращает
    ----------
        Выбранную функцию начального распределения q(x).
    """
    match idv:
        case IDV.PARABOLIC:
            return lambda x: \
                np.where((x >= 9) & (x <= 11), (x - 9)*(11 - x), 0.)
        case IDV.STEP:
            return lambda x: \
                np.where((x >= 0) & (x <= 3), 3., 0.)
        case IDV.LINEAR:
            return lambda x: \
                np.where((x >= 0) & (x <= 3), 2*x + 3, 0.)


def solve(mesh: np.ndarray,
          initial_s: np.ndarray,
          duration: float,
          speed: float,
          courant: float):
    """Функция решателя уравнения переноса.

    Параметры
    ---------
    mesh
        Расчётная сетка.
    initial_s
        Начальное распределение функции s на расчётной сетке.
    duration
        Интервал интегрирования по времени.
    speed
        Скорость переноса.
    courant
        Число Куранта.

    Возвращает
    ----------
        Решение уравнения переноса.
    """
    # Начальные условия
    s = initial_s.copy()
    # Пространственный шаг
    dx = mesh[1] - mesh[0]
    # Шаг по времени
    dt = calc_time_step(dx, speed, courant)
    # Счётчик времени
    t = 0.0

    # Цикл численного интегрирования
    while t < duration:
        s -= eq(s, speed, dt, dx)
        t += dt
    
    # Возвращение решения дифференциального уравнения
    return s


def eq(s: np.ndarray, speed: float, dt: float, dx: float):
    """Уравнение переноса величины `s` со скоростью `speed`
    за время `dt` на расчётной сетке с шагом `dx`.
    """
    return speed * dt/dx * (s - np.roll(s, 1))


def calc_time_step(dx: float, speed: float, courant: float):
    """Рассчитать шаг по времени для обеспечения устойчивости численного решения.

    Параметры
    ---------
    dx
        Шаг по координате.
    speed
        Скорость переноса.
    courant
        Число Куранта-Фридрихса-Леви.

    Возвращает
    ----------
        Временной шаг.
    """
    return abs(courant * dx / speed)


if __name__ == "__main__":
    import json
    import os


    DIR_NAME = os.path.dirname(__file__)
    
    path = os.path.join(DIR_NAME, "params", "transition_eq.json")
    with open(path, "r") as f:
        data = json.load(f)
        
    x_min, x_max = data["x_min"], data["x_max"]
    n = data["nodes"]
    duration = data["duration"]
    speed = data["speed"]
    courant = data["courant"]

    
    mesh = np.linspace(x_min, x_max, n)
    q = get_initial_distrib(IDV.PARABOLIC)
    s_0 = q(mesh)
    s = solve(mesh, s_0, duration, speed, courant)


    PICS_DIR = os.path.join(DIR_NAME, "pics")
    try:
        os.mkdir(PICS_DIR)
    except OSError:
        pass

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots()

        ax.plot(mesh, s_0, label="$s^0$")
        ax.plot(mesh, s, label="$s$")
        
        ax.set(xlabel="$x$", ylabel="$s$")
        ax.legend()

        path = os.path.join(PICS_DIR, "transition_eq.png")
        fig.savefig(path, dpi=300)

    plt.show()
