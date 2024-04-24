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


def get_initial_distribution(v: InitialDistributionVariant):
    """Получить начальное распределение q(x).

    Параметры
    ---------
    v
        Вариант функции начального распределения.

    Возвращает
    ----------
        Выбранную функцию начального распределения q(x).
    """
    match v:
        case InitialDistributionVariant.PARABOLIC:
            return \
                lambda x: np.where((x >= 9) & (x <= 11), (x - 9)*(11 - x), 0.0)
        case InitialDistributionVariant.STEP:
            return lambda x: np.where((x >= 0) & (x <= 3), 3.0, 0.0)
        case InitialDistributionVariant.LINEAR:
            return lambda x: np.where((x >= 0) & (x <= 3), 2*x + 3, 0.0)


def solve(mesh: np.ndarray,
          initial_s: np.ndarray,
          duration: float,
          speed: float,
          cfl: float):
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
    cfl
        Число CFL.

    Возвращает
    ----------
        Решение уравнения переноса.
    """
    dx = mesh[1] - mesh[0]
    t = 0.0
    dt = calc_time_step(dx, speed, cfl)
    s = initial_s.copy()

    while t < duration:
        s -= speed * dt/dx * (s - np.roll(s, 1))
        t += dt
    
    return s


def calc_time_step(dx: float, speed: float, cfl: float):
    """Рассчитать шаг по времени для обеспечения устойчивости численного решения.

    Параметры
    ---------
    dx
        Шаг по координате.
    speed
        Скорость переноса.
    cfl
        Число Куранта-Фридрихса-Леви.

    Возвращает
    ----------
        Временной шаг.
    """
    return abs(cfl * dx / speed)


def plot_s(x: np.ndarray, s: np.ndarray, initial_s: np.ndarray):
    """Построить графики начального и итогового распределения.

    Параметры
    ---------
    x
        Расчётная сетка.
    s
        Итоговое распределение.
    initial_s
        Начальное распределение.

    Возвращает
    ----------
        Объекты графика типа figure и axis.
    """
    fig, ax = plt.subplots(num="transition_equation")
    ax.plot(x, s, label="$s(x)$")
    ax.plot(x, initial_s, label="$s^0(x)$")
    return fig, ax


if __name__ == "__main__":
    n = 300
    x_min, x_max = 0.0, 20.0
    duration = 0.2
    speed = 15.0
    cfl = 0.8

    mesh = np.linspace(x_min, x_max, n + 1)
    q = get_initial_distribution(InitialDistributionVariant.PARABOLIC)
    initial_s = q(mesh)
    s = solve(mesh, initial_s, duration, speed, cfl)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plot_s(mesh, s, initial_s)
        ax.set(xlabel="$x$", ylabel="$s$")
        ax.legend()
        fig.savefig("labs-1/pics/transition_equation")

    plt.show()
