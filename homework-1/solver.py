import numpy as np
import scipy.optimize as opti
from dataclasses import dataclass

import gd_func as gdf
from nozzle import Nozzle
from task import Task


@dataclass(frozen=True)
class Solution:
    x: np.ndarray
    square: np.ndarray
    mach: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    specific_mass: np.ndarray
    mass_consumption: float
    speed_out: float
    pressure_out: float
    thrust: float
    thrust_specific: float
    thrust_space: float
    thrust_specific_space: float
    ideal_speed: float
    adapted_level: float

    def __str__(self):
        return f"Решение:\n{self.as_dict()}"
    
    def as_dict(self):
        return {
            "Массовый расход, кг/с": float(self.mass_consumption),
            "Скорость потока на выходе, м/с": float(self.speed_out),
            "Давление на выходе, Па": float(self.pressure_out),
            "Тяга, Н": float(self.thrust),
            "Удельная тяга, м/с": float(self.thrust_specific),
            "Тяга в пустоте, Н": float(self.thrust_space),
            "Удельная тяга в пустоте, м/с": float(self.thrust_specific_space),
            "Скорость идеальной ракеты, м/с": float(self.ideal_speed),
            "Степень нерасчётности сопла": \
                f"{float(self.adapted_level)} " + (
                    "(недорасширение)" if self.adapted_level < 1 
                    else "(перерасширение)"
                )
        }


def solve(task: Task, noz: Nozzle, p_a: float):
    x = np.linspace(0, noz.length, 300)
    S = 0.25 * np.pi * noz.diameter_at(x)**2
    mach = []
    for xi, si in zip(x, S):
        mach.append(opti.root_scalar(
            area_equation,
            args=(si, 0.25*np.pi*noz.d_critic**2, task.k),
            bracket=[1e-6, 1] if xi < noz.x_critic else [1, 10]
        ).root)
    mach = np.array(mach)
    p = task.p0 / gdf.press_ratio(mach, task.k)
    rho0 = task.p0 / (task.R * task.T0)
    rho = rho0 / gdf.dens_ration(mach, task.k)
    T = task.T0 / gdf.temp_ratio(mach, task.k)
    v = mach * gdf.sonic(task.k, task.R, T)
    j = rho * v
    G = task.p0*noz.area_critic \
        * np.sqrt(task.k / task.R / task.T0) \
        * (2 / (task.k + 1))**(0.5*(task.k + 1)/(task.k - 1))
    v_out = v[-1]
    P = G*v_out + S[-1]*(p[-1] - p_a)
    P_spec = P / G
    P_space = G*v_out + S[-1]*p[-1]
    P_space_spec = P_space / G
    v_ideal = -P_space_spec*np.log(1 - task.rel_propel_mass)
    return Solution(
        x, S, mach, p, rho, T, j,
        G, v_out, p[-1],
        P, P_spec, P_space, P_space_spec, v_ideal,
        p_a / p[-1]
    )


def area_equation(mach: float, s: float, s_critic: float, k: float):
    """Уравнение для нахождения числа Маха `mach`
    в сечении с произвольной координатой.

    На входе
    ----------
    mach : float
        Переменная, для которой ищется решение (число Маха).
    s : float
        Площадь поперечного сечения с произвольной координатой, м^2.
    s_critic : float
        Площадь критического сечения, м^2.
    k : float
        Показатель адиабаты газа.

    На выходе
    -------
    float
        Значение выражения.
    """    
    return s - s_critic*gdf.square_ratio(mach, k)


@dataclass(frozen=True)
class AdaptedNozzle:
    mach_out: float
    d_out: float
    x_out: float
    dx: float
    dx_percents: float

    def as_dict(self):
        return {
            "Число Маха на выходе": float(self.mach_out),
            "Диаметр выходного сечения, м": float(self.d_out),
            "Координата выходного сечения, м": float(self.x_out),
            "Удлинение (+) или укорочение (-) сопла, м": float(self.dx),
            "Относительное удлинение (+) или укорочение (-) сопла, %": float(self.dx_percents)
        }


def adapt_nozzle(task: Task, noz: Nozzle, p_a: float):
    mach_out = np.sqrt(
        2 / (task.k - 1) * (
            (task.p0 / p_a)**((task.k - 1)/task.k) - 1
        )
    )
    S_out = noz.area_critic * gdf.square_ratio(mach_out, task.k)
    d_out = 2 * np.sqrt(S_out / np.pi)
    x_out = noz.x_critic + 0.5*(d_out - noz.d_critic)/ np.tan(noz.beta)
    dx = x_out - noz.x_out
    dx_percents = dx / noz.x_out * 100
    return AdaptedNozzle(mach_out, d_out, x_out, dx, dx_percents)
