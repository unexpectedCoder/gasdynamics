#!.venv/bin/python

import numpy as np
from math import sqrt, tan, degrees, pi

from task import Task


class Nozzle:
    """Класс, описывающий сопло Лаваля в одномерной постановке.

    Параметры конструктора
    ----------------------
    d_chamber
        Диаметр камеры сгорания двигателя, м.
    d_critic
        Диаметр критического сечения, м.
    area_ratio
        Отношение площадей выходного и критического сечений.
    confuser_angle
        Угол конуса конфузора, рад.
    diffuser_angle
        Угол конуса диффузора, рад.
    """    

    def __init__(self,
                 d_chamber: float,
                 d_critic: float,
                 area_ratio: float,
                 confuser_angle: float,
                 diffuser_angle: float):
        self.d_chamber = d_chamber
        self.d_critic = d_critic
        self.area_ratio = area_ratio
        self.alpha = confuser_angle
        self.beta = diffuser_angle
        
        self._init_geometry()
    
    def _init_geometry(self):
        self.d_out = self.d_critic * sqrt(self.area_ratio)
        self.x_critic = 0.5*(self.d_chamber - self.d_critic) / tan(self.alpha)
        self.x_out = self.x_critic + 0.5*(self.d_out - self.d_critic) / tan(self.beta)
        self.area_critic = 0.25 * pi * self.d_critic**2

    @classmethod
    def from_task(cls, task: Task):
        """Создать объект сопла, используя объект задачи.

        На входе
        --------
        task
            Задача.

        На выходе
        ---------
            Сопло.
        """
        return cls(
            task.d_chamber,
            task.d_critic,
            task.area_ratio,
            task.alpha,
            task.beta
        )
    
    def diameter_at(self, x: float):
        """Рассчитать величину диаметра поперечного сечения сопла на заданной координате.

        На входе
        --------
        x
            Координата сечения, м.

        На выходе
        ---------
            Диаметр сечения, м.
        """
        x = np.asarray(x)
        d = np.empty_like(x, dtype=np.float64)
        confuser = x < self.x_critic
        d[confuser] = \
            self.d_chamber \
            - (self.d_chamber - self.d_critic) * x[confuser]/self.x_critic
        diffuser = x >= self.x_critic
        d[diffuser] = \
            self.d_critic \
            + (self.d_out - self.d_critic) \
            * (x[diffuser] - self.x_critic)/(self.x_out - self.x_critic)
        return d
    
    def as_dict(self):
        """Возвращает словарь с информацией о геометрии сопла.
        """        
        return {
            "Диаметр камеры сгорания, м": self.d_chamber,
            "Диаметр критического сечения, м": self.d_critic,
            "Коэффициент уширения сопла": self.area_ratio,
            "Угол конфузора, градус": degrees(self.alpha),
            "Угол диффузора, градус": degrees(self.beta),
            "Диаметр выходного сечения, м": self.d_out,
            "Координата критического сечения, м": self.x_critic,
            "Координата выходного сечения, м": self.x_out,
            "Площадь критического сечения, м^2": self.area_critic
        }
    
    def __str__(self):
        return self.geometry_info()
    
    @property
    def length(self):
        """Длина сопла, м."""
        return self.x_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    path = os.path.join("initdata", "variants.yaml")
    task = Task.from_file(path, variant=0)
    noz = Nozzle.from_task(task)

    x = np.linspace(0, noz.length, 200)
    d = noz.diameter_at(x)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots()
        ax.plot(x, 0.25 * pi * d**2, c="k")
        ax.set(xlabel="$x$, м", ylabel="$S(x)$, м$^2$")
        ax.grid(True, ls=":", c="k")
        plt.show()
