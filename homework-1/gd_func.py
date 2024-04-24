import numpy as np


def temp_ratio(mach: float, k: float):
    """Функция температуры - отношение полной температуры потока к статической.

    На входе
    --------
    mach
        Число Маха.
    k : float
        Показатель адиабаты газа.

    На выходе
    ---------
        Отношение полной температуры потока к статической.
    """    
    return 1 + 0.5*(k - 1) * mach**2


def dens_ration(mach: float, k: float):
    """Функция плотности -
    отношение полной плотности потока газа к статической.

    На входе
    --------
    mach
        Число Маха.
    k
        Показатель адиабаты газа.

    На выходе
    ---------
        Отношение полной плотности потока к статической.
    """    
    return temp_ratio(mach, k)**(1/(k - 1))


def press_ratio(mach: float, k: float):
    """Функция давления - отношения полного давления потока к статическому.

    На входе
    --------
    mach
        Число Маха.
    k
        Показатель адиабаты газа.

    На выходе
    ---------
        Отношение полного давления к статическому.
    """
    return temp_ratio(mach, k)**(k/(k - 1))


def square_ratio(mach: float, k: float):
    """Отношение площади поперечного сечения к площади критического.

    На входе
    --------
    mach
        Число Маха.
    k
        Показатель адиабаты газа.

    На выходе
    ---------
        Отношение площади поперечного сечения к площади критического.
    """    
    a = (k + 1) / (2*(k - 1))
    return 1/mach * (2/(k + 1))**a * temp_ratio(mach, k)**a


def lambda2mach(lam: float, k: float):
    """Перевод безразмерной скорости потока `lam`
    в число Маха (`k` - показатель адиабаты газа).
    """
    return lam * np.sqrt(2/(k + 1) / (1 - (k - 1)/(k + 1) * lam**2))


def mach2lambda(mach: float, k: float):
    """Перевод числа Маха `mach` в безразмерную скорость потока (`k` - показатель адиабаты газа).
    """
    return mach * np.sqrt(0.5*(k + 1) / temp_ratio(mach, k))


def sonic(k: float, R: float, T: float):
    """Расчёт скорости звука.

    На входе
    --------
    k
        Показатель адиабаты
    R
        Удельная газовая постоянная, Дж/(кг К)
    T
        Статическая температура газа, К.

    На выходе
    ---------
        Местная скорость звука, м/с.
    """    
    return np.sqrt(k * R * T)
