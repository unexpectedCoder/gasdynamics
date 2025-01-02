def accurate_throwing(x0: float,
                      c0: float,
                      k: float,
                      m_gas: float,
                      m_piston: float):
    """Функция точного решения баллистической задачи Лагранжа.

    Параметры
    ---------
    x0
        Начальная координата поршня, м.
    c0
        Скорость звука в сжатом газе, м/с.
    k
        Показатель адиабаты.
    m_gas
        Масса газа, кг.
    m_piston
        Масса поршня, кг.

    Возвращает
    ----------
        Кортеж
        (`интервал времени точного решения`,
        `функция координаты поршня от времени`,
        `функция скорости поршня от времени`).
    """
    gamma = (k + 1)/(2*k)
    m_ratio = m_gas / m_piston
    tau = x0 / c0
    duration = tau*(2 + gamma*m_ratio)

    bracket = lambda t: 1 + gamma*m_ratio * t/tau
    x = lambda t: x0 + 2*c0*t/(k - 1) + 2*k/(k - 1)/m_ratio*x0 * (
        1 - bracket(t)**(2/(k+1))
    )
    v = lambda t: 2*c0/(k - 1) * (
        1 - bracket(t)**((1-k)/(k+1))
    )

    return duration, x, v
