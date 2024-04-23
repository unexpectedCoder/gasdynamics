def accurate_throwing_solution(x0: float,
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
        Кортеж (интервал времени точного решения, функция координаты поршня от времени, функция скорости поршня от времени).
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


if __name__ == "__main__":
    import sympy as sp
    from IPython.display import display

    sp.init_printing(use_unicode=True)

    p, eps = sp.symbols(r"p_{i+1/2}^{n+1} \varepsilon_{i+1/2}^{n+1}")

    p_curr, eps_curr = sp.symbols(r"p_{i+1/2}^n \varepsilon_{i+1/2}^n")
    u_curr_i, u_next_i = sp.symbols(r"u_i^{n+1} u_{i+1}^{n+1}")
    tau = sp.symbols(r"\tau^n")
    q = sp.symbols(r"\xi_{i+1/2}")
    rho = sp.symbols(r"\rho_{i+1/2}^{n+1}")
    s, k = sp.symbols("S k")

    p_eq = sp.Eq(p, (k - 1)*rho*eps)
    eps_eq = sp.Eq(
        eps,
        eps_curr - tau/q * s * (p + p_curr)/2 * (u_next_i - u_curr_i)
    )

    sol = sp.solve([p_eq, eps_eq], [p, eps])

    display(sol)
    display(sp.simplify(sol[eps]))
    display(sp.simplify(sol[p]))
