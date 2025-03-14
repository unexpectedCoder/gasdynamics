import matplotlib.pyplot as plt
import numpy as np


def initialize(mesh: np.ndarray,
               p0: float,
               temp0: float,
               gas_const: float,
               k: float):
    """Инициализация начального состояния (вектор q и давление p).

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    p0
        Начальное давление, Па.
    temp0
        Начальная температура, К.
    gas_const
        Удельная газовая постоянная, Дж/(кг К).
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Кортеж (вектор консервативных переменных, давление) в контрольных объёмах.
        Важно! Массивы содержат две фиктивные ячейки (крайне левая и крайне правая).
    """
    n = mesh.size + 1
    p = np.full(n, p0)
    rho = p / temp0 / gas_const
    e = p / rho / (k - 1)
    q = np.vstack([rho, np.zeros(n, dtype=np.float64), rho*e])
    return q, p


def solve(mesh: np.ndarray,
          q0: np.ndarray,
          p0: np.ndarray,
          k: float,
          piston_mass: float,
          piston_area: float,
          tube_length: float,
          courant: float,
          alpha: float,
          beta: float):
    """Функция решателя баллистической задачи Лагранжа в эйлеровых координатах.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    q0
        Начальный вектор консервативных переменных в контрольных объёмах.
    p0
        Начальное распределение давления по контрольным объёмам, Па.
    k
        Показатель адиабаты.
    piston_mass
        Масса поршня, кг.
    piston_area
        Площадь поршня, м^2.
    tube_length
        Длина трубы, м.
    courant
        Число Куранта-Фридрихса-Леви.
    alpha, optional
        Коэффициент в функции расчёта давления на интерфейсах, по умолчанию 3/16.
    beta, optional
        Коэффициент в функции расчёта числа Маха на интерфейсах, по умолчанию 1/4.

    Возвращает
    ----------
        Кортеж массивов (время, расчётная сетка, вектор консервативных переменных, давление, скорость поршня).
    """
    t, q, p = 0.0, q0, p0
    dx = mesh[1] - mesh[0]
    piston_x, piston_v = mesh[-1], 0.0

    t_store = [t]
    mesh_store = [mesh.copy()]
    q_store = [q[:, 1:-1].copy()]
    p_store = [p[1:-1].copy()]
    piston_v_store = [piston_v]

    while mesh[-1] < tube_length:
        p = equation_of_state(q, k)
        c = calc_sonic(p, q[0], k)
        dt = calc_time_step(q, dx, c, courant)

        piston_v += dt * p[-2] * piston_area / piston_mass
        piston_x += dt * piston_v

        new_mesh, new_dx = np.linspace(0, piston_x, mesh.size, retstep=True)
        u_borders = np.linspace(0, piston_v, mesh.size)

        q[:, 0] = np.array([q[0, 1], -q[1, 1], q[2, 1]])
        q[:, -1] = np.array([
            q[0, -2],
            -q[1, -2] + 2*q[0, -2]*piston_v,
            q[2, -2]
        ])
        flux = ausm_plus(q, u_borders, c, p, alpha, beta)
        f_left, f_right = flux[:, :-1], flux[:, 1:]
        q = step_up_euler(new_dx, dx, q, f_left, f_right, dt)
        
        t += dt
        mesh, dx = new_mesh, new_dx

        t_store.append(t)
        mesh_store.append(mesh.copy())
        q_store.append(q[:, 1:-1].copy())
        p_store.append(p[1:-1].copy())
        piston_v_store.append(piston_v)

    return (
        np.array(t_store),
        np.array(mesh_store),
        np.array(q_store),
        np.array(p_store),
        np.array(piston_v_store)
    )


def equation_of_state(q: np.ndarray, k: float):
    """Уравнение состояния для расчёта давления газа.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Давление газа в контрольных объёмах, Па.
    """
    u = q[1] / q[0]
    e = q[2]/q[0] - 0.5*u**2
    return (k - 1)*e*q[0]


def calc_sonic(p: np.ndarray, rho: np.ndarray, k: float):
    """Рассчитать скорость звука в контрольных объёмах.

    Параметры
    ---------
    p
        Давление газа в контрольных объёмах, Па.
    rho
        Плотность газа в контрольных объёмах, кг/м^3.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Скорость звука в контрольных объёмах, м/с.
    """
    return np.sqrt(k * p / rho)


def ausm_plus(q: np.ndarray,
              u_borders: np.ndarray,
              c: np.ndarray,
              p: np.ndarray,
              alpha: float,
              beta: float):
    """Реализация метода AUSM+.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    u_borders
        Скорость интерфейсов, м/с.
    c
        Скорость звука в контрольных объёмах, м/с.
    p
        Давление газа в контрольных объёмах, Па.
    alpha
        Коэффициент в функции расчёта давления на интерфейсах.
    beta
        Коэффициент в функции расчёта числа Маха на интерфейсах.

    Возвращает
    ----------
        Вектор потока на интерфейсах.
    """
    interface_sonic = 0.5*(c[:-1] + c[1:])
    u = q[1] / q[0]
    mach_left = (u[:-1] - u_borders) / interface_sonic
    mach_right = (u[1:] - u_borders) / interface_sonic

    q_left, p_left = q[:, :-1], p[:-1]
    flux_left = calc_ausm_flux(q_left, p_left)
    q_right, p_right = q[:, 1:], p[1:]
    flux_right = calc_ausm_flux(q_right, p_right)

    interface_mach = calc_interface_mach(mach_left, mach_right, beta)

    interface_pressure = calc_interface_pressure(
        mach_left, mach_right, p, alpha
    )

    return 0.5*interface_sonic * (
        interface_mach*(flux_right + flux_left)
        - np.abs(interface_mach) * (flux_right - flux_left)
    ) + np.vstack([
        np.zeros_like(interface_pressure),
        interface_pressure,
        interface_pressure * u_borders
    ])


def calc_ausm_flux(q: np.ndarray, p: np.ndarray):
    """Рассчитать вектор потока для метода AUSM+.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    p
        Давление газа в контрольных объёмах, Па.

    Возвращает
    ----------
        Вектор потока для метода AUSM+.
    """
    return np.vstack([q[0], q[1], q[2] + p])


def calc_interface_mach(mach_left: np.ndarray,
                        mach_right: np.ndarray,
                        beta: float):
    """Рассчитать число Маха на интерфейсе.

    Параметры
    ---------
    mach_left
        Число Маха в объёме слева от интерфейса.
    mach_right
        Число Маха в объёме справа от интерфейса.
    beta
        Коэффициент функции расчёта.

    Возвращает
    ----------
        Число Маха на интерфейсе.
    """
    return f_beta(mach_left, "+", beta) + f_beta(mach_right, "-", beta)


def f_beta(mach: np.ndarray, sign: str, beta: float):
    """Функция расчёта числа Маха относительно интерфейса.

    Параметры
    ---------
    mach
        Число Маха в контрольном объёме.
    sign
        Вариант (знак) функции: "+" или "-".
    beta
        Коэффициент (параметр) функции.

    Возвращает
    ----------
        Относительное число Маха.
    """
    abs_mach = np.abs(mach)
    if sign == "+":
        return np.where(
            abs_mach >= 1,
            0.5*(mach + abs_mach),
            0.25*(mach + 1)**2 * (1 + 4*beta*(mach - 1)**2)
        )
    return np.where(
        abs_mach >= 1,
        0.5*(mach - abs_mach),
        -0.25*(mach - 1)**2 * (1 + 4*beta*(mach + 1)**2)
    )


def calc_interface_pressure(mach_left: np.ndarray,
                            mach_right: np.ndarray,
                            p: np.ndarray,
                            alpha: float):
    """Рассчитать давление на интерфейсе.

    Параметры
    ---------
    mach_left
        Число Маха в объёме слева от интерфейса.
    mach_right
        Число Маха в объёме справа от интерфейса.
    p
        Давление газа в контрольном объёме, Па.
    alpha
        Коэффициент функции пересчёта.

    Возвращает
    ----------
        Давление газа на интерфейсе, Па.
    """
    return \
        g_alpha(mach_left, "+", alpha)*p[:-1] \
        + g_alpha(mach_right, "-", alpha)*p[1:]


def g_alpha(mach: np.ndarray, sign: str, alpha: float):
    """Функция расчёта давления газа относительно интерфейса.

    Параметры
    ---------
    mach
        Число Маха в контрольном объёме.
    sign
        Вариант (знак) функции: "+" или "-".
    alpha
        Коэффициент (параметр) функции.

    Возвращает
    ----------
        Относительное давление на интерфейсе.
    """
    abs_mach = np.abs(mach)
    if sign == "+":
        return np.where(
            abs_mach >= 1,
            (mach + abs_mach) / (2*mach),
            (mach + 1)**2 * ((2 - mach)/4 + alpha*mach*(mach - 1)**2)
        )
    return np.where(
        abs_mach >= 1,
        (mach - abs_mach) / (2*mach),
        (mach - 1)**2 * ((2 + mach)/4 - alpha*mach*(mach + 1)**2)
    )


def calc_time_step(q: np.ndarray, dx: float, c: np.ndarray, courant: float):
    """Рассчитать величину временного шага для обеспечения устойчивости численного метода.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    dx
        Шаг расчётной сетки, м.
    c
        Скорость звука в контрольных объёмах, м/с.
    courant
        Число CFL.

    Возвращает
    ----------
        Шаг по времени, с.
    """
    return courant * np.min(dx / (c + np.abs(q[1]/q[0])))


def step_up_euler(new_dx: np.ndarray,
                  dx: np.ndarray,
                  q: np.ndarray,
                  f_left: np.ndarray,
                  f_right: np.ndarray,
                  dt: float):
    """Рассчитать вектор консервативных переменных на новом временном шаге.

    Допущение: используется равномерная расчётная сетка.

    Параметры
    ---------
    new_dx
        Шаг новой расчётной сетки, м.
    dx
        Шаг прошлой расчётной сетки, м.
    q
        Текущий вектор консервативных переменных.
    f_left
        Потоки через левые интерфейсы объёмов.
    f_right
        Потоки через правые интерфейсы объёмов.
    dt
        Шаг по времени, с.

    Возвращает
    ----------
        Новый вектор консервативных переменных.
    """
    new_q = q.copy()
    new_q[:, 1:-1] = dx/new_dx * (q[:, 1:-1] - dt/dx * (f_right - f_left))
    return new_q


if __name__ == "__main__":
    import json
    import os

    from utils import accurate_throwing


    DIR_NAME = os.path.dirname(__file__)


    PARAMS_DIR = os.path.join(DIR_NAME, "params")
    path = os.path.join(PARAMS_DIR, "euler.json")
    with open(path, "r") as f:
        data = json.load(f)

    n = data["nodes"]
    x_0 = data["piston_x_0"]
    m_p = data["piston_mass"]
    d = data["tube_d"]
    tube_len = data["tube_len"]
    p_0 = data["p_0"]
    T_0 = data["T_0"]
    R = data["R"]
    k = data["k"]
    alpha, beta = data["alpha"], data["beta"]
    courant = data["courant"]


    # Численное решение
    S = 0.25 * np.pi * d**2
    mesh = np.linspace(0, x_0, n + 1)
    q, p = initialize(mesh, p_0, T_0, R, k)
    times, meshes, qs, ps, u_p = solve(
        mesh, q, p, k, m_p, S, tube_len, courant, alpha, beta
    )


    # Точное решение
    rho_0 = p_0 / (R * T_0)
    sonic_0 = calc_sonic(p_0, rho_0, k)
    m_g = rho_0 * S * x_0
    duration_acc, x_p_acc, u_p_acc = accurate_throwing(
        x_0, sonic_0, k, m_g, m_p
    )


    acc_index = np.argwhere(times < duration_acc)
    t_acc = times[acc_index]
    acc_speed = u_p_acc(t_acc)
    num_speed = u_p[acc_index]
    acc_x = x_p_acc(t_acc)
    num_x = meshes[acc_index, -1]


    PICS_DIR = os.path.join(DIR_NAME, "pics")
    try:
        os.mkdir(PICS_DIR)
    except OSError:
        pass

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num="euler_piston_x")
        ax.plot(times*1e3, meshes[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, x_p_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$x_\mathrm{п}$, м")
        ax.legend()
        fig.savefig(f"{os.path.join(PICS_DIR, fig.get_label())}.png", dpi=300)

        fig, ax = plt.subplots(num="euler_p_p_contour")
        t_arr = np.repeat(times.reshape((times.size, 1)), p[0].size, axis=1)
        x = 0.5*(meshes[:, 1:] + meshes[:, :-1])
        img = ax.pcolormesh(
            x, t_arr*1e3, ps*1e-6, cmap="Blues", shading="gouraud"
        )
        fig.colorbar(img, ax=ax, label="$p$, МПа")
        ax.set(xlabel="$x$, м", ylabel="$t$, мс")
        ax.grid(False)
        fig.savefig(f"{os.path.join(PICS_DIR, fig.get_label())}.png", dpi=300)

        fig, ax = plt.subplots(num="euler_u_p")
        ax.plot(times*1e3, u_p, label="Численное решение")
        ax.plot(t_acc*1e3, u_p_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel="$u_\mathrm{п}$, м/с")
        ax.legend()
        fig.savefig(f"{os.path.join(PICS_DIR, fig.get_label())}.png", dpi=300)

        fig, ax = plt.subplots(num="euler_p")
        ax.plot(times*1e3, ps[:, 0]*1e-6, label="На дно канала")
        ax.plot(times*1e3, ps[:, -1]*1e-6, label="На поршень")
        ax.set(xlabel="$t$, мс", ylabel="$p$, МПа")
        ax.legend()
        fig.savefig(f"{os.path.join(PICS_DIR, fig.get_label())}.png", dpi=300)

        fig, ax = plt.subplots(num="euler_compare")
        delta_speed = np.abs(acc_speed - num_speed)/acc_speed * 100
        ax.plot(t_acc*1e3, delta_speed, label=r"$\mathrm{\delta} u_\mathrm{п}$")
        delta_x = np.abs(acc_x - num_x)/acc_x * 100
        ax.plot(t_acc*1e3, delta_x, label=r"$\mathrm{\delta} x_\mathrm{п}$")
        ax.set(
            xlabel="$t$, мс",
            ylabel=r"Относительная погрешность $\mathrm{\delta}$, %"
        )
        ax.legend()
        fig.savefig(os.path.join(PICS_DIR, f"{fig.get_label()}.png"), dpi=300)
