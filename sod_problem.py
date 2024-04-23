import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def get_initial_conditions(mesh: np.ndarray, k: float):
    """Получить начальные распределения вектора консервативных переменных q и давления газа p.

    Параметры
    ---------
    mesh
        Расчётная сетка.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Начальные распределения q и p.
    """
    n = mesh.size - 1
    middle = n // 2
    # Плотность
    rho = np.zeros(n, dtype=np.float64)
    rho[:middle] = 1.0
    rho[middle:] = 0.125
    # Давление
    p = np.zeros_like(rho)
    p[:middle] = 1.0
    p[middle:] = 0.1
    # Скорость
    v = np.zeros_like(p)
    # Полная энергия газа
    e = 1/(k-1) * p/rho + 0.5 * v**2
    return np.vstack([rho, rho*v, rho*e]), p


def solve(mesh: np.ndarray,
          initial_q: np.ndarray,
          initial_pressure: np.ndarray,
          duration: float,
          k: float,
          cfl: float):
    """Функция решателя системы дифференциальных уравнений с частными производными нестационарной газовой динамики.

    Параметры
    ---------
    mesh
        Расчётная сетка.
    initial_q
        Начальное распределение консервативных переменных (вектора q).
    initial_pressure
        Начальное распределение давления газа.
    duration
        Интервал интегрирования по времени.
    cfl
        Число CFL.

    Возвращает
    ----------
        Кортеж (t_store, q_store, p_store) с результатами интегрирования на каждом временном шаге. В t_store хранятся шаги по времени, в q_store - соответствующие значения вектора q, в p_store - соответствующее распределение давления.
    """
    dx = mesh[1] - mesh[0]
    t = 0.0
    q = initial_q.copy()
    p = initial_pressure.copy()

    t_store = [t]
    q_store = [q]
    p_store = [p]

    while t < duration:
        p = equation_of_state(q, k)
        c = calc_sonic(q, p, k)

        f_left, f_right = rusanov(q, c, p)
        dt = calc_time_step(q[1]/q[0], c, dx, cfl)
        q = step_up_euler(q, f_left, f_right, dt, dx)

        t += dt

        p_store.append(p)
        q_store.append(q)
        t_store.append(t)
    
    return t_store, q_store, p_store


def equation_of_state(q: np.ndarray, k: float):
    """Калорическое уравнение состояния.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Давление газа.
    """
    kin = 0.5 * (q[1] / q[0])**2
    eps = q[2] / q[0] - kin
    return eps * (k - 1) * q[0]


def calc_sonic(q: np.ndarray, p: np.ndarray, k: float):
    """Рассчитать скорость звука.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    p
        Распределение давления по контрольным объёмам.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Скорость звука в контрольных объёмах.
    """
    return np.sqrt(k * p / q[0])


def rusanov(q: np.ndarray, c: np.ndarray, p: np.ndarray):
    """Рассчитать потоки на интерфейсах (границах) контрольных объёмов по методу Русанова.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    c
        Скорость звука в контрольных объёмах.
    p
        Давление в контрольных объёмах.

    Возвращает
    ----------
        Кортеж (потоки через левые границы, потоки через правые границы).
    """
    # В текущем объёме
    f = calc_flux(q, p)
    # В левом объёме
    q_left = np.roll(q, 1, axis=1)
    q_left[:, 0] = q_left[:, 1]
    p_left = np.roll(p, 1)
    p_left[0] = p_left[1]
    f_left = calc_flux(q_left, p_left)
    # В правом объёме
    q_right = np.roll(q, -1, axis=1)
    q_right[:, -1] = q_right[:, -2]
    p_right = np.roll(p, -1)
    p_right[-1] = p_right[-2]
    f_right = calc_flux(q_right, p_right)
    # In result...
    c_max = np.max(c + np.abs(q[1] / q[0]))
    return (
        0.5 * (f + f_left - c_max*(q - q_left)),
        0.5 * (f + f_right - c_max*(q_right - q))
    )


def calc_flux(q: np.ndarray, p: np.ndarray):
    """Рассчитать потоки в контрольных объёмах.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    p
        Давление в контрольных объёмах.

    Возвращает
    ----------
        Потоки в контрольных объёмах.
    """
    e = q[2] / q[0]
    h = e + p / q[0]
    return np.vstack([
        q[1],
        q[1]**2 / q[0] + p,
        q[1] * h
    ])


def calc_time_step(v: np.ndarray, c: np.ndarray, dx: float, cfl: float):
    """Рассчитать величину временного шага для обеспечения устойчивости численного метода.

    Параметры
    ---------
    v
        Скорость газа в контрольных объёмах.
    c
        Скорость звука в контрольных объёмах.
    dx
        Шаг по координате.
    cfl
        Число CFL.

    Возвращает
    ----------
        Величину временного шага.
    """
    return cfl * dx / np.max(c + np.abs(v))


def step_up_euler(q: np.ndarray,
                  f_left: np.ndarray,
                  f_right: np.ndarray,
                  dt: float,
                  dx: float):
    """Интегрирование системы дифференциальных уравнений методом Годунова.

    Параметры
    ---------
    q
        Вектор консервативных переменных.
    f_left
        Потоки через левые границы объёмов.
    f_right
        Потоки через правые границы объёмов.
    dt
        Шаг по времени.
    dx
        Шаг по координате.

    Возвращает
    ----------
        Значение вектора консервативных переменных q на следующем временном слое (шаге).
    """
    return q - dt/dx * (f_right - f_left)


def plot_graphs(x: np.ndarray, q: np.ndarray, p: np.ndarray):
    """Построение необходимых графиков.

    Параметры
    ---------
    x
        Расчётная сетка.
    q
        Вектор консервативных переменных.
    p
        Давление газа в контрольных объёмах.

    Возвращает
    ----------
        Кортеж объектов графика типа figure и axis.
    """
    x = 0.5 * (x[1:] + x[:-1])
    fig, ax = plt.subplots(
        num="sod_problem",
        nrows=2,
        ncols=2,
        figsize=(10, 6),
        sharex=True
    )
    # Давление
    ax[0, 0].plot(x, p, c="k")
    ax[0, 0].set(ylabel=r'$p$')
    # Скорость
    ax[0, 1].plot(x, q[1] / q[0], c="k")
    ax[0, 1].set(ylabel=r'$u$')
    # Плотность
    ax[1, 0].plot(x, q[0], c="k")
    ax[1, 0].set(xlabel=r'$x$', ylabel=r'$\rho$')
    # Внутренняя энергия газа
    ax[1, 1].plot(x, q[2]/q[0] - 0.5*(q[1] / q[0])**2, c="k")
    ax[1, 1].set(xlabel=r'$x$', ylabel=r'$\varepsilon$')

    for a in ax.flat:
        a.axvline(0.5, c="grey", ls=":")

    fig.savefig("pics/sod_problem.png", dpi=300)
    return fig, ax


def animate(x: np.ndarray, qs: np.ndarray, ps: np.ndarray):
    x = 0.5 * (x[1:] + x[:-1])
    fig, axes = plt.subplots(
        num="sod_problem_animation",
        nrows=2,
        ncols=2,
        figsize=(10, 6),
        sharex=True
    )
    xdata = x
    ydata = [
        ps[0],
        qs[0][1] / qs[0][0],
        qs[0][0],
        qs[0][2]/qs[0][0] - 0.5*(qs[0][1] / qs[0][0])**2
    ]
    qs.pop(0)
    lns = [ax.plot(xdata, yd, c="b")[0] for ax, yd in zip(axes.flat, ydata)]

    def init():
        for ax in axes.flat:
            ax.axvline(0.5, c="k", ls=":")
        axes[0, 0].set(ylabel=r"$p$")
        axes[0, 1].set(ylabel=r"$u$", ylim=(-1, 1))
        axes[1, 0].set(xlabel=r"$x$", ylabel=r"$\rho$")
        axes[1, 1].set(xlabel=r"$x$", ylabel=r"$\varepsilon$", ylim=(0, 3))
        return lns

    def update(frame):
        q, p = frame
        lns[0].set_data(xdata, p)
        lns[1].set_data(xdata, q[1]/q[0])
        lns[2].set_data(xdata, q[0])
        lns[3].set_data(xdata, q[2]/q[0] - 0.5*(q[1] / q[0])**2)
        return lns

    ani = FuncAnimation(
        fig, update, frames=list(zip(qs, ps)),
        init_func=init, blit=True, interval=40
    )
    ani.save("pics/sod_problem.gif", dpi=100)


if __name__ == "__main__":
    n = 100
    duration = 0.15
    cfl = 0.75
    k_adiabatic = 1.4

    mesh = np.linspace(0, 1, n + 1)
    q0, p0 = get_initial_conditions(mesh, k_adiabatic)
    time, q, pressure = solve(mesh, q0, p0, duration, k_adiabatic, cfl)

    with plt.style.context("sciart.mplstyle"):
        plot_graphs(mesh, q[-1], pressure[-1])
        animate(mesh, q, pressure)
    
    plt.show()
