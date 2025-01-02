import matplotlib.pyplot as plt
import numpy as np
from celluloid import Camera


def initialize(mesh: np.ndarray,
               p0: float,
               rho0: float,
               piston_mass: float,
               tube_area: float,
               k: float):
    """Инициализация начального состояния.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    p0
        Начальное давление сжатого газа, Па.
    rho0
        Начальная плотность газа, кг/м^3.
    piston_mass
        Масса поршня, кг.
    tube_area
        Площадь поршня, м^2.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Кортеж
        (`скорость границ`,
        `давление`,
        `плотность`,
        `внутренняя энергия`,
        `массовая лагранжева координата`,
        `масса перегородок`).
    """
    u_borders = np.zeros_like(mesh)
    n = mesh.size - 1
    p = np.full(n, p0)
    rho = np.full(n, rho0)
    inner_e = p / rho / (k - 1)
    
    dx = mesh[1] - mesh[0]
    lagrange = rho * tube_area * dx
    interface_lagrange = np.zeros_like(mesh)
    interface_lagrange[-1] = piston_mass

    return u_borders, p, rho, inner_e, lagrange, interface_lagrange


def solve(mesh: np.ndarray,
          u_borders: np.ndarray,
          p: np.ndarray,
          rho: np.ndarray,
          inner_e: np.ndarray,
          lagrange: np.ndarray,
          interface_lagrange: np.ndarray,
          tube_length: float,
          tube_area: float,
          k: float,
          courant: float):
    """Функция решателя баллистической задачи Лагранжа.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    u_borders
        Начальная скорость границ объёмов.
    p
        Начальное давление в контрольных объёмах, Па.
    rho
        Плотность в контрольных объёмах, кг/м^3.
    inner_e
        Внутренняя энергия контрольных объёмов, Дж/кг.
    lagrange
        Массовая лагранжева координата контрольных объёмов, кг.
    interface_lagrange
        Массы перегородок, кг.
    tube_length
        Длина трубы, м.
    tube_area
        Площадь поршня (поперечного сечения трубы), м^2.
    k
        Показатель адиабаты.
    courant
        Число Куранта.

    Возвращает
    ----------
        Кортеж
        (`шаги по времени`,
        `сетка`,
        `скорости объёмов`,
        `скорости границ`,
        `давление`,
        `плотность`,
        `внутренняя энергия`).
    """
    t = 0.0
    u = np.zeros_like(p)

    t_store = [t]
    mesh_store = [mesh.copy()]
    u_store = [u.copy()]
    u_borders_store = [u_borders.copy()]
    p_store = [p.copy()]
    rho_store = [rho.copy()]
    inner_e_store = [inner_e.copy()]
    
    while mesh[-1] < tube_length:
        c = calc_sonic(p, rho, k)
        dt = calc_time_step(mesh, u, c, courant)

        mesh, u_borders, p, rho, inner_e = step_up_lagrange(
            mesh,
            u_borders,
            p,
            rho,
            inner_e,
            k,
            lagrange,
            interface_lagrange,
            tube_area,
            dt
        )

        u = 0.5*(u_borders[:-1] + u_borders[1:])
        t += dt

        t_store.append(t)
        mesh_store.append(mesh.copy())
        u_store.append(u.copy())
        u_borders_store.append(u_borders.copy())
        p_store.append(p.copy())
        rho_store.append(rho.copy())
        inner_e_store.append(inner_e.copy())
    
    return (
        np.array(t_store),
        np.array(mesh_store),
        np.array(u_store),
        np.array(u_borders_store),
        np.array(p_store),
        np.array(rho_store),
        np.array(inner_e_store)
    )


def calc_sonic(p: np.ndarray, rho: np.ndarray, k: float):
    """Рассчитать скорость звука.

    Параметры
    ---------
    p
        Давление в контрольных объёмах, Па.
    rho
        Плотность в контрольных объёмах, кг/м^3.
    k
        Показатель адиабаты.

    Возвращает
    ----------
        Скорость звука, м/с.
    """
    return np.sqrt(k * p / rho)


def calc_time_step(mesh: np.ndarray, u: np.ndarray, c: np.ndarray, courant: float):
    """Рассчитать временной шаг, гарантирующий устойчивость численного метода.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    u
        Скорость газа в контрольных объёмах, м/с.
    c
        Скорость звука в контрольных объёмах, м/с.
    courant
        Число Куранта.

    Возвращает
    ----------
        Величина временного шага, с.
    """
    return courant * np.min(
        (mesh[1:] - mesh[:-1]) / (c + np.abs(u))
    )


def step_up_lagrange(mesh: np.ndarray,
                     u_borders: np.ndarray,
                     p: np.ndarray,
                     rho: np.ndarray,
                     inner_e: np.ndarray,
                     k: float,
                     lagrange: np.ndarray,
                     interface_lagrange: np.ndarray,
                     tube_area: float,
                     dt: float):
    """Функция интегрирования уравнений с массовой лагранжевой переменной.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    u_borders
        Скорости границ объёмов, м/с.
    p
        Давление в контрольных объёмах, Па.
    rho
        Плотность в контрольных объёмах, кг/м^3.
    inner_e
        Внутренняя энергия в контрольных объёмах, Дж/кг.
    k
        Показатель адиабаты.
    lagrange
        Массовая лагранжева переменная, кг.
    interface_lagrange
        Массы перегородок, кг.
    tube_area
        Площадь поршня (поперечного сечения трубы).
    dt
        Шаг по времени, с.

    Возвращает
    ----------
        Кортеж (новая сетка, новые скорости границ, новое давление, новая плотность, новая внутренняя энергия).
    """
    delta_p = p[1:] - p[:-1]
    u_borders[1:-1] -= dt * delta_p * tube_area / (
        lagrange[1:] + interface_lagrange[1:-1]
    )
    m_piston = interface_lagrange[-1]
    u_borders[-1] += p[-1] * tube_area * dt / (0.5*lagrange[-1] + m_piston)
    
    mesh += dt * u_borders

    delta_u = u_borders[1:] - u_borders[:-1]
    e_num = 2*inner_e*lagrange - dt*tube_area*p * delta_u
    delta_x = mesh[1:] - mesh[:-1]
    rho = lagrange / tube_area / delta_x
    e_denom = 2*lagrange + (k - 1) * tube_area*rho*dt*delta_u
    inner_e = e_num / e_denom
    p = (k - 1)*inner_e*rho

    return mesh, u_borders, p, rho, inner_e


def make_wave_gif(mesh: np.ndarray, p: np.ndarray):
    fig, ax = plt.subplots(num="pressure_wave")
    ax.set(xlabel="$x$, м", ylabel="$p$, МПа")

    camera = Camera(fig)
    for mi, pi in zip(mesh, p):
        x = 0.5*(mi[:-1] + mi[1:])
        ax.plot(x, pi*1e-6, c="b")
        camera.snap()

    return fig, camera.animate()


if __name__ == "__main__":
    import json
    import os

    from utils import accurate_throwing


    DIR_NAME = os.path.dirname(__file__)


    PARAMS_DIR = os.path.join(DIR_NAME, "params")
    path = os.path.join(PARAMS_DIR, "lagrange.json")

    with open(path, "r") as f:
        data = json.load(f)

    n = data["nodes"]
    x_0 = data["piston_x_0"]
    m_p = data["piston_mass"]
    p_0 = data["p_0"]
    T_0 = data["T_0"]
    R = data["R"]
    k = data["k"]
    tube_len = data["tube_len"]
    S = 0.25 * np.pi * data["tube_d"]**2
    courant = data["courant"]


    # Численное решение
    mesh = np.linspace(0, x_0, n + 1)
    rho_0 = p_0 / (R * T_0)
    init_data = initialize(mesh, p_0, rho_0, m_p, S, k)
    times, meshes, u, u_borders, p, rho, inner_e = solve(
        mesh, *init_data, tube_len, S, k, courant
    )


    # Точное решение
    sonic_0 = calc_sonic(p_0, rho_0, k)
    m_g = rho_0 * S * x_0
    duration_acc, x_p_acc, u_p_acc = accurate_throwing(
        x_0, sonic_0, k, m_g, m_p
    )


    acc_index = np.argwhere(times < duration_acc)
    t_acc = times[acc_index]
    acc_speed = u_p_acc(t_acc)
    num_speed = u_borders[acc_index, -1]
    acc_x = x_p_acc(t_acc)
    num_x = meshes[acc_index, -1]


    PICS_DIR = os.path.join(DIR_NAME, "pics")
    try:
        os.mkdir(PICS_DIR)
    except OSError:
        pass

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num="lagrange_x_p")

        ax.plot(times*1e3, meshes[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, x_p_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$x_\mathrm{п}$, м")
        ax.legend()
        fig.savefig(os.path.join(PICS_DIR, f"{fig.get_label()}.png"), dpi=300)

        fig, ax = plt.subplots(num="lagrange_p_p_contourf")
        t_arr = np.repeat(times.reshape((times.size, 1)), p[0].size, axis=1)
        x = 0.5*(meshes[:, 1:] + meshes[:, :-1])
        ax.contour(x, t_arr*1e3, p*1e-6, cmap="pink")
        img = ax.pcolormesh(
            x, t_arr*1e3, p*1e-6, cmap="Blues", shading="gouraud"
        )
        fig.colorbar(img, ax=ax, label="$p$, МПа")
        ax.set(xlabel="$x$, м", ylabel="$t$, мс")
        ax.grid(False)
        fig.savefig(os.path.join(PICS_DIR, f"{fig.get_label()}.png"), dpi=300)

        fig, ax = plt.subplots(num="lagrange_u_p")
        ax.plot(times*1e3, u_borders[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, u_p_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$u_\mathrm{п}$, м/с")
        ax.legend()
        fig.savefig(os.path.join(PICS_DIR, f"{fig.get_label()}.png"), dpi=300)

        fig, ax = plt.subplots(num="lagrange_p")
        ax.plot(times*1e3, p[:, 0]*1e-6, label="На дно канала")
        ax.plot(times*1e3, p[:, -1]*1e-6, label="На поршень")
        ax.set(xlabel="$t$, мс", ylabel="Давление $p$, МПа")
        ax.legend()
        fig.savefig(os.path.join(PICS_DIR, f"{fig.get_label()}.png"), dpi=300)

        fig, ax = plt.subplots(num="lagrange_compare")
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

        fig, ani = make_wave_gif(meshes, p)
        ani.save(f"{os.path.join(PICS_DIR, fig.get_label())}.gif", dpi=100, fps=30)
