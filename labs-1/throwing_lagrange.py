import matplotlib.pyplot as plt
import numpy as np


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
        Кортеж (скорость границ, давление, плотность, внутренняя энергия, массовая лагранжева координата, масса перегородок).
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
          cfl: float):
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
    cfl
        Число CFL.

    Возвращает
    ----------
        Кортеж (шаги по времени, сетка, скорости объёмов, скорости границ, давление, плотность, внутренняя энергия).
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
        dt = calc_time_step(mesh, u, c, cfl)

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


def calc_time_step(mesh: np.ndarray, u: np.ndarray, c: np.ndarray, cfl: float):
    """Рассчитать временной шаг, гарантирующий устойчивость численного метода.

    Параметры
    ---------
    mesh
        Расчётная сетка, м.
    u
        Скорость газа в контрольных объёмах, м/с.
    c
        Скорость звука в контрольных объёмах, м/с.
    cfl
        Число CFL.

    Возвращает
    ----------
        Величина временного шага, с.
    """
    return cfl * np.min(
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


if __name__ == "__main__":
    from utils import accurate_throwing_solution

    n = 300
    piston_x0, piston_mass = 0.5, 0.1
    p0, temperature0, gas_const = 5e6, 300, 287
    rho0 = p0 / gas_const / temperature0
    k = 1.4
    tube_length, tube_area = 2.0, 0.25 * np.pi * 0.03**2
    cfl = 0.5

    mesh = np.linspace(0, piston_x0, n + 1)
    init_data = initialize(mesh, p0, rho0, piston_mass, tube_area, k)
    t, mesh, u, u_borders, p, rho, inner_e = solve(
        mesh, *init_data, tube_length, tube_area, k, cfl
    )

    sonic0 = calc_sonic(p0, rho0, k)
    gas_mass = rho0 * tube_area * piston_x0
    duration_acc, x_piston_acc, u_piston_acc = accurate_throwing_solution(
        piston_x0, sonic0, k, gas_mass, piston_mass
    )
    t_acc = np.linspace(0, duration_acc, n)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num="throwing_lagrange_piston_x")
        ax.plot(t*1e3, mesh[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, x_piston_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$x_\mathrm{п}$, м")
        ax.legend()
        fig.savefig("labs-1/pics/throwing_lagrange_piston_x")

        fig, ax = plt.subplots(num="throwing_lagrange_piston_p_contourf")
        t_arr = np.repeat(t.reshape((t.size, 1)), p[0].size, axis=1)
        x = 0.5*(mesh[:, 1:] + mesh[:, :-1])
        img = ax.pcolormesh(
            x, t_arr*1e3, p*1e-6, cmap="Blues", shading="gouraud"
        )
        fig.colorbar(img, ax=ax, label="$p$, МПа")
        ax.set(xlabel="$x$, м", ylabel="$t$, мс")
        ax.grid(False)
        fig.savefig("labs-1/pics/throwing_lagrange_piston_p_contourf")

        fig, ax = plt.subplots(num="throwing_lagrange_piston_speed")
        ax.plot(t*1e3, u_borders[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, u_piston_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$u_\mathrm{п}$, м/с")
        ax.legend()
        fig.savefig("labs-1/pics/throwing_lagrange_piston_speed")

        fig, ax = plt.subplots(num="throwing_lagrange_pressure")
        ax.plot(t*1e3, p[:, 0]*1e-6, label="На дно канала")
        ax.plot(t*1e3, p[:, -1]*1e-6, label="На поршень")
        ax.set(xlabel="$t$, мс", ylabel="Давление $p$, МПа")
        ax.legend()
        fig.savefig("labs-1/pics/throwing_lagrange_pressure")

    plt.show()
