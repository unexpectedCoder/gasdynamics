import matplotlib.pyplot as plt
import numpy as np


def initialize(mesh: np.ndarray,
               p0: float,
               temp0: float,
               gas_const: float,
               k: float):
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
          cfl: float,
          alpha=3/16,
          beta=1/4):
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
        dt = calc_time_step(q, dx, c, cfl)

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
    u = q[1] / q[0]
    e = q[2]/q[0] - 0.5*u**2
    return (k - 1)*e*q[0]


def calc_sonic(p: np.ndarray, rho: np.ndarray, k: float):
    return np.sqrt(k * p / rho)


def ausm_plus(q: np.ndarray,
              u_borders: np.ndarray,
              c: np.ndarray,
              p: np.ndarray,
              alpha: float,
              beta: float):
    interface_sonic = 0.5*(c[:-1] + c[1:])
    u = q[1] / q[0]

    mach_left = (u[:-1] - u_borders) / interface_sonic
    mach_right = (u[1:] - u_borders) / interface_sonic
    interface_mach = calc_interface_mach(mach_left, mach_right, beta)

    interface_pressure = calc_interface_pressure(
        mach_left, mach_right, p, alpha
    )

    q_left, p_left = q[:, :-1], p[:-1]
    flux_left = calc_ausm_flux(q_left, p_left)
    q_right, p_right = q[:, 1:], p[1:]
    flux_right = calc_ausm_flux(q_right, p_right)

    return 0.5*interface_sonic * (
        interface_mach*(flux_right + flux_left)
        - np.abs(interface_mach) * (flux_right - flux_left)
    ) + np.vstack([
        np.zeros_like(interface_pressure),
        interface_pressure,
        interface_pressure * u_borders
    ])


def calc_ausm_flux(q: np.ndarray, p: np.ndarray):
    return np.vstack([q[0], q[1], q[2] + p])


def calc_interface_mach(mach_left: np.ndarray,
                        mach_right: np.ndarray,
                        beta=1/4):
    return f_beta(mach_left, "+", beta) + f_beta(mach_right, "-", beta)


def f_beta(mach: np.ndarray, sign: str, beta: float):
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
    return \
        g_alpha(mach_left, "+", alpha)*p[:-1] \
        + g_alpha(mach_right, "-", alpha)*p[1:]


def g_alpha(mach: np.ndarray, sign: str, alpha: float):
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


def calc_time_step(q: np.ndarray, dx: float, c: np.ndarray, cfl: float):
    return cfl * np.min(dx / (c + np.abs(q[1]/q[0])))


def step_up_euler(new_dx: np.ndarray,
                  dx: np.ndarray,
                  q: np.ndarray,
                  f_left: np.ndarray,
                  f_right: np.ndarray,
                  dt: float):
    new_q = q.copy()
    new_q[:, 1:-1] = dx/new_dx * (q[:, 1:-1] - dt/dx * (f_right - f_left))
    return new_q


if __name__ == "__main__":
    from utils import accurate_throwing_solution

    n = 300
    piston_x0, piston_mass, d = 0.5, 0.1, 0.03
    piston_area = 0.25 * np.pi * d**2
    tube_length = 2.0
    p0, temp0, gas_const, k = 5e6, 300, 287, 1.4
    cfl = 0.5

    mesh = np.linspace(0, piston_x0, n + 1)
    q, p = initialize(mesh, p0, temp0, gas_const, k)

    times, meshes, qs, ps, piston_vs = solve(
        mesh, q, p, k, piston_mass, piston_area, tube_length, cfl
    )

    rho0 = p0 / temp0 / gas_const
    sonic0 = calc_sonic(p0, rho0, k)
    gas_mass = rho0 * piston_area * piston_x0
    duration_acc, x_piston_acc, u_piston_acc = accurate_throwing_solution(
        piston_x0, sonic0, k, gas_mass, piston_mass
    )
    t_acc = np.linspace(0, duration_acc, n)
    # print(piston_x0, sonic0, k, gas_mass, piston_mass)
    # exit(0)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num="throwing_euler_piston_x")
        ax.plot(times*1e3, meshes[:, -1], label="Численное решение")
        ax.plot(t_acc*1e3, x_piston_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel=r"$x_\mathrm{п}$, м")
        ax.legend()
        fig.savefig("pics/throwing_euler_piston_x")

        fig, ax = plt.subplots(num="throwing_euler_piston_p_contourf")
        t_arr = np.repeat(times.reshape((times.size, 1)), p[0].size, axis=1)
        x = 0.5*(meshes[:, 1:] + meshes[:, :-1])
        img = ax.pcolormesh(
            x, t_arr*1e3, ps*1e-6, cmap="Blues", shading="gouraud"
        )
        fig.colorbar(img, ax=ax, label="$p$, МПа")
        ax.set(xlabel="$x$, м", ylabel="$t$, мс")
        ax.grid(False)
        fig.savefig("pics/throwing_euler_piston_p_contourf")

        fig, ax = plt.subplots(num="throwing_euler_piston_speed")
        ax.plot(times*1e3, piston_vs, label="Численное решение")
        ax.plot(t_acc*1e3, u_piston_acc(t_acc), ls="--", label="Точное решение")
        ax.set(xlabel="$t$, мс", ylabel="$u_\mathrm{п}$, м/с")
        ax.legend()
        fig.savefig("pics/throwing_euler_piston_speed")

        fig, ax = plt.subplots(num="throwing_euler_pressure")
        ax.plot(times*1e3, ps[:, 0]*1e-6, label="На дно канала")
        ax.plot(times*1e3, ps[:, -1]*1e-6, label="На поршень")
        ax.set(xlabel="$t$, мс", ylabel="$p$, МПа")
        ax.legend()
        fig.savefig("pics/throwing_euler_pressure")

    plt.show()
