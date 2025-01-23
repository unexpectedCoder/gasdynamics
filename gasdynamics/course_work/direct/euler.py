import matplotlib.pyplot as plt
import numpy as np
from collections import namedtuple


EuSolution = namedtuple(
    "EuSolution",
    "t_store mesh q_store p_store x_p_store v_p_store"
)


def solve(x_0, m_p, tube_len, p_0, T_0, data: dict):
    n = data["volumes"]
    w = data["piston"]["w"]
    d = data["tube"]["d"]
    R = data["gas"]["R"]
    k = data["gas"]["k"]
    alpha, beta = data["alpha"], data["beta"]
    courant = data["courant"]
    S = 0.25 * np.pi * d**2
    mesh = np.linspace(0, x_0, n + 1)
    q0, p0 = initialize(mesh, p_0, T_0, R, k)

    t, q, p = 0.0, q0, p0
    dx = mesh[1] - mesh[0]
    piston_x, piston_v = mesh[-1], 0.0
    visc = data["visc"]
    A, B, h = visc["A"], visc["B"], visc["h"]

    t_store = [t]
    q_store = [q[:, 1:-1].copy()]
    p_store = [p[1:-1].copy()]
    x_p_store = [piston_x]
    v_p_store = [piston_v]

    i, i_max = 0, data["maxiter"]
    while piston_x < tube_len and i <= i_max:
        p = equation_of_state(q, k)
        c = calc_sonic(p, q[0], k)
        dt = calc_time_step(q, dx, c, courant)

        friction = calc_friction(piston_v, d, w, h, T_0, A, B)
        piston_v += dt / m_p * (p[-2]*S - friction)
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
        q_store.append(q[:, 1:-1].copy())
        p_store.append(p[1:-1].copy())
        x_p_store.append(piston_x)
        v_p_store.append(piston_v)

        i += 1

    return EuSolution(
        np.array(t_store),
        mesh,
        np.array(q_store),
        np.array(p_store),
        np.array(x_p_store),
        np.array(v_p_store)
    )


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


def calc_friction(v_p: float,
                  d: float,
                  w: float,
                  h: float,
                  T: float,
                  A: float,
                  B: float):
    mu = calc_visc(T, A, B)
    S = np.pi * d * w
    return mu * S * v_p / h


def calc_visc(T: float, A: float, B: float):
    return 10**((A + B/T)**2) / 1000


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
    return np.vstack([q[0], q[1], q[2] + p])


def calc_interface_mach(mach_left: np.ndarray,
                        mach_right: np.ndarray,
                        beta: float):
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


def calc_time_step(q: np.ndarray, dx: float, c: np.ndarray, courant: float):
    return courant * np.min(dx / (c + np.abs(q[1]/q[0])))


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
    import json
    import os


    DIR_NAME = os.path.join(os.path.dirname(__file__), os.pardir)
    PARAMS_DIR = os.path.join(DIR_NAME, "euler_tasks")
    path = os.path.join(PARAMS_DIR, "euler.json")
    with open(path, "r") as f:
        data = json.load(f)

    # Численное решение
    x_0 = data["piston"]["x_0"]
    m_p = data["piston"]["m"]
    tube_len = data["tube"]["L"]
    p_0 = data["gas"]["p_0"]
    T_0 = data["gas"]["T_0"]
    t, mesh, qs, ps, x_p, v_p = solve(
        x_0, m_p, tube_len, p_0, T_0, data
    )

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num="euler_p")
        ax.plot(t*1e3, ps[:, 0]*1e-6, label="На дно канала")
        ax.plot(t*1e3, ps[:, -1]*1e-6, label="На поршень")
        ax.set(xlabel="$t$, мс", ylabel="$p$, МПа")
        ax.legend()

        fig, ax = plt.subplots(num="euler_v_p")
        ax.plot(t*1e3, v_p)
        ax.set(xlabel="$t$, мс", ylabel="$u_\mathrm{п}$, м/с")
    
    plt.show()
