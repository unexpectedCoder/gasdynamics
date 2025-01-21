import numpy as np
from scipy import optimize as opti

from gasdynamics.course_work.direct import euler as eu


def optimize(data: dict):
    cons = data["constraints"]

    p = 1e5, cons["p_max"]
    T = cons["T_min"], cons["T_max"]
    m_p = cons["piston_m_min"], cons["piston_m_max"]
    bounds = [p, T, m_p]
    sol = opti.differential_evolution(
        solve, bounds, args=(data,), workers=8
    )

    return sol.x, -sol.fun


def solve(x: list, data: dict):
    p_0, T_0, m_p = x

    cons = data["constraints"]
    x_0 = data["piston"]["x_0"]
    tube_len = data["tube"]["d"] * cons["n_tube_len"]

    sol = eu.solve(x_0, m_p, tube_len, p_0, T_0, data)
    v_p = sol.v_p_store[-1]
    E_kin = 0.5 * m_p * v_p**2
    print(x, round(E_kin, 1), round(v_p, 2), sep="\t")

    return -E_kin


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import os


    var = int(input("Вариант > "))

    DIRNAME = os.path.dirname(__file__)
    path = os.path.join(DIRNAME, "tasks", f"{var}.json")
    with open(path, "r") as f:
        data = json.load(f)
    
    x, E_kin = optimize(data)

    p_0, T_0, m_p = x
    tube_len = data["tube"]["L"]
    x_0 = data["piston"]["x_0"]
    sol = eu.solve(x_0, m_p, tube_len, p_0, T_0, data)

    results = {
        "E_kin": E_kin,
        "p_0": p_0,
        "T_0": T_0,
        "m_p": m_p,
        "x_0": x_0,
        "x_p": sol.x_p_store[-1],
        "v_p": sol.v_p_store[-1],
        "t": sol.t_store[-1]
    }

    RES_DIR = os.path.join(DIRNAME, "results")
    try:
        os.mkdir(RES_DIR)
    except OSError:
        pass

    path = os.path.join(RES_DIR, f"{var}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots(num=f"{var}_pressure")
        ax.plot(sol.t_store*1e3, sol.p_store[:, 1]*1e-6, label="На дно канала")
        ax.plot(sol.t_store*1e3, sol.p_store[:, -2]*1e-6, label="На поршень")
        ax.set(xlabel=r"$t$, мс", ylabel=r"$p$, МПа")
        ax.legend()
        fig.savefig(f"{os.path.join(RES_DIR, fig.get_label())}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(num=f"{var}_coordinate")
        ax.plot(sol.t_store*1e3, sol.x_p_store)
        ax.set(xlabel="$t$, мс", ylabel=r"$x_\mathrm{п}$, м")
        fig.savefig(f"{os.path.join(RES_DIR, fig.get_label())}.png", dpi=300)
        plt.close(fig)

        fig, ax = plt.subplots(num=f"{var}_speed")
        ax.plot(sol.t_store*1e3, sol.v_p_store)
        ax.set(xlabel="$t$, мс", ylabel="$u_\mathrm{п}$, м/с")
        fig.savefig(f"{os.path.join(RES_DIR, fig.get_label())}.png", dpi=300)
        plt.close(fig)
