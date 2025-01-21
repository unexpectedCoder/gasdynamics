import numpy as np
from scipy import optimize as opti

from gasdynamics.course_work.direct import euler as eu


def optimize(data: dict):
    tube = data["tube"]
    cons = data["constraints"]

    p = 1e5, cons["p_max"]
    T = cons["T_min"], cons["T_max"]
    x_0 = 0.001, tube["d"]*cons["n_tube_len"]*cons["x_0_ratio"]
    bounds = [p, T, x_0]
    
    sol = opti.differential_evolution(
        solve, bounds, args=(data,), workers=8
    )
    return sol.x, -sol.fun


def solve(x: list, data: dict):
    p_0, T_0, x_0 = x

    cons = data["constraints"]
    tube_len = data["tube"]["d"] * cons["n_tube_len"]
    m_p = data["piston"]["m"]

    sol = eu.solve(x_0, m_p, tube_len, p_0, T_0, data)
    v_p = sol.v_p_store[-1]
    x_p = sol.x_p_store[-1]

    W_0 = 0.25 * np.pi * data["tube"]["d"]**2 * x_0
    E_0 = p_0 * W_0 / (data["gas"]["k"] - 1)
    E_kin = 0.5 * data["piston"]["m"] * v_p**2
    v_p_req = cons["v_p"]

    eta = E_kin / E_0
    x_0_max = x_p * cons["x_0_ratio"]
    
    if v_p < v_p_req:
        res = -eta + 0.0002*(v_p_req - v_p)**2
    elif x_0 > x_0_max:
        res = -eta + 1000*(x_0 - x_0_max)**2
    else:
        res = -eta
    print(x, -res, round(v_p, 2), x_0/x_p < 1/3, sep="\t")
    return res


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import os


    var = int(input("Вариант > "))

    DIRNAME = os.path.dirname(__file__)
    path = os.path.join(DIRNAME, "tasks", f"{var}.json")
    with open(path, "r") as f:
        data = json.load(f)
    
    x, eta = optimize(data)

    p_0, T_0, x_0 = x
    m_p = data["piston"]["m"]
    tube_len = x_0 / data["constraints"]["x_0_ratio"]
    sol = eu.solve(x_0, m_p, tube_len, p_0, T_0, data)

    results = {
        "eta": eta,
        "p_0": p_0,
        "T_0": T_0,
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
