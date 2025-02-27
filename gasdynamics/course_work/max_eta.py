import numpy as np
from scipy import optimize as opti

from gasdynamics.course_work.direct import euler as eu


def optimize(data: dict):
    tube = data["tube"]
    cons = data["constraints"]

    max_tube_len = tube["d"]*cons["n_tube_len"]
    p = 1e5, cons["p_max"]
    T = cons["T_min"], cons["T_max"]
    x_0 = 0.001, max_tube_len * cons["x_0_ratio"]
    tube_len = max_tube_len * cons["x_0_ratio"] + 0.01, max_tube_len
    bounds = [p, T, x_0, tube_len]
    
    sol = opti.differential_evolution(
        solve, bounds, args=(data,), workers=-1, callback=cb, disp=True,
        popsize=120, mutation=(0.67, 1), tol=0.001, polish=False,
        constraints=opti.LinearConstraint(
            (0, 0, 1/cons["x_0_ratio"], -1), ub=0
        )
    )
    return sol.x, -sol.fun


def cb(intermediate_result: opti.OptimizeResult):
    x_store.append(intermediate_result.x)
    eta_store.append(-intermediate_result.fun)


def solve(x: list, data: dict):
    p_0, T_0, x_0, tube_len = x

    cons = data["constraints"]
    m_p = data["piston"]["m"]

    sol = eu.solve(x_0, m_p, tube_len, p_0, T_0, data)
    # x_p = sol.x_p_store[-1]
    v_p = sol.v_p_store[-1]

    W_0 = 0.25 * np.pi * data["tube"]["d"]**2 * x_0
    E_0 = p_0 * W_0 / (data["gas"]["k"] - 1)
    E_kin = 0.5 * data["piston"]["m"] * v_p**2
    v_p_req = cons["v_p"]

    eta = E_kin / E_0
    # x_0_max = x_p * cons["x_0_ratio"]
    
    penalty = 0
    if v_p < v_p_req:
        penalty += 0.001*(v_p_req - v_p)**2
    # print(
    #     eta, eta - penalty, x_0 / x_p, x_0 <= x_0_max, v_p >= v_p_req, sep="\t"
    # )
    return -eta + penalty


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import os


    x_store, eta_store = [], []

    var = int(input("Вариант > "))

    DIRNAME = os.path.dirname(__file__)
    path = os.path.join(DIRNAME, "tasks", f"{var}.json")
    with open(path, "r") as f:
        data = json.load(f)
    
    x, eta = optimize(data)
    x_store = np.array(x_store)
    eta_store = np.array(eta_store)

    p_0, T_0, x_0, tube_len = x
    m_p = data["piston"]["m"]
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

        p = x_store[:, 0] * 1e-6
        T = x_store[:, 1]
        m = x_store[:, 2]
        fig, axes = plt.subplots(
            ncols=3, num=f"{var}_eta", figsize=(14, 5)
        )

        axes[0].plot(p, T, alpha=0.3)
        axes[0].scatter(p, T, c=eta_store, marker=".")
        axes[0].set(xlabel=r"$p_0$, МПа", ylabel=r"$T_0$, К")

        axes[1].plot(p, m, alpha=0.3)
        axes[1].scatter(p, m, c=eta_store, marker=".")
        axes[1].set(xlabel=r"$p_0$, МПа", ylabel=r"$m$, кг")

        axes[2].plot(T, m, alpha=0.3)
        img = axes[2].scatter(T, m, c=eta_store, marker=".")
        axes[2].set(xlabel=r"$T_0$, К", ylabel=r"$m$, кг")
        fig.colorbar(img, ax=axes[2], label=r"$\eta$")

        fig.savefig(f"{os.path.join(RES_DIR, fig.get_label())}.png", dpi=300)
        plt.close(fig)
