import numpy as np
from scipy import optimize as opti

from gasdynamics.course_work.direct import euler as eu


def optimize(data: dict):
    tube = data["tube"]
    cons = data["constraints"]

    p = 1e5, cons["p_max"]
    T = 273, 500
    x_0 = 0.001, tube["d"]*cons["n_tube_len"]*cons["x_0_ratio"]
    bounds = [p, T, x_0]
    
    quasi_sol = opti.differential_evolution(solve, bounds, args=(data,))
    quasi_x = quasi_sol.x
    quasi_f = -quasi_sol.fun
    
    p = (
        max(quasi_x[0] - 0.5e6, 1e5),
        min(quasi_x[0] + 0.5e6, p[1])
    )
    T = (
        max(quasi_x[1] - 5, T[0]),
        min(quasi_x[1] + 5, T[1])
    )
    x_0 = (
        max(quasi_x[2] - 1e-2, x_0[0]),
        min(quasi_x[2] + 1e-2, x_0[1])
    )
    ranges = [p, T, x_0]
    sol = opti.brute(solve, ranges, args=(data,), Ns=10, workers=-1)
    x, f = sol[0], -sol[1]

    return x, f if f > quasi_f else quasi_x, quasi_f


def solve(x: list, data: dict):
    p_0, T_0, x_0 = x

    cons = data["constraints"]
    tube_len = data["tube"]["d"] * cons["n_tube_len"]
    data["gas"]["p_0"] = p_0
    data["gas"]["T_0"] = T_0
    data["tube"]["L"] = tube_len
    data["piston"]["x_0"] = x_0

    sol = eu.solve(data)
    v_p = sol.v_p_store[-1]
    x_p = sol.mesh_store[-1, -1]

    W_0 = 0.25 * np.pi * data["tube"]["d"]**2 * x_0
    E_0 = p_0 * W_0 / (data["gas"]["k"] - 1)
    E_kin = 0.5 * data["piston"]["m"] * v_p**2
    v_p_req = cons["v_p"]

    eta = E_kin / E_0
    x_0_max = x_p*cons["x_0_ratio"]
    
    if v_p < v_p_req:
        return -eta + 0.001*(v_p_req - v_p)**2
    if x_0 < x_0_max:
        return -eta
    return -eta + 100*(x_0 - x_0_max)**2


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    import os


    DIRNAME = os.path.dirname(__file__)
    TASK_PATH = os.path.join(DIRNAME, "euler_tasks", "euler.json")

    with open(TASK_PATH, "r") as f:
        data = json.load(f)
    x, f = optimize(data)

    p_0, T_0, x_0 = x
    data["gas"]["p_0"] = p_0
    data["gas"]["T_0"] = T_0
    data["piston"]["x_0"] = x_0
    sol = eu.solve(data)

    with plt.style.context("sciart.mplstyle"):
        fig, ax = plt.subplots()
        ax.plot(sol.t_store*1e-3, sol.p_store[:, -2]*1e-6)
        ax.set(xlabel=r"$t$, мс", ylabel=r"$p$, МПа")
    
    plt.show()
