import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm

import solver
from gd_func import mach2lambda
from nozzle import Nozzle
from solver import Solution
from task import Task


DIR_NAME = os.path.dirname(__file__)
with open(os.path.join(DIR_NAME, "settings.yml"), "r") as f:
    config = yaml.safe_load(f)


def main(data_path: str, variants: set[int]):
    save_dir = os.path.join(DIR_NAME, config["solutions_dir"])
    try:
        os.mkdir(save_dir)
    except OSError:
        pass

    tasks = Task.from_file(data_path, variants)
    for task in tqdm(tasks, desc="Solve variants"):
        pics_dir = os.path.join(save_dir, f"{config['pics_dir']}_{task.variant}")
        try:
            os.mkdir(pics_dir)
        except OSError:
            pass

        solve(task, save_dir, pics_dir)


def solve(task: Task, save_dir: str, pics_dir: str):
    # Initialization
    noz = Nozzle.from_task(task)
    p_a = 1e5
    # Solving
    sol = solver.solve(task, noz, p_a)
    adapted = solver.adapt_nozzle(task, noz, p_a)
    # Saving
    sol_dict = {
        "Вариант": task.variant,
        "Сопло": noz.as_dict(),
        "Физика": sol.as_dict(),
        "Улучшенное сопло": adapted.as_dict()
    }
    with open(os.path.join(save_dir, f"{task.variant}.yml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(sol_dict, f, allow_unicode=True, sort_keys=False)
    # Visualization
    with plt.style.context("sciart.mplstyle"):
        make_plots(task, noz, sol, pics_dir)


def make_plots(task: Task,
               noz: Nozzle,
               sol: Solution,
               pics_dir: str = None):
    fig_s, ax_s = plt.subplots(num="square")
    ax_s.plot(sol.x, sol.square, c="k")
    ax_s.axvline(noz.x_critic, ls="--", c="k")
    ax_s.set(xlabel="$x$, м", ylabel="$S(x)$, м$^2$")
    ax_s.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_s.savefig(os.path.join(pics_dir, fig_s.get_label()))
        plt.close(fig_s)

    fig_mach, ax_mach = plt.subplots(num="Mach")
    ax_mach.plot(sol.x, sol.mach, c="k")
    ax_mach.set(xlabel="$x$, м", ylabel=r"$\mathrm{M}$")
    ax_mach.axvline(noz.x_critic, ls="--", c="k")
    ax_mach.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_mach.savefig(os.path.join(pics_dir, fig_mach.get_label()))
        plt.close(fig_mach)

    fig_lambda, ax_lambda = plt.subplots(num="lambda")
    ax_lambda.plot(sol.x, mach2lambda(sol.mach, task.k), c="k")
    ax_lambda.set(xlabel="$x$, м", ylabel=r"$\lambda$")
    ax_lambda.axvline(noz.x_critic, ls="--", c="k")
    ax_lambda.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_lambda.savefig(os.path.join(pics_dir, fig_lambda.get_label()))
        plt.close(fig_lambda)

    fig_p, ax_p = plt.subplots(num="pressure")
    ax_p.plot(sol.x, sol.pressure*1e-6, c="k")
    ax_p.set(xlabel="$x$, м", ylabel=r"$p$, МПа")
    ax_p.axvline(noz.x_critic, ls="--", c="k")
    ax_p.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_p.savefig(os.path.join(pics_dir, fig_p.get_label()))
        plt.close(fig_p)

    fig_dens, ax_dens = plt.subplots(num="density")
    ax_dens.plot(sol.x, sol.density, c="k")
    ax_dens.set(xlabel="$x$, м", ylabel=r"$\rho$, кг/м$^3$")
    ax_dens.axvline(noz.x_critic, ls="--", c="k")
    ax_dens.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_dens.savefig(os.path.join(pics_dir, fig_dens.get_label()))
        plt.close(fig_dens)

    fig_temp, ax_temp = plt.subplots(num="temperature")
    ax_temp.plot(sol.x, sol.temperature, c="k")
    ax_temp.set(xlabel="$x$, м", ylabel=r"$T$, К")
    ax_temp.axvline(noz.x_critic, ls="--", c="k")
    ax_temp.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_temp.savefig(os.path.join(pics_dir, fig_temp.get_label()))
        plt.close(fig_temp)

    fig_j, ax_j = plt.subplots(num="j")
    ax_j.plot(sol.x, sol.specific_mass, c="k")
    ax_j.set(xlabel="$x$, м", ylabel=r"$j$, кг/(с $\cdot$ м$^2$)")
    ax_j.axvline(noz.x_critic, ls="--", c="k")
    ax_j.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_j.savefig(os.path.join(pics_dir, fig_j.get_label()))
        plt.close(fig_j)

    fig_G, ax_G = plt.subplots(num="G")
    ax_G.plot(sol.x, sol.specific_mass*sol.square, c="k")
    ax_G.set(xlabel="$x$, м", ylabel=r"$G$, кг/с")
    ax_G.axvline(noz.x_critic, ls="--", c="k")
    ax_G.grid(True, ls=":", c="k")
    if pics_dir is not None:
        fig_G.savefig(os.path.join(pics_dir, fig_G.get_label()))
        plt.close(fig_G)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser("GD-Homework")
    parser.add_argument("-v", default=-1, type=int)
    args = parser.parse_args()

    data_path = os.path.join(
        DIR_NAME, config["variants_dir"], config["variants_file"]
    )
    main(data_path, args.v if args.v >= 0 else None)
