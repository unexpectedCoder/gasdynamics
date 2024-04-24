#!.venv/bin/python

import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import solver
from gd_func import mach2lambda
from nozzle import Nozzle
from solver import Solution
from task import Task


def main(data_path: str, variants: set[int] | int = None):
    # Настраиваем окружение
    save_dir = os.path.join("homework-1", "results")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if isinstance(variants, int):
        variants = [variants]
    # Решаем заданные варианты
    tasks = Task.from_file(data_path, variants)
    for task in tqdm(tasks, desc="Решение вариантов"):
        var_save_dir = os.path.join(save_dir, str(task.variant))
        if not os.path.isdir(var_save_dir):
            os.mkdir(var_save_dir)
        pics_dir = os.path.join(var_save_dir, "pics")
        if not os.path.isdir(pics_dir):
            os.mkdir(pics_dir)
        do_task(task, var_save_dir, pics_dir)


def do_task(task: Task, save_dir: str, pics_dir: str):
    # Инициализируем задачу (исходные данные)
    noz = Nozzle.from_task(task)
    p_a = 1e5
    # Решаем задачу
    sol = solver.solve(task, noz, p_a)
    adapted = solver.adapt_nozzle(task, noz, p_a)
    # Сохраняем результаты
    with open(os.path.join(save_dir, "solution.txt"), "w") as f:
        for k, v in sol.as_dict().items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(save_dir, "adapted.txt"), "w") as f:
        for k, v in adapted.as_dict().items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(save_dir, "nozzle.txt"), "w") as f:
        for k, v in noz.as_dict().items():
            f.write(f"{k}: {v}\n")
    # Строим графики
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

    data_path = os.path.join("homework-1", "initdata", "variants.csv")
    main(data_path, args.v if args.v >= 0 else None)
