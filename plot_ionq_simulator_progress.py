#!/usr/bin/env python3
"""Plot the four resumed IonQ simulator energy histories."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "fig"

STEM = (
    "Si_sc1x1x1_k1x1x1_qudft_au-fixed_P4_9b32cd20_"
    "fixed-occ_ionq-simulator_shots1000"
)

RUNS = (
    ("Run 1", DATA / f"{STEM}_rep03.out"),
    ("Run 2", DATA / f"{STEM}_rep02.out"),
    ("Run 3", DATA / f"{STEM}_rep01.out"),
    ("Run 4", DATA / f"{STEM}.out"),
)

PYSCF_ETOT = -7.110403267952345
OUTPUT = FIG / "Si_ionq_simulator_resume_progress.png"
OUTPUT_LAST10 = FIG / "Si_ionq_simulator_resume_progress_last10.png"


def load_history(path):
    with h5py.File(path, "r") as handle:
        history = handle["history"]
        iteration = np.asarray(history["iteration"], dtype=int)
        energy = np.asarray(history["etot"], dtype=float)
    return iteration, energy


def main():
    FIG.mkdir(exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors

    for index, (label, path) in enumerate(RUNS):
        iteration, energy = load_history(path)
        color = colors[index]
        axis.plot(
            iteration,
            energy,
            color=color,
            linewidth=1.1,
            marker="o",
            markersize=2.2,
            markeredgewidth=0,
            alpha=0.82,
            label=f"{label} (itr {iteration[-1]})",
        )
        axis.scatter(
            iteration[-1],
            energy[-1],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            s=42,
            zorder=4,
        )

    axis.axhline(
        PYSCF_ETOT,
        color="0.45",
        linestyle="--",
        linewidth=1.6,
        label=f"PySCF GS ({PYSCF_ETOT:.6f} $E_h$)",
        zorder=1,
    )

    axis.set_title("Si Gamma-point primitive cell: IonQ simulator progress")
    axis.set_xlabel("Optimization iteration")
    axis.set_ylabel(r"Total energy ($E_h$)")
    axis.grid(True, color="0.88", linewidth=0.7)
    axis.legend(loc="best", frameon=True, fontsize=9)
    axis.margins(x=0.01)

    figure.savefig(OUTPUT, dpi=220)
    print(OUTPUT)

    recent_figure, axes = plt.subplots(
        2, 2, figsize=(10, 7), sharey=True, constrained_layout=True)
    recent_energies = []

    for index, ((label, path), axis) in enumerate(zip(RUNS, axes.flat)):
        iteration, energy = load_history(path)
        iteration = iteration[-10:]
        energy = energy[-10:]
        recent_energies.append(energy)
        color = colors[index]

        axis.plot(
            iteration,
            energy,
            color=color,
            linewidth=1.4,
            marker="o",
            markersize=4.2,
            label=label,
        )
        axis.scatter(
            iteration[-1],
            energy[-1],
            color=color,
            edgecolor="black",
            linewidth=0.6,
            s=55,
            zorder=4,
        )
        axis.axhline(
            PYSCF_ETOT,
            color="0.45",
            linestyle="--",
            linewidth=1.4,
            label=(f"PySCF GS ({PYSCF_ETOT:.6f} $E_h$)"
                   if index == 0 else None),
            zorder=1,
        )
        axis.set_title(f"{label}: itr {iteration[0]}–{iteration[-1]}")
        axis.set_xlabel("Optimization iteration")
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)

        if index == 0:
            axis.legend(loc="best", frameon=True, fontsize=8)

    all_recent = np.concatenate(recent_energies)
    lower = min(PYSCF_ETOT, float(all_recent.min()))
    upper = max(PYSCF_ETOT, float(all_recent.max()))
    padding = max(0.01, 0.08 * (upper - lower))
    axes[0, 0].set_ylim(lower - padding, upper + padding)
    axes[0, 0].set_ylabel(r"Total energy ($E_h$)")
    axes[1, 0].set_ylabel(r"Total energy ($E_h$)")
    recent_figure.suptitle(
        "Si Gamma-point primitive cell: latest 10 simulator iterations")
    recent_figure.savefig(OUTPUT_LAST10, dpi=220)
    print(OUTPUT_LAST10)


if __name__ == "__main__":
    main()
