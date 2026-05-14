"""[CLASSICAL] Task 03 -- 计算精确经典参考能量。

为了知道 VQE 算得准不准，我们先用经典线性代数方法精确对角化小规模 H2
Hamiltonian。H2 很小，所以这一步很快；大分子时这类精确对角化会指数变难。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, hartree_to_ev, print_banner, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", type=Path, required=True, help="task_02 输出的 problem.dill")
    p.add_argument("--output", type=Path, required=True, help="输出 reference.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出参考能量对比 PNG")
    return p.parse_args()


def _plot_reference(hf_energy: float, exact_energy: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    names = ["Hartree-Fock", "Exact"]
    values = [hf_energy, exact_energy]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(names, values, color=["#9ecae1", "#31a354"], edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.6f}", ha="center", fontsize=10)
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Classical reference energy")
    ax.grid(True, axis="y", alpha=0.25)
    y_min = min(values) - 0.06
    y_max = max(values) + 0.06
    ax.set_ylim(y_min, y_max)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 03 / 精确经典参考")
    print(f"  problem = {args.problem}")

    from qiskit_algorithms import NumPyMinimumEigensolver

    with args.problem.open("rb") as f:
        pkg = dill.load(f)
    qubit_operator = pkg["qubit_operator"]
    metadata = pkg["metadata"]

    with Timer("NumPy 精确对角化"):
        solver = NumPyMinimumEigensolver()
        result = solver.compute_minimum_eigenvalue(qubit_operator)

    nuclear = float(metadata["nuclear_repulsion_energy"])
    exact_electronic = float(np.real(result.eigenvalue))
    exact_total = exact_electronic + nuclear
    hf_total = float(metadata["hartree_fock_total_energy"])
    hf_error = hf_total - exact_total

    payload = {
        "method": "NumPyMinimumEigensolver",
        "exact_total_energy_hartree": exact_total,
        "exact_electronic_energy_hartree": exact_electronic,
        "nuclear_repulsion_energy_hartree": nuclear,
        "hartree_fock_total_energy_hartree": hf_total,
        "hartree_fock_error_hartree": hf_error,
        "hartree_fock_error_ev": hartree_to_ev(hf_error),
    }
    print(f"  精确总能量 = {exact_total:.12f} Hartree")
    print(f"  精确电子能 = {exact_electronic:.12f} Hartree")
    print(f"  Hartree-Fock 误差 = {hf_error:.6e} Hartree")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制经典参考图"):
        _plot_reference(hf_total, exact_total, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
