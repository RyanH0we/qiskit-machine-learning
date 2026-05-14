"""[QUANTUM] Task 05 -- 用本地量子模拟器评估初始能量。

这一步第一次真正调用 Qiskit Estimator primitive。Estimator 会对给定 ansatz
和 Hamiltonian 计算期望值 <psi(theta)|H|psi(theta)>。默认使用本地
StatevectorEstimator，不连接真实量子硬件。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import dill
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, create_estimator, ensure_parent_dir, print_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", type=Path, required=True, help="task_02 输出的 problem.dill")
    p.add_argument("--ansatz", type=Path, required=True, help="task_04 输出的 ansatz.dill")
    p.add_argument("--initial-point", type=Path, required=True, help="task_04 输出的 initial_point.npy")
    p.add_argument("--reference", type=Path, default=None, help="可选：task_03 输出的 reference.json")
    p.add_argument("--output", type=Path, required=True, help="输出 initial_energy.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出初始能量图 PNG")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _extract_estimator_value(pub_result) -> tuple[float, float]:
    evs = np.asarray(pub_result.data.evs).reshape(-1)
    stds = np.asarray(pub_result.data.stds).reshape(-1)
    ev = float(np.real(evs[0]))
    std = float(np.real(stds[0])) if len(stds) else 0.0
    return ev, std


def _plot_initial(payload: dict, reference: dict | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    names = ["Initial"]
    values = [payload["initial_total_energy_hartree"]]
    colors = ["#756bb1"]
    if reference is not None:
        names.append("Exact")
        values.append(reference["exact_total_energy_hartree"])
        colors.append("#31a354")

    out = ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.6f}", ha="center", fontsize=10)
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Initial ansatz energy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(min(values) - 0.06, max(values) + 0.06)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 05 / 本地模拟器评估初始能量")
    print(f"  estimator = StatevectorEstimator(seed={args.seed})")

    with args.problem.open("rb") as f:
        problem_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    qubit_operator = problem_pkg["qubit_operator"]
    nuclear = float(problem_pkg["metadata"]["nuclear_repulsion_energy"])
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)

    estimator = create_estimator(seed=args.seed)
    with Timer("Estimator 计算初始期望值"):
        result = estimator.run([(ansatz, qubit_operator, initial_point)]).result()
    electronic_energy, std = _extract_estimator_value(result[0])
    total_energy = electronic_energy + nuclear

    payload = {
        "estimator": "StatevectorEstimator",
        "seed": args.seed,
        "initial_electronic_energy_hartree": electronic_energy,
        "nuclear_repulsion_energy_hartree": nuclear,
        "initial_total_energy_hartree": total_energy,
        "estimator_std": std,
        "initial_point": initial_point.tolist(),
    }
    print(f"  初始电子能 = {electronic_energy:.12f} Hartree")
    print(f"  初始总能量 = {total_energy:.12f} Hartree")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    reference = read_json(args.reference) if args.reference else None
    with Timer("绘制初始能量图"):
        _plot_initial(payload, reference, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
