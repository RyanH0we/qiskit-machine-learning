"""[QUANTUM] Task 05 -- 用本地量子模拟器评估初始 ansatz 的 QUBO 能量。

这一步第一次真正调用 Qiskit Sampler primitive。默认使用本地
StatevectorSampler，不连接真实量子硬件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    bitstring_to_bits,
    create_sampler,
    decode_schedule,
    ensure_parent_dir,
    print_banner,
    qubo_energy,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instance", type=Path, required=True, help="task_01 输出的 instance.json")
    p.add_argument("--hamiltonian", type=Path, required=True, help="task_02 输出的 hamiltonian.dill")
    p.add_argument("--ansatz", type=Path, required=True, help="task_04 输出的 ansatz.dill")
    p.add_argument("--initial-point", type=Path, required=True, help="task_04 输出的 initial_point.npy")
    p.add_argument("--output", type=Path, required=True, help="输出 initial_energy.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出初始采样图 PNG")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _sample_counts(ansatz, initial_point: np.ndarray, shots: int, seed: int) -> dict[str, int]:
    circuit = ansatz.assign_parameters(initial_point, inplace=False)
    circuit.measure_all()
    sampler = create_sampler(shots=shots, seed=seed)
    result = sampler.run([circuit]).result()[0]
    return result.data.meas.get_counts()


def _summarize_counts(instance: dict, qubo: dict, counts: dict[str, int]) -> tuple[float, list[dict]]:
    shots = sum(counts.values())
    rows: list[dict] = []
    mean_energy = 0.0
    for bitstring, count in counts.items():
        bits = bitstring_to_bits(bitstring, int(qubo["num_variables"]))
        energy = qubo_energy(qubo, bits)
        prob = count / shots
        decoded = decode_schedule(instance, qubo["variables"], bits)
        mean_energy += prob * energy
        rows.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": float(prob),
                "qubo_energy": float(energy),
                "feasible": bool(decoded["feasible"]),
                "makespan": decoded["actual_makespan"],
                "selected_cmax": decoded["selected_cmax"],
                "violations": decoded["violations"][:5],
            }
        )
    rows.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return float(mean_energy), rows


def _plot_initial(rows: list[dict], mean_energy: float, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    top = rows[: min(10, len(rows))]
    labels = [row["bitstring"] for row in top]
    probs = [row["probability"] for row in top]
    colors = ["#31a354" if row["feasible"] else "#756bb1" for row in top]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    bars = ax.bar(range(len(top)), probs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, top):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"E={row['qubo_energy']:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Sample probability")
    ax.set_title(f"Initial ansatz samples (mean QUBO energy = {mean_energy:.3f})")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 05 / 本地模拟器评估初始能量")
    print(f"  sampler = StatevectorSampler(shots={args.shots}, seed={args.seed})")
    instance = read_json(args.instance)

    with args.hamiltonian.open("rb") as f:
        hamiltonian_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)

    with Timer("Sampler 初始采样"):
        counts = _sample_counts(ansatz, initial_point, args.shots, args.seed)
    mean_energy, rows = _summarize_counts(instance, qubo, counts)
    best_by_energy = min(rows, key=lambda row: (row["qubo_energy"], -row["probability"], row["bitstring"]))
    print(f"  初始采样均值 QUBO energy = {mean_energy:.6f}")
    print(f"  初始样本中最低能量 = {best_by_energy['qubo_energy']:.6f}")
    print(f"  初始样本中最低能量是否可行 = {best_by_energy['feasible']}")

    payload = {
        "sampler": "StatevectorSampler",
        "shots": int(args.shots),
        "seed": int(args.seed),
        "initial_mean_qubo_energy": mean_energy,
        "initial_best_sample": best_by_energy,
        "num_unique_bitstrings": len(rows),
        "top_samples": rows[:20],
        "initial_point": initial_point.tolist(),
    }
    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制初始采样分布"):
        _plot_initial(rows, mean_energy, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
