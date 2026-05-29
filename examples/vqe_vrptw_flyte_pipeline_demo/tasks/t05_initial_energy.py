"""[QUANTUM] Task 05 -- 用本地量子模拟器评估初始 ansatz 的 QUBO 能量。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import (
    CACHE_VERSION,
    TaskKind,
    bitstring_to_bits,
    create_sampler,
    decode_vrptw_solution,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_vrptw_image,
)


class InitialEnergyOut(NamedTuple):
    initial_energy_json: FlyteFile
    initial_energy_png: FlyteFile


def _sample_counts(ansatz, initial_point: np.ndarray, shots: int, seed: int) -> dict[str, int]:
    circuit = ansatz.assign_parameters(initial_point, inplace=False)
    circuit.measure_all()
    sampler = create_sampler(shots=shots, seed=seed)
    result = sampler.run([circuit]).result()[0]
    return result.data.meas.get_counts()


def summarize_counts(instance: dict, qubo: dict, counts: dict[str, int]) -> tuple[float, list[dict]]:
    shots = sum(counts.values())
    rows: list[dict] = []
    mean_energy = 0.0
    for bitstring, count in counts.items():
        bits = bitstring_to_bits(bitstring, int(qubo["num_variables"]))
        energy = qubo_energy(qubo, bits)
        prob = count / shots
        decoded = decode_vrptw_solution(instance, qubo["variables"], bits)
        mean_energy += prob * energy
        rows.append(
            {
                "bitstring": bitstring,
                "count": int(count),
                "probability": float(prob),
                "qubo_energy": float(energy),
                "feasible": bool(decoded["feasible"]),
                "total_distance": float(decoded["total_distance"]),
                "num_vehicles_used": int(decoded["num_vehicles_used"]),
                "num_customers_served": int(decoded["num_customers_served"]),
                "violations": decoded["violations"][:5],
            }
        )
    rows.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return float(mean_energy), rows


def _plot_initial(rows: list[dict], mean_energy: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = rows[: min(10, len(rows))]
    labels = [row["bitstring"] for row in top]
    probs = [row["probability"] for row in top]
    colors = ["#31a354" if row["feasible"] else "#756bb1" for row in top]

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    bars = ax.bar(range(len(top)), probs, color=colors, edgecolor="black", linewidth=0.5)
    for bar, row in zip(bars, top):
        tag = f"d={row['total_distance']:.1f}" if row["feasible"] else f"E={row['qubo_energy']:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, tag, ha="center", fontsize=8)
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("sample probability")
    ax.set_title(f"Initial ansatz samples / mean QUBO energy={mean_energy:.3f}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_vrptw_image,
    requests=Resources(cpu="1", mem="4Gi"),
    limits=Resources(cpu="2", mem="8Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=15),
)
def t05_initial_energy(
    instance_json: FlyteFile,
    hamiltonian_dill: FlyteFile,
    ansatz_dill: FlyteFile,
    initial_point_npy: FlyteFile,
    shots: int = 4096,
    seed: int = 42,
) -> InitialEnergyOut:
    """[QUANTUM] 对初始参数线路进行本地 sampler 采样。"""

    print_banner(TaskKind.QUANTUM, "Task 05 / 本地模拟器评估初始能量")
    print(f"  sampler = StatevectorSampler(shots={shots}, seed={seed})", flush=True)
    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    with open(hamiltonian_dill.download(), "rb") as f:
        hamiltonian_pkg = dill.load(f)
    with open(ansatz_dill.download(), "rb") as f:
        ansatz_pkg = dill.load(f)
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(initial_point_npy.download())

    counts = _sample_counts(ansatz, initial_point, shots, seed)
    mean_energy, rows = summarize_counts(instance, qubo, counts)
    best_by_energy = min(rows, key=lambda row: (row["qubo_energy"], -row["probability"], row["bitstring"]))
    feasible_rows = [row for row in rows if row["feasible"]]
    best_feasible = min(feasible_rows, key=lambda row: (row["total_distance"], row["qubo_energy"])) if feasible_rows else None

    payload = {
        "sampler": "StatevectorSampler",
        "shots": int(shots),
        "seed": int(seed),
        "initial_mean_qubo_energy": mean_energy,
        "initial_best_sample_by_energy": best_by_energy,
        "initial_best_feasible_sample": best_feasible,
        "num_unique_bitstrings": len(rows),
        "top_samples": rows[:30],
        "initial_point": initial_point.tolist(),
    }

    work = task_workdir("t05")
    json_path = work / "initial_energy.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    png_path = work / "05_initial_energy.png"
    _plot_initial(rows, mean_energy, png_path)

    print(f"  初始采样均值 QUBO energy = {mean_energy:.6f}", flush=True)
    print(f"  初始样本中最低能量 = {best_by_energy['qubo_energy']:.6f}", flush=True)
    print(f"  初始样本中是否出现可行解 = {best_feasible is not None}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)
    return InitialEnergyOut(
        initial_energy_json=FlyteFile(str(json_path)),
        initial_energy_png=FlyteFile(str(png_path)),
    )
