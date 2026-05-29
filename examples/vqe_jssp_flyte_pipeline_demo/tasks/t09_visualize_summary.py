"""[CLASSICAL] Task 09 -- 生成最终可视化总览图。"""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, print_banner, task_workdir, vqe_jssp_image


def _read_trace(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "eval_count": float(row["eval_count"]),
                    "mean_qubo_energy": float(row["mean_qubo_energy"]),
                }
            )
    return rows


def _read_samples(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "bitstring": row["bitstring"],
                    "probability": float(row["probability"]),
                    "qubo_energy": float(row["qubo_energy"]),
                    "feasible": row["feasible"] == "True",
                    "makespan": None if row["makespan"] in ("", "None") else int(float(row["makespan"])),
                }
            )
    return rows


def _plot_dashboard(
    instance: dict,
    qubo: dict,
    hamiltonian: dict,
    ansatz: dict,
    reference: dict,
    initial: dict,
    vqe: dict,
    decoded: dict,
    metrics: dict,
    trace: list[dict[str, float]],
    samples: list[dict],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    ax_setup, ax_energy, ax_conv, ax_probs, ax_gantt, ax_terms = axes.ravel()

    ax_setup.axis("off")
    setup_text = (
        f"Instance: {len(instance['jobs'])} jobs x {len(instance['machines'])} machines\n"
        f"horizon: {instance['horizon']}\n"
        f"QUBO variables / qubits: {qubo['num_variables']}\n"
        f"Pauli terms: {hamiltonian['num_pauli_terms']}\n"
        f"ansatz params: {ansatz['num_parameters']}\n"
        f"shots: {vqe['shots']}, optimizer: {vqe['optimizer']}"
    )
    ax_setup.text(0.02, 0.92, setup_text, va="top", fontsize=11)
    ax_setup.set_title("Pipeline setup")

    names = ["Initial mean", "VQE mean", "Exact QUBO"]
    values = [
        initial["initial_mean_qubo_energy"],
        vqe["vqe_mean_qubo_energy"],
        reference["optimal_qubo_energy"],
    ]
    colors = ["#9ecae1", "#756bb1", "#31a354"]
    bars = ax_energy.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax_energy.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.2f}", ha="center", fontsize=8)
    ax_energy.set_ylabel("QUBO energy")
    ax_energy.set_title("Energy comparison")
    ax_energy.grid(True, axis="y", alpha=0.25)

    if trace:
        ax_conv.plot(
            [row["eval_count"] for row in trace],
            [row["mean_qubo_energy"] for row in trace],
            color="#756bb1",
            marker="o",
            markersize=3,
            linewidth=1.1,
        )
    ax_conv.axhline(reference["optimal_qubo_energy"], color="#31a354", linestyle="--", linewidth=1.2)
    ax_conv.set_xlabel("eval")
    ax_conv.set_ylabel("QUBO energy")
    ax_conv.set_title("Hybrid optimization")
    ax_conv.grid(True, alpha=0.25)

    top = samples[: min(8, len(samples))]
    if top:
        ax_probs.bar(
            range(len(top)),
            [row["probability"] for row in top],
            color=["#31a354" if row["feasible"] else "#756bb1" for row in top],
            edgecolor="black",
            linewidth=0.5,
        )
        ax_probs.set_xticks(range(len(top)))
        ax_probs.set_xticklabels([row["bitstring"] for row in top], rotation=60, ha="right", fontsize=7)
    ax_probs.set_ylabel("probability")
    ax_probs.set_title("VQE candidates")
    ax_probs.grid(True, axis="y", alpha=0.25)

    ax_gantt.set_title("VQE best feasible schedule")
    best = decoded["best_feasible_sample"]
    if best is None:
        ax_gantt.axis("off")
        ax_gantt.text(0.5, 0.5, "No feasible sample found", ha="center", va="center")
    else:
        machines = [int(m) for m in instance["machines"]]
        job_colors = {0: "#4e79a7", 1: "#f28e2b"}
        for row in best["schedule"]:
            y = machines.index(int(row["machine"]))
            ax_gantt.barh(
                y,
                int(row["duration"]),
                left=int(row["start"]),
                height=0.55,
                color=job_colors[int(row["job"])],
                edgecolor="black",
                linewidth=0.7,
            )
            ax_gantt.text(
                int(row["start"]) + int(row["duration"]) / 2,
                y,
                row["label"],
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                weight="bold",
            )
        ax_gantt.axvline(best["makespan"], color="#d62728", linestyle="--", linewidth=1.1)
        ax_gantt.set_yticks(range(len(machines)))
        ax_gantt.set_yticklabels([f"M{m}" for m in machines])
        ax_gantt.set_xlim(0, int(instance["horizon"]) + 0.5)
        ax_gantt.set_xlabel("time")
        ax_gantt.grid(True, axis="x", alpha=0.25)

    top_terms = hamiltonian["pauli_terms"][: min(10, len(hamiltonian["pauli_terms"]))]
    labels = [term["pauli"] for term in top_terms]
    coeffs = [term["coefficient"]["real"] for term in top_terms]
    y = np.arange(len(labels))
    ax_terms.barh(
        y,
        coeffs,
        color=["#2ca25f" if coeff >= 0 else "#de2d26" for coeff in coeffs],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_terms.axvline(0, color="black", linewidth=0.8)
    ax_terms.set_yticks(y)
    ax_terms.set_yticklabels(labels, fontsize=7)
    ax_terms.invert_yaxis()
    ax_terms.set_xlabel("coefficient")
    status = "PASS" if metrics["decoded"]["best_measurement_qubo_energy_is_optimal"] else "CHECK"
    ax_terms.set_title(f"Largest Pauli terms / {status}")
    ax_terms.grid(True, axis="x", alpha=0.25)

    fig.suptitle("VQE for Job Shop Scheduling Problem (Flyte orchestrated)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=10),
)
def t09_visualize_summary(
    instance_json: FlyteFile,
    qubo_json: FlyteFile,
    hamiltonian_json: FlyteFile,
    ansatz_json: FlyteFile,
    reference_json: FlyteFile,
    initial_energy_json: FlyteFile,
    vqe_result_json: FlyteFile,
    decoded_solution_json: FlyteFile,
    metrics_json: FlyteFile,
    vqe_trace_csv: FlyteFile,
    samples_csv: FlyteFile,
) -> FlyteFile:
    """[CLASSICAL] 把关键 JSON/CSV 汇总为单张 dashboard PNG。"""

    print_banner(TaskKind.CLASSICAL, "Task 09 / 最终可视化总览")
    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    qubo = json.loads(Path(qubo_json.download()).read_text(encoding="utf-8"))
    hamiltonian = json.loads(Path(hamiltonian_json.download()).read_text(encoding="utf-8"))
    ansatz = json.loads(Path(ansatz_json.download()).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    initial = json.loads(Path(initial_energy_json.download()).read_text(encoding="utf-8"))
    vqe = json.loads(Path(vqe_result_json.download()).read_text(encoding="utf-8"))
    decoded = json.loads(Path(decoded_solution_json.download()).read_text(encoding="utf-8"))
    metrics = json.loads(Path(metrics_json.download()).read_text(encoding="utf-8"))
    trace = _read_trace(Path(vqe_trace_csv.download()))
    samples = _read_samples(Path(samples_csv.download()))

    out_path = task_workdir("t09") / "09_summary_dashboard.png"
    _plot_dashboard(
        instance,
        qubo,
        hamiltonian,
        ansatz,
        reference,
        initial,
        vqe,
        decoded,
        metrics,
        trace,
        samples,
        out_path,
    )
    print(f"  -> 写入 {out_path}", flush=True)
    return FlyteFile(str(out_path))
