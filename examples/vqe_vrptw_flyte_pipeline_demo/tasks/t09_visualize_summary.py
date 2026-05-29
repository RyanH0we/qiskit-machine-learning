"""[CLASSICAL] Task 09 -- 生成最终可视化总览图。"""

from __future__ import annotations

import csv
import json
from datetime import timedelta
from pathlib import Path

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, customer_by_id, print_banner, task_workdir, vqe_vrptw_image


def _read_trace(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"eval_count": float(row["eval_count"]), "mean_qubo_energy": float(row["mean_qubo_energy"])})
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
                    "total_distance": float(row["total_distance"]),
                    "num_customers_served": int(float(row["num_customers_served"])),
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

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.8))
    ax_setup, ax_energy, ax_conv, ax_samples, ax_routes, ax_terms = axes.ravel()

    ax_setup.axis("off")
    setup_text = (
        f"Instance: {len(instance['vehicles'])} vehicles x {len(instance['customers'])} customers\n"
        f"max stops/vehicle: {instance['solver_config']['max_stops_per_vehicle']}\n"
        f"QUBO variables / qubits: {qubo['num_variables']}\n"
        f"x/y/slack: {qubo['num_x_variables']}/{qubo['num_y_variables']}/{qubo['num_capacity_slack_variables']}\n"
        f"Pauli terms: {hamiltonian['num_pauli_terms']}\n"
        f"ansatz params: {ansatz['num_parameters']}\n"
        f"shots: {vqe['shots']}, optimizer: {vqe['optimizer']}"
    )
    ax_setup.text(0.02, 0.94, setup_text, va="top", fontsize=10.5)
    ax_setup.set_title("Pipeline setup")

    names = ["Initial mean", "VQE mean", "Reference"]
    values = [initial["initial_mean_qubo_energy"], vqe["vqe_mean_qubo_energy"], reference["best_feasible_qubo_energy"]]
    colors = ["#9ecae1", "#756bb1", "#31a354"]
    bars = ax_energy.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax_energy.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.2f}", ha="center", fontsize=8)
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
    ax_conv.axhline(reference["best_feasible_qubo_energy"], color="#31a354", linestyle="--", linewidth=1.2)
    ax_conv.set_xlabel("eval")
    ax_conv.set_ylabel("QUBO energy")
    ax_conv.set_title("SamplingVQE convergence")
    ax_conv.grid(True, alpha=0.25)

    top = samples[: min(8, len(samples))]
    if top:
        ax_samples.bar(
            range(len(top)),
            [row["probability"] for row in top],
            color=["#31a354" if row["feasible"] else "#756bb1" for row in top],
            edgecolor="black",
            linewidth=0.5,
        )
        ax_samples.set_xticks(range(len(top)))
        ax_samples.set_xticklabels([row["bitstring"] for row in top], rotation=60, ha="right", fontsize=7)
    ax_samples.set_ylabel("probability")
    ax_samples.set_title("VQE candidates")
    ax_samples.grid(True, axis="y", alpha=0.25)

    ax_routes.set_title("VQE best feasible routes")
    best = decoded["best_feasible_sample"]
    depot = instance["depot"]
    customers = customer_by_id(instance)
    ax_routes.scatter([depot["x"]], [depot["y"]], s=120, marker="s", color="#525252", edgecolor="black")
    ax_routes.text(depot["x"] + 0.05, depot["y"] + 0.05, depot["id"], fontsize=9, weight="bold")
    for customer in instance["customers"]:
        ax_routes.scatter([customer["x"]], [customer["y"]], s=90, color="#9ecae1", edgecolor="black")
        ax_routes.text(customer["x"] + 0.05, customer["y"] + 0.05, customer["id"], fontsize=9, weight="bold")
    if best is not None:
        route_colors = ["#3182bd", "#31a354", "#de2d26"]
        for i, route in enumerate(best["routes"]):
            if not route["events"]:
                continue
            points = [depot, *[customers[event["customer_id"]] for event in route["events"]], depot]
            ax_routes.plot(
                [p["x"] for p in points],
                [p["y"] for p in points],
                color=route_colors[i % len(route_colors)],
                marker="o",
                linewidth=2.0,
                label=route["vehicle_id"],
            )
        ax_routes.text(
            0.02,
            0.04,
            f"distance={best['total_distance']:.3f}\nfeasible={best['feasible']}",
            transform=ax_routes.transAxes,
            fontsize=9,
            va="bottom",
        )
    ax_routes.set_aspect("equal")
    ax_routes.grid(True, alpha=0.25)
    if best is not None:
        ax_routes.legend(fontsize=8)

    top_terms = hamiltonian["pauli_terms"][: min(10, len(hamiltonian["pauli_terms"]))]
    y = np.arange(len(top_terms))
    coeffs = [term["coefficient"]["real"] for term in top_terms]
    ax_terms.barh(
        y,
        coeffs,
        color=["#2ca25f" if coeff >= 0 else "#de2d26" for coeff in coeffs],
        edgecolor="black",
        linewidth=0.4,
    )
    ax_terms.axvline(0, color="black", linewidth=0.8)
    ax_terms.set_yticks(y)
    ax_terms.set_yticklabels([term["pauli"] for term in top_terms], fontsize=7)
    ax_terms.invert_yaxis()
    ax_terms.set_xlabel("coefficient")
    status = "PASS" if metrics["acceptance"]["best_feasible_found"] else "CHECK"
    ax_terms.set_title(f"Largest Pauli terms / {status}")
    ax_terms.grid(True, axis="x", alpha=0.25)

    fig.suptitle("VQE for VRPTW Solver: time-indexed QUBO + SamplingVQE (Flyte)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_vrptw_image,
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
    """[CLASSICAL] 把关键 JSON/CSV 汇总为 dashboard PNG。"""

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
    _plot_dashboard(instance, qubo, hamiltonian, ansatz, reference, initial, vqe, decoded, metrics, trace, samples, out_path)
    print(f"  -> 写入 {out_path}", flush=True)
    return FlyteFile(str(out_path))
