"""[CLASSICAL] Task 08 -- 生成最终可视化总览图。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, print_banner, task_workdir, vqe_image


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
                    "total_energy_hartree": float(row["total_energy_hartree"]),
                }
            )
    return rows


def _plot_dashboard(
    molecule: dict,
    hamiltonian: dict,
    reference: dict,
    ansatz: dict,
    initial: dict,
    vqe: dict,
    metrics: dict,
    trace: list[dict[str, float]],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    ax_mol, ax_energy, ax_conv, ax_terms = axes.ravel()

    bond = molecule["bond_length_angstrom"]
    ax_mol.plot([0, bond], [0, 0], color="#4d4d4d", linewidth=3)
    ax_mol.scatter([0, bond], [0, 0], s=1100, color="#6baed6", edgecolor="black", linewidth=1.2)
    ax_mol.text(0, 0, "H", ha="center", va="center", fontsize=18, weight="bold")
    ax_mol.text(bond, 0, "H", ha="center", va="center", fontsize=18, weight="bold")
    ax_mol.text(bond / 2, 0.22, f"bond = {bond:.3f} A", ha="center", fontsize=10)
    summary = (
        f"basis: {molecule['basis']}\n"
        f"qubits: {hamiltonian['num_qubits']}\n"
        f"Pauli terms: {hamiltonian['num_pauli_terms']}\n"
        f"ansatz params: {ansatz['num_parameters']}\n"
        "tasks: 6 classical + 1 quantum + 1 hybrid"
    )
    ax_mol.text(0.02, 0.05, summary, transform=ax_mol.transAxes, fontsize=10, va="bottom")
    ax_mol.set_xlim(-0.35, bond + 0.35)
    ax_mol.set_ylim(-0.35, 0.55)
    ax_mol.set_aspect("equal")
    ax_mol.axis("off")
    ax_mol.set_title("Problem setup")

    names = ["HF", "Initial", "VQE", "Exact"]
    values = [
        reference["hartree_fock_total_energy_hartree"],
        initial["initial_total_energy_hartree"],
        vqe["vqe_total_energy_hartree"],
        reference["exact_total_energy_hartree"],
    ]
    colors = ["#9ecae1", "#756bb1", "#fd8d3c", "#31a354"]
    bars = ax_energy.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax_energy.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.6f}", ha="center", fontsize=8)
    ax_energy.set_ylabel("Hartree")
    ax_energy.set_title("Energy comparison")
    ax_energy.set_ylim(min(values) - 0.06, max(values) + 0.06)
    ax_energy.grid(True, axis="y", alpha=0.25)
    status = "PASS" if metrics["vqe_within_chemical_accuracy"] else "CHECK"
    ax_energy.text(
        0.03,
        0.06,
        f"VQE abs error = {metrics['errors_hartree']['vqe_abs_error']:.2e} Ha\n{status}: chemical accuracy",
        transform=ax_energy.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )

    if trace:
        ax_conv.plot(
            [row["eval_count"] for row in trace],
            [row["total_energy_hartree"] for row in trace],
            color="#756bb1",
            marker="o",
            markersize=3,
            linewidth=1.2,
        )
    exact = reference["exact_total_energy_hartree"]
    chem = metrics["chemical_accuracy_hartree"]
    ax_conv.axhline(exact, color="#31a354", linestyle="--", linewidth=1.2)
    ax_conv.axhspan(exact - chem, exact + chem, color="#a1d99b", alpha=0.18)
    ax_conv.set_xlabel("Function evaluation")
    ax_conv.set_ylabel("Hartree")
    ax_conv.set_title("Hybrid VQE loop")
    ax_conv.grid(True, alpha=0.25)

    top_terms = hamiltonian["pauli_terms"][: min(10, len(hamiltonian["pauli_terms"]))]
    labels = [t["pauli"] for t in top_terms]
    coeffs = [t["coefficient"]["real"] for t in top_terms]
    y = np.arange(len(labels))
    ax_terms.barh(y, coeffs, color=["#2ca25f" if c >= 0 else "#de2d26" for c in coeffs], edgecolor="black", linewidth=0.4)
    ax_terms.axvline(0, color="black", linewidth=0.8)
    ax_terms.set_yticks(y)
    ax_terms.set_yticklabels(labels, fontsize=8)
    ax_terms.invert_yaxis()
    ax_terms.set_xlabel("Coefficient")
    ax_terms.set_title("Largest Pauli terms")
    ax_terms.grid(True, axis="x", alpha=0.25)

    fig.suptitle("VQE for H2 ground-state energy (Flyte orchestrated)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="500m", mem="1Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t08_visualize_summary(
    molecule_json: FlyteFile,
    hamiltonian_json: FlyteFile,
    reference_json: FlyteFile,
    ansatz_json: FlyteFile,
    initial_energy_json: FlyteFile,
    vqe_result_json: FlyteFile,
    metrics_json: FlyteFile,
    vqe_trace_csv: FlyteFile,
) -> FlyteFile:
    """[CLASSICAL] 把 7 个 JSON + 1 个 CSV 汇总为单张 dashboard。"""

    print_banner(TaskKind.CLASSICAL, "Task 08 / 最终可视化总览")
    molecule = json.loads(Path(molecule_json.download()).read_text(encoding="utf-8"))
    hamiltonian = json.loads(Path(hamiltonian_json.download()).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    ansatz = json.loads(Path(ansatz_json.download()).read_text(encoding="utf-8"))
    initial = json.loads(Path(initial_energy_json.download()).read_text(encoding="utf-8"))
    vqe = json.loads(Path(vqe_result_json.download()).read_text(encoding="utf-8"))
    metrics = json.loads(Path(metrics_json.download()).read_text(encoding="utf-8"))
    trace = _read_trace(Path(vqe_trace_csv.download()))

    out_path = task_workdir("t08") / "07_summary_dashboard.png"
    _plot_dashboard(molecule, hamiltonian, reference, ansatz, initial, vqe, metrics, trace, out_path)
    print(f"  -> 写入 {out_path}", flush=True)
    return FlyteFile(str(out_path))
