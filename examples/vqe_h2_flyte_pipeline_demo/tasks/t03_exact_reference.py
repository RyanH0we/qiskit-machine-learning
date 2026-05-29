"""[CLASSICAL] Task 03 -- 用经典精确对角化得到 H2 基态参考能量。

对 ``t02`` 输出的 4-qubit Hamiltonian 做精确对角化（NumPyMinimumEigensolver）。
仅适合 H2 这种小体系做教学校验，作为后续 VQE 结果的"真值"参照。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, hartree_to_ev, print_banner, task_workdir, vqe_image


class ReferenceOut(NamedTuple):
    reference_json: FlyteFile
    reference_png: FlyteFile


def _plot_reference(hf_energy: float, exact_energy: float, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="500m", mem="1Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t03_exact_reference(problem_dill: FlyteFile) -> ReferenceOut:
    """[CLASSICAL] NumPyMinimumEigensolver 精确对角化。"""

    from qiskit_algorithms import NumPyMinimumEigensolver

    print_banner(TaskKind.CLASSICAL, "Task 03 / 精确经典参考")

    with open(problem_dill.download(), "rb") as f:
        pkg = dill.load(f)
    qubit_operator = pkg["qubit_operator"]
    metadata = pkg["metadata"]

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
    print(f"  精确总能量 = {exact_total:.12f} Hartree", flush=True)
    print(f"  精确电子能 = {exact_electronic:.12f} Hartree", flush=True)
    print(f"  Hartree-Fock 误差 = {hf_error:.6e} Hartree", flush=True)

    work = task_workdir("t03")
    json_path = work / "reference.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "03_reference_energy.png"
    _plot_reference(hf_total, exact_total, png_path)

    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return ReferenceOut(
        reference_json=FlyteFile(str(json_path)),
        reference_png=FlyteFile(str(png_path)),
    )
