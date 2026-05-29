"""[QUANTUM] Task 05 -- 用本地量子模拟器计算初始 ansatz 的能量期望值。

这是流水线第一次真正调用 Qiskit 的 Estimator primitive。默认使用
``StatevectorEstimator``——一个本地、精确、无噪声的状态向量模拟器，
不涉及任何真实量子硬件。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, create_estimator, print_banner, task_workdir, vqe_image


class InitialEnergyOut(NamedTuple):
    initial_energy_json: FlyteFile
    initial_energy_png: FlyteFile


def _extract_estimator_value(pub_result) -> tuple[float, float]:
    evs = np.asarray(pub_result.data.evs).reshape(-1)
    stds = np.asarray(pub_result.data.stds).reshape(-1)
    ev = float(np.real(evs[0]))
    std = float(np.real(stds[0])) if len(stds) else 0.0
    return ev, std


def _plot_initial(payload: dict, reference: dict | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["Initial"]
    values = [payload["initial_total_energy_hartree"]]
    colors = ["#756bb1"]
    if reference is not None:
        names.append("Exact")
        values.append(reference["exact_total_energy_hartree"])
        colors.append("#31a354")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.6f}", ha="center", fontsize=10)
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("Initial ansatz energy")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(min(values) - 0.06, max(values) + 0.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="1", mem="2Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t05_initial_energy(
    problem_dill: FlyteFile,
    ansatz_dill: FlyteFile,
    initial_point_npy: FlyteFile,
    reference_json: FlyteFile,
    seed: int = 42,
) -> InitialEnergyOut:
    """[QUANTUM] StatevectorEstimator 计算初始能量期望值。"""

    import warnings

    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    print_banner(TaskKind.QUANTUM, "Task 05 / 本地模拟器评估初始能量")
    print(f"  estimator = StatevectorEstimator(seed={seed})", flush=True)

    with open(problem_dill.download(), "rb") as f:
        problem_pkg = dill.load(f)
    with open(ansatz_dill.download(), "rb") as f:
        ansatz_pkg = dill.load(f)

    qubit_operator = problem_pkg["qubit_operator"]
    nuclear = float(problem_pkg["metadata"]["nuclear_repulsion_energy"])
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(initial_point_npy.download())

    estimator = create_estimator(seed=seed)
    result = estimator.run([(ansatz, qubit_operator, initial_point)]).result()
    electronic_energy, std = _extract_estimator_value(result[0])
    total_energy = electronic_energy + nuclear

    payload = {
        "estimator": "StatevectorEstimator",
        "seed": seed,
        "initial_electronic_energy_hartree": electronic_energy,
        "nuclear_repulsion_energy_hartree": nuclear,
        "initial_total_energy_hartree": total_energy,
        "estimator_std": std,
        "initial_point": initial_point.tolist(),
    }
    print(f"  初始电子能 = {electronic_energy:.12f} Hartree", flush=True)
    print(f"  初始总能量 = {total_energy:.12f} Hartree", flush=True)

    work = task_workdir("t05")
    json_path = work / "initial_energy.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))

    png_path = work / "05_initial_energy.png"
    _plot_initial(payload, reference, png_path)

    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return InitialEnergyOut(
        initial_energy_json=FlyteFile(str(json_path)),
        initial_energy_png=FlyteFile(str(png_path)),
    )
