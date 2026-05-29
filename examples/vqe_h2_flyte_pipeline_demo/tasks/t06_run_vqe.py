"""[HYBRID] Task 06 -- 运行 VQE：量子估能量 + 经典优化器调参数。

VQE 是量子-经典混合算法的代表：

* 量子侧（``StatevectorEstimator``）负责对参数化线路 ansatz(theta) 测
  Hamiltonian 的期望值 E(theta)。
* 经典侧（SLSQP/COBYLA/L_BFGS_B）根据 E(theta) 更新 theta。

这两步在一个循环里反复迭代，直到能量尽可能低。
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, create_estimator, print_banner, task_workdir, vqe_image


class VQEOut(NamedTuple):
    vqe_result_json: FlyteFile
    vqe_result_dill: FlyteFile
    vqe_trace_csv: FlyteFile
    vqe_convergence_png: FlyteFile


def _make_optimizer(name: str, maxiter: int):
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP

    if name == "SLSQP":
        return SLSQP(maxiter=maxiter)
    if name == "COBYLA":
        return COBYLA(maxiter=maxiter)
    if name == "L_BFGS_B":
        return L_BFGS_B(maxiter=maxiter)
    raise ValueError(f"不支持的 optimizer: {name}")


def _as_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(np.real(value))
    except Exception:  # noqa: BLE001
        return None


def _extract_estimator_value(pub_result) -> float:
    evs = np.asarray(pub_result.data.evs).reshape(-1)
    return float(np.real(evs[0]))


def _write_trace(trace: list[dict], out_path: Path, num_parameters: int) -> None:
    fields = ["eval_count", "electronic_energy_hartree", "total_energy_hartree"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            flat = {
                "eval_count": row["eval_count"],
                "electronic_energy_hartree": row["electronic_energy_hartree"],
                "total_energy_hartree": row["total_energy_hartree"],
            }
            if len(row["parameters"]) != num_parameters:
                raise ValueError(
                    f"trace 参数长度 {len(row['parameters'])} 与 ansatz 参数数 {num_parameters} 不一致"
                )
            for i, value in enumerate(row["parameters"]):
                flat[f"theta_{i}"] = value
            writer.writerow(flat)


def _plot_convergence(
    trace: list[dict],
    final_energy: float,
    reference: dict | None,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if trace:
        xs = [row["eval_count"] for row in trace]
        ys = [row["total_energy_hartree"] for row in trace]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.3, color="#756bb1", label="VQE evaluations")
    else:
        ax.scatter([1], [final_energy], color="#756bb1", label="VQE final")

    if reference is not None:
        exact = reference["exact_total_energy_hartree"]
        chemical_accuracy = 0.0016
        ax.axhline(exact, color="#31a354", linestyle="--", linewidth=1.3, label="Exact")
        ax.axhspan(exact - chemical_accuracy, exact + chemical_accuracy, color="#a1d99b", alpha=0.18, label="Chemical accuracy")

    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title("VQE convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="2", mem="3Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t06_run_vqe(
    problem_dill: FlyteFile,
    ansatz_dill: FlyteFile,
    initial_point_npy: FlyteFile,
    reference_json: FlyteFile,
    optimizer: str = "SLSQP",
    maxiter: int = 100,
    seed: int = 42,
) -> VQEOut:
    """[HYBRID] 量子估值 + 经典优化的 VQE 主循环。"""

    import warnings

    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    print_banner(TaskKind.HYBRID, "Task 06 / 运行 VQE 混合优化")
    print(f"  optimizer = {optimizer}, maxiter = {maxiter}", flush=True)
    print(f"  estimator = StatevectorEstimator(seed={seed})", flush=True)

    with open(problem_dill.download(), "rb") as f:
        problem_pkg = dill.load(f)
    with open(ansatz_dill.download(), "rb") as f:
        ansatz_pkg = dill.load(f)
    qubit_operator = problem_pkg["qubit_operator"]
    nuclear = float(problem_pkg["metadata"]["nuclear_repulsion_energy"])
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(initial_point_npy.download())

    trace: list[dict] = []
    estimator = create_estimator(seed=seed)
    optimizer_inst = _make_optimizer(optimizer, maxiter)

    def objective(parameters: np.ndarray) -> float:
        point = np.asarray(parameters, dtype=float).reshape(-1)
        result = estimator.run([(ansatz, qubit_operator, point)]).result()
        electronic = _extract_estimator_value(result[0])
        eval_count = len(trace) + 1
        trace.append(
            {
                "eval_count": eval_count,
                "electronic_energy_hartree": electronic,
                "total_energy_hartree": electronic + nuclear,
                "parameters": [float(x) for x in point],
            }
        )
        if eval_count == 1 or eval_count % 5 == 0:
            print(f"  eval {eval_count:03d}: total energy = {electronic + nuclear:.12f} Hartree", flush=True)
        return electronic

    t0 = time.perf_counter()
    result = optimizer_inst.minimize(fun=objective, x0=np.asarray(initial_point, dtype=float))
    optimizer_time = time.perf_counter() - t0

    final_electronic = float(np.real(result.fun))
    final_total = final_electronic + nuclear
    optimal_point = getattr(result, "x", None)
    optimal_point_list = (
        [float(x) for x in np.asarray(optimal_point).reshape(-1)] if optimal_point is not None else []
    )

    payload = {
        "algorithm": "VQE",
        "estimator": "StatevectorEstimator",
        "optimizer": optimizer,
        "maxiter": maxiter,
        "seed": seed,
        "vqe_total_energy_hartree": final_total,
        "vqe_electronic_energy_hartree": final_electronic,
        "nuclear_repulsion_energy_hartree": nuclear,
        "optimal_point": optimal_point_list,
        "num_parameters": int(ansatz.num_parameters),
        "num_evaluations_recorded": len(trace),
        "cost_function_evals": _as_float(getattr(result, "nfev", None)),
        "optimizer_time_seconds": optimizer_time,
        "raw_optimal_value_hartree": _as_float(getattr(result, "fun", None)),
    }

    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    payload["exact_total_energy_hartree"] = reference["exact_total_energy_hartree"]
    payload["absolute_error_hartree"] = abs(final_total - reference["exact_total_energy_hartree"])

    print(f"  VQE 最终总能量 = {final_total:.12f} Hartree", flush=True)
    print(f"  与精确参考误差 = {payload['absolute_error_hartree']:.6e} Hartree", flush=True)

    work = task_workdir("t06")
    json_path = work / "vqe_result.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dill_path = work / "vqe_result.dill"
    with dill_path.open("wb") as f:
        dill.dump({"result": result, "payload": payload, "trace": trace}, f)

    csv_path = work / "vqe_trace.csv"
    _write_trace(trace, csv_path, int(ansatz.num_parameters))

    png_path = work / "06_vqe_convergence.png"
    _plot_convergence(trace, final_total, reference, png_path)

    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {dill_path}", flush=True)
    print(f"  -> 写入 {csv_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return VQEOut(
        vqe_result_json=FlyteFile(str(json_path)),
        vqe_result_dill=FlyteFile(str(dill_path)),
        vqe_trace_csv=FlyteFile(str(csv_path)),
        vqe_convergence_png=FlyteFile(str(png_path)),
    )
