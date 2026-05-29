"""[HYBRID] Task 06 -- 运行 SamplingVQE 优化 JSSP-QUBO。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import (
    CACHE_VERSION,
    TaskKind,
    bitstring_to_bits,
    create_sampler,
    decode_schedule,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_jssp_image,
    write_trace_csv,
)


class VQEOut(NamedTuple):
    vqe_result_json: FlyteFile
    vqe_result_dill: FlyteFile
    vqe_trace_csv: FlyteFile
    vqe_convergence_png: FlyteFile


def _make_optimizer(name: str, maxiter: int):
    from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B, SLSQP

    if name == "COBYLA":
        return COBYLA(maxiter=maxiter)
    if name == "SLSQP":
        return SLSQP(maxiter=maxiter)
    if name == "L_BFGS_B":
        return L_BFGS_B(maxiter=maxiter)
    raise ValueError(f"不支持的 optimizer: {name}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(np.real(value))
    except Exception:  # noqa: BLE001
        return None


def _serialize_best_measurement(best: dict | None) -> dict | None:
    if best is None:
        return None
    value = best.get("value")
    return {
        "state": int(best["state"]) if "state" in best else None,
        "bitstring": str(best.get("bitstring")),
        "probability": float(best.get("probability", 0.0)),
        "value": _as_float(value),
    }


def _summarize_eigenstate(instance: dict, qubo: dict, eigenstate: dict) -> list[dict]:
    rows: list[dict] = []
    for bitstring, probability in eigenstate.items():
        bits = bitstring_to_bits(str(bitstring), int(qubo["num_variables"]))
        energy = qubo_energy(qubo, bits)
        decoded = decode_schedule(instance, qubo["variables"], bits)
        rows.append(
            {
                "bitstring": str(bitstring),
                "probability": float(probability),
                "qubo_energy": float(energy),
                "feasible": bool(decoded["feasible"]),
                "makespan": decoded["actual_makespan"],
                "selected_cmax": decoded["selected_cmax"],
                "violations": decoded["violations"][:5],
            }
        )
    rows.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return rows


def _plot_convergence(trace: list[dict], reference: dict | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    if trace:
        xs = [row["eval_count"] for row in trace]
        ys = [row["mean_qubo_energy"] for row in trace]
        ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.2, color="#756bb1", label="VQE evaluations")
    if reference is not None:
        exact = reference["optimal_qubo_energy"]
        ax.axhline(exact, color="#31a354", linestyle="--", linewidth=1.3, label="Classical optimum")
    ax.set_xlabel("Function evaluation")
    ax.set_ylabel("QUBO energy")
    ax.set_title("SamplingVQE convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="6Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=30),
)
def t06_run_vqe(
    instance_json: FlyteFile,
    hamiltonian_dill: FlyteFile,
    ansatz_dill: FlyteFile,
    initial_point_npy: FlyteFile,
    reference_json: FlyteFile,
    optimizer: str = "COBYLA",
    maxiter: int = 120,
    shots: int = 4096,
    seed: int = 42,
) -> VQEOut:
    """[HYBRID] 量子采样估能量 + 经典优化器更新参数。"""

    print_banner(TaskKind.HYBRID, "Task 06 / 运行 SamplingVQE 混合优化")
    print(f"  optimizer = {optimizer}, maxiter = {maxiter}", flush=True)
    print(f"  sampler = StatevectorSampler(shots={shots}, seed={seed})", flush=True)

    from qiskit_algorithms import SamplingVQE

    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    with open(hamiltonian_dill.download(), "rb") as f:
        hamiltonian_pkg = dill.load(f)
    with open(ansatz_dill.download(), "rb") as f:
        ansatz_pkg = dill.load(f)

    operator = hamiltonian_pkg["operator"]
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(initial_point_npy.download())
    trace: list[dict] = []

    def callback(eval_count: int, parameters: np.ndarray, energy: float, metadata: dict) -> None:  # noqa: ARG001
        row = {
            "eval_count": int(eval_count),
            "mean_qubo_energy": float(np.real(energy)),
            "parameters": [float(x) for x in np.asarray(parameters).reshape(-1)],
        }
        trace.append(row)
        if eval_count == 1 or eval_count % 10 == 0:
            print(f"  eval {eval_count:03d}: mean QUBO energy = {row['mean_qubo_energy']:.6f}", flush=True)

    sampler = create_sampler(shots=shots, seed=seed)
    optimizer_inst = _make_optimizer(optimizer, maxiter)
    vqe = SamplingVQE(
        sampler,
        ansatz,
        optimizer_inst,
        initial_point=initial_point,
        callback=callback,
    )
    result = vqe.compute_minimum_eigenvalue(operator)

    eigenstate = {str(k): float(v) for k, v in dict(result.eigenstate).items()}
    top_samples = _summarize_eigenstate(instance, qubo, eigenstate)
    final_energy = float(np.real(result.eigenvalue))
    optimal_point = getattr(result, "optimal_point", None)
    optimal_point_list = [] if optimal_point is None else [float(x) for x in np.asarray(optimal_point).reshape(-1)]

    payload = {
        "algorithm": "SamplingVQE",
        "sampler": "StatevectorSampler",
        "optimizer": optimizer,
        "maxiter": int(maxiter),
        "shots": int(shots),
        "seed": int(seed),
        "vqe_mean_qubo_energy": final_energy,
        "optimal_point": optimal_point_list,
        "num_parameters": int(ansatz.num_parameters),
        "num_evaluations_recorded": len(trace),
        "cost_function_evals": _as_float(getattr(result, "cost_function_evals", None)),
        "optimizer_time_seconds": _as_float(getattr(result, "optimizer_time", None)),
        "raw_optimal_value": _as_float(getattr(result, "optimal_value", None)),
        "best_measurement": _serialize_best_measurement(getattr(result, "best_measurement", None)),
        "eigenstate": eigenstate,
        "top_samples": top_samples[:30],
        "classical_optimal_qubo_energy": reference["optimal_qubo_energy"],
        "energy_gap_to_classical_optimum": final_energy - reference["optimal_qubo_energy"],
    }

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
    write_trace_csv(trace, csv_path, int(ansatz.num_parameters))

    png_path = work / "06_vqe_convergence.png"
    _plot_convergence(trace, reference, png_path)

    print(f"  VQE 最终均值 QUBO energy = {final_energy:.6f}", flush=True)
    if payload["best_measurement"] is not None:
        best = payload["best_measurement"]
        print(f"  最佳测量 bitstring = {best['bitstring']}, energy = {best['value']}", flush=True)
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
