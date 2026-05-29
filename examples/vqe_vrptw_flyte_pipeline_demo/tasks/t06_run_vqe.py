"""[HYBRID] Task 06 -- 运行 sampler-based VQE 优化 VRPTW-QUBO。"""

from __future__ import annotations

import json
import time
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
    decode_vrptw_solution,
    print_banner,
    qubo_energy,
    task_workdir,
    vqe_vrptw_image,
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


def _summarize_eigenstate(instance: dict, qubo: dict, eigenstate: dict) -> list[dict]:
    rows: list[dict] = []
    for bitstring, probability in eigenstate.items():
        bits = bitstring_to_bits(str(bitstring), int(qubo["num_variables"]))
        energy = qubo_energy(qubo, bits)
        decoded = decode_vrptw_solution(instance, qubo["variables"], bits)
        rows.append(
            {
                "bitstring": str(bitstring),
                "probability": float(probability),
                "qubo_energy": float(energy),
                "feasible": bool(decoded["feasible"]),
                "total_distance": float(decoded["total_distance"]),
                "num_customers_served": int(decoded["num_customers_served"]),
                "num_vehicles_used": int(decoded["num_vehicles_used"]),
                "violations": decoded["violations"][:5],
            }
        )
    rows.sort(key=lambda row: (-row["probability"], row["qubo_energy"], row["bitstring"]))
    return rows


def _sample_counts(sampler, ansatz, parameters: np.ndarray) -> dict[str, int]:
    circuit = ansatz.assign_parameters(np.asarray(parameters, dtype=float), inplace=False)
    circuit.measure_all()
    result = sampler.run([circuit]).result()[0]
    return result.data.meas.get_counts()


def _product_ansatz_expected_energy(qubo: dict, parameters: np.ndarray, num_qubits: int) -> float:
    """快速计算 RY product ansatz 对角 QUBO 的精确期望值。"""

    point = np.asarray(parameters, dtype=float).reshape(-1)
    if len(point) % num_qubits != 0:
        raise ValueError("参数数必须是 qubit 数的整数倍")
    reps = len(point) // num_qubits
    probabilities = np.zeros(num_qubits, dtype=float)
    for qubit in range(num_qubits):
        angle = sum(point[layer * num_qubits + qubit] for layer in range(reps))
        probabilities[qubit] = np.sin(angle / 2.0) ** 2

    energy = float(qubo.get("offset", 0.0))
    for term in qubo.get("linear", []):
        energy += float(term["coefficient"]) * probabilities[int(term["index"])]
    for term in qubo.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        energy += float(term["coefficient"]) * probabilities[i] * probabilities[j]
    return float(energy)


def _plot_convergence(trace: list[dict], reference: dict | None, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    if trace:
        ax.plot(
            [row["eval_count"] for row in trace],
            [row["mean_qubo_energy"] for row in trace],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color="#756bb1",
            label="VQE evaluations",
        )
    if reference is not None:
        ax.axhline(
            reference["best_feasible_qubo_energy"],
            color="#31a354",
            linestyle="--",
            linewidth=1.3,
            label="Classical feasible optimum",
        )
    ax.set_xlabel("function evaluation")
    ax.set_ylabel("QUBO energy")
    ax.set_title("SamplingVQE convergence")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_vrptw_image,
    requests=Resources(cpu="2", mem="4Gi"),
    limits=Resources(cpu="4", mem="8Gi"),
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
    """[HYBRID] 经典优化器更新参数，最终用本地 sampler 采样候选解。"""

    print_banner(TaskKind.HYBRID, "Task 06 / 运行 sampler-based VQE 混合优化")
    print(f"  optimizer = {optimizer}, maxiter = {maxiter}", flush=True)
    print(f"  sampler = StatevectorSampler(shots={shots}, seed={seed})", flush=True)
    print("  objective = exact product-state QUBO expectation; final candidates use sampler samples", flush=True)

    instance = json.loads(Path(instance_json.download()).read_text(encoding="utf-8"))
    reference = json.loads(Path(reference_json.download()).read_text(encoding="utf-8"))
    with open(hamiltonian_dill.download(), "rb") as f:
        hamiltonian_pkg = dill.load(f)
    with open(ansatz_dill.download(), "rb") as f:
        ansatz_pkg = dill.load(f)
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(initial_point_npy.download())

    trace: list[dict] = []
    sampler = create_sampler(shots=shots, seed=seed)
    optimizer_inst = _make_optimizer(optimizer, maxiter)

    def objective(parameters: np.ndarray) -> float:
        point = np.asarray(parameters, dtype=float).reshape(-1)
        energy = _product_ansatz_expected_energy(qubo, point, int(ansatz.num_qubits))
        eval_count = len(trace) + 1
        row = {
            "eval_count": int(eval_count),
            "mean_qubo_energy": float(np.real(energy)),
            "parameters": [float(x) for x in point],
        }
        trace.append(row)
        if eval_count == 1 or eval_count % 10 == 0:
            print(f"  eval {eval_count:03d}: mean QUBO energy = {row['mean_qubo_energy']:.6f}", flush=True)
        return energy

    t0 = time.perf_counter()
    result = optimizer_inst.minimize(fun=objective, x0=np.asarray(initial_point, dtype=float))
    optimizer_time = time.perf_counter() - t0

    optimal_point = getattr(result, "x", None)
    optimal_point_array = np.asarray(initial_point if optimal_point is None else optimal_point, dtype=float).reshape(-1)
    final_counts = _sample_counts(sampler, ansatz, optimal_point_array)
    final_shots = sum(final_counts.values())
    eigenstate = {str(k): float(v) / final_shots for k, v in final_counts.items()}
    top_samples = _summarize_eigenstate(instance, qubo, eigenstate)
    final_energy = float(np.real(result.fun))
    optimal_point_list = [float(x) for x in optimal_point_array]
    best_sample = min(top_samples, key=lambda row: (row["qubo_energy"], -row["probability"], row["bitstring"])) if top_samples else None

    payload = {
        "algorithm": "sampler-based VQE",
        "sampler": "StatevectorSampler",
        "objective_evaluator": "exact product-state QUBO expectation for diagonal Hamiltonian",
        "optimizer": optimizer,
        "maxiter": int(maxiter),
        "shots": int(shots),
        "seed": int(seed),
        "vqe_mean_qubo_energy": final_energy,
        "optimal_point": optimal_point_list,
        "num_parameters": int(ansatz.num_parameters),
        "num_evaluations_recorded": len(trace),
        "cost_function_evals": _as_float(getattr(result, "nfev", None)),
        "optimizer_time_seconds": optimizer_time,
        "raw_optimal_value": _as_float(getattr(result, "fun", None)),
        "best_measurement": None
        if best_sample is None
        else {
            "state": None,
            "bitstring": best_sample["bitstring"],
            "probability": best_sample["probability"],
            "value": best_sample["qubo_energy"],
        },
        "eigenstate": eigenstate,
        "top_samples": top_samples[:40],
        "classical_best_feasible_qubo_energy": reference["best_feasible_qubo_energy"],
        "energy_gap_to_classical_feasible": final_energy - reference["best_feasible_qubo_energy"],
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
