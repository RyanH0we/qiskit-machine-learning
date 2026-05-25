"""[HYBRID] Task 06 -- 运行 SamplingVQE 优化 VRPTW-QUBO。"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    bitstring_to_bits,
    create_sampler,
    decode_vrptw_solution,
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
    p.add_argument("--reference", type=Path, default=None, help="可选：task_03 输出的 reference.json")
    p.add_argument("--output-json", type=Path, required=True, help="输出 vqe_result.json")
    p.add_argument("--output-dill", type=Path, required=True, help="输出 vqe_result.dill")
    p.add_argument("--trace-output", type=Path, required=True, help="输出 vqe_trace.csv")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 VQE 收敛曲线 PNG")
    p.add_argument("--optimizer", default="COBYLA", choices=["COBYLA", "SLSQP", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=120, help="经典优化器最大迭代次数")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    return {
        "state": int(best["state"]) if "state" in best else None,
        "bitstring": str(best.get("bitstring")),
        "probability": float(best.get("probability", 0.0)),
        "value": _as_float(best.get("value")),
    }


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


def _mean_energy_from_counts(qubo: dict, counts: dict[str, int]) -> float:
    shots = sum(counts.values())
    mean = 0.0
    for bitstring, count in counts.items():
        bits = bitstring_to_bits(str(bitstring), int(qubo["num_variables"]))
        mean += (count / shots) * qubo_energy(qubo, bits)
    return float(mean)


def _product_ansatz_expected_energy(qubo: dict, parameters: np.ndarray, num_qubits: int) -> float:
    """快速计算 RY product ansatz 对角 QUBO 的精确期望值。

    本示例的 ansatz 没有纠缠，Hamiltonian 只含 Z/ZZ 项，所以每个 qubit 的
    取 1 概率可以直接由 RY 总角度得到。这和用本地 statevector 对同一线路
    计算对角 QUBO 期望值等价，但快很多，适合教学流水线。
    """

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


def _write_trace(trace: list[dict], out_path: Path, num_parameters: int) -> None:
    out = ensure_parent_dir(out_path)
    fields = ["eval_count", "mean_qubo_energy"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            if len(row["parameters"]) != num_parameters:
                raise ValueError("trace 参数长度和 ansatz 参数数不一致")
            flat = {"eval_count": row["eval_count"], "mean_qubo_energy": row["mean_qubo_energy"]}
            for i, value in enumerate(row["parameters"]):
                flat[f"theta_{i}"] = value
            writer.writerow(flat)


def _plot_convergence(trace: list[dict], reference: dict | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
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
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.HYBRID, "Task 06 / 运行 sampler-based VQE 混合优化")
    print(f"  optimizer = {args.optimizer}, maxiter = {args.maxiter}")
    print(f"  sampler = StatevectorSampler(shots={args.shots}, seed={args.seed})")
    print("  objective = exact product-state QUBO expectation; final candidates use sampler samples")

    instance = read_json(args.instance)
    with args.hamiltonian.open("rb") as f:
        hamiltonian_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)
    reference = read_json(args.reference) if args.reference else None

    trace: list[dict] = []

    sampler = create_sampler(shots=args.shots, seed=args.seed)
    optimizer = _make_optimizer(args.optimizer, args.maxiter)

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
            print(f"  eval {eval_count:03d}: mean QUBO energy = {row['mean_qubo_energy']:.6f}")
        return energy

    with Timer("VQE 量子-经典迭代"):
        t0 = time.perf_counter()
        result = optimizer.minimize(fun=objective, x0=np.asarray(initial_point, dtype=float))
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
        "optimizer": args.optimizer,
        "maxiter": int(args.maxiter),
        "shots": int(args.shots),
        "seed": int(args.seed),
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
    }
    if reference is not None:
        payload["classical_best_feasible_qubo_energy"] = reference["best_feasible_qubo_energy"]
        payload["energy_gap_to_classical_feasible"] = final_energy - reference["best_feasible_qubo_energy"]

    print(f"  VQE 最终均值 QUBO energy = {final_energy:.6f}")
    if payload["best_measurement"] is not None:
        best = payload["best_measurement"]
        print(f"  最佳测量 bitstring = {best['bitstring']}, energy = {best['value']}")

    json_out = write_json(args.output_json, payload)
    print(f"  -> 写入 {json_out}")

    dill_out = ensure_parent_dir(args.output_dill)
    with dill_out.open("wb") as f:
        dill.dump({"result": result, "payload": payload, "trace": trace}, f)
    print(f"  -> 写入 {dill_out}")

    _write_trace(trace, args.trace_output, int(ansatz.num_parameters))
    print(f"  -> 写入 {args.trace_output}")

    with Timer("绘制 VQE 收敛曲线"):
        _plot_convergence(trace, reference, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
