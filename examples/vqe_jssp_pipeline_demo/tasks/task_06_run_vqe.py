"""[HYBRID] Task 06 -- 运行 SamplingVQE 优化 JSSP-QUBO。

VQE 是量子-经典混合算法：量子侧对参数化线路进行采样并估计 QUBO/Ising
能量，经典优化器根据能量结果更新参数，并反复迭代直到能量尽量低。
"""

from __future__ import annotations

import argparse
import csv
import sys
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
    decode_schedule,
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


def _write_trace(trace: list[dict], out_path: Path, num_parameters: int) -> None:
    out = ensure_parent_dir(out_path)
    fields = ["eval_count", "mean_qubo_energy"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            flat = {"eval_count": row["eval_count"], "mean_qubo_energy": row["mean_qubo_energy"]}
            for i, value in enumerate(row["parameters"]):
                flat[f"theta_{i}"] = value
            writer.writerow(flat)


def _plot_convergence(trace: list[dict], reference: dict | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
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
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.HYBRID, "Task 06 / 运行 SamplingVQE 混合优化")
    print(f"  optimizer = {args.optimizer}, maxiter = {args.maxiter}")
    print(f"  sampler = StatevectorSampler(shots={args.shots}, seed={args.seed})")
    instance = read_json(args.instance)

    from qiskit_algorithms import SamplingVQE

    with args.hamiltonian.open("rb") as f:
        hamiltonian_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    operator = hamiltonian_pkg["operator"]
    qubo = hamiltonian_pkg["qubo"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)

    trace: list[dict] = []

    def callback(eval_count: int, parameters: np.ndarray, energy: float, metadata: dict) -> None:  # noqa: ARG001
        row = {
            "eval_count": int(eval_count),
            "mean_qubo_energy": float(np.real(energy)),
            "parameters": [float(x) for x in np.asarray(parameters).reshape(-1)],
        }
        trace.append(row)
        if eval_count == 1 or eval_count % 10 == 0:
            print(f"  eval {eval_count:03d}: mean QUBO energy = {row['mean_qubo_energy']:.6f}")

    sampler = create_sampler(shots=args.shots, seed=args.seed)
    optimizer = _make_optimizer(args.optimizer, args.maxiter)
    vqe = SamplingVQE(
        sampler,
        ansatz,
        optimizer,
        initial_point=initial_point,
        callback=callback,
    )

    with Timer("SamplingVQE 量子-经典迭代"):
        result = vqe.compute_minimum_eigenvalue(operator)

    eigenstate = {str(k): float(v) for k, v in dict(result.eigenstate).items()}
    top_samples = _summarize_eigenstate(instance, qubo, eigenstate)
    final_energy = float(np.real(result.eigenvalue))
    optimal_point = getattr(result, "optimal_point", None)
    optimal_point_list = [] if optimal_point is None else [float(x) for x in np.asarray(optimal_point).reshape(-1)]
    reference = read_json(args.reference) if args.reference else None

    payload = {
        "algorithm": "SamplingVQE",
        "sampler": "StatevectorSampler",
        "optimizer": args.optimizer,
        "maxiter": int(args.maxiter),
        "shots": int(args.shots),
        "seed": int(args.seed),
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
    }
    if reference is not None:
        payload["classical_optimal_qubo_energy"] = reference["optimal_qubo_energy"]
        payload["energy_gap_to_classical_optimum"] = final_energy - reference["optimal_qubo_energy"]

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
