"""[HYBRID] Task 06 -- 运行 VQE 优化。

VQE 是量子-经典混合算法：量子侧负责对参数化线路测能量期望值，经典优化器
根据能量结果更新参数，并反复迭代直到能量尽量低。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import dill
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    create_estimator,
    ensure_parent_dir,
    print_banner,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", type=Path, required=True, help="task_02 输出的 problem.dill")
    p.add_argument("--ansatz", type=Path, required=True, help="task_04 输出的 ansatz.dill")
    p.add_argument("--initial-point", type=Path, required=True, help="task_04 输出的 initial_point.npy")
    p.add_argument("--reference", type=Path, default=None, help="可选：task_03 输出的 reference.json")
    p.add_argument("--output-json", type=Path, required=True, help="输出 vqe_result.json")
    p.add_argument("--output-dill", type=Path, required=True, help="输出 vqe_result.dill")
    p.add_argument("--trace-output", type=Path, required=True, help="输出 vqe_trace.csv")
    p.add_argument("--figure-output", type=Path, required=True, help="输出收敛曲线 PNG")
    p.add_argument("--optimizer", default="SLSQP", choices=["SLSQP", "COBYLA", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=100, help="经典优化器最大迭代次数")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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
    out = ensure_parent_dir(out_path)
    fields = ["eval_count", "electronic_energy_hartree", "total_energy_hartree"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out.open("w", newline="", encoding="utf-8") as f:
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


def _plot_convergence(trace: list[dict], final_energy: float, reference: dict | None, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
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
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.HYBRID, "Task 06 / 运行 VQE 混合优化")
    print(f"  optimizer = {args.optimizer}, maxiter = {args.maxiter}")
    print(f"  estimator = StatevectorEstimator(seed={args.seed})")

    with args.problem.open("rb") as f:
        problem_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    qubit_operator = problem_pkg["qubit_operator"]
    nuclear = float(problem_pkg["metadata"]["nuclear_repulsion_energy"])
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)

    trace: list[dict] = []

    estimator = create_estimator(seed=args.seed)
    optimizer = _make_optimizer(args.optimizer, args.maxiter)

    def objective(parameters: np.ndarray) -> float:
        point = np.asarray(parameters, dtype=float).reshape(-1)
        result = estimator.run([(ansatz, qubit_operator, point)]).result()
        electronic = _extract_estimator_value(result[0])
        flat_parameters = np.asarray(parameters, dtype=float).reshape(-1)
        eval_count = len(trace) + 1
        trace.append(
            {
                "eval_count": eval_count,
                "electronic_energy_hartree": electronic,
                "total_energy_hartree": electronic + nuclear,
                "parameters": [float(x) for x in flat_parameters],
            }
        )
        if eval_count == 1 or eval_count % 5 == 0:
            print(f"  eval {eval_count:03d}: total energy = {electronic + nuclear:.12f} Hartree")
        return electronic

    with Timer("VQE 量子-经典迭代"):
        t0 = time.perf_counter()
        result = optimizer.minimize(fun=objective, x0=np.asarray(initial_point, dtype=float))
        optimizer_time = time.perf_counter() - t0

    final_electronic = float(np.real(result.fun))
    final_total = final_electronic + nuclear
    optimal_point = getattr(result, "x", None)
    if optimal_point is None:
        optimal_point_list: list[float] = []
    else:
        optimal_point_list = [float(x) for x in np.asarray(optimal_point).reshape(-1)]

    payload = {
        "algorithm": "VQE",
        "estimator": "StatevectorEstimator",
        "optimizer": args.optimizer,
        "maxiter": args.maxiter,
        "seed": args.seed,
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
    if args.reference:
        reference = read_json(args.reference)
        payload["exact_total_energy_hartree"] = reference["exact_total_energy_hartree"]
        payload["absolute_error_hartree"] = abs(final_total - reference["exact_total_energy_hartree"])
    else:
        reference = None

    print(f"  VQE 最终总能量 = {final_total:.12f} Hartree")
    if "absolute_error_hartree" in payload:
        print(f"  与精确参考误差 = {payload['absolute_error_hartree']:.6e} Hartree")

    json_out = write_json(args.output_json, payload)
    print(f"  -> 写入 {json_out}")

    dill_out = ensure_parent_dir(args.output_dill)
    with dill_out.open("wb") as f:
        dill.dump({"result": result, "payload": payload, "trace": trace}, f)
    print(f"  -> 写入 {dill_out}")

    _write_trace(trace, args.trace_output, int(ansatz.num_parameters))
    print(f"  -> 写入 {args.trace_output}")

    with Timer("绘制 VQE 收敛曲线"):
        _plot_convergence(trace, final_total, reference, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
