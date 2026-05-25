"""[HYBRID] Task 07 -- 运行 VQE 优化。

VQE 是量子-经典混合算法：量子侧计算当前参数下的 Hamiltonian 期望值，经典
优化器根据期望值更新参数，并反复迭代。
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import (
    TaskKind,
    Timer,
    create_estimator,
    ensure_parent_dir,
    print_banner,
    read_json,
    summarize_state_probabilities,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hamiltonian-dill", type=Path, required=True, help="task_03 输出的 hamiltonian.dill")
    p.add_argument("--qubo", type=Path, required=True, help="task_03 输出的 qubo.json")
    p.add_argument("--routes", type=Path, required=True, help="task_02 输出的 routes.json")
    p.add_argument("--ansatz", type=Path, required=True, help="task_05 输出的 ansatz.dill")
    p.add_argument("--initial-point", type=Path, required=True, help="task_05 输出的 initial_point.npy")
    p.add_argument("--reference", type=Path, required=True, help="task_04 输出的 reference.json")
    p.add_argument("--output-json", type=Path, required=True, help="输出 vqe_result.json")
    p.add_argument("--output-dill", type=Path, required=True, help="输出 vqe_result.dill")
    p.add_argument("--trace-output", type=Path, required=True, help="输出 vqe_trace.csv")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 VQE 收敛图 PNG")
    p.add_argument("--optimizer", default="COBYLA", choices=["COBYLA", "SLSQP", "L_BFGS_B"])
    p.add_argument("--maxiter", type=int, default=120, help="经典优化器最大迭代次数")
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
    fields = ["eval_count", "energy"]
    fields += [f"theta_{i}" for i in range(num_parameters)]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trace:
            flat = {"eval_count": row["eval_count"], "energy": row["energy"]}
            if len(row["parameters"]) != num_parameters:
                raise ValueError(
                    f"trace 参数长度 {len(row['parameters'])} 与 ansatz 参数数 {num_parameters} 不一致"
                )
            for i, value in enumerate(row["parameters"]):
                flat[f"theta_{i}"] = value
            writer.writerow(flat)


def _plot_vqe(payload: dict, reference: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    trace = payload["trace"]
    route_probs = payload["probability_summary"]["route_probabilities"]
    exact = reference["best"]["energy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax_conv, ax_prob = axes

    if trace:
        ax_conv.plot(
            [row["eval_count"] for row in trace],
            [row["energy"] for row in trace],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color="#756bb1",
        )
    ax_conv.axhline(exact, color="#31a354", linestyle="--", linewidth=1.2, label="Exact best")
    ax_conv.set_xlabel("function evaluation")
    ax_conv.set_ylabel("QUBO / Hamiltonian energy")
    ax_conv.set_title("Hybrid VQE convergence")
    ax_conv.grid(True, alpha=0.25)
    ax_conv.legend(fontsize=9)

    labels = [row["route_id"] + "\n" + row["route_label"] for row in route_probs]
    probs = [row["probability"] for row in route_probs]
    colors = ["#de2d26" if row["route_id"] == reference["best"]["selected_route_id"] else "#9ecae1" for row in route_probs]
    x = np.arange(len(route_probs))
    ax_prob.bar(x, probs, color=colors, edgecolor="black", linewidth=0.6)
    ax_prob.set_xticks(x)
    ax_prob.set_xticklabels(labels, fontsize=8)
    ax_prob.set_ylabel("probability")
    ax_prob.set_title("Final one-hot route probabilities")
    ax_prob.set_ylim(0, max(probs + [0.1]) * 1.25)
    ax_prob.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"Task 07: VQE final energy = {payload['vqe_energy']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.HYBRID, "Task 07 / 运行 VQE 混合优化")
    print(f"  optimizer = {args.optimizer}, maxiter = {args.maxiter}")
    print(f"  estimator = StatevectorEstimator(seed={args.seed})")

    with args.hamiltonian_dill.open("rb") as f:
        hamiltonian_pkg = dill.load(f)
    with args.ansatz.open("rb") as f:
        ansatz_pkg = dill.load(f)
    operator = hamiltonian_pkg["operator"]
    ansatz = ansatz_pkg["ansatz"]
    initial_point = np.load(args.initial_point)
    qubo = read_json(args.qubo)
    routes = read_json(args.routes)["routes"]
    reference = read_json(args.reference)

    trace: list[dict] = []

    estimator = create_estimator(seed=args.seed)
    optimizer = _make_optimizer(args.optimizer, args.maxiter)

    def objective(parameters: np.ndarray) -> float:
        point = np.asarray(parameters, dtype=float).reshape(-1)
        result = estimator.run([(ansatz, operator, point)]).result()
        value = _extract_estimator_value(result[0])
        eval_count = len(trace) + 1
        trace.append(
            {
                "eval_count": eval_count,
                "energy": value,
                "parameters": [float(x) for x in point],
            }
        )
        if eval_count == 1 or eval_count % 10 == 0:
            print(f"  eval {eval_count:03d}: energy = {value:.6f}")
        return value

    with Timer("VQE 量子-经典迭代"):
        t0 = time.perf_counter()
        result = optimizer.minimize(fun=objective, x0=np.asarray(initial_point, dtype=float))
        optimizer_time = time.perf_counter() - t0

    final_energy = float(np.real(result.fun))
    optimal_point = getattr(result, "x", None)
    if optimal_point is None:
        optimal_point_array = np.asarray(initial_point, dtype=float)
    else:
        optimal_point_array = np.asarray(optimal_point, dtype=float).reshape(-1)

    with Timer("提取 VQE 最终概率分布"):
        probability_summary = summarize_state_probabilities(ansatz, optimal_point_array, qubo, routes)

    payload = {
        "algorithm": "VQE",
        "estimator": "StatevectorEstimator",
        "optimizer": args.optimizer,
        "maxiter": args.maxiter,
        "seed": args.seed,
        "vqe_energy": final_energy,
        "exact_best_energy": reference["best"]["energy"],
        "energy_gap_to_exact": final_energy - reference["best"]["energy"],
        "optimal_point": [float(x) for x in optimal_point_array],
        "num_parameters": int(ansatz.num_parameters),
        "num_evaluations_recorded": len(trace),
        "cost_function_evals": _as_float(getattr(result, "nfev", None)),
        "optimizer_time_seconds": optimizer_time,
        "raw_optimal_value": _as_float(getattr(result, "fun", None)),
        "probability_summary": probability_summary,
        "trace": trace,
    }
    best_route = probability_summary["best_probability_route"]
    print(f"  VQE 最终能量 = {final_energy:.6f}")
    if best_route is not None:
        print(
            "  最高概率 one-hot 路线 = "
            f"{best_route['route_id']} / {best_route['route_label']} / p={best_route['probability']:.3f}"
        )
    print(f"  与精确最优能量差 = {payload['energy_gap_to_exact']:.6f}")

    json_out = write_json(args.output_json, payload)
    print(f"  -> 写入 {json_out}")

    dill_out = ensure_parent_dir(args.output_dill)
    with dill_out.open("wb") as f:
        dill.dump({"result": result, "payload": payload, "trace": trace}, f)
    print(f"  -> 写入 {dill_out}")

    _write_trace(trace, args.trace_output, int(ansatz.num_parameters))
    print(f"  -> 写入 {args.trace_output}")

    with Timer("绘制 VQE 收敛曲线和路线概率"):
        _plot_vqe(payload, reference, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
