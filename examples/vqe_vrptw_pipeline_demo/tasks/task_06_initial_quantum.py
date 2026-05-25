"""[QUANTUM] Task 06 -- 用本地量子模拟器评估初始 ansatz。

这是流水线第一次真正调用 Qiskit Estimator primitive。它计算
<psi(theta)|H|psi(theta)>，并把初始状态下各条路线的概率画出来。
"""

from __future__ import annotations

import argparse
import sys
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
    p.add_argument("--output", type=Path, required=True, help="输出 initial_quantum.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出初始概率图 PNG")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _extract_estimator_value(pub_result) -> tuple[float, float]:
    evs = np.asarray(pub_result.data.evs).reshape(-1)
    stds = np.asarray(pub_result.data.stds).reshape(-1)
    ev = float(np.real(evs[0]))
    std = float(np.real(stds[0])) if len(stds) else 0.0
    return ev, std


def _plot_initial_probabilities(payload: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    out = ensure_parent_dir(out_path)
    route_probs = payload["probability_summary"]["route_probabilities"]
    labels = [row["route_id"] + "\n" + row["route_label"] for row in route_probs]
    probs = [row["probability"] for row in route_probs]

    top_states = payload["probability_summary"]["top_states"][:10]
    state_labels = [row["bitstring"] for row in top_states]
    state_probs = [row["probability"] for row in top_states]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_route, ax_state = axes

    x = np.arange(len(route_probs))
    ax_route.bar(x, probs, color="#9ecae1", edgecolor="black", linewidth=0.6)
    ax_route.set_xticks(x)
    ax_route.set_xticklabels(labels, fontsize=8)
    ax_route.set_ylabel("probability")
    ax_route.set_title("One-hot route probabilities")
    ax_route.set_ylim(0, max(probs + [0.1]) * 1.25)
    ax_route.grid(True, axis="y", alpha=0.25)

    y = np.arange(len(top_states))
    ax_state.barh(y, state_probs, color="#756bb1", edgecolor="black", linewidth=0.5)
    ax_state.set_yticks(y)
    ax_state.set_yticklabels(state_labels, fontsize=8)
    ax_state.invert_yaxis()
    ax_state.set_xlabel("probability")
    ax_state.set_title("Top computational states")
    ax_state.grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"Task 06: initial quantum expectation = {payload['initial_energy']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 06 / 本地模拟器评估初始状态")
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

    estimator = create_estimator(seed=args.seed)
    with Timer("Estimator 计算初始 QUBO 期望值"):
        result = estimator.run([(ansatz, operator, initial_point)]).result()
    initial_energy, std = _extract_estimator_value(result[0])

    with Timer("从 statevector 提取初始概率分布"):
        probability_summary = summarize_state_probabilities(ansatz, initial_point, qubo, routes)

    payload = {
        "estimator": "StatevectorEstimator",
        "seed": args.seed,
        "initial_energy": initial_energy,
        "estimator_std": std,
        "initial_point": initial_point.tolist(),
        "probability_summary": probability_summary,
    }
    print(f"  初始期望值 = {initial_energy:.6f}")
    print(f"  初始 invalid probability = {probability_summary['invalid_probability']:.3f}")

    out = write_json(args.output, payload)
    print(f"  -> 写入 {out}")

    with Timer("绘制初始概率分布"):
        _plot_initial_probabilities(payload, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
