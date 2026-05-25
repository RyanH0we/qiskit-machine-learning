"""[CLASSICAL] Task 04 -- 构建 VRPTW-QUBO 使用的参数化量子线路。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, read_json, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hamiltonian", type=Path, required=True, help="task_02 输出的 hamiltonian.json")
    p.add_argument("--qubo", type=Path, default=None, help="可选：task_02 输出的 qubo.json，用于生成教学 warm start")
    p.add_argument("--output", type=Path, required=True, help="输出 ansatz.dill")
    p.add_argument("--metadata-output", type=Path, required=True, help="输出 ansatz.json")
    p.add_argument("--initial-point-output", type=Path, required=True, help="输出 initial_point.npy")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 ansatz 电路图 PNG")
    p.add_argument("--ansatz-reps", type=int, default=2, help="RY product ansatz 重复层数")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_ansatz(num_qubits: int, reps: int):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    theta = ParameterVector("theta", length=num_qubits * reps)
    qc = QuantumCircuit(num_qubits, name="VRPTW_RY_Product")
    cursor = 0
    for _layer in range(reps):
        for qubit in range(num_qubits):
            qc.ry(theta[cursor], qubit)
            cursor += 1
    return qc


def _draw_ansatz(ansatz, out_path: Path) -> bool:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    try:
        fig = ansatz.draw(output="mpl", style="iqp", fold=80)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 电路图绘制失败: {exc}")
        return False


def _warm_start_from_qubo(qubo: dict, num_qubits: int, reps: int) -> tuple[np.ndarray, dict]:
    """根据 QUBO 变量生成一个朴素路线 warm start。"""

    variables = qubo["variables"]
    target_bits = [0 for _ in range(num_qubits)]
    assigned_customers: set[str] = set()
    x_vars = [v for v in variables if v["kind"] == "x"]
    y_by_slot = {(int(v["vehicle"]), int(v["position"])): v for v in variables if v["kind"] == "y"}

    for vehicle, position in sorted(y_by_slot):
        candidates = [
            v
            for v in x_vars
            if int(v["vehicle"]) == vehicle
            and int(v["position"]) == position
            and str(v["customer_id"]) not in assigned_customers
        ]
        if not candidates:
            continue
        chosen = sorted(candidates, key=lambda v: (str(v["customer_id"]), float(v["start_time"])))[0]
        target_bits[int(chosen["index"])] = 1
        target_bits[int(y_by_slot[(vehicle, position)]["index"])] = 1
        assigned_customers.add(str(chosen["customer_id"]))

    p_one = 0.92
    p_zero = 0.03
    point = np.zeros(num_qubits * reps, dtype=float)
    for qubit, bit in enumerate(target_bits):
        probability = p_one if bit else p_zero
        total_angle = 2.0 * np.arcsin(np.sqrt(probability))
        for layer in range(reps):
            point[layer * num_qubits + qubit] = total_angle / reps
    metadata = {
        "strategy": "greedy_route_warm_start_from_qubo_variables",
        "target_bits": target_bits,
        "target_one_probability": p_one,
        "target_zero_probability": p_zero,
        "assigned_customers": sorted(assigned_customers),
    }
    return point, metadata


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 04 / 构建 VRPTW VQE ansatz")
    hamiltonian = read_json(args.hamiltonian)
    num_qubits = int(hamiltonian["num_qubits"])
    if args.ansatz_reps < 1:
        raise ValueError("--ansatz-reps 必须 >= 1")

    with Timer("创建 RY product 参数化线路"):
        ansatz = _build_ansatz(num_qubits, args.ansatz_reps)
        if args.qubo is None:
            rng = np.random.default_rng(args.seed)
            initial_point = rng.uniform(-0.25, 0.25, size=ansatz.num_parameters)
            warm_start = {"strategy": "seeded_uniform(-0.25, 0.25)"}
        else:
            qubo = read_json(args.qubo)
            initial_point, warm_start = _warm_start_from_qubo(qubo, num_qubits, args.ansatz_reps)

    metadata = {
        "ansatz": "RY product ansatz",
        "num_qubits": int(ansatz.num_qubits),
        "num_parameters": int(ansatz.num_parameters),
        "ansatz_reps": int(args.ansatz_reps),
        "initial_point_strategy": warm_start["strategy"],
        "warm_start": warm_start,
        "seed": int(args.seed),
        "parameters": [str(p) for p in ansatz.parameters],
        "why_product_ansatz": "VRPTW-QUBO Hamiltonian 是对角的，最终解是计算基态 bitstring；产品态 ansatz 易于教学和本地快速验证。",
    }
    print(f"  qubit 数 = {metadata['num_qubits']}")
    print(f"  参数数 = {metadata['num_parameters']}, reps = {metadata['ansatz_reps']}")

    out = ensure_parent_dir(args.output)
    with out.open("wb") as f:
        dill.dump({"ansatz": ansatz, "metadata": metadata}, f)
    print(f"  -> 写入 {out}")

    np_out = ensure_parent_dir(args.initial_point_output)
    np.save(np_out, initial_point)
    print(f"  -> 写入 {np_out}")

    json_out = write_json(args.metadata_output, metadata)
    print(f"  -> 写入 {json_out}")

    with Timer("绘制 ansatz 电路"):
        ok = _draw_ansatz(ansatz, args.figure_output)
    if ok:
        print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
