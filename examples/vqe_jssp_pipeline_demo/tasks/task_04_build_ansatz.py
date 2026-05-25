"""[CLASSICAL] Task 04 -- 构建用于 JSSP-QUBO 的参数化量子线路。

Ansatz 是带参数的量子线路模板。构建线路本身是经典工作，真正的采样和
能量估计会在后续 task 中发生。
"""

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
    p.add_argument("--hamiltonian-metadata", type=Path, required=True, help="task_02 输出的 hamiltonian.json")
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
    qc = QuantumCircuit(num_qubits, name="JSSP_RY_Product")
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


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 04 / 构建 JSSP VQE ansatz")
    metadata_from_hamiltonian = read_json(args.hamiltonian_metadata)
    num_qubits = int(metadata_from_hamiltonian["num_qubits"])
    if args.ansatz_reps < 1:
        raise ValueError("--ansatz-reps 必须 >= 1")

    with Timer("创建 RY product 参数化线路"):
        ansatz = _build_ansatz(num_qubits, args.ansatz_reps)
        rng = np.random.default_rng(args.seed)
        initial_point = rng.uniform(-0.2, 0.2, size=ansatz.num_parameters)

    metadata = {
        "ansatz": "RY product ansatz",
        "num_qubits": int(ansatz.num_qubits),
        "num_parameters": int(ansatz.num_parameters),
        "ansatz_reps": int(args.ansatz_reps),
        "initial_point_strategy": "seeded_uniform(-0.2, 0.2)",
        "seed": int(args.seed),
        "parameters": [str(p) for p in ansatz.parameters],
    }
    print(f"  qubit 数 = {metadata['num_qubits']}")
    print(f"  参数数 = {metadata['num_parameters']}, reps = {metadata['ansatz_reps']}")
    print("  初始参数 = 固定随机小角度；RY product ansatz 适合对角 QUBO 采样")

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
