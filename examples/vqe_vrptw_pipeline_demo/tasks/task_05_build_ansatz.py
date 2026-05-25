"""[CLASSICAL] Task 05 -- 构建 VQE 使用的参数化量子线路。

构建线路本身是经典工作。真正的量子期望值计算会在后续 task 中发生。
本示例默认使用无纠缠 RealAmplitudes，因为路线选择 QUBO 的最优解是一个
计算基态 bitstring，产品态已经足够表达，COBYLA 也更容易收敛。
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
    p.add_argument("--hamiltonian", type=Path, required=True, help="task_03 输出的 hamiltonian.json")
    p.add_argument("--output", type=Path, required=True, help="输出 ansatz.dill")
    p.add_argument("--metadata-output", type=Path, required=True, help="输出 ansatz.json")
    p.add_argument("--initial-point-output", type=Path, required=True, help="输出 initial_point.npy")
    p.add_argument("--figure-output", type=Path, required=True, help="输出线路图 PNG")
    p.add_argument("--reps", type=int, default=2, help="RealAmplitudes 层数")
    return p.parse_args()


def _build_ansatz(num_qubits: int, reps: int):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import real_amplitudes

    circuit = QuantumCircuit(num_qubits, name="H+RealAmplitudes")
    circuit.h(range(num_qubits))
    circuit.compose(
        real_amplitudes(num_qubits, reps=reps, entanglement=[]),
        inplace=True,
    )
    return circuit


def _draw_ansatz(ansatz, out_path: Path) -> bool:
    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    try:
        fig = ansatz.decompose().draw(output="mpl", style="iqp", fold=-1)
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 电路图绘制失败: {exc}")
        return False


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 05 / 构建 H + RealAmplitudes ansatz")
    hamiltonian = read_json(args.hamiltonian)
    num_qubits = int(hamiltonian["num_qubits"])

    with Timer("创建参数化量子线路"):
        ansatz = _build_ansatz(num_qubits, args.reps)
        initial_point = np.zeros(ansatz.num_parameters)

    metadata = {
        "ansatz": "Hadamard initial layer + RealAmplitudes",
        "num_qubits": num_qubits,
        "num_parameters": int(ansatz.num_parameters),
        "reps": args.reps,
        "entanglement": "none",
        "initial_point": initial_point.tolist(),
        "parameters": [str(p) for p in ansatz.parameters],
        "why_hadamards": "初始态接近所有路线均匀叠加，方便新手观察优化前后的概率变化。",
        "why_no_entanglement": "路线选择 QUBO 的精确最优解是一个计算基态，产品态 ansatz 足够表达且更利于教学默认参数收敛。",
    }
    print(f"  qubit 数 = {num_qubits}")
    print(f"  参数数 = {metadata['num_parameters']}")
    print("  初始参数 = 全 0")

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
