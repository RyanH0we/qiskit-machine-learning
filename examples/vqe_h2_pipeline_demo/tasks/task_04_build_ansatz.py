"""[CLASSICAL] Task 04 -- 构建 Hartree-Fock 初态和 UCCSD ansatz。

Ansatz 是带参数的量子线路模板。这里使用量子化学中常见的 UCCSD，并把
Hartree-Fock 态作为线路初态。构建线路本身是经典工作，真正的量子求期望值
会在后续 task 中发生。
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import dill
import numpy as np
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", type=Path, required=True, help="task_02 输出的 problem.dill")
    p.add_argument("--output", type=Path, required=True, help="输出 ansatz.dill")
    p.add_argument("--metadata-output", type=Path, required=True, help="输出 ansatz.json")
    p.add_argument("--initial-point-output", type=Path, required=True, help="输出 initial_point.npy")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 ansatz 电路图 PNG")
    return p.parse_args()


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
    print_banner(TaskKind.CLASSICAL, "Task 04 / 构建 UCCSD ansatz")
    print(f"  problem = {args.problem}")

    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    with args.problem.open("rb") as f:
        pkg = dill.load(f)
    metadata_from_problem = pkg["metadata"]
    mapper = JordanWignerMapper()
    num_spatial_orbitals = int(metadata_from_problem["num_spatial_orbitals"])
    num_particles = tuple(int(x) for x in metadata_from_problem["num_particles"])

    with Timer("创建 Hartree-Fock 初态和 UCCSD 线路"):
        initial_state = HartreeFock(num_spatial_orbitals, num_particles, mapper)
        ansatz = UCCSD(
            num_spatial_orbitals,
            num_particles,
            mapper,
            initial_state=initial_state,
        )
        initial_point = np.zeros(ansatz.num_parameters)

    metadata = {
        "ansatz": "UCCSD",
        "initial_state": "HartreeFock",
        "num_qubits": int(ansatz.num_qubits),
        "num_parameters": int(ansatz.num_parameters),
        "num_spatial_orbitals": num_spatial_orbitals,
        "num_particles": [int(num_particles[0]), int(num_particles[1])],
        "initial_point": initial_point.tolist(),
        "parameters": [str(p) for p in ansatz.parameters],
    }
    print(f"  qubit 数 = {metadata['num_qubits']}")
    print(f"  参数数 = {metadata['num_parameters']}")
    print("  初始参数 = 全 0，表示从 Hartree-Fock 态开始")

    out = ensure_parent_dir(args.output)
    with out.open("wb") as f:
        dill.dump({"ansatz": ansatz, "initial_state": initial_state, "metadata": metadata}, f)
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
