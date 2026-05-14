"""[CLASSICAL] Task 02 -- 构造 H2 电子结构问题并映射到 qubit Hamiltonian。

这一步使用 PySCF 在经典计算机上计算分子积分，然后用 Jordan-Wigner 映射把
电子的费米子 Hamiltonian 转成量子比特上的 Pauli 算符。它仍然是经典预处理。
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
    complex_to_record,
    ensure_parent_dir,
    print_banner,
    read_json,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--molecule", type=Path, required=True, help="task_01 输出的 molecule.json")
    p.add_argument("--problem-output", type=Path, required=True, help="输出 problem.dill")
    p.add_argument("--hamiltonian-output", type=Path, required=True, help="输出 hamiltonian.json")
    p.add_argument("--figure-output", type=Path, required=True, help="输出 Pauli 系数图 PNG")
    return p.parse_args()


def _plot_pauli_terms(terms: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [t["pauli"] for t in terms]
    coeffs = [t["coefficient"]["real"] for t in terms]
    colors = ["#2ca25f" if c >= 0 else "#de2d26" for c in coeffs]

    out = ensure_parent_dir(out_path)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(labels)), coeffs, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient (Hartree)")
    ax.set_title("Qubit Hamiltonian Pauli terms")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 02 / 构造并映射 Hamiltonian")
    molecule = read_json(args.molecule)
    print(f"  输入分子 = {molecule['atom']}")
    print(f"  基组 = {molecule['basis']}, 映射器 = Jordan-Wigner")

    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    with Timer("PySCF 电子结构计算"):
        driver = PySCFDriver(
            atom=molecule["atom"],
            charge=int(molecule["charge"]),
            spin=int(molecule["spin"]),
            basis=molecule["basis"],
        )
        problem = driver.run()

    mapper = JordanWignerMapper()
    with Timer("Jordan-Wigner 映射到量子比特"):
        electronic_hamiltonian = problem.second_q_ops()[0]
        qubit_operator = mapper.map(electronic_hamiltonian)

    labels = qubit_operator.paulis.to_labels()
    coeffs = qubit_operator.coeffs
    order = np.argsort(-np.abs(coeffs))
    terms = [
        {
            "pauli": labels[i],
            "coefficient": complex_to_record(complex(coeffs[i])),
            "abs_coefficient": float(abs(coeffs[i])),
        }
        for i in order
    ]

    metadata = {
        "molecule": molecule,
        "mapper": "JordanWignerMapper",
        "num_spatial_orbitals": int(problem.num_spatial_orbitals),
        "num_spin_orbitals": int(problem.num_spin_orbitals),
        "num_particles": [int(problem.num_particles[0]), int(problem.num_particles[1])],
        "num_qubits": int(qubit_operator.num_qubits),
        "num_pauli_terms": int(len(qubit_operator)),
        "nuclear_repulsion_energy": float(problem.nuclear_repulsion_energy),
        "hartree_fock_total_energy": float(problem.reference_energy),
        "pauli_terms": terms,
    }

    print(f"  空间轨道数 = {metadata['num_spatial_orbitals']}")
    print(f"  电子数(alpha, beta) = {tuple(metadata['num_particles'])}")
    print(f"  qubit 数 = {metadata['num_qubits']}, Pauli 项数 = {metadata['num_pauli_terms']}")
    print(f"  核间排斥能 = {metadata['nuclear_repulsion_energy']:.12f} Hartree")
    print(f"  Hartree-Fock 总能量 = {metadata['hartree_fock_total_energy']:.12f} Hartree")

    problem_out = ensure_parent_dir(args.problem_output)
    with problem_out.open("wb") as f:
        dill.dump(
            {
                "qubit_operator": qubit_operator,
                "metadata": metadata,
            },
            f,
        )
    print(f"  -> 写入 {problem_out}")

    json_out = write_json(args.hamiltonian_output, metadata)
    print(f"  -> 写入 {json_out}")

    with Timer("绘制 Pauli 系数图"):
        _plot_pauli_terms(terms, args.figure_output)
    print(f"  -> 写入 {args.figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
