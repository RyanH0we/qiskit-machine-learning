"""[CLASSICAL] Task 02 -- 构造 H2 电子结构问题并映射为 qubit Hamiltonian。

调用 PySCF 做电子结构积分，再用 Jordan-Wigner 映射成量子比特上的 Pauli 算符。
仍然是经典预处理。``problem.dill`` 保留了完整的 ``ElectronicStructureProblem``
对象供 ``t04`` 用来构造 UCCSD ansatz；``hamiltonian.json`` 是给人看的元数据。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, complex_to_record, print_banner, task_workdir, vqe_image


class HamiltonianOut(NamedTuple):
    problem_dill: FlyteFile
    hamiltonian_json: FlyteFile
    hamiltonian_png: FlyteFile


def _plot_pauli_terms(terms: list[dict], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [t["pauli"] for t in terms]
    coeffs = [t["coefficient"]["real"] for t in terms]
    colors = ["#2ca25f" if c >= 0 else "#de2d26" for c in coeffs]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(np.arange(len(labels)), coeffs, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Coefficient (Hartree)")
    ax.set_title("Qubit Hamiltonian Pauli terms")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


@task(
    container_image=vqe_image,
    requests=Resources(cpu="1", mem="2Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t02_build_hamiltonian(molecule_json: FlyteFile) -> HamiltonianOut:
    """[CLASSICAL] PySCF 电子结构 + Jordan-Wigner 映射。"""

    import warnings

    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    print_banner(TaskKind.CLASSICAL, "Task 02 / 构造并映射 Hamiltonian")
    molecule = json.loads(Path(molecule_json.download()).read_text(encoding="utf-8"))
    print(f"  输入分子 = {molecule['atom']}", flush=True)
    print(f"  基组 = {molecule['basis']}, 映射器 = Jordan-Wigner", flush=True)

    driver = PySCFDriver(
        atom=molecule["atom"],
        charge=int(molecule["charge"]),
        spin=int(molecule["spin"]),
        basis=molecule["basis"],
    )
    problem = driver.run()

    mapper = JordanWignerMapper()
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

    print(f"  空间轨道数 = {metadata['num_spatial_orbitals']}", flush=True)
    print(f"  电子数(alpha, beta) = {tuple(metadata['num_particles'])}", flush=True)
    print(f"  qubit 数 = {metadata['num_qubits']}, Pauli 项数 = {metadata['num_pauli_terms']}", flush=True)
    print(f"  核间排斥能 = {metadata['nuclear_repulsion_energy']:.12f} Hartree", flush=True)
    print(f"  Hartree-Fock 总能量 = {metadata['hartree_fock_total_energy']:.12f} Hartree", flush=True)

    work = task_workdir("t02")
    problem_path = work / "problem.dill"
    with problem_path.open("wb") as f:
        dill.dump({"qubit_operator": qubit_operator, "metadata": metadata}, f)

    json_path = work / "hamiltonian.json"
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "02_hamiltonian_terms.png"
    _plot_pauli_terms(terms, png_path)

    print(f"  -> 写入 {problem_path}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return HamiltonianOut(
        problem_dill=FlyteFile(str(problem_path)),
        hamiltonian_json=FlyteFile(str(json_path)),
        hamiltonian_png=FlyteFile(str(png_path)),
    )
