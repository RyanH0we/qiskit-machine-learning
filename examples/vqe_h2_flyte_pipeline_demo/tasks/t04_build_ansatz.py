"""[CLASSICAL] Task 04 -- 构造 Hartree-Fock 初态与 UCCSD ansatz。

Ansatz 是带参数的量子线路模板。这里使用量子化学中常用的 UCCSD，并把 Hartree-
Fock 态作为线路的起点。构造线路本身仍是经典工作，真正的量子求期望值在 ``t05``
和 ``t06`` 中发生。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import TaskKind, print_banner, task_workdir, vqe_image


class AnsatzOut(NamedTuple):
    ansatz_dill: FlyteFile
    ansatz_json: FlyteFile
    initial_point_npy: FlyteFile
    ansatz_png: FlyteFile


def _draw_ansatz(ansatz, out_path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig = ansatz.decompose().draw(output="mpl", style="iqp", fold=-1)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 电路图绘制失败: {exc}", flush=True)
        return False


@task(
    container_image=vqe_image,
    requests=Resources(cpu="500m", mem="1Gi"),
    cache=True,
    cache_version="v1",
    retries=1,
)
def t04_build_ansatz(problem_dill: FlyteFile) -> AnsatzOut:
    """[CLASSICAL] 构造 HartreeFock + UCCSD。"""

    import warnings

    from scipy.sparse import SparseEfficiencyWarning

    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    print_banner(TaskKind.CLASSICAL, "Task 04 / 构建 UCCSD ansatz")

    with open(problem_dill.download(), "rb") as f:
        pkg = dill.load(f)
    metadata_from_problem = pkg["metadata"]
    mapper = JordanWignerMapper()
    num_spatial_orbitals = int(metadata_from_problem["num_spatial_orbitals"])
    num_particles = tuple(int(x) for x in metadata_from_problem["num_particles"])

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
    print(f"  qubit 数 = {metadata['num_qubits']}", flush=True)
    print(f"  参数数 = {metadata['num_parameters']}", flush=True)
    print("  初始参数 = 全 0，从 Hartree-Fock 态开始", flush=True)

    work = task_workdir("t04")
    dill_path = work / "ansatz.dill"
    with dill_path.open("wb") as f:
        dill.dump({"ansatz": ansatz, "initial_state": initial_state, "metadata": metadata}, f)

    json_path = work / "ansatz.json"
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    npy_path = work / "initial_point.npy"
    np.save(npy_path, initial_point)

    png_path = work / "04_ansatz_circuit.png"
    _draw_ansatz(ansatz, png_path)

    print(f"  -> 写入 {dill_path}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {npy_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)

    return AnsatzOut(
        ansatz_dill=FlyteFile(str(dill_path)),
        ansatz_json=FlyteFile(str(json_path)),
        initial_point_npy=FlyteFile(str(npy_path)),
        ansatz_png=FlyteFile(str(png_path)),
    )
