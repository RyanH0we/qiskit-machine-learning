"""[CLASSICAL] Task 04 -- 构建用于 JSSP-QUBO 的参数化量子线路。"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, print_banner, task_workdir, vqe_jssp_image


class AnsatzOut(NamedTuple):
    ansatz_dill: FlyteFile
    ansatz_json: FlyteFile
    initial_point_npy: FlyteFile
    ansatz_png: FlyteFile


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
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        fig = ansatz.draw(output="mpl", style="iqp", fold=80)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] 电路图绘制失败: {exc}", flush=True)
        return False


@task(
    container_image=vqe_jssp_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
    timeout=timedelta(minutes=10),
)
def t04_build_ansatz(
    hamiltonian_json: FlyteFile,
    ansatz_reps: int = 2,
    seed: int = 42,
) -> AnsatzOut:
    """[CLASSICAL] 构建 RY product ansatz 与初始参数。"""

    print_banner(TaskKind.CLASSICAL, "Task 04 / 构建 JSSP VQE ansatz")
    metadata_from_hamiltonian = json.loads(Path(hamiltonian_json.download()).read_text(encoding="utf-8"))
    num_qubits = int(metadata_from_hamiltonian["num_qubits"])
    if ansatz_reps < 1:
        raise ValueError("ansatz_reps 必须 >= 1")

    ansatz = _build_ansatz(num_qubits, ansatz_reps)
    rng = np.random.default_rng(seed)
    initial_point = rng.uniform(-0.2, 0.2, size=ansatz.num_parameters)

    metadata = {
        "ansatz": "RY product ansatz",
        "num_qubits": int(ansatz.num_qubits),
        "num_parameters": int(ansatz.num_parameters),
        "ansatz_reps": int(ansatz_reps),
        "initial_point_strategy": "seeded_uniform(-0.2, 0.2)",
        "seed": int(seed),
        "parameters": [str(p) for p in ansatz.parameters],
    }

    work = task_workdir("t04")
    dill_path = work / "ansatz.dill"
    with dill_path.open("wb") as f:
        dill.dump({"ansatz": ansatz, "metadata": metadata}, f)

    npy_path = work / "initial_point.npy"
    np.save(npy_path, initial_point)

    json_path = work / "ansatz.json"
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    png_path = work / "04_ansatz_circuit.png"
    _draw_ansatz(ansatz, png_path)

    print(f"  qubit 数 = {metadata['num_qubits']}", flush=True)
    print(f"  参数数 = {metadata['num_parameters']}, reps = {metadata['ansatz_reps']}", flush=True)
    print(f"  -> 写入 {dill_path}", flush=True)
    print(f"  -> 写入 {npy_path}", flush=True)
    print(f"  -> 写入 {json_path}", flush=True)
    print(f"  -> 写入 {png_path}", flush=True)
    return AnsatzOut(
        ansatz_dill=FlyteFile(str(dill_path)),
        ansatz_json=FlyteFile(str(json_path)),
        initial_point_npy=FlyteFile(str(npy_path)),
        ansatz_png=FlyteFile(str(png_path)),
    )
