"""[QUANTUM] Task 03 -- 用本地量子模拟器计算量子核矩阵。"""

from __future__ import annotations

from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class QuantumKernelOut(NamedTuple):
    kernel_train_npz: FlyteFile
    kernel_test_npz: FlyteFile
    feature_map_dill: FlyteFile
    circuit_png: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="1", mem="1Gi"),
    limits=Resources(cpu="2", mem="2Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t03_quantum_kernel(
    data_npz: FlyteFile,
    reps: int = 2,
    entanglement: str = "linear",
    seed: int = 42,
) -> QuantumKernelOut:
    """[QUANTUM] 计算 K_train / K_test，并保存 feature map 与电路图。"""

    print_banner(TaskKind.QUANTUM, "Task 03 / compute quantum kernel matrix")
    print(f"  data = {data_npz.path}", flush=True)
    print(f"  ZZFeatureMap(reps={reps}, entanglement={entanglement!r})", flush=True)
    if entanglement not in {"linear", "circular", "full"}:
        raise ValueError("entanglement 只能是 linear、circular 或 full")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed

    npz = np.load(data_npz.download(), allow_pickle=True)
    x_train, x_test = npz["x_train"], npz["x_test"]
    y_train, y_test = npz["y_train"], npz["y_test"]
    n_features = x_train.shape[1]
    print(
        f"  n_features={n_features}, n_train={len(x_train)}, n_test={len(x_test)}",
        flush=True,
    )

    feature_map = zz_feature_map(n_features, reps=reps, entanglement=entanglement)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    k_train = quantum_kernel.evaluate(x_vec=x_train)
    k_test = quantum_kernel.evaluate(x_vec=x_test, y_vec=x_train)
    print(f"  K_train.shape = {k_train.shape}", flush=True)
    print(f"  K_test.shape  = {k_test.shape}", flush=True)

    work = task_workdir("t03")
    kernel_train = work / "kernel_train.npz"
    kernel_test = work / "kernel_test.npz"
    feature_map_path = work / "feature_map.dill"
    circuit_path = work / "circuit.png"

    np.savez_compressed(kernel_train, kernel=k_train, x=x_train, y=y_train, kind="quantum")
    np.savez_compressed(
        kernel_test,
        kernel=k_test,
        x_test=x_test,
        y_test=y_test,
        x_train=x_train,
        y_train=y_train,
        kind="quantum",
    )
    with feature_map_path.open("wb") as f:
        dill.dump(
            {
                "feature_map": feature_map,
                "reps": reps,
                "entanglement": entanglement,
                "n_features": n_features,
            },
            f,
        )

    fig = feature_map.decompose().draw(output="mpl", style="iqp", fold=-1)
    fig.savefig(circuit_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"  -> wrote {kernel_train}", flush=True)
    print(f"  -> wrote {kernel_test}", flush=True)
    print(f"  -> wrote {feature_map_path}", flush=True)
    print(f"  -> wrote {circuit_path}", flush=True)
    return QuantumKernelOut(
        kernel_train_npz=FlyteFile(str(kernel_train)),
        kernel_test_npz=FlyteFile(str(kernel_test)),
        feature_map_dill=FlyteFile(str(feature_map_path)),
        circuit_png=FlyteFile(str(circuit_path)),
    )
