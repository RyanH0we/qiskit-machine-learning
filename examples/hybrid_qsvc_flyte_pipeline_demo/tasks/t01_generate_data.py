"""[CLASSICAL] Task 01 -- 生成 ad_hoc 合成数据集。"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class DataOut(NamedTuple):
    data_npz: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t01_generate_data(
    n: int = 2,
    train: int = 20,
    test: int = 10,
    gap: float = 0.3,
    seed: int = 42,
) -> DataOut:
    """[CLASSICAL] 生成 ad_hoc 数据集，输出 ``data.npz``。"""

    print_banner(TaskKind.CLASSICAL, "Task 01 / generate ad_hoc data")
    print(
        f"  n={n}, train(per class)={train}, test(per class)={test}, "
        f"gap={gap}, seed={seed}",
        flush=True,
    )

    from qiskit_machine_learning.datasets import ad_hoc_data
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed
    np.random.seed(seed)

    x_train, y_train, x_test, y_test = ad_hoc_data(
        training_size=train,
        test_size=test,
        n=n,
        gap=gap,
        one_hot=False,
    )

    work = task_workdir("t01")
    out = work / "data.npz"
    np.savez_compressed(
        out,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        meta=np.array(
            {
                "n": n,
                "train_per_class": train,
                "test_per_class": test,
                "gap": gap,
                "seed": seed,
            },
            dtype=object,
        ),
    )
    print(f"  x_train.shape = {x_train.shape}, x_test.shape = {x_test.shape}", flush=True)
    print(f"  -> wrote {out}", flush=True)
    return DataOut(data_npz=FlyteFile(str(out)))
