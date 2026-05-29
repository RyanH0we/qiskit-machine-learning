"""[CLASSICAL] Task 06 -- 经典 RBF 核 SVC 基线。"""

from __future__ import annotations

from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import (
    CACHE_VERSION,
    TaskKind,
    hybrid_qsvc_image,
    parse_gamma,
    print_banner,
    task_workdir,
)


class ClassicalSVCOut(NamedTuple):
    classical_svc_dill: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t06_train_classical_svc(
    data_npz: FlyteFile,
    c: float = 1.0,
    rbf_gamma: str = "scale",
    seed: int = 42,
) -> ClassicalSVCOut:
    """[CLASSICAL] 训练 sklearn ``SVC(kernel='rbf')`` 对照组。"""

    print_banner(TaskKind.CLASSICAL, "Task 06 / train classical RBF SVC baseline")
    print(f"  C={c}, gamma={rbf_gamma}, seed={seed}", flush=True)

    from sklearn.svm import SVC

    npz = np.load(data_npz.download(), allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]
    gamma_val = parse_gamma(rbf_gamma)

    model = SVC(kernel="rbf", C=c, gamma=gamma_val, random_state=seed)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    train_acc = float((y_train_pred == y_train).mean())
    test_acc = float((y_test_pred == y_test).mean())
    bundle = {
        "name": "Classical SVC (RBF)",
        "kind": "classical",
        "model": model,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "C": c,
        "gamma": gamma_val,
    }

    work = task_workdir("t06")
    out = work / "classical_svc.dill"
    with out.open("wb") as f:
        dill.dump(bundle, f)
    print(f"  train_acc = {train_acc:.4f}", flush=True)
    print(f"  test_acc  = {test_acc:.4f}", flush=True)
    print(f"  -> wrote {out}", flush=True)
    return ClassicalSVCOut(classical_svc_dill=FlyteFile(str(out)))
