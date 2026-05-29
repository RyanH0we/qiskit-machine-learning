"""[HYBRID] Task 05 -- 用预先算好的量子核训练 SVC。"""

from __future__ import annotations

from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class QSVCOut(NamedTuple):
    qsvc_dill: FlyteFile


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="500m", mem="512Mi"),
    limits=Resources(cpu="1", mem="1Gi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t05_train_qsvc(
    kernel_train_npz: FlyteFile,
    kernel_test_npz: FlyteFile,
    c: float = 1.0,
    seed: int = 42,
) -> QSVCOut:
    """[HYBRID] 量子核矩阵 + 经典 SVM dual 优化。"""

    print_banner(TaskKind.HYBRID, "Task 05 / train QSVC = quantum kernel + SVM")
    print(f"  C={c}, seed={seed}", flush=True)

    from sklearn.svm import SVC

    npz_tr = np.load(kernel_train_npz.download(), allow_pickle=True)
    npz_te = np.load(kernel_test_npz.download(), allow_pickle=True)
    k_train = npz_tr["kernel"]
    y_train = npz_tr["y"]
    k_test = npz_te["kernel"]
    y_test = npz_te["y_test"]

    model = SVC(kernel="precomputed", C=c, random_state=seed)
    model.fit(k_train, y_train)
    y_train_pred = model.predict(k_train)
    y_test_pred = model.predict(k_test)

    train_acc = float((y_train_pred == y_train).mean())
    test_acc = float((y_test_pred == y_test).mean())
    bundle = {
        "name": "QSVC",
        "kind": "hybrid",
        "model": model,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "n_support_vectors": int(model.support_.size),
        "C": c,
    }

    work = task_workdir("t05")
    out = work / "qsvc.dill"
    with out.open("wb") as f:
        dill.dump(bundle, f)
    print(f"  train_acc = {train_acc:.4f}", flush=True)
    print(f"  test_acc  = {test_acc:.4f}", flush=True)
    print(f"  -> wrote {out}", flush=True)
    return QSVCOut(qsvc_dill=FlyteFile(str(out)))
