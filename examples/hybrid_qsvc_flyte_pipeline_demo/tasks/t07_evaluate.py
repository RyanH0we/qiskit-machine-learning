"""[CLASSICAL] Task 07 -- 聚合各模型指标到 metrics.json。"""

from __future__ import annotations

import json
from typing import NamedTuple

import dill
import numpy as np
from flytekit import Resources, task
from flytekit.types.file import FlyteFile

from pipeline_lib import CACHE_VERSION, TaskKind, hybrid_qsvc_image, print_banner, task_workdir


class MetricsOut(NamedTuple):
    metrics_json: FlyteFile


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    cm = [[0, 0], [0, 0]]
    for true_value, pred_value in zip(y_true.astype(int).tolist(), y_pred.astype(int).tolist()):
        cm[true_value][pred_value] += 1
    return cm


def _summarize(name: str, bundle: dict) -> dict:
    return {
        "name": name,
        "kind": bundle["kind"],
        "train_acc": float(bundle["train_acc"]),
        "test_acc": float(bundle["test_acc"]),
        "confusion_matrix_train": _confusion(bundle["y_train"], bundle["y_train_pred"]),
        "confusion_matrix_test": _confusion(bundle["y_test"], bundle["y_test_pred"]),
    }


@task(
    container_image=hybrid_qsvc_image,
    requests=Resources(cpu="200m", mem="256Mi"),
    limits=Resources(cpu="500m", mem="512Mi"),
    cache=True,
    cache_version=CACHE_VERSION,
    retries=1,
)
def t07_evaluate(qsvc_dill: FlyteFile, classical_svc_dill: FlyteFile) -> MetricsOut:
    """[CLASSICAL] 生成包含准确率、混淆矩阵、量子优势的 JSON。"""

    print_banner(TaskKind.CLASSICAL, "Task 07 / aggregate metrics into metrics.json")

    with open(qsvc_dill.download(), "rb") as f:
        qsvc = dill.load(f)
    with open(classical_svc_dill.download(), "rb") as f:
        csvc = dill.load(f)

    qsvc_summary = _summarize("QSVC", qsvc)
    csvc_summary = _summarize("Classical SVC (RBF)", csvc)
    quantum_advantage = qsvc_summary["test_acc"] - csvc_summary["test_acc"]
    metrics = {
        "models": {
            "qsvc": qsvc_summary,
            "classical_svc": csvc_summary,
        },
        "quantum_advantage_test_acc": float(quantum_advantage),
    }

    work = task_workdir("t07")
    out = work / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  QSVC test_acc = {qsvc_summary['test_acc']:.4f}", flush=True)
    print(f"  Classical test_acc = {csvc_summary['test_acc']:.4f}", flush=True)
    print(f"  quantum advantage = {quantum_advantage:+.4f}", flush=True)
    print(f"  -> wrote {out}", flush=True)
    return MetricsOut(metrics_json=FlyteFile(str(out)))
