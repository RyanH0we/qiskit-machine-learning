"""[CLASSICAL] Task 07 -- 聚合各模型指标到 metrics.json。

读 task_05 的 QSVC 结果与 task_06 的经典 SVC 结果, 把 train_acc / test_acc /
混淆矩阵 / 量子优势 (qsvc_test_acc - classical_test_acc) 落到 JSON。
metrics.json 是后续可视化与流水线断言的唯一真相源。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_parent_dir, print_banner


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """二分类混淆矩阵: rows=true, cols=pred, 顺序 [0, 1]."""
    cm = [[0, 0], [0, 0]]
    for t, p in zip(y_true.astype(int).tolist(), y_pred.astype(int).tolist()):
        cm[t][p] += 1
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qsvc", type=Path, required=True, help="task_05 输出 .dill")
    p.add_argument("--classical", type=Path, required=True, help="task_06 输出 .dill")
    p.add_argument("--output", type=Path, required=True, help="输出 metrics.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.CLASSICAL, "Task 07 / aggregate metrics into metrics.json")
    print(f"  qsvc      = {args.qsvc}")
    print(f"  classical = {args.classical}")
    print(f"  output    = {args.output}")

    with args.qsvc.open("rb") as f:
        qsvc = dill.load(f)
    with args.classical.open("rb") as f:
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

    out = ensure_parent_dir(args.output)
    with Timer("write json"):
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"  -> wrote {out}")
    print()
    print("  Summary:")
    for key in ("qsvc", "classical_svc"):
        s = metrics["models"][key]
        print(f"    {s['name']:24s} [{s['kind']:8s}] train={s['train_acc']:.4f}  test={s['test_acc']:.4f}")
    print(f"    >> quantum advantage (test_acc):  {quantum_advantage:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
