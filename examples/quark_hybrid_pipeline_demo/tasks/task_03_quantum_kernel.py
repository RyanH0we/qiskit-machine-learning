"""[QUANTUM] Task 03 -- 用 Quark SDK 在公司量子计算机上计算量子核矩阵。

对每对 (x_i, x_j) 计算 fidelity:

    K[i, j] = |<phi(x_i) | phi(x_j)>|^2

其中 phi 用 ZZFeatureMap(n=2, reps=2) 编码。本任务会构造
U(x_i)·U†(x_j) 的 compute-uncompute 电路,测量全 0 态概率 P(00...0),
并把它作为 kernel matrix 的元素。

使用前请把公司 Quark token 填到下面的 QUARK_TOKEN 字符串中。

输出:
  - kernel_train.npz  : K_train (n_train x n_train), 用于 SVC.fit
  - kernel_test.npz   : K_test  (n_test x n_train),  用于 SVC.predict
  - quark_tasks.jsonl : 每个硬件任务的 tid / 矩阵坐标 / counts 摘要
  - circuit.png       : ZZFeatureMap 的电路图
  - feature_map.dill  : feature map 配置,供复现实验时检查
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_dir, ensure_parent_dir, print_banner


# 请把公司 Quark token 填到这里,例如:
# QUARK_TOKEN = "你的 token"
QUARK_TOKEN = "your_token_here"

FAILED_STATUSES = {"failed", "error", "cancelled", "canceled", "timeout"}


@dataclass(frozen=True)
class QuarkKernelResult:
    """一次 Quark 硬件任务解析后的最小结果。"""

    task_id: str
    status: str
    counts: dict[str, float]
    fidelity: float
    raw_preview: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="task_01 输出的 .npz")
    p.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--reps", type=int, default=2, help="ZZFeatureMap 的重复次数")
    p.add_argument(
        "--entanglement",
        default="linear",
        choices=["linear", "circular", "full"],
        help="ZZFeatureMap 纠缠拓扑",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chip", default="Baihua", help="Quark 量子芯片名称")
    p.add_argument("--shots", type=int, default=1024, help="shots, 必须是 1024 的正整数倍")
    p.add_argument(
        "--compiler",
        default="quarkcircuit",
        choices=["none", "quarkcircuit", "qsteed", "qiskit"],
        help="Quark 编译器; none 会传 None",
    )
    p.add_argument("--correct", action="store_true", help="开启 readout error correction")
    p.add_argument(
        "--open-dd",
        default="none",
        choices=["none", "XY4", "CPMG"],
        help="dynamical decoupling; none 会传 None",
    )
    p.add_argument(
        "--target-qubits",
        default="",
        help="物理比特映射,逗号分隔,例如 '0,1'; 留空表示由后端决定",
    )
    p.add_argument("--poll-interval", type=float, default=1.0, help="任务状态轮询间隔(秒)")
    return p.parse_args()


def parse_target_qubits(value: str) -> list[int]:
    """把 CLI 中的 '0,1,2' 解析为 [0, 1, 2]。"""
    if value.strip() == "":
        return []
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip() != ""]
    except ValueError as exc:
        raise ValueError("--target-qubits 必须是逗号分隔的整数,例如 '0,1'") from exc


def _none_if_text(value: str) -> str | None:
    return None if value.lower() == "none" else value


def validate_runtime_args(args: argparse.Namespace, n_features: int) -> list[int]:
    if QUARK_TOKEN.strip() == "":
        raise ValueError(
            "QUARK_TOKEN 为空。请打开 "
            "examples/quark_hybrid_pipeline_demo/tasks/task_03_quantum_kernel.py, "
            "把文件顶部的 QUARK_TOKEN = \"\" 改成你的公司 Quark token。"
        )
    if args.shots <= 0 or args.shots % 1024 != 0:
        raise ValueError("--shots 必须是 1024 的正整数倍")
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval 必须大于 0")
    target_qubits = parse_target_qubits(args.target_qubits)
    if target_qubits and len(target_qubits) != n_features:
        raise ValueError(
            f"--target-qubits 长度必须等于特征维度/量子比特数 {n_features}, "
            f"当前为 {len(target_qubits)}"
        )
    return target_qubits


def build_compute_uncompute_qasm(feature_map: Any, x: np.ndarray, y: np.ndarray) -> str:
    """构造 U(x)·U†(y) 并导出 Quark SDK 可提交的 OpenQASM 2.0 字符串。"""
    from qiskit import QuantumCircuit, transpile
    import qiskit.qasm2 as qasm2

    n_qubits = int(len(x))
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.compose(feature_map.assign_parameters([float(v) for v in x]), inplace=True)
    qc.compose(feature_map.assign_parameters([float(v) for v in y]).inverse(), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    # Quark 示例使用 OpenQASM 2.0。这里固定到 qelib1 中最通用的 u3/cx/measure。
    compiled = transpile(qc, basis_gates=["u3", "cx"], optimization_level=0)
    return qasm2.dumps(compiled)


def _raw_preview(value: Any, limit: int = 1000) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _normalize_count_key(key: Any) -> str:
    if isinstance(key, (int, np.integer)):
        return format(int(key), "b")
    text = str(key).strip().replace(" ", "")
    if text.lower().startswith("0x"):
        return format(int(text, 16), "b")
    if text.lower().startswith("0b"):
        return text[2:]
    return text


def _looks_like_counts(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for key, count in value.items():
        normalized = _normalize_count_key(key)
        if normalized == "" or any(ch not in "01" for ch in normalized):
            return False
        if not _is_number(count):
            return False
    return True


def _find_counts(value: Any) -> dict[str, float] | None:
    """宽松查找 counts,兼容 {'00': 10} / {'counts': {...}} / 嵌套 result。"""
    if _looks_like_counts(value):
        return {_normalize_count_key(key): float(count) for key, count in value.items()}

    if isinstance(value, dict):
        preferred_keys = (
            "counts",
            "count",
            "measurement_counts",
            "measurements",
            "measure",
            "result",
            "results",
            "data",
        )
        for key in preferred_keys:
            if key in value:
                found = _find_counts(value[key])
                if found is not None:
                    return found
        for nested in value.values():
            found = _find_counts(nested)
            if found is not None:
                return found

    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_counts(item)
            if found is not None:
                return found

    return None


def extract_counts(result: Any) -> dict[str, float]:
    counts = _find_counts(result)
    if counts is None:
        raise ValueError(f"无法从 Quark result 中解析 counts: {_raw_preview(result)}")
    return counts


def _is_zero_key(key: str, n_qubits: int) -> bool:  # noqa: ARG001
    normalized = _normalize_count_key(key)
    if normalized == "":
        return False
    if any(ch not in "01" for ch in normalized):
        return False
    return int(normalized, 2) == 0


def fidelity_from_counts(counts: dict[str, float], n_qubits: int, shots: int) -> float:
    zero_count = sum(count for key, count in counts.items() if _is_zero_key(key, n_qubits))
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        raise ValueError("counts 总和为 0,无法计算 fidelity")

    # 有些 SDK 在开启校正后可能返回概率; 若总和约等于 1,直接使用全 0 概率。
    if total <= 1.000001:
        fidelity = zero_count
    else:
        fidelity = zero_count / float(shots)
    return float(np.clip(fidelity, 0.0, 1.0))


def _counts_preview(counts: dict[str, float], limit: int = 6) -> dict[str, float]:
    items = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:limit]
    return {key: value for key, value in items}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def append_task_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def run_quark_circuit(
    tmgr: Any,
    *,
    qasm: str,
    n_qubits: int,
    task_name: str,
    args: argparse.Namespace,
    target_qubits: list[int],
) -> QuarkKernelResult:
    task = {
        "chip": args.chip,
        "name": task_name,
        "circuit": qasm,
        "shots": args.shots,
        "options": {
            "compiler": _none_if_text(args.compiler),
            "correct": bool(args.correct),
            "open_dd": _none_if_text(args.open_dd),
            "target_qubits": target_qubits,
        },
    }

    tid = tmgr.run(task)
    print(f"    submitted {task_name}: tid={tid}")

    last_status = None
    while True:
        status = str(tmgr.status(tid))
        if status != last_status:
            print(f"    status({tid}) = {status}")
            last_status = status
        status_key = status.lower()
        if status == "Finished":
            break
        if status_key in FAILED_STATUSES:
            raise RuntimeError(f"Quark task {tid} ended with status {status}")
        time.sleep(args.poll_interval)

    raw_result = tmgr.result(tid)
    counts = extract_counts(raw_result)
    fidelity = fidelity_from_counts(counts, n_qubits=n_qubits, shots=args.shots)
    return QuarkKernelResult(
        task_id=str(tid),
        status=last_status or "Finished",
        counts=counts,
        fidelity=fidelity,
        raw_preview=_raw_preview(raw_result),
    )


def run_kernel_pair(
    tmgr: Any,
    *,
    feature_map: Any,
    x: np.ndarray,
    y: np.ndarray,
    n_qubits: int,
    task_name: str,
    args: argparse.Namespace,
    target_qubits: list[int],
) -> QuarkKernelResult:
    qasm = build_compute_uncompute_qasm(feature_map, x, y)
    return run_quark_circuit(
        tmgr,
        qasm=qasm,
        n_qubits=n_qubits,
        task_name=task_name,
        args=args,
        target_qubits=target_qubits,
    )


def log_kernel_result(
    log_path: Path,
    *,
    matrix: str,
    row: int,
    col: int,
    result: QuarkKernelResult,
) -> None:
    append_task_record(
        log_path,
        {
            "matrix": matrix,
            "row": row,
            "col": col,
            "task_id": result.task_id,
            "status": result.status,
            "fidelity": result.fidelity,
            "counts_preview": _counts_preview(result.counts),
        },
    )


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 03 / compute quantum kernel on Quark hardware")
    print(f"  input         = {args.input}")
    print(f"  output-dir    = {args.output_dir}")
    print(f"  chip          = {args.chip}")
    print(f"  shots         = {args.shots}")
    print(f"  compiler      = {_none_if_text(args.compiler)}")
    print(f"  correct       = {args.correct}")
    print(f"  open_dd       = {_none_if_text(args.open_dd)}")
    print(f"  target_qubits = {args.target_qubits or '[]'}")
    print(f"  ZZFeatureMap(reps={args.reps}, entanglement={args.entanglement!r})")

    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = args.seed

    npz = np.load(args.input, allow_pickle=True)
    x_train, x_test = npz["x_train"], npz["x_test"]
    y_train, y_test = npz["y_train"], npz["y_test"]
    n_features = int(x_train.shape[1])
    print(f"  n_features = {n_features}, n_train = {len(x_train)}, n_test = {len(x_test)}")

    try:
        target_qubits = validate_runtime_args(args, n_features)
    except ValueError as exc:
        print(f"  [error] {exc}")
        return 2

    try:
        from quark import Task
    except ImportError:
        print("  [error] 未能导入 Quark SDK: from quark import Task")
        print("          请先在 qml conda 环境中安装公司内部 Quark Python SDK。")
        return 2

    feature_map = zz_feature_map(n_features, reps=args.reps, entanglement=args.entanglement)
    out_dir = ensure_dir(args.output_dir)
    task_log = out_dir / "quark_tasks.jsonl"
    task_log.write_text("", encoding="utf-8")

    tmgr = Task(QUARK_TOKEN)

    k_train = np.eye(len(x_train), dtype=float)
    with Timer("evaluate K_train on Quark hardware"):
        for i in range(len(x_train)):
            for j in range(i + 1, len(x_train)):
                task_name = f"qkernel_train_{i:03d}_{j:03d}"
                print(f"  K_train[{i}, {j}]")
                result = run_kernel_pair(
                    tmgr,
                    feature_map=feature_map,
                    x=x_train[i],
                    y=x_train[j],
                    n_qubits=n_features,
                    task_name=task_name,
                    args=args,
                    target_qubits=target_qubits,
                )
                k_train[i, j] = result.fidelity
                k_train[j, i] = result.fidelity
                log_kernel_result(task_log, matrix="train", row=i, col=j, result=result)

    k_test = np.empty((len(x_test), len(x_train)), dtype=float)
    with Timer("evaluate K_test on Quark hardware"):
        for i in range(len(x_test)):
            for j in range(len(x_train)):
                task_name = f"qkernel_test_{i:03d}_{j:03d}"
                print(f"  K_test[{i}, {j}]")
                result = run_kernel_pair(
                    tmgr,
                    feature_map=feature_map,
                    x=x_test[i],
                    y=x_train[j],
                    n_qubits=n_features,
                    task_name=task_name,
                    args=args,
                    target_qubits=target_qubits,
                )
                k_test[i, j] = result.fidelity
                log_kernel_result(task_log, matrix="test", row=i, col=j, result=result)

    print(f"  K_train.shape = {k_train.shape}, diag mean = {k_train.diagonal().mean():.4f}")
    print(f"  K_train range = [{k_train.min():.4f}, {k_train.max():.4f}]")
    print(f"  K_test.shape  = {k_test.shape}, range = [{k_test.min():.4f}, {k_test.max():.4f}]")

    metadata = {
        "backend": "quark",
        "chip": args.chip,
        "shots": args.shots,
        "compiler": _none_if_text(args.compiler),
        "correct": args.correct,
        "open_dd": _none_if_text(args.open_dd),
        "target_qubits": target_qubits,
        "reps": args.reps,
        "entanglement": args.entanglement,
    }
    np.savez_compressed(
        out_dir / "kernel_train.npz",
        kernel=k_train,
        x=x_train,
        y=y_train,
        kind="quantum",
        backend="quark",
        chip=args.chip,
        shots=args.shots,
        metadata=np.array(metadata, dtype=object),
    )
    np.savez_compressed(
        out_dir / "kernel_test.npz",
        kernel=k_test,
        x_test=x_test,
        y_test=y_test,
        x_train=x_train,
        y_train=y_train,
        kind="quantum",
        backend="quark",
        chip=args.chip,
        shots=args.shots,
        metadata=np.array(metadata, dtype=object),
    )
    print(f"  -> wrote {out_dir / 'kernel_train.npz'}")
    print(f"  -> wrote {out_dir / 'kernel_test.npz'}")
    print(f"  -> wrote {task_log}")

    circuit_png = out_dir / "circuit.png"
    try:
        with Timer("draw circuit"):
            fig = feature_map.decompose().draw(output="mpl", style="iqp", fold=-1)
            fig.savefig(circuit_png, dpi=120, bbox_inches="tight")
            import matplotlib.pyplot as plt

            plt.close(fig)
        print(f"  -> wrote {circuit_png}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] could not draw circuit: {exc}")

    fm_pkl = out_dir / "feature_map.dill"
    with ensure_parent_dir(fm_pkl).open("wb") as f:
        dill.dump(
            {
                "feature_map": feature_map,
                "reps": args.reps,
                "entanglement": args.entanglement,
                "n_features": n_features,
                "backend": "quark",
            },
            f,
        )
    print(f"  -> wrote {fm_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
