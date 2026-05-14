"""[QUANTUM] Task 03 -- 用 gjq-client 在 Yudu 量子真机上计算量子核矩阵。

对每对 (x_i, x_j) 计算 fidelity:

    K[i, j] = |<phi(x_i) | phi(x_j)>|^2

其中 phi 用 ZZFeatureMap(n=2, reps=2) 编码。本任务会构造
U(x_i)·U†(x_j) 的 compute-uncompute 电路，测量全 0 态概率 P(00...0)，
并把它作为 kernel matrix 的元素。

本示例固定使用公司量子真机 Yudu，不使用本地量子模拟器，也不使用云端模拟器。

输出:
  - kernel_train.npz : K_train (n_train x n_train)，用于 SVC.fit
  - kernel_test.npz  : K_test  (n_test x n_train)，用于 SVC.predict
  - gjq_tasks.jsonl  : 每个 Yudu 任务的 instanceId / 矩阵坐标 / counts 摘要
  - circuit.png      : ZZFeatureMap 的电路图
  - feature_map.dill : feature map 配置，供复现实验时检查
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dill
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_utils import TaskKind, Timer, ensure_dir, ensure_parent_dir, print_banner


GJQ_API_KEY = ""
GJQ_BACKEND_NAME = "Yudu"


@dataclass(frozen=True)
class GJQKernelResult:
    """一次 GJQ 真机任务解析后的最小结果。"""

    instance_id: str
    counts: dict[str, float]
    fidelity: float
    raw_preview: str


def parse_optimization_level(value: str) -> int | None:
    """解析 SDK 支持的转译优化等级。"""
    text = value.strip().lower()
    if text == "none":
        return None
    try:
        level = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--optimization-level 必须是 none、0、1、2 或 3") from exc
    if level not in {0, 1, 2, 3}:
        raise argparse.ArgumentTypeError("--optimization-level 必须是 none、0、1、2 或 3")
    return level


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
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--shots", type=int, default=1024, help="采样次数 shots")
    p.add_argument(
        "--optimization-level",
        type=parse_optimization_level,
        default=2,
        help="GJQ 转译优化等级：none、0、1、2 或 3，默认 2",
    )
    return p.parse_args()


def validate_runtime_args(args: argparse.Namespace) -> None:
    if GJQ_API_KEY.strip() == "":
        raise ValueError("GJQ_API_KEY 为空，请先在脚本顶部填入可用 API key。")
    if args.shots <= 0:
        raise ValueError("--shots 必须是正整数")


def build_compute_uncompute_circuit(feature_map: Any, x: np.ndarray, y: np.ndarray) -> Any:
    """构造 U(x)·U†(y) 并添加显式测量位，供 GJQ Sampler 提交。"""
    from qiskit import QuantumCircuit

    n_qubits = int(len(x))
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.compose(feature_map.assign_parameters([float(v) for v in x]), inplace=True)
    qc.compose(feature_map.assign_parameters([float(v) for v in y]).inverse(), inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


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
    """宽松查找 counts，兼容 {'counts': {...}} 或嵌套 result 结构。"""
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
        raise ValueError(f"无法从 GJQ result 中解析 counts: {_raw_preview(result)}")
    return counts


def _find_instance_id(value: Any) -> str | None:
    if isinstance(value, dict):
        for key in ("instanceId", "instance_id", "task_id", "taskId", "job_id", "jobId"):
            if key in value and value[key] is not None:
                return str(value[key])
        for nested in value.values():
            found = _find_instance_id(nested)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_instance_id(item)
            if found is not None:
                return found
    return None


def extract_instance_id(result: Any, job: Any) -> str:
    found = _find_instance_id(result)
    if found is not None:
        return found
    for attr in ("instance_id", "instanceId", "job_id", "jobId", "task_id", "taskId"):
        value = getattr(job, attr, None)
        if callable(value):
            try:
                value = value()
            except TypeError:
                value = None
        if value is not None:
            return str(value)
    return "unknown"


def _is_zero_key(key: str) -> bool:
    normalized = _normalize_count_key(key)
    if normalized == "":
        return False
    if any(ch not in "01" for ch in normalized):
        return False
    return int(normalized, 2) == 0


def fidelity_from_counts(counts: dict[str, float]) -> float:
    zero_count = sum(count for key, count in counts.items() if _is_zero_key(key))
    total = sum(float(v) for v in counts.values())
    if total <= 0:
        raise ValueError("counts 总和为 0，无法计算 fidelity")
    return float(np.clip(zero_count / total, 0.0, 1.0))


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


def run_gjq_circuit(
    sampler: Any,
    pass_manager: Any,
    *,
    circuit: Any,
    task_name: str,
    shots: int,
) -> GJQKernelResult:
    isa_circuit = pass_manager.run(circuit)
    job = sampler.run(isa_circuit, shots=shots)
    print(f"    已提交 {task_name}，等待 Yudu 返回结果")

    raw_result = job.result()
    counts = extract_counts(raw_result)
    fidelity = fidelity_from_counts(counts)
    instance_id = extract_instance_id(raw_result, job)
    return GJQKernelResult(
        instance_id=instance_id,
        counts=counts,
        fidelity=fidelity,
        raw_preview=_raw_preview(raw_result),
    )


def run_kernel_pair(
    sampler: Any,
    pass_manager: Any,
    *,
    feature_map: Any,
    x: np.ndarray,
    y: np.ndarray,
    task_name: str,
    shots: int,
) -> GJQKernelResult:
    circuit = build_compute_uncompute_circuit(feature_map, x, y)
    return run_gjq_circuit(
        sampler,
        pass_manager,
        circuit=circuit,
        task_name=task_name,
        shots=shots,
    )


def log_kernel_result(
    log_path: Path,
    *,
    matrix: str,
    row: int,
    col: int,
    result: GJQKernelResult,
) -> None:
    append_task_record(
        log_path,
        {
            "matrix": matrix,
            "row": row,
            "col": col,
            "backend": GJQ_BACKEND_NAME,
            "instanceId": result.instance_id,
            "fidelity": result.fidelity,
            "counts_preview": _counts_preview(result.counts),
        },
    )


def main() -> int:
    args = parse_args()
    print_banner(TaskKind.QUANTUM, "Task 03 / 在 GJQ Yudu 真机上计算量子核")
    print(f"  input              = {args.input}")
    print(f"  output-dir         = {args.output_dir}")
    print(f"  backend            = {GJQ_BACKEND_NAME}")
    print(f"  shots              = {args.shots}")
    print(f"  optimization-level = {args.optimization_level}")
    print(f"  ZZFeatureMap(reps={args.reps}, entanglement={args.entanglement!r})")

    try:
        validate_runtime_args(args)
    except ValueError as exc:
        print(f"  [error] {exc}")
        return 2

    try:
        from gjq_client import GJQRuntimeService, Sampler, generate_preset_pass_manager
    except ImportError:
        print("  [error] 未能导入 gjq-client SDK")
        print("          请先在 qml-gjq 环境中安装：pip install gjq-client")
        return 2

    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = args.seed

    npz = np.load(args.input, allow_pickle=True)
    x_train, x_test = npz["x_train"], npz["x_test"]
    y_train, y_test = npz["y_train"], npz["y_test"]
    n_features = int(x_train.shape[1])
    print(f"  n_features = {n_features}, n_train = {len(x_train)}, n_test = {len(x_test)}")

    feature_map = zz_feature_map(n_features, reps=args.reps, entanglement=args.entanglement)
    out_dir = ensure_dir(args.output_dir)
    task_log = out_dir / "gjq_tasks.jsonl"
    task_log.write_text("", encoding="utf-8")

    with Timer("初始化 GJQ Runtime Service"):
        service = GJQRuntimeService(api_key=GJQ_API_KEY)
        backend = service.backend(GJQ_BACKEND_NAME)
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=args.optimization_level,
        )
        sampler = Sampler(backend=backend)

    k_train = np.eye(len(x_train), dtype=float)
    with Timer("在 Yudu 上计算 K_train"):
        for i in range(len(x_train)):
            for j in range(i + 1, len(x_train)):
                task_name = f"qkernel_train_{i:03d}_{j:03d}"
                print(f"  K_train[{i}, {j}]")
                result = run_kernel_pair(
                    sampler,
                    pass_manager,
                    feature_map=feature_map,
                    x=x_train[i],
                    y=x_train[j],
                    task_name=task_name,
                    shots=args.shots,
                )
                k_train[i, j] = result.fidelity
                k_train[j, i] = result.fidelity
                log_kernel_result(task_log, matrix="train", row=i, col=j, result=result)

    k_test = np.empty((len(x_test), len(x_train)), dtype=float)
    with Timer("在 Yudu 上计算 K_test"):
        for i in range(len(x_test)):
            for j in range(len(x_train)):
                task_name = f"qkernel_test_{i:03d}_{j:03d}"
                print(f"  K_test[{i}, {j}]")
                result = run_kernel_pair(
                    sampler,
                    pass_manager,
                    feature_map=feature_map,
                    x=x_test[i],
                    y=x_train[j],
                    task_name=task_name,
                    shots=args.shots,
                )
                k_test[i, j] = result.fidelity
                log_kernel_result(task_log, matrix="test", row=i, col=j, result=result)

    print(f"  K_train.shape = {k_train.shape}, diag mean = {k_train.diagonal().mean():.4f}")
    print(f"  K_train range = [{k_train.min():.4f}, {k_train.max():.4f}]")
    print(f"  K_test.shape  = {k_test.shape}, range = [{k_test.min():.4f}, {k_test.max():.4f}]")

    metadata = {
        "backend": GJQ_BACKEND_NAME,
        "sdk": "gjq-client",
        "shots": args.shots,
        "optimization_level": args.optimization_level,
        "reps": args.reps,
        "entanglement": args.entanglement,
    }
    np.savez_compressed(
        out_dir / "kernel_train.npz",
        kernel=k_train,
        x=x_train,
        y=y_train,
        kind="quantum",
        backend=GJQ_BACKEND_NAME,
        sdk="gjq-client",
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
        backend=GJQ_BACKEND_NAME,
        sdk="gjq-client",
        shots=args.shots,
        metadata=np.array(metadata, dtype=object),
    )
    print(f"  -> wrote {out_dir / 'kernel_train.npz'}")
    print(f"  -> wrote {out_dir / 'kernel_test.npz'}")
    print(f"  -> wrote {task_log}")

    circuit_png = out_dir / "circuit.png"
    try:
        with Timer("绘制 feature map 电路"):
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
                "backend": GJQ_BACKEND_NAME,
                "sdk": "gjq-client",
            },
            f,
        )
    print(f"  -> wrote {fm_pkl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
