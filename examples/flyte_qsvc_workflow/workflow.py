"""flyte 适配版 QSVC 量子-经典混合流水线。

与 ``examples/hybrid_pipeline_demo/`` 一比一同源(8 个任务 / 同样数据流 / 同样 4 张图):
  - 1 个 [QUANTUM]   t03_quantum_kernel
  - 1 个 [HYBRID]    t05_train_qsvc
  - 6 个 [CLASSICAL] t01/t02/t04/t06/t07/t08

区别只在 orchestrator:
  - hybrid_pipeline_demo: subprocess 顺序调用 + 本地文件路径
  - 本 demo            : flytekit @task + @workflow + ImageSpec + FlyteFile

每个 @task 是一个独立容器,artifact 通过 ``FlyteFile`` 在 task 之间自动传递
(flyte 后端把文件推到 BlobStore / S3 / GCS, 下游 task 自动 download)。
本机不实际运行 —— 只做静态语法验证;真正的运行请按 README 在你自己的
flyte sandbox 或远程集群里执行。
"""

from __future__ import annotations

from typing import NamedTuple

from flytekit import ImageSpec, Resources, task, workflow
from flytekit.types.file import FlyteFile


# ---------------------------------------------------------------------------
# ImageSpec: 声明式定义 task 容器镜像
# ---------------------------------------------------------------------------
# - flyte 在第一次 `pyflyte run --remote` / `pyflyte register` 时会
#   按这里的 packages 自动 build & push 一个镜像,后续 task 复用。
# - registry 必须改成你有 push 权限的镜像仓(README 第 5 节有说明)。
# - 想加 GPU / apt 包 / 私有 wheel,见 ImageSpec 全部参数:
#   https://docs.flyte.org/en/latest/api/flytekit/generated/flytekit.image_spec.image_spec.ImageSpec.html
qml_image = ImageSpec(
    name="qml-flyte-demo",
    python_version="3.12",
    packages=[
        "qiskit>=2.0",
        "qiskit-aer>=0.15",
        "qiskit-machine-learning>=0.8",
        "numpy>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.2",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "dill>=0.3",
        "pylatexenc>=2.10",
    ],
    # TODO(user): 改成你自己的 registry,例如 "ghcr.io/your-username"
    # 或 "<aws-account>.dkr.ecr.<region>.amazonaws.com"
    registry="ghcr.io/your-org",
)


# ---------------------------------------------------------------------------
# Tuple 返回类型: flyte 推荐用 NamedTuple 给多返回值起名,UI 与 cli 都更可读
# ---------------------------------------------------------------------------
class QuantumKernelOut(NamedTuple):
    kernel_train: FlyteFile      # kernel_train.npz
    kernel_test: FlyteFile       # kernel_test.npz
    circuit: FlyteFile           # circuit.png (ZZFeatureMap 电路图)
    feature_map: FlyteFile       # feature_map.dill (供 t08 重建量子核)


class FinalFigures(NamedTuple):
    bars: FlyteFile              # 03_results.png
    decision: FlyteFile          # 04_decision.png


class WorkflowOutputs(NamedTuple):
    metrics: FlyteFile           # metrics.json
    fig_data: FlyteFile          # 01_data.png
    fig_kernels: FlyteFile       # 02_kernels.png
    fig_bars: FlyteFile          # 03_results.png
    fig_decision: FlyteFile      # 04_decision.png


# ---------------------------------------------------------------------------
# t01 [CLASSICAL]: ad_hoc_data 生成
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="500m", mem="512Mi"))
def t01_generate_data(
    n: int = 2,
    train: int = 20,
    test: int = 10,
    gap: float = 0.3,
    seed: int = 42,
) -> FlyteFile:
    """[CLASSICAL] 生成 ad_hoc 数据集。

    返回 data.npz, 内含 x_train/y_train/x_test/y_test 四个 ndarray。
    """
    import numpy as np
    from qiskit_machine_learning.datasets import ad_hoc_data
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed
    np.random.seed(seed)

    x_train, y_train, x_test, y_test = ad_hoc_data(
        training_size=train, test_size=test, n=n, gap=gap, one_hot=False,
    )
    out = "data.npz"
    np.savez_compressed(
        out, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
    )
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t02 [CLASSICAL]: 数据散点图
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="500m", mem="512Mi"))
def t02_visualize_data(data: FlyteFile) -> FlyteFile:
    """[CLASSICAL] 训练/测试集 2D 散点图,按真实标签上色。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    npz = np.load(data.download(), allow_pickle=True)
    x_tr, y_tr = npz["x_train"], npz["y_train"]
    x_te, y_te = npz["x_test"], npz["y_test"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, (x, y, title) in zip(
        axes,
        [(x_tr, y_tr, f"Training set (n={len(y_tr)})"),
         (x_te, y_te, f"Test set (n={len(y_te)})")],
    ):
        for cls, marker, color in [(0, "o", "tab:blue"), (1, "s", "tab:red")]:
            mask = y == cls
            ax.scatter(x[mask, 0], x[mask, 1], marker=marker, c=color, s=70,
                       edgecolors="black", linewidth=0.6, label=f"class {cls}")
        ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
        ax.set_title(title)
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        ax.grid(True, alpha=0.3); ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("ad_hoc_data: non-trivially separable in $[0, 2\\pi]^2$", fontsize=12)
    fig.tight_layout()

    out = "01_data.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t03 [QUANTUM]: 流水线唯一一个真正运行量子电路的 task
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="1", mem="1Gi"))
def t03_quantum_kernel(
    data: FlyteFile,
    reps: int = 2,
    entanglement: str = "linear",
    seed: int = 42,
) -> QuantumKernelOut:
    """[QUANTUM] 用 FidelityQuantumKernel + ZZFeatureMap 计算量子核矩阵。

    每个矩阵元素 K[i,j] = |<phi(x_i)|phi(x_j)>|^2 = 一次量子电路求值
    (默认 statevector 模拟器; 想跑 IBM 真实硬件,把 FidelityQuantumKernel 的
    fidelity 参数换成 qiskit_ibm_runtime.SamplerV2 的 ComputeUncompute 即可)。
    """
    import dill
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit.circuit.library import zz_feature_map
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = seed

    npz = np.load(data.download(), allow_pickle=True)
    x_train, x_test = npz["x_train"], npz["x_test"]
    y_train, y_test = npz["y_train"], npz["y_test"]

    feature_map = zz_feature_map(x_train.shape[1], reps=reps, entanglement=entanglement)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    k_train = quantum_kernel.evaluate(x_vec=x_train)
    k_test = quantum_kernel.evaluate(x_vec=x_test, y_vec=x_train)

    np.savez_compressed(
        "kernel_train.npz",
        kernel=k_train, x=x_train, y=y_train, kind="quantum",
    )
    np.savez_compressed(
        "kernel_test.npz",
        kernel=k_test, x_test=x_test, y_test=y_test,
        x_train=x_train, y_train=y_train, kind="quantum",
    )

    fig = feature_map.decompose().draw(output="mpl", style="iqp", fold=-1)
    fig.savefig("circuit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    with open("feature_map.dill", "wb") as f:
        dill.dump(
            {"feature_map": feature_map, "reps": reps,
             "entanglement": entanglement, "n_features": int(x_train.shape[1])},
            f,
        )

    return QuantumKernelOut(
        kernel_train=FlyteFile("kernel_train.npz"),
        kernel_test=FlyteFile("kernel_test.npz"),
        circuit=FlyteFile("circuit.png"),
        feature_map=FlyteFile("feature_map.dill"),
    )


# ---------------------------------------------------------------------------
# t04 [CLASSICAL]: 量子核 vs 经典 RBF 核 热力图对比
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="500m", mem="512Mi"))
def t04_visualize_kernel(
    quantum_kernel: FlyteFile,
    rbf_gamma: float = 1.0,
) -> FlyteFile:
    """[CLASSICAL] 量子核 vs 经典 RBF 核 训练矩阵 热力图并排对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics.pairwise import rbf_kernel

    npz = np.load(quantum_kernel.download(), allow_pickle=True)
    k_quantum = npz["kernel"]
    x_train = npz["x"]
    y_train = npz["y"]

    order = np.argsort(y_train, kind="stable")
    k_quantum_sorted = k_quantum[np.ix_(order, order)]
    x_train_sorted = x_train[order]
    y_train_sorted = y_train[order]
    k_rbf_sorted = rbf_kernel(x_train_sorted, gamma=rbf_gamma)
    n0 = int((y_train_sorted == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, mat, title in [
        (axes[0], k_quantum_sorted, "Quantum kernel (Fidelity, ZZFeatureMap)"),
        (axes[1], k_rbf_sorted, f"Classical kernel (RBF, gamma={rbf_gamma})"),
    ]:
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.axhline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
        ax.axvline(n0 - 0.5, color="white", linewidth=1.0, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("sample idx (sorted by label)")
        ax.set_ylabel("sample idx (sorted by label)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Training kernel matrices (top-left {n0}x{n0} = class 0)",
        fontsize=12,
    )
    fig.tight_layout()

    out = "02_kernels.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t05 [HYBRID]: 量子核 + sklearn SVC(precomputed) = QSVC
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="500m", mem="512Mi"))
def t05_train_qsvc(
    kernel_train: FlyteFile,
    kernel_test: FlyteFile,
    c: float = 1.0,
    seed: int = 42,
) -> FlyteFile:
    """[HYBRID] 上半身量子(已预算 K),下半身经典(SVM dual 优化)。"""
    import dill
    import numpy as np
    from sklearn.svm import SVC

    npz_tr = np.load(kernel_train.download(), allow_pickle=True)
    npz_te = np.load(kernel_test.download(), allow_pickle=True)
    k_train = npz_tr["kernel"]; y_train = npz_tr["y"]
    k_test = npz_te["kernel"]; y_test = npz_te["y_test"]

    model = SVC(kernel="precomputed", C=c, random_state=seed)
    model.fit(k_train, y_train)
    y_train_pred = model.predict(k_train)
    y_test_pred = model.predict(k_test)

    bundle = {
        "name": "QSVC (quantum kernel + SVC)",
        "kind": "hybrid",
        "model": model,
        "y_train": y_train, "y_train_pred": y_train_pred,
        "y_test": y_test, "y_test_pred": y_test_pred,
        "train_acc": float((y_train_pred == y_train).mean()),
        "test_acc": float((y_test_pred == y_test).mean()),
        "n_support_vectors": int(model.support_.size),
        "C": c,
    }
    out = "qsvc.dill"
    with open(out, "wb") as f:
        dill.dump(bundle, f)
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t06 [CLASSICAL]: 经典 RBF SVC 基线
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="500m", mem="512Mi"))
def t06_train_classical_svc(
    data: FlyteFile,
    c: float = 1.0,
    rbf_gamma: float = -1.0,    # -1 表示用 sklearn 默认的 'scale'
    seed: int = 42,
) -> FlyteFile:
    """[CLASSICAL] sklearn SVC(kernel='rbf') 基线对照。"""
    import dill
    import numpy as np
    from sklearn.svm import SVC

    npz = np.load(data.download(), allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]

    gamma_val: float | str = "scale" if rbf_gamma < 0 else float(rbf_gamma)
    model = SVC(kernel="rbf", C=c, gamma=gamma_val, random_state=seed)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    bundle = {
        "name": "Classical SVC (RBF)",
        "kind": "classical",
        "model": model,
        "y_train": y_train, "y_train_pred": y_train_pred,
        "y_test": y_test, "y_test_pred": y_test_pred,
        "train_acc": float((y_train_pred == y_train).mean()),
        "test_acc": float((y_test_pred == y_test).mean()),
        "C": c, "gamma": gamma_val,
    }
    out = "classical_svc.dill"
    with open(out, "wb") as f:
        dill.dump(bundle, f)
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t07 [CLASSICAL]: 聚合指标到 metrics.json
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="200m", mem="256Mi"))
def t07_evaluate(qsvc: FlyteFile, classical: FlyteFile) -> FlyteFile:
    """[CLASSICAL] 把两个模型的 acc / 混淆矩阵 / 量子优势 写到 metrics.json。"""
    import json
    import dill
    import numpy as np

    def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
        cm = [[0, 0], [0, 0]]
        for t, p in zip(y_true.astype(int).tolist(), y_pred.astype(int).tolist()):
            cm[t][p] += 1
        return cm

    def _summarize(b: dict) -> dict:
        return {
            "name": b["name"], "kind": b["kind"],
            "train_acc": float(b["train_acc"]),
            "test_acc": float(b["test_acc"]),
            "confusion_matrix_train": _confusion(b["y_train"], b["y_train_pred"]),
            "confusion_matrix_test": _confusion(b["y_test"], b["y_test_pred"]),
        }

    with open(qsvc.download(), "rb") as f:
        qsvc_bundle = dill.load(f)
    with open(classical.download(), "rb") as f:
        csvc_bundle = dill.load(f)

    qsvc_summary = _summarize(qsvc_bundle)
    csvc_summary = _summarize(csvc_bundle)
    metrics = {
        "models": {"qsvc": qsvc_summary, "classical_svc": csvc_summary},
        "quantum_advantage_test_acc": float(
            qsvc_summary["test_acc"] - csvc_summary["test_acc"]
        ),
    }

    out = "metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return FlyteFile(out)


# ---------------------------------------------------------------------------
# t08 [CLASSICAL]: 准确率柱状图 + 决策边界 (重算量子核 grid -> 第二次跑量子电路)
# ---------------------------------------------------------------------------
@task(container_image=qml_image, requests=Resources(cpu="1", mem="1Gi"))
def t08_visualize_results(
    data: FlyteFile,
    metrics: FlyteFile,
    qsvc: FlyteFile,
    classical: FlyteFile,
    feature_map: FlyteFile,
    grid: int = 30,
) -> FinalFigures:
    """[CLASSICAL] 03_results.png + 04_decision.png。

    决策边界绘制需要在 grid x grid 个新点上重算量子核 K(grid, train) ——
    这是流水线里第二次(也是最后一次)调用量子电路。grid=30 在 statevector
    模拟器上约 30s。
    """
    import json
    import dill
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    # ---- 准确率柱状图 ------------------------------------------------------
    metrics_dict = json.loads(open(metrics.download()).read())
    qsvc_m = metrics_dict["models"]["qsvc"]
    csvc_m = metrics_dict["models"]["classical_svc"]
    advantage = metrics_dict["quantum_advantage_test_acc"]

    names = [qsvc_m["name"], csvc_m["name"]]
    train_accs = [qsvc_m["train_acc"], csvc_m["train_acc"]]
    test_accs = [qsvc_m["test_acc"], csvc_m["test_acc"]]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    x = np.arange(len(names))
    bars_tr = ax.bar(x - width / 2, train_accs, width,
                     color=["#d2b4de", "#aed6f1"], edgecolor="black",
                     linewidth=0.6, label="train acc")
    bars_te = ax.bar(x + width / 2, test_accs, width,
                     color=["#9b59b6", "#3498db"], edgecolor="black",
                     linewidth=0.6, label="test acc")
    for bars, vals in ((bars_tr, train_accs), (bars_te, test_accs)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n[{metrics_dict['models'][k]['kind']}]"
                        for n, k in zip(names, ["qsvc", "classical_svc"])])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(loc="upper right", fontsize=10)
    ax.set_title(f"Test accuracy: quantum advantage = {advantage:+.2%}", fontsize=12)
    fig.tight_layout()
    fig.savefig("03_results.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ---- 决策边界 ----------------------------------------------------------
    npz = np.load(data.download(), allow_pickle=True)
    x_train, y_train = npz["x_train"], npz["y_train"]
    x_test, y_test = npz["x_test"], npz["y_test"]
    with open(feature_map.download(), "rb") as f:
        fm_pkg = dill.load(f)
    fm = fm_pkg["feature_map"]
    with open(qsvc.download(), "rb") as f:
        qsvc_b = dill.load(f)
    with open(classical.download(), "rb") as f:
        csvc_b = dill.load(f)

    g = grid
    xs = np.linspace(0, 2 * np.pi, g)
    ys = np.linspace(0, 2 * np.pi, g)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.c_[XX.ravel(), YY.ravel()]

    quantum_kernel = FidelityQuantumKernel(feature_map=fm)
    k_grid = quantum_kernel.evaluate(x_vec=grid_pts, y_vec=x_train)

    z_q = qsvc_b["model"].predict(k_grid).reshape(g, g)
    z_c = csvc_b["model"].predict(grid_pts).reshape(g, g)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, z, title, kind in [
        (axes[0], z_q, qsvc_b["name"], qsvc_b["kind"]),
        (axes[1], z_c, csvc_b["name"], csvc_b["kind"]),
    ]:
        ax.contourf(XX, YY, z, levels=[-0.5, 0.5, 1.5],
                    colors=["#cce5ff", "#ffd2cf"], alpha=0.7)
        for cls, marker, color, label in [
            (0, "o", "tab:blue", "class 0"),
            (1, "s", "tab:red", "class 1"),
        ]:
            mtr = y_train == cls
            mte = y_test == cls
            ax.scatter(x_train[mtr, 0], x_train[mtr, 1],
                       c=color, marker=marker, s=55, edgecolors="black",
                       linewidth=0.6, label=f"{label} (train)")
            ax.scatter(x_test[mte, 0], x_test[mte, 1],
                       c=color, marker=marker, s=120, edgecolors="black",
                       linewidth=1.6, alpha=0.95, label=f"{label} (test)")
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
        ax.set_xlabel("$x_0$"); ax.set_ylabel("$x_1$")
        ax.set_title(f"{title}  [{kind}]")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Decision boundaries on $[0, 2\\pi]^2$", fontsize=12)
    fig.tight_layout()
    fig.savefig("04_decision.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return FinalFigures(
        bars=FlyteFile("03_results.png"),
        decision=FlyteFile("04_decision.png"),
    )


# ---------------------------------------------------------------------------
# @workflow: 用数据依赖把 8 个 task 串成 DAG
# ---------------------------------------------------------------------------
@workflow
def qsvc_workflow(
    n: int = 2,
    train: int = 20,
    test: int = 10,
    gap: float = 0.3,
    reps: int = 2,
    c: float = 1.0,
    rbf_gamma: float = -1.0,    # -1 -> sklearn 'scale'
    seed: int = 42,
    grid: int = 30,
) -> WorkflowOutputs:
    """端到端 flyte workflow。flyte 会按数据依赖自动并行/调度。

    数据流(=DAG):
        t01 -> t02
        t01 -> t03 -> t04
        t01 -> t06
        t03 -> t05
        {t05, t06} -> t07
        {t01, t03, t05, t06, t07} -> t08
    """
    data = t01_generate_data(n=n, train=train, test=test, gap=gap, seed=seed)
    fig_data = t02_visualize_data(data=data)
    qk_out = t03_quantum_kernel(data=data, reps=reps, seed=seed)
    fig_kernels = t04_visualize_kernel(quantum_kernel=qk_out.kernel_train, rbf_gamma=1.0)
    qsvc = t05_train_qsvc(
        kernel_train=qk_out.kernel_train,
        kernel_test=qk_out.kernel_test,
        c=c, seed=seed,
    )
    csvc = t06_train_classical_svc(data=data, c=c, rbf_gamma=rbf_gamma, seed=seed)
    metrics = t07_evaluate(qsvc=qsvc, classical=csvc)
    figs = t08_visualize_results(
        data=data, metrics=metrics, qsvc=qsvc, classical=csvc,
        feature_map=qk_out.feature_map, grid=grid,
    )
    return WorkflowOutputs(
        metrics=metrics,
        fig_data=fig_data,
        fig_kernels=fig_kernels,
        fig_bars=figs.bars,
        fig_decision=figs.decision,
    )


# ---------------------------------------------------------------------------
# 本地直接 `python workflow.py` 入口 -- 走 flyte local execution mode
# (不需要 k8s / sandbox; flytekit 会在本地进程内顺序跑所有 task)
#
# 这只是 sanity check; 真正的"并行调度 + 容器化 + UI 可观测"请按 README
# 用 `pyflyte run --remote` 提交到 sandbox 或远程集群。
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running flyte workflow in local mode (no k8s/sandbox)...")
    out = qsvc_workflow()
    print("done.")
    print(f"  metrics      = {out.metrics.path}")
    print(f"  fig_data     = {out.fig_data.path}")
    print(f"  fig_kernels  = {out.fig_kernels.path}")
    print(f"  fig_bars     = {out.fig_bars.path}")
    print(f"  fig_decision = {out.fig_decision.path}")
