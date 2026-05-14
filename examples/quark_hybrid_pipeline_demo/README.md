# Quark Hardware Hybrid Pipeline Demo

这个示例基于 [`examples/hybrid_pipeline_demo`](../hybrid_pipeline_demo/)，但把量子核矩阵计算从本地模拟器替换为公司 Quark Python SDK 调用真实量子计算机。

原始 demo 仍然保留在 `examples/hybrid_pipeline_demo/`。本目录是独立的新 Example。

## 做了什么

流水线仍然拆成 8 个独立 CLI task，由 `main.py` 顺序编排：

| # | Task | Kind | 说明 |
|---|------|------|------|
| 1 | `task_01_generate_data.py` | CLASSICAL | 生成 `ad_hoc_data` 二分类数据 |
| 2 | `task_02_visualize_data.py` | CLASSICAL | 画数据散点图 |
| 3 | `task_03_quantum_kernel.py` | QUANTUM | 用 Quark SDK 提交 compute-uncompute 电路，计算量子核矩阵 |
| 4 | `task_04_visualize_kernel.py` | CLASSICAL | 对比量子核和经典 RBF 核热力图 |
| 5 | `task_05_train_qsvc.py` | HYBRID | 用 Quark 返回的核矩阵训练 `SVC(kernel="precomputed")` |
| 6 | `task_06_train_classical_svc.py` | CLASSICAL | 训练经典 RBF SVC 基线 |
| 7 | `task_07_evaluate.py` | CLASSICAL | 汇总准确率和混淆矩阵 |
| 8 | `task_08_visualize_results.py` | CLASSICAL | 只画准确率柱状图 |

真机版默认不画 2D 决策边界，因为那需要额外提交 `grid^2 * n_train` 个量子核任务。这里刻意避免任何本地量子模拟器调用。

## 准备环境

本示例使用独立的 `qml-quark` conda 环境测试和运行。它和原始 `examples/hybrid_pipeline_demo` 使用的 `qml` 环境分开，避免 `pyquafu` 的 `numpy<2.0` 约束影响原 demo。

```bash
conda env create -f environment.yml
conda activate qml-quark
pip install -e ../.. --no-deps
```

`environment.yml` 已包含 Quark SDK 相关依赖 `quarkstudio` 和 `pyquafu`，所以新机器可以直接用上面的命令复现环境。安装后确认下面命令可用：

```bash
python -c "from quark import Task; print(Task)"
```

注意：`pyquafu` 当前要求 `numpy<2.0`，而仓库包声明 `qiskit-machine-learning` 需要 `numpy>=2.0`。本示例已在 `qml-quark` 环境的 `numpy 1.26.4` 下完成真机 smoke test；因此这里用 `pip install -e ../.. --no-deps` 安装本仓库，避免 pip 把 numpy 升回 2.x。

## Token 位置

Token 写在脚本里。打开：

```text
examples/quark_hybrid_pipeline_demo/tasks/task_03_quantum_kernel.py
```

找到文件顶部：

```python
QUARK_TOKEN = "..."
```

后续如果要更换 token，改这里即可。

## 一键运行

```bash
cd examples/quark_hybrid_pipeline_demo
conda run -n qml-quark python main.py --clean
```

默认参数是小规模真机 workload：

```text
--n 2
--train 2      # 每类 2 个训练样本，训练集共 4 个
--test 1       # 每类 1 个测试样本，测试集共 2 个
--shots 1024
--chip Baihua
--compiler quarkcircuit
```

默认会提交：

```text
K_train: 4 个训练样本，只提交非对角上三角 = 6 个 Quark 任务
K_test : 2 个测试样本 x 4 个训练样本 = 8 个 Quark 任务
合计   : 14 个 Quark 任务
```

## 常用参数

```text
--chip            Quark 芯片名，默认 Baihua
--shots           shots，必须是 1024 的正整数倍
--compiler        none | quarkcircuit | qsteed | qiskit，默认 quarkcircuit
--correct         开启 readout error correction
--open-dd         none | XY4 | CPMG，默认 none
--target-qubits   物理比特映射，例如 "0,1"，留空表示由后端决定
--poll-interval   Quark 任务状态轮询间隔秒数，默认 1.0
```

示例：

```bash
conda run -n qml-quark python main.py --clean --chip Baihua --shots 2048 --target-qubits 0,1
```

## 输出

跑完后会生成：

```text
artifacts/
├── data/data.npz
├── kernel/kernel_train.npz
├── kernel/kernel_test.npz
├── kernel/quark_tasks.jsonl
├── kernel/circuit.png
├── kernel/feature_map.dill
├── models/qsvc.dill
├── models/classical_svc.dill
├── figures/01_data.png
├── figures/02_kernels.png
├── figures/03_results.png
└── metrics.json
```

`quark_tasks.jsonl` 每行记录一个硬件任务，包括矩阵位置、Quark task id、最终状态、fidelity 和 counts 摘要，方便排查真机返回。

## 单独运行 Task 03

如果已经有 `artifacts/data/data.npz`，可以只重算 Quark 量子核：

```bash
conda run -n qml-quark python tasks/task_03_quantum_kernel.py \
    --input artifacts/data/data.npz \
    --output-dir artifacts/kernel \
    --chip Baihua \
    --shots 1024
```

## 验证命令

```bash
conda run -n qml-quark python -m py_compile main.py
conda run -n qml-quark python -m py_compile tasks/*.py
conda run -n qml-quark python main.py --help
conda run -n qml-quark python tasks/task_03_quantum_kernel.py --help
```

## 注意

- 本示例的量子核来自真实 Quark 后端，结果会受硬件噪声、排队时间、编译器和比特映射影响。
- `kernel_train.npz` / `kernel_test.npz` 的核心字段与原 demo 兼容，下游训练任务不关心量子核来自模拟器还是真机。
- 如果 `tmgr.result(tid)` 的格式不是常见 counts 形态，task 03 会报出结果结构摘要，便于补充解析器。
