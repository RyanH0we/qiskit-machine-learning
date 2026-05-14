# GJQ Yudu 真机混合流水线示例

本示例基于 [`examples/hybrid_pipeline_demo`](../hybrid_pipeline_demo/)，但把量子核矩阵计算从本地量子模拟器替换为通过 `gjq-client` 调用公司量子真机 `Yudu`。

原始 demo 仍然保留在 `examples/hybrid_pipeline_demo/`。本目录是独立的新 Example。

## 做了什么

流水线仍然拆成 8 个独立 CLI task，由 `main.py` 顺序编排：

| # | Task | Kind | 说明 |
|---|------|------|------|
| 1 | `task_01_generate_data.py` | CLASSICAL | 生成 `ad_hoc_data` 二分类数据 |
| 2 | `task_02_visualize_data.py` | CLASSICAL | 绘制数据散点图 |
| 3 | `task_03_quantum_kernel.py` | QUANTUM | 用 `gjq-client` 在 `Yudu` 上提交 compute-uncompute 电路，计算量子核矩阵 |
| 4 | `task_04_visualize_kernel.py` | CLASSICAL | 对比量子核和经典 RBF 核热力图 |
| 5 | `task_05_train_qsvc.py` | HYBRID | 用 Yudu 返回的核矩阵训练 `SVC(kernel="precomputed")` |
| 6 | `task_06_train_classical_svc.py` | CLASSICAL | 训练经典 RBF SVC 基线 |
| 7 | `task_07_evaluate.py` | CLASSICAL | 汇总准确率和混淆矩阵 |
| 8 | `task_08_visualize_results.py` | CLASSICAL | 只绘制准确率柱状图 |

真机版默认不画 2D 决策边界，因为那需要额外提交 `grid^2 * n_train` 个量子核任务。这里刻意避免任何本地量子模拟器调用。

## 准备环境

本示例使用独立的 `qml-gjq` conda 环境测试和运行，避免破坏已有环境。

在仓库根目录执行：

```bash
conda env create -f examples/gjq_hybrid_pipeline_demo/environment.yml
conda activate qml-gjq
pip install -e . --no-deps
```

安装后确认 SDK 可导入：

```bash
python -c "from gjq_client import GJQRuntimeService, Sampler, generate_preset_pass_manager; print('gjq-client ok')"
```

## API Key 与设备

测试 API key 写在：

```text
examples/gjq_hybrid_pipeline_demo/tasks/task_03_quantum_kernel.py
```

文件顶部常量：

```python
GJQ_API_KEY = "your_api_key_here"
GJQ_BACKEND_NAME = "Yudu"
```

本示例固定使用 `Yudu`。其他设备暂不开放，也不会使用 `SAS-CPU`、`FAS-CPU` 或本地量子模拟器。

## 一键运行

```bash
cd examples/gjq_hybrid_pipeline_demo
conda run -n qml-gjq python main.py --clean
```

默认参数是小规模真机 workload：

```text
--n 2
--train 2
--test 1
--shots 1024
--optimization-level 2
```

默认会提交：

```text
K_train: 4 个训练样本，只提交非对角上三角 = 6 个 Yudu 任务
K_test : 2 个测试样本 x 4 个训练样本 = 8 个 Yudu 任务
合计   : 14 个 Yudu 任务
```

## 常用参数

```text
--shots                 采样次数，默认 1024
--optimization-level    GJQ 转译优化等级：none、0、1、2 或 3，默认 2
--train                 每类训练样本数，默认 2
--test                  每类测试样本数，默认 1
--reps                  ZZFeatureMap reps，默认 2
--clean                 开始前删除 artifacts/
```

示例：

```bash
conda run -n qml-gjq python main.py --clean --shots 2048 --optimization-level 1
```

## 输出

跑完后会生成：

```text
artifacts/
├── data/data.npz
├── kernel/kernel_train.npz
├── kernel/kernel_test.npz
├── kernel/gjq_tasks.jsonl
├── kernel/circuit.png
├── kernel/feature_map.dill
├── models/qsvc.dill
├── models/classical_svc.dill
├── figures/01_data.png
├── figures/02_kernels.png
├── figures/03_results.png
└── metrics.json
```

`gjq_tasks.jsonl` 每行记录一个 Yudu 任务，包括矩阵位置、`instanceId`、fidelity 和 counts 摘要，方便排查真机返回。

## 单独运行 Task 03

如果已经有 `artifacts/data/data.npz`，可以只重算 GJQ 量子核：

```bash
conda run -n qml-gjq python tasks/task_03_quantum_kernel.py \
    --input artifacts/data/data.npz \
    --output-dir artifacts/kernel \
    --shots 1024 \
    --optimization-level 2
```

## 验证命令

```bash
conda run -n qml-gjq python -m py_compile examples/gjq_hybrid_pipeline_demo/main.py
conda run -n qml-gjq python -m py_compile examples/gjq_hybrid_pipeline_demo/tasks/*.py
conda run -n qml-gjq python examples/gjq_hybrid_pipeline_demo/main.py --help
conda run -n qml-gjq python examples/gjq_hybrid_pipeline_demo/tasks/task_03_quantum_kernel.py --help
```

真机小规模全流程：

```bash
cd examples/gjq_hybrid_pipeline_demo
conda run -n qml-gjq python main.py --clean
```

## 注意事项

- 真机结果会受硬件噪声、排队时间、转译优化等级和设备状态影响。
- `kernel_train.npz` / `kernel_test.npz` 的核心字段与原 demo 兼容，下游训练任务不关心量子核来自模拟器还是真机。
- 如果 `job.result()` 返回结构不是常见 counts 形态，Task 03 会报出结果结构摘要，便于补充解析器。
- API key 是测试凭据；生产或公开发布前建议改为环境变量或密钥管理服务注入。
