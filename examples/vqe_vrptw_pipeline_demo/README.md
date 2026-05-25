# VQE 求解带时间窗车辆路径规划 VRPTW 示例

这是一个面向新手的量子-经典混合计算示例：用 **VQE（Variational Quantum Eigensolver，变分量子本征求解器）** 在本地量子模拟器上求一个小型 **VRPTW（Vehicle Routing Problem with Time Windows，带时间窗车辆路径规划）**。

本示例遵循现有 `examples/vqe_h2_pipeline_demo/` 和 `examples/hybrid_pipeline_demo/` 的任务化风格：

1. 每一步都是 `tasks/*.py` 中的独立 CLI。
2. 任务之间只通过 `artifacts/` 下的文件传递结果。
3. 每个任务启动时都会标出 `[CLASSICAL]`、`[QUANTUM]` 或 `[HYBRID]`。
4. 运行结束会生成 JSON、CSV、DILL 和 PNG 可视化结果。

如果你还不了解 VRPTW、QUBO、Ising Hamiltonian 或 VQE，建议先读 [EXPLAINER.md](EXPLAINER.md)。

---

## 1. 环境创建

从仓库根目录运行：

```bash
conda env create -f examples/vqe_vrptw_pipeline_demo/environment.yml
conda activate qml-vqe-vrptw
pip install -e .
```

如果你已经进入本示例目录：

```bash
conda env create -f environment.yml
conda activate qml-vqe-vrptw
pip install -e ../..
```

本示例默认使用本地 `StatevectorEstimator` 模拟器，不连接真实量子计算机。

---

## 2. 一键运行

```bash
cd examples/vqe_vrptw_pipeline_demo
python main.py --clean --maxiter 120 --seed 42
```

常用参数：

```bash
python main.py \
  --optimizer COBYLA \
  --maxiter 120 \
  --ansatz-reps 2 \
  --wait-weight 0.2 \
  --late-weight 8.0 \
  --seed 42 \
  --clean
```

默认问题是 1 辆车服务 3 个客户。程序会枚举 6 条候选路线，把“选择哪条路线”建成 6 个 qubit 的 QUBO/Ising Hamiltonian，再用 VQE 找最低能量态。代码内部会把路线成本除以最大路线成本做归一化，让 COBYLA 这类无梯度优化器更容易收敛；`routes.json` 和 `metrics.json` 仍保留原始路线成本。

这个 demo 是“先枚举完整路线，再选择一条路线”的 toy problem，不是通用 VRPTW 求解器。它没有处理多车、容量约束、大规模路线集合或列生成；如果客户数变多，不能简单依赖全排列枚举。

---

## 3. 目录结构

```text
examples/vqe_vrptw_pipeline_demo/
├── README.md
├── EXPLAINER.md
├── environment.yml
├── main.py
├── pipeline_utils.py
├── tasks/
│   ├── task_01_define_instance.py
│   ├── task_02_enumerate_routes.py
│   ├── task_03_build_qubo.py
│   ├── task_04_exact_reference.py
│   ├── task_05_build_ansatz.py
│   ├── task_06_initial_quantum.py
│   ├── task_07_run_vqe.py
│   └── task_08_visualize_summary.py
└── artifacts/                 # 运行后自动生成
```

---

## 4. 子任务说明

| # | 脚本 | 类型 | 输入 | 输出 | 作用 |
|---|---|---|---|---|---|
| 1 | `task_01_define_instance.py` | CLASSICAL | CLI 参数 | `instance.json`, `01_instance.png` | 定义 depot、客户坐标、服务时间和时间窗 |
| 2 | `task_02_enumerate_routes.py` | CLASSICAL | `instance.json` | `routes.json`, `routes.csv`, `02_routes.png` | 枚举 6 条访问顺序，计算距离、等待、迟到和路线代价 |
| 3 | `task_03_build_qubo.py` | CLASSICAL | `routes.json` | `qubo.json`, `hamiltonian.json`, `hamiltonian.dill`, `03_qubo_hamiltonian.png` | 构造 one-hot 路线选择 QUBO，并映射为 Pauli-Z Hamiltonian |
| 4 | `task_04_exact_reference.py` | CLASSICAL | `qubo.json`, `routes.json` | `reference.json`, `04_exact_reference.png` | 穷举 64 个 bitstring，得到经典精确最优路线 |
| 5 | `task_05_build_ansatz.py` | CLASSICAL | `hamiltonian.json` | `ansatz.dill`, `ansatz.json`, `initial_point.npy`, `05_ansatz_circuit.png` | 构建 Hadamard + 无纠缠 RealAmplitudes 参数化线路 |
| 6 | `task_06_initial_quantum.py` | QUANTUM | Hamiltonian、ansatz、初始参数 | `initial_quantum.json`, `06_initial_quantum.png` | 本地模拟器计算初始期望值和初始路线概率 |
| 7 | `task_07_run_vqe.py` | HYBRID | Hamiltonian、ansatz、参考答案 | `vqe_result.json`, `vqe_result.dill`, `vqe_trace.csv`, `07_vqe_convergence.png` | 运行 VQE：量子估能量，经典优化器调参数 |
| 8 | `task_08_visualize_summary.py` | CLASSICAL | 上游 JSON/CSV | `metrics.json`, `08_summary_dashboard.png`, `09_best_route_map.png` | 汇总指标，展示最终路线和 VQE 收敛结果 |

---

## 5. 单独运行某个 task

每个 task 都可以独立运行，方便调试，也方便迁移到工作流编排引擎。

例如只重跑 VQE：

```bash
python tasks/task_07_run_vqe.py \
  --hamiltonian-dill artifacts/qubo/hamiltonian.dill \
  --qubo artifacts/qubo/qubo.json \
  --routes artifacts/routes/routes.json \
  --ansatz artifacts/ansatz/ansatz.dill \
  --initial-point artifacts/ansatz/initial_point.npy \
  --reference artifacts/results/reference.json \
  --output-json artifacts/results/vqe_result.json \
  --output-dill artifacts/results/vqe_result.dill \
  --trace-output artifacts/results/vqe_trace.csv \
  --figure-output artifacts/figures/07_vqe_convergence.png \
  --optimizer COBYLA \
  --maxiter 120 \
  --seed 42
```

查看任意任务参数：

```bash
python tasks/task_07_run_vqe.py --help
```

---

## 6. 输出结果

运行完成后重点查看：

```text
artifacts/metrics.json
artifacts/results/vqe_result.json
artifacts/results/vqe_trace.csv
artifacts/figures/08_summary_dashboard.png
artifacts/figures/09_best_route_map.png
artifacts/environment.lock.yml
```

`metrics.json` 中最重要的字段：

- `exact_best_route_id`：经典穷举得到的最低代价路线。
- `exact_best_route_cost`：这条路线的原始路线成本。
- `vqe_best_probability_route_id`：VQE 最终态中概率最高的一条 one-hot 路线。
- `vqe_route_cost`：VQE 最高概率路线的原始路线成本。
- `route_cost_gap`：VQE 最高概率路线成本与精确最优路线成本的差。
- `one_hot_success_probability`：最终态落在合法 one-hot 路线上的总概率。
- `invalid_probability`：最终态落在非法 bitstring 上的概率。
- `vqe_matches_exact_best_route`：VQE 是否选中了经典精确最优路线。
- `vqe_energy_lower_than_initial`：VQE 优化后能量是否低于初始量子态。
- `route_probabilities`：每条候选路线在最终量子态中的概率。

`qubo.json` 中的 `penalty` 默认自动设为 `2.0`。因为路线成本已经归一化到 `[0, 1]`，这个惩罚足以让 one-hot 约束压过选多条或不选路线的非法状态。换成别的实例后，应重新检查 `invalid_probability` 和精确枚举结果，确认 penalty 仍然足够。

---

## 7. 为什么这里是“量子-经典混合”

本示例把 VRPTW 转成 Hamiltonian 后，VQE 的循环可以理解成：

```text
经典端给出参数 theta
        ↓
量子端运行 ansatz(theta)，测路线选择 Hamiltonian 的期望值 E(theta)
        ↓
经典优化器根据 E(theta) 更新 theta
        ↓
重复，直到能量尽量低
```

所以：

- `task_06` 是 `[QUANTUM]`：它真正调用 estimator 计算量子期望值。
- `task_07` 是 `[HYBRID]`：它把“量子估能量”和“经典优化参数”放进同一个迭代循环。
- 其他任务主要是经典预处理、经典参考计算或结果可视化。

---

## 8. 迁移到 Flyte 的思路

本示例暂不新增完整 Flyte workflow，但任务边界已经按 Flyte 友好方式设计。迁移时可以把每个 CLI 包成一个 `@task`：

```python
from flytekit import task, workflow
from flytekit.types.file import FlyteFile
import subprocess
import sys

@task
def t03_build_qubo(routes: FlyteFile) -> FlyteFile:
    out = "/tmp/qubo.json"
    ham = "/tmp/hamiltonian.json"
    ham_dill = "/tmp/hamiltonian.dill"
    fig = "/tmp/qubo.png"
    subprocess.run(
        [
            sys.executable,
            "tasks/task_03_build_qubo.py",
            "--routes",
            routes,
            "--qubo-output",
            out,
            "--hamiltonian-output",
            ham,
            "--hamiltonian-dill",
            ham_dill,
            "--figure-output",
            fig,
        ],
        check=True,
    )
    return FlyteFile(out)
```

生产化时建议把 JSON、CSV、DILL、PNG 都作为明确的 Flyte 输入输出对象传递，而不是让任务共享本地目录。

---

## 9. 切换到真实量子硬件

当前示例的 estimator 创建集中在 [pipeline_utils.py](pipeline_utils.py) 的 `create_estimator()` 中。默认实现是：

```python
from qiskit.primitives import StatevectorEstimator
return StatevectorEstimator(seed=seed)
```

以后要接真实量子硬件，可以把这里替换成 IBM Runtime 的 `EstimatorV2`，并在 `main.py` / task 参数里加入 backend、shots、session 等配置。上游的 VRPTW 建模、QUBO、Hamiltonian 和下游评估逻辑不需要大改。
