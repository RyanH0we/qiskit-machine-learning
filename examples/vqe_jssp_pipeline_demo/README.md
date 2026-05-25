# VQE 求解车间调度 JSSP 的任务化示例

这是一个面向新手的量子-经典混合计算示例：用 **VQE（Variational Quantum Eigensolver，变分量子本征求解器）** 的组合优化版本 `SamplingVQE`，求解一个小型 **车间调度问题（Job Shop Scheduling Problem, JSSP）**。

本示例遵循现有 `examples/vqe_h2_pipeline_demo/` 的任务化风格：

1. 每一步都是 `tasks/*.py` 中的独立 CLI。
2. 任务之间只通过 `artifacts/` 下的文件传递结果。
3. 每个任务启动时都会标出 `[CLASSICAL]`、`[QUANTUM]` 或 `[HYBRID]`。
4. 运行结束会生成 JSON、CSV、DILL 和 PNG 可视化结果。

如果你不了解 JSSP、QUBO、Ising Hamiltonian 或 VQE，建议先读 [EXPLAINER.md](EXPLAINER.md)。

---

## 1. 环境创建

从仓库根目录运行：

```bash
conda env create -f examples/vqe_jssp_pipeline_demo/environment.yml
conda activate qml-vqe-jssp
pip install -e .
```

如果你已经进入本示例目录：

```bash
conda env create -f environment.yml
conda activate qml-vqe-jssp
pip install -e ../..
```

本示例默认使用本地 `StatevectorSampler` 模拟器，不连接真实量子计算机。

---

## 2. 一键运行

```bash
cd examples/vqe_jssp_pipeline_demo
python main.py --clean
```

常用参数：

```bash
python main.py \
  --horizon 5 \
  --penalty 10.0 \
  --ansatz-reps 2 \
  --optimizer COBYLA \
  --maxiter 120 \
  --shots 4096 \
  --seed 42 \
  --clean
```

默认实例是 `2 jobs x 2 machines`，经典最优 makespan 为 `4`，经典最优 QUBO energy 为 `4.0`。由于 VQE 使用采样和经典优化，最终均值能量可能不会精确等于 `4`；`decoded_solution.json` 和 `metrics.json` 会分别检查“是否采到最优 makespan 排程”和“是否达到最优 QUBO energy”。这两个判断不要混在一起：某个排程的实际 makespan 可能是 4，但如果它选择了 `cmax_5`，QUBO energy 仍不是最优。

---

## 3. 目录结构

```text
examples/vqe_jssp_pipeline_demo/
├── README.md
├── EXPLAINER.md
├── environment.yml
├── main.py
├── pipeline_utils.py
├── tasks/
│   ├── task_01_define_instance.py
│   ├── task_02_build_qubo.py
│   ├── task_03_exact_reference.py
│   ├── task_04_build_ansatz.py
│   ├── task_05_initial_energy.py
│   ├── task_06_run_vqe.py
│   ├── task_07_decode_solution.py
│   ├── task_08_evaluate.py
│   └── task_09_visualize_summary.py
└── artifacts/                 # 运行后自动生成
```

---

## 4. 子任务说明

| # | 脚本 | 类型 | 输入 | 输出 | 作用 |
|---|---|---|---|---|---|
| 1 | `task_01_define_instance.py` | CLASSICAL | CLI 参数 | `instance.json`, `01_instance.png` | 定义 2x2 JSSP 实例 |
| 2 | `task_02_build_qubo.py` | CLASSICAL | `instance.json` | `qubo.json`, `hamiltonian.dill`, `hamiltonian.json`, `02_qubo_matrix.png` | 手写时间索引 QUBO，并映射到 Ising Hamiltonian |
| 3 | `task_03_exact_reference.py` | CLASSICAL | `instance.json`, `qubo.json` | `reference.json`, `03_reference_gantt.png` | 经典暴力枚举得到参考最优解 |
| 4 | `task_04_build_ansatz.py` | CLASSICAL | `hamiltonian.json` | `ansatz.dill`, `ansatz.json`, `initial_point.npy`, `04_ansatz_circuit.png` | 构建 RY product 参数化线路 |
| 5 | `task_05_initial_energy.py` | QUANTUM | Hamiltonian、ansatz、初始参数 | `initial_energy.json`, `05_initial_energy.png` | 本地 sampler 初始采样并估计 QUBO 能量 |
| 6 | `task_06_run_vqe.py` | HYBRID | Hamiltonian、ansatz、初始参数 | `vqe_result.json`, `vqe_result.dill`, `vqe_trace.csv`, `06_vqe_convergence.png` | 运行 SamplingVQE：量子采样估能量，经典优化器调参数 |
| 7 | `task_07_decode_solution.py` | CLASSICAL | VQE 结果、QUBO、实例 | `decoded_solution.json`, `samples.csv`, `07_solution_probabilities.png`, `08_vqe_gantt.png` | 把 bitstring 解码成排程并绘制甘特图 |
| 8 | `task_08_evaluate.py` | CLASSICAL | reference、initial、VQE、decoded | `metrics.json` | 汇总可行性、最优性和能量差距 |
| 9 | `task_09_visualize_summary.py` | CLASSICAL | 全部关键 JSON/CSV | `09_summary_dashboard.png` | 生成最终可视化总览图 |

---

## 5. 单独运行某个 task

每个 task 都可以独立运行，方便调试，也方便迁移到工作流编排引擎。

例如只重跑 VQE：

```bash
python tasks/task_06_run_vqe.py \
  --instance artifacts/instance/instance.json \
  --hamiltonian artifacts/qubo/hamiltonian.dill \
  --ansatz artifacts/ansatz/ansatz.dill \
  --initial-point artifacts/ansatz/initial_point.npy \
  --reference artifacts/results/reference.json \
  --output-json artifacts/results/vqe_result.json \
  --output-dill artifacts/results/vqe_result.dill \
  --trace-output artifacts/results/vqe_trace.csv \
  --figure-output artifacts/figures/06_vqe_convergence.png \
  --optimizer COBYLA \
  --maxiter 120 \
  --shots 4096 \
  --seed 42
```

查看任意任务参数：

```bash
python tasks/task_06_run_vqe.py --help
```

---

## 6. 输出结果

运行完成后重点查看：

```text
artifacts/metrics.json
artifacts/results/decoded_solution.json
artifacts/results/samples.csv
artifacts/figures/09_summary_dashboard.png
artifacts/environment.lock.yml
```

`metrics.json` 中最重要的字段：

- `reference.optimal_makespan`：经典暴力枚举得到的最优 makespan，默认应为 `4`。
- `decoded.best_feasible_found`：VQE 样本中是否找到了可行排程。
- `decoded.best_feasible_makespan`：VQE 样本中最好的可行排程 makespan。
- `decoded.best_feasible_makespan_is_optimal`：VQE 样本中最好的可行排程是否达到经典最优 makespan。
- `decoded.best_feasible_qubo_energy_is_optimal`：VQE 样本中最好的可行 bitstring 是否也达到经典最优 QUBO energy。
- `decoded.best_measurement_qubo_energy_is_optimal`：`SamplingVQE` 返回的最佳测量 bitstring 是否达到经典最优 QUBO energy。
- `decoded.most_likely_sample_is_optimal`：最高概率样本是否同时可行且 QUBO 最优。
- `acceptance.vqe_top_candidates_include_makespan_4`：默认实例下，候选解列表是否包含 makespan 为 `4` 的排程。

默认运行可能出现这种情况：最高概率样本是可行但 makespan 为 `5`，低概率样本里包含 makespan 为 `4` 的排程，而 `best_measurement` 达到 QUBO 最优 energy。此时应表述为“采样/最佳测量中出现了最优解信号”，而不是“最终概率分布已经集中到最优排程”。

---

## 7. 为什么这里是“量子-经典混合”

JSSP 的 VQE 循环可以理解成：

```text
经典端给出参数 theta
        ↓
量子端运行 ansatz(theta)，采样 bitstring，并估计 QUBO/Ising 能量
        ↓
经典优化器根据能量更新 theta
        ↓
重复，直到采样分布更偏向低能量排程
```

所以：

- `task_05` 是 `[QUANTUM]`：它真正调用 sampler 对量子线路采样。
- `task_06` 是 `[HYBRID]`：它把“量子采样估能量”和“经典优化参数”放进同一个迭代循环。
- 其他任务主要是经典建模、经典参考计算、解码或可视化。

---

## 8. 迁移到 Flyte 的思路

本示例暂不新增完整 Flyte workflow，但任务边界已经按 Flyte 友好方式设计。迁移时可以把每个 CLI 包成一个 `@task`：

```python
from flytekit import task
from flytekit.types.file import FlyteFile
import subprocess
import sys

@task
def t01_define_instance(horizon: int) -> FlyteFile:
    out = "/tmp/instance.json"
    fig = "/tmp/instance.png"
    subprocess.run(
        [
            sys.executable,
            "tasks/task_01_define_instance.py",
            "--horizon",
            str(horizon),
            "--output",
            out,
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

当前示例的 sampler 创建集中在 [pipeline_utils.py](pipeline_utils.py) 的 `create_sampler()` 中。默认实现是：

```python
from qiskit.primitives import StatevectorSampler
return StatevectorSampler(default_shots=shots, seed=seed)
```

以后要接真实量子硬件，可以把这里替换成 IBM Runtime 的 `SamplerV2`，并在 `main.py` / task 参数里加入 backend、shots、session 等配置。上游 JSSP/QUBO 建模、Hamiltonian、ansatz 和下游解码评估逻辑不需要大改。
