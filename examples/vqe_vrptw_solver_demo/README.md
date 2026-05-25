# VQE 求解通用 VRPTW 的任务化示例

这是一个面向新手的量子-经典混合计算示例：用 **VQE 思路** 和本地 Qiskit 量子模拟器求解一个小型 **VRPTW（Vehicle Routing Problem with Time Windows，带时间窗车辆路径规划）**。

和 `examples/vqe_vrptw_pipeline_demo/` 不同，本示例不是“先枚举完整路线，再 one-hot 选择一条路线”的 toy problem。这里直接把车辆、访问序位、客户、服务开始时间、容量、路径连续性和时间窗约束建成 QUBO。

默认实例是 `2` 辆车、`4` 个客户、每辆车容量 `2`、每辆车最多 `2` 个 stop。默认 QUBO 有 `20` 个变量，也就是 `20` 个 qubit。

---

## 1. 创建环境

从仓库根目录运行：

```bash
conda create -y -n qml-vrptw-solver python=3.12
conda run -n qml-vrptw-solver python -m pip install -e .
conda run -n qml-vrptw-solver python -m pip install \
  "qiskit>=2,<3" "qiskit-algorithms==0.4.0" "qiskit-aer>=0.17" \
  "numpy>=2.0" "scipy>=1.10" "matplotlib>=3.7" "dill>=0.3.4" "pylatexenc>=2.10"
```

也可以用本目录的环境文件：

```bash
conda env create -f examples/vqe_vrptw_solver_demo/environment.yml
conda activate qml-vrptw-solver
pip install -e .
```

后续测试建议都通过 `conda run -n qml-vrptw-solver ...` 执行，避免和其他示例环境混在一起。

---

## 2. 一键运行

```bash
cd examples/vqe_vrptw_solver_demo
conda run -n qml-vrptw-solver python main.py --clean
```

常用参数：

```bash
conda run -n qml-vrptw-solver python main.py \
  --time-granularity 1 \
  --max-stops-per-vehicle 2 \
  --penalty 0 \
  --ansatz-reps 2 \
  --optimizer COBYLA \
  --maxiter 120 \
  --shots 4096 \
  --seed 42 \
  --clean
```

`--penalty 0` 表示自动设置约束惩罚权重。默认实例会暴力枚举 `2^20` 个 bitstring 作为经典参考，因此第一次运行大约需要几十秒。

---

## 3. 子任务说明

| # | 脚本 | 类型 | 输入 | 输出 | 作用 |
|---|---|---|---|---|---|
| 1 | `task_01_define_or_load_instance.py` | CLASSICAL | CLI 或自定义 JSON | `instance.json`, `01_instance.png` | 定义/读取 VRPTW 实例，补全距离和 travel time 矩阵 |
| 2 | `task_02_build_time_indexed_qubo.py` | CLASSICAL | `instance.json` | `qubo.json`, `hamiltonian.json`, `hamiltonian.dill`, `02_qubo_hamiltonian.png` | 构造时间索引 QUBO，并映射成 Pauli-Z Hamiltonian |
| 3 | `task_03_exact_reference.py` | CLASSICAL | `instance.json`, `qubo.json` | `reference.json`, `03_reference_routes.png` | 小规模暴力枚举，得到经典参考解 |
| 4 | `task_04_build_ansatz.py` | CLASSICAL | Hamiltonian/QUBO 元数据 | `ansatz.dill`, `ansatz.json`, `initial_point.npy`, `04_ansatz_circuit.png` | 构建 RY product ansatz 和教学 warm start |
| 5 | `task_05_initial_energy.py` | QUANTUM | Hamiltonian、ansatz、初始参数 | `initial_energy.json`, `05_initial_energy.png` | 使用本地 `StatevectorSampler` 对初始线路采样 |
| 6 | `task_06_run_vqe.py` | HYBRID | Hamiltonian、ansatz、参考解 | `vqe_result.json`, `vqe_result.dill`, `vqe_trace.csv`, `06_vqe_convergence.png` | 经典优化器更新参数，量子线路采样得到最终候选 |
| 7 | `task_07_decode_solution.py` | CLASSICAL | VQE 结果、QUBO、实例 | `decoded_solution.json`, `samples.csv`, `solution_routes.csv`, `07_solution_probabilities.png`, `08_vqe_routes.png` | 把 bitstring 解码成车辆路径，并检查约束 |
| 8 | `task_08_evaluate.py` | CLASSICAL | reference、initial、VQE、decoded | `metrics.json` | 汇总可行性、能量、距离和验收指标 |
| 9 | `task_09_visualize_summary.py` | CLASSICAL | 全部关键产物 | `09_summary_dashboard.png` | 生成最终可视化总览图 |

这些 task 都是独立 CLI，只通过文件传递结果，后续接入 Flyte 时可以把每个脚本包装成一个 `@task`。

---

## 4. 单独运行某个 task

例如只重跑 VQE：

```bash
conda run -n qml-vrptw-solver python tasks/task_06_run_vqe.py \
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
conda run -n qml-vrptw-solver python tasks/task_06_run_vqe.py --help
```

---

## 5. 输出结果

重点查看：

```text
artifacts/metrics.json
artifacts/results/decoded_solution.json
artifacts/results/samples.csv
artifacts/results/solution_routes.csv
artifacts/figures/09_summary_dashboard.png
artifacts/environment.lock.yml
```

`metrics.json` 中常用字段：

- `reference.best_feasible_total_distance`：经典暴力枚举得到的参考最优总距离。
- `vqe.mean_qubo_energy`：VQE 最终参数下的采样/期望均值能量，不等于某个单独路径的距离。
- `decoded.best_feasible_found`：VQE 最终样本中是否出现可行 VRPTW 路径。
- `decoded.best_feasible_total_distance`：VQE 样本里最好的可行路径总距离。
- `acceptance.all_customers_served_once`：是否每个客户恰好服务一次。
- `acceptance.capacity_feasible`：车辆容量是否满足。
- `acceptance.time_window_feasible`：服务开始时间是否满足客户时间窗。
- `acceptance.returns_to_depot`：车辆是否能在 depot 时间窗内返回。

注意：VQE 的均值能量、最高概率样本、best measurement、经典参考解是不同概念。这个示例会把它们分开记录，避免把“采样中出现可行/最优信号”误读成“整个概率分布已经完全集中到最优解”。

---

## 6. 切换到真实量子硬件

当前示例只使用本地模拟器。量子后端入口集中在 [pipeline_utils.py](pipeline_utils.py) 的 `create_sampler()`：

```python
from qiskit.primitives import StatevectorSampler

return StatevectorSampler(default_shots=shots, seed=seed)
```

以后要接真实量子计算机时，可以把这里替换为 IBM Runtime 的 SamplerV2，并把 backend、session、shots 等参数从 CLI 传进来。其他 task 的文件边界不需要大改。

---

## 7. 适用范围

这是一个教学级、小规模、可视化友好的 VRPTW-QUBO solver 示例。它展示的是“如何把 VRPTW 直接建模成 QUBO，并用量子-经典混合流程求候选解”。它不是工业级大规模 VRPTW 求解器，也不表示当前 VQE 能直接替代成熟的 OR-Tools、列生成或大型混合整数规划求解器。
