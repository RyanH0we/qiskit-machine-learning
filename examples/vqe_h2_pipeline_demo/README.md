# VQE 计算氢分子 H2 基态能量示例

这是一个面向新手的量子-经典混合计算示例：用 **VQE（Variational Quantum Eigensolver，变分量子本征求解器）** 估算氢分子 H2 的基态能量。

本示例遵循现有 `examples/hybrid_pipeline_demo/` 的任务化风格：

1. 每一步都是 `tasks/*.py` 中的独立 CLI。
2. 任务之间只通过 `artifacts/` 下的文件传递结果。
3. 每个任务启动时都会标出 `[CLASSICAL]`、`[QUANTUM]` 或 `[HYBRID]`。
4. 运行结束会生成 JSON、CSV、DILL 和 PNG 可视化结果。

如果你还不了解 VQE、Hamiltonian、ansatz 或量子化学，建议先读 [EXPLAINER.md](EXPLAINER.md)。

---

## 1. 环境创建

从仓库根目录运行：

```bash
conda env create -f examples/vqe_h2_pipeline_demo/environment.yml
conda activate qml-vqe-h2
pip install -e .
```

如果你已经进入本示例目录：

```bash
conda env create -f environment.yml
conda activate qml-vqe-h2
pip install -e ../..
```

本示例默认使用本地 `StatevectorEstimator` 模拟器，不连接真实量子计算机。

---

## 2. 一键运行

```bash
cd examples/vqe_h2_pipeline_demo
python main.py --clean
```

常用参数：

```bash
python main.py \
  --bond-length 0.735 \
  --basis sto3g \
  --optimizer SLSQP \
  --maxiter 100 \
  --seed 42 \
  --clean
```

默认设置下，H2 的精确参考总能量约为 `-1.137306 Hartree`，VQE 结果应非常接近该值。

这里的“精确参考”是对当前 4-qubit Hamiltonian 做经典精确对角化。H2/STO-3G 很小，所以这一步适合作为教学校验；更大的分子不能依赖这种全空间精确对角化。

---

## 3. 目录结构

```text
examples/vqe_h2_pipeline_demo/
├── README.md
├── EXPLAINER.md
├── environment.yml
├── main.py
├── pipeline_utils.py
├── tasks/
│   ├── task_01_define_molecule.py
│   ├── task_02_build_hamiltonian.py
│   ├── task_03_exact_reference.py
│   ├── task_04_build_ansatz.py
│   ├── task_05_initial_energy.py
│   ├── task_06_run_vqe.py
│   ├── task_07_evaluate.py
│   └── task_08_visualize_summary.py
└── artifacts/                 # 运行后自动生成
```

---

## 4. 子任务说明

| # | 脚本 | 类型 | 输入 | 输出 | 作用 |
|---|---|---|---|---|---|
| 1 | `task_01_define_molecule.py` | CLASSICAL | CLI 参数 | `molecule.json`, `01_molecule.png` | 定义 H2 几何、键长、基组、电荷、自旋 |
| 2 | `task_02_build_hamiltonian.py` | CLASSICAL | `molecule.json` | `problem.dill`, `hamiltonian.json`, `02_hamiltonian_terms.png` | 用 PySCF 构造电子结构问题，再用 Jordan-Wigner 映射为 qubit Hamiltonian |
| 3 | `task_03_exact_reference.py` | CLASSICAL | `problem.dill` | `reference.json`, `03_reference_energy.png` | 用经典精确对角化得到参考答案 |
| 4 | `task_04_build_ansatz.py` | CLASSICAL | `problem.dill` | `ansatz.dill`, `ansatz.json`, `initial_point.npy`, `04_ansatz_circuit.png` | 构造 Hartree-Fock 初态和 UCCSD 参数化线路 |
| 5 | `task_05_initial_energy.py` | QUANTUM | `problem.dill`, `ansatz.dill`, `initial_point.npy` | `initial_energy.json`, `05_initial_energy.png` | 本地模拟器第一次计算量子期望值 |
| 6 | `task_06_run_vqe.py` | HYBRID | Hamiltonian、ansatz、初始参数 | `vqe_result.json`, `vqe_result.dill`, `vqe_trace.csv`, `06_vqe_convergence.png` | 运行 VQE：量子估能量，经典优化器调参数 |
| 7 | `task_07_evaluate.py` | CLASSICAL | `reference.json`, `initial_energy.json`, `vqe_result.json` | `metrics.json` | 汇总误差、是否达到 chemical accuracy |
| 8 | `task_08_visualize_summary.py` | CLASSICAL | 上游 JSON/CSV | `07_summary_dashboard.png` | 生成最终可视化总览图 |

---

## 5. 单独运行某个 task

每个 task 都可以独立运行，方便调试，也方便迁移到工作流编排引擎。

例如只重跑 VQE：

```bash
python tasks/task_06_run_vqe.py \
  --problem artifacts/hamiltonian/problem.dill \
  --ansatz artifacts/ansatz/ansatz.dill \
  --initial-point artifacts/ansatz/initial_point.npy \
  --reference artifacts/results/reference.json \
  --output-json artifacts/results/vqe_result.json \
  --output-dill artifacts/results/vqe_result.dill \
  --trace-output artifacts/results/vqe_trace.csv \
  --figure-output artifacts/figures/06_vqe_convergence.png \
  --optimizer SLSQP \
  --maxiter 100 \
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
artifacts/results/vqe_result.json
artifacts/results/vqe_trace.csv
artifacts/figures/07_summary_dashboard.png
artifacts/environment.lock.yml
```

`metrics.json` 中最重要的字段：

- `energies_hartree.vqe`：VQE 得到的 H2 总能量。
- `energies_hartree.exact`：经典精确参考总能量。
- `errors_hartree.vqe_abs_error`：VQE 与精确值的绝对误差。
- `vqe_within_chemical_accuracy`：误差是否小于 `0.0016 Hartree`。

建议复核这些关系：

- `energies_hartree.hartree_fock` 应等于 `energies_hartree.initial_ansatz`，因为初始参数全 0 时就是 Hartree-Fock 初态。
- `errors_hartree.vqe_abs_error` 应小于 `1e-3 Hartree`，默认运行通常会远小于 chemical accuracy。
- `artifacts/ansatz/ansatz.json` 中应显示 `num_qubits=4`、`num_parameters=3`。UCCSD 在 Qiskit 中先保存为高层模板，原始模板深度可能很小；若展开到基础门，当前默认结果约为深度 112、包含 56 个 `cx`。

---

## 7. 为什么这里是“量子-经典混合”

VQE 的循环可以理解成：

```text
经典端给出参数 theta
        ↓
量子端运行 ansatz(theta)，测 Hamiltonian 的期望值 E(theta)
        ↓
经典优化器根据 E(theta) 更新 theta
        ↓
重复，直到能量尽量低
```

所以：

- `task_05` 是 `[QUANTUM]`：它真正调用 estimator 计算量子期望值。
- `task_06` 是 `[HYBRID]`：它把“量子估能量”和“经典优化参数”放进同一个迭代循环。
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
def t01_define_molecule(bond_length: float) -> FlyteFile:
    out = "/tmp/molecule.json"
    fig = "/tmp/molecule.png"
    subprocess.run(
        [
            sys.executable,
            "tasks/task_01_define_molecule.py",
            "--bond-length",
            str(bond_length),
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

当前示例的 estimator 创建集中在 [pipeline_utils.py](pipeline_utils.py) 的 `create_estimator()` 中。默认实现是：

```python
from qiskit.primitives import StatevectorEstimator
return StatevectorEstimator(seed=seed)
```

以后要接真实量子硬件，可以把这里替换成 IBM Runtime 的 `EstimatorV2`，并在 `main.py` / task 参数里加入 backend、shots、session 等配置。示例的上游分子建模、Hamiltonian、ansatz 和下游评估逻辑不需要大改。
