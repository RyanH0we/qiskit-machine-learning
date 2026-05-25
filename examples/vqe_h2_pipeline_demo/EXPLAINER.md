# VQE 计算 H2 基态能量：零基础讲解

这份讲解假设你没有量子计算背景。目标不是把所有物理和数学细节讲完，而是让你能看懂本示例每个 task 为什么存在、输入输出是什么、哪些地方是经典计算、哪些地方是真正的量子计算。

---

## 1. 我们到底在算什么

氢分子 H2 由两个氢原子组成。分子的“基态能量”可以理解为：

> 这个分子在最稳定、最低能量状态下的总能量。

在量子化学里，电子不是沿固定轨道转圈的小球，而是由波函数描述。要求基态能量，本质上是在求一个 Hamiltonian 的最小本征值。

Hamiltonian 可以先粗略理解为“能量规则表”：给它一个量子态，它能告诉你这个量子态对应的能量期望值。

---

## 2. 为什么需要 VQE

直接精确求解分子 Hamiltonian 会随着分子变大迅速变难。H2 很小，经典电脑可以轻松精确求解；但我们故意用它做 VQE 示例，因为它足够简单，适合入门。

VQE 的核心想法是：

1. 准备一个带参数的量子线路，叫 ansatz。
2. 给参数一个初始值，在线路上制备一个候选量子态。
3. 测量这个态的能量。
4. 经典优化器根据能量调整参数。
5. 重复直到能量尽量低。

用一句话说：

> 量子计算机负责“给定参数时估计能量”，经典计算机负责“决定下一组参数”。

---

## 3. 关键概念

### Hamiltonian

Hamiltonian 是系统能量的数学表示。对 H2 来说，它包含：

- 电子的动能；
- 电子和原子核之间的吸引；
- 电子之间的排斥；
- 两个原子核之间的排斥。

量子计算机不能直接吃“电子 Hamiltonian”，所以我们要先把它变成 qubit 上的 Pauli 算符。

### Jordan-Wigner 映射

电子是费米子，量子计算机的基本对象是 qubit。Jordan-Wigner 映射的作用是：

> 把电子问题翻译成量子比特问题。

本示例里，H2 在 `sto3g` 基组下会变成 4 个 qubit 上的 Pauli Hamiltonian。

### Ansatz

Ansatz 是“试探波函数”的线路模板。它不是唯一的，不同 ansatz 表示不同搜索空间。

本示例使用：

- `HartreeFock`：一个合理的经典初始态；
- `UCCSD`：量子化学中常用、物理含义较强的 ansatz。

### Chemical accuracy

化学里常用 `0.0016 Hartree` 作为一个经验精度门槛。误差小于这个值，通常就说达到了 chemical accuracy。

---

## 4. 8 个 task 在做什么

### Task 01：定义分子

输入是键长、基组、电荷、自旋。输出 `molecule.json`。

这一步没有量子计算，只是在告诉程序：我们要研究的是 H2，两个氢原子相距 `0.735 Å`。

### Task 02：构造 Hamiltonian

输入 `molecule.json`，输出 `problem.dill` 和 `hamiltonian.json`。

这一步做两件事：

1. PySCF 计算分子的电子结构积分。
2. Jordan-Wigner 把电子 Hamiltonian 映射成 qubit Hamiltonian。

这仍然是经典计算。输出图 `02_hamiltonian_terms.png` 展示每个 Pauli 项的系数。

### Task 03：计算精确参考

输入 `problem.dill`，输出 `reference.json`。

H2 很小，所以可以用经典 NumPy 精确对角化得到参考答案。这个答案用来检查 VQE 算得准不准。

注意：这里精确对角化的是已经映射到 qubit 的 4-qubit Hamiltonian。这个做法在 H2/STO-3G 上可行，也方便教学；分子变大后 Hilbert 空间会快速膨胀，通常不能再用这种全空间精确对角化做日常参考。

### Task 04：构建 ansatz

输入 `problem.dill`，输出 `ansatz.dill`、`initial_point.npy` 和线路图。

这里创建 UCCSD 参数化线路。初始参数全为 0，表示从 Hartree-Fock 态开始。

### Task 05：评估初始能量

输入 Hamiltonian、ansatz 和初始参数，输出 `initial_energy.json`。

这是第一次真正调用量子 primitive。默认使用本地 `StatevectorEstimator`，所以它是模拟器，不是真机。

这一步回答的问题是：

> 如果不优化参数，只用初始 Hartree-Fock 态，能量是多少？

默认配置下，这个初始总能量应等于 `reference.json` 中的 Hartree-Fock 总能量。这个等式是一个很有用的 sanity check：如果两者不一致，应优先检查 ansatz 初态、初始参数和核排斥能是否处理一致。

### Task 06：运行 VQE

输入 Hamiltonian、ansatz、初始参数，输出 VQE 结果和收敛曲线。

这是混合计算核心：

- 量子侧：计算当前参数下的能量期望值。
- 经典侧：SLSQP 优化器决定下一组参数。

`vqe_trace.csv` 记录每次能量评估，`06_vqe_convergence.png` 展示能量如何下降。

另一个容易混淆的点是线路深度：`UCCSD` 在 Qiskit 中首先是一个高层线路模板，原始模板深度可能显示得很小；把它展开到基础门后，默认 H2 线路约为深度 112，并包含 56 个 `cx`。做硬件资源估算时应看展开后的门级线路。

### Task 07：汇总指标

输入参考能量、初始能量、VQE 能量，输出 `metrics.json`。

它会计算：

- VQE 和精确参考之间的误差；
- 是否小于 `1e-3 Hartree`；
- 是否达到 chemical accuracy。

### Task 08：最终总览图

输入前面所有关键结果，输出 `07_summary_dashboard.png`。

这张图把分子结构、能量对比、VQE 收敛、Hamiltonian Pauli 项放在一起，方便新手快速检查整个流程。

---

## 5. 为什么 H2 适合作为第一个 VQE 例子

H2 有几个优点：

- 物理意义清楚：两个氢原子组成一个分子。
- qubit 数少：默认设置下只需要 4 个 qubit。
- 经典精确答案容易得到，方便验证。
- VQE 通常能收敛到非常接近精确答案。

它不是为了证明量子计算已经超过经典计算，而是为了展示 VQE 的标准工作流。

---

## 6. 读结果时看什么

最重要的是 `artifacts/metrics.json`：

```json
{
  "vqe_within_chemical_accuracy": true,
  "errors_hartree": {
    "vqe_abs_error": 0.0000000001
  }
}
```

如果 `vqe_within_chemical_accuracy` 是 `true`，说明 VQE 结果已经非常接近精确参考。

再看 `artifacts/figures/06_vqe_convergence.png`：曲线越往下，表示优化器找到了能量更低的量子态。

---

## 7. 接下来可以改什么

你可以尝试：

- 改 `--bond-length`，观察 H2 拉伸后能量如何变化。
- 把 `--optimizer` 改成 `COBYLA` 或 `L_BFGS_B`，比较收敛速度。
- 增大 `--maxiter`，观察是否能更稳定地达到精确参考。
- 在 `pipeline_utils.py` 中替换 `create_estimator()`，尝试接入真实量子硬件。

这些实验会帮助你理解：VQE 的结果不只取决于量子线路，也取决于 ansatz、优化器、初始点和后端噪声。
