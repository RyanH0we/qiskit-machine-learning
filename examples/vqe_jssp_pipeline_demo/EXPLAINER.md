# 用 VQE 求解 JSSP：零基础讲解

这份讲解假设你没有量子计算背景，也不熟悉调度优化。目标不是把所有数学细节讲完，而是让你能看懂本示例每个 task 为什么存在、输入输出是什么、哪些地方是经典计算、哪些地方是真正的量子计算。

---

## 1. 我们到底在算什么

JSSP 是 Job Shop Scheduling Problem，中文常叫“车间调度问题”。你可以把它想成：

> 有一些作业，每个作业由多道工序组成；每道工序必须在指定机器上运行一段时间；同一台机器同一时刻只能做一件事；同一个作业的工序必须按顺序做完。我们希望总完工时间尽量短。

本示例默认只有 2 个作业、2 台机器：

```text
Job 0: O0 在 M0 上运行 2 个时间片 -> O1 在 M1 上运行 1 个时间片
Job 1: O0 在 M1 上运行 1 个时间片 -> O1 在 M0 上运行 2 个时间片
```

这个小问题的最优 makespan 是 `4`。makespan 可以理解为“最后一道工序完成的时间”。

---

## 2. 为什么要先变成 QUBO

量子线路不能直接理解“作业、机器、工序”这些业务对象。我们需要把调度问题翻译成二进制变量：

```text
x_{job,op,t} = 1  表示某个工序在时间 t 开始
x_{job,op,t} = 0  表示它不在时间 t 开始
```

例如 `x_j0_o0_t1 = 1` 表示 Job 0 的第 0 道工序从时间 1 开始。

还需要一组变量选择 makespan：

```text
cmax_4 = 1  表示选择 Cmax = 4
cmax_5 = 1  表示选择 Cmax = 5
```

QUBO 的全称是 Quadratic Unconstrained Binary Optimization，可以粗略理解为：

> 写出一个只包含 0/1 变量的一元项和二元项的能量函数。能量越低，排程越好。

---

## 3. 约束如何变成“惩罚”

原始 JSSP 有很多约束：

- 每道工序必须且只能选择一个开始时间。
- 每个作业内部，后一工序不能早于前一工序完成。
- 同一台机器上的工序不能时间重叠。
- 所有工序必须在选定的 Cmax 前完成。

QUBO 的做法不是单独保存“约束”，而是把违反约束的情况变成高能量惩罚。

例如“某道工序只能开始一次”可以写成：

```text
penalty * (x_t0 + x_t1 + x_t2 - 1)^2
```

如果刚好选了一个开始时间，这一项为 0；如果一个都没选或选了多个，这一项就会变大。

机器冲突和工序顺序也类似：只要某两个选择会导致非法排程，就加入：

```text
penalty * x_a * x_b
```

当两个非法选择同时为 1 时，能量就被加高。

---

## 4. QUBO 如何变成 Ising Hamiltonian

量子计算里常用 Hamiltonian 表示能量规则。为了让量子线路能评估 QUBO，我们把每个二进制变量映射到一个 qubit：

```text
x = (1 - Z) / 2
```

这里的 `Z` 是 Pauli-Z 算符。直观理解：

- qubit 测到 `0` 时，`Z` 的本征值是 `+1`，所以 `x = 0`。
- qubit 测到 `1` 时，`Z` 的本征值是 `-1`，所以 `x = 1`。

这样，QUBO 能量函数就能被改写成一串 Pauli 项，例如：

```text
3.5 * I
-1.2 * Z0
 2.5 * Z0 Z3
```

`task_02_build_qubo.py` 会保存 `qubo.json` 和 `hamiltonian.json`，让你能看到变量、系数和 Pauli 项。

---

## 5. VQE 在这里做什么

VQE 的核心思想是：

1. 准备一个带参数的量子线路，叫 ansatz。
2. 给参数一组初始值。
3. 运行线路，采样 bitstring。
4. 用 Hamiltonian/QUBO 给这些 bitstring 计算平均能量。
5. 经典优化器根据能量调整参数。
6. 重复，直到采样分布更偏向低能量 bitstring。

本示例使用的是 `SamplingVQE`。它适合组合优化问题，因为组合优化最终要读出来的就是 bitstring，而 bitstring 可以直接解码成排程。

---

## 6. 9 个 task 在做什么

### Task 01：定义 JSSP 实例

输入是 `horizon`。输出 `instance.json`。

这一步只是写下问题本身：有几个作业、几台机器、每道工序在哪台机器上运行多久。

### Task 02：构造 QUBO 和 Hamiltonian

输入 `instance.json`，输出 `qubo.json`、`hamiltonian.dill` 和 `hamiltonian.json`。

这一步是经典建模：

1. 生成时间索引变量。
2. 写入目标函数：尽量选择更小的 Cmax。
3. 写入约束惩罚项。
4. 用 `x=(1-Z)/2` 映射成 Ising Hamiltonian。

### Task 03：计算经典参考

输入 `instance.json` 和 `qubo.json`，输出 `reference.json`。

默认实例只有 14 个变量，所以可以枚举 `2^14 = 16384` 个 bitstring。这样能得到可靠参考答案，方便判断 VQE 结果好不好。

### Task 04：构建 ansatz

输入 `hamiltonian.json`，输出 `ansatz.dill`、`ansatz.json` 和 `initial_point.npy`。

这里创建一个简单的 RY product 参数化线路。JSSP-QUBO 的 Hamiltonian 是对角的，RY product ansatz 可以直接控制每个 bit 被采样为 1 的概率，适合作为新手示例。构建线路本身仍然是经典工作。

### Task 05：评估初始能量

输入 Hamiltonian、ansatz 和初始参数，输出 `initial_energy.json`。

这是第一次真正调用量子 primitive。默认使用本地 `StatevectorSampler`，所以它是模拟器，不是真机。

这一步回答的问题是：

> 还没优化参数时，量子线路采样出来的 bitstring 平均能量是多少？

### Task 06：运行 VQE

输入 Hamiltonian、ansatz、初始参数，输出 VQE 结果和收敛曲线。

这是混合计算核心：

- 量子侧：采样当前参数下的 bitstring 分布，并估计 QUBO 能量。
- 经典侧：COBYLA / SLSQP / L-BFGS-B 优化器决定下一组参数。

`vqe_trace.csv` 记录每次能量评估，`06_vqe_convergence.png` 展示能量如何变化。

### Task 07：解码 bitstring

输入 VQE 结果、QUBO 和实例，输出 `decoded_solution.json`、`samples.csv` 和甘特图。

VQE 不会直接输出“排程”，它输出的是 bitstring。解码步骤会检查：

- 每道工序是否刚好选择一个开始时间。
- 是否满足作业顺序。
- 同一机器上是否有重叠。
- makespan 是否符合选择的 Cmax。

### Task 08：汇总指标

输入参考答案、初始能量、VQE 结果和解码结果，输出 `metrics.json`。

它会告诉你：

- VQE 样本中是否找到可行排程。
- 最佳可行排程 makespan 是多少。
- 是否达到经典最优 makespan。

### Task 09：最终总览图

输入前面所有关键结果，输出 `09_summary_dashboard.png`。

这张图把问题规模、能量对比、VQE 收敛、候选 bitstring、VQE 甘特图和 Hamiltonian Pauli 项放在一起，方便新手快速检查整个流程。

---

## 7. 为什么 VQE 不一定每次都“完美”

这个示例不是要证明量子计算已经比经典调度算法更强。默认问题很小，经典枚举反而最稳。

VQE 的结果会受到这些因素影响：

- ansatz 是否容易表示好解；
- 优化器是否找到好参数；
- shots 是否足够；
- penalty 是否足够大；
- 量子硬件或模拟器是否有噪声。

所以你应该同时看两个东西：

1. `vqe_mean_qubo_energy`：VQE 的采样平均能量是否下降。
2. `decoded_solution.json`：采样分布里是否真的出现了可行好排程。

组合优化里，第二点经常比平均能量更直观。

---

## 8. 接下来可以改什么

你可以尝试：

- 增大 `--maxiter`，观察 VQE 是否更稳定地找到低能量候选。
- 增大 `--shots`，减少采样随机性。
- 改 `--penalty`，观察惩罚太小会不会让非法解看起来更“便宜”。
- 改 `--ansatz-reps`，比较线路表达能力和优化难度。
- 在 `pipeline_utils.py` 中替换 `create_sampler()`，尝试接入真实量子硬件。

这些实验会帮助你理解：VQE 的结果不只取决于量子线路，也取决于建模、惩罚权重、优化器、初始点和采样次数。
