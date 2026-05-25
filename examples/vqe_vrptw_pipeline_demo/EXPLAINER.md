# VQE 求解 VRPTW：零基础讲解

这份讲解假设你没有量子计算背景。目标不是把所有数学细节讲完，而是让你看懂本示例每个 task 为什么存在、输入输出是什么、哪些地方是经典计算、哪些地方是真正的量子计算。

---

## 1. 我们到底在算什么

VRPTW 的全称是 Vehicle Routing Problem with Time Windows，中文常叫“带时间窗的车辆路径规划”。

你可以把它想成一个配送问题：

> 一辆车从仓库出发，要访问几个客户。每个客户都有允许服务的时间窗，比如 A 必须在 2.5 到 6.5 之间服务。我们想找到一条总成本最低、尽量不迟到的路线。

本示例为了新手友好，只做最小规模：

- 1 个仓库 depot；
- 3 个客户 A、B、C；
- 1 辆车；
- 6 条候选访问顺序：A-B-C、A-C-B、B-A-C、B-C-A、C-A-B、C-B-A。

真实 VRPTW 会更大、更复杂，但这个小实例已经足够展示“经典建模 -> QUBO -> 量子 Hamiltonian -> VQE”的完整工作流。

也要注意边界：本示例不是通用 VRPTW 求解器。它把 3 个客户的 6 条完整访问顺序全部枚举出来，再用量子优化选择其中一条；它没有覆盖多车、容量约束、大规模路线集合或列生成。

---

## 2. 为什么要先枚举候选路线

如果直接把 VRPTW 的每个客户、每个位置、每个时间都做成变量，模型会很快变大，不适合零基础入门。

本示例采用更直观的“路线列选择”方式：

1. 先用经典程序枚举所有候选路线。
2. 对每条路线计算成本：距离 + 等待惩罚 + 迟到惩罚。
3. 让量子优化只回答一个问题：应该选择哪一条路线？

因此我们有 6 个二进制变量：

```text
x_R01 = 1 表示选择路线 R01
x_R02 = 1 表示选择路线 R02
...
x_R06 = 1 表示选择路线 R06
```

最终我们希望“恰好一个变量为 1”，也就是只选择一条路线。

---

## 3. QUBO 是什么

QUBO 的全称是 Quadratic Unconstrained Binary Optimization，意思是：

> 只用 0/1 变量，并且目标函数最多包含二次项的无约束优化问题。

本示例的 QUBO 是：

```text
min  sum(normalized_route_cost[r] * x_r) + penalty * (sum(x_r) - 1)^2
```

前半部分 `sum(normalized_route_cost[r] * x_r)` 表示“选中的路线越便宜越好”。代码里会把每条路线成本除以最大路线成本做归一化，这不会改变哪条路线最优，但会让 Hamiltonian 的数值尺度更温和，方便默认的 COBYLA 优化器收敛。

后半部分 `penalty * (sum(x_r) - 1)^2` 是 one-hot 约束惩罚：

- 如果没有选路线，`sum(x)=0`，会被罚；
- 如果选了两条或更多路线，`sum(x)>1`，会被罚；
- 如果恰好选一条路线，`sum(x)=1`，惩罚为 0。

这就是为什么它是“无约束”的：约束没有消失，而是被放进了目标函数里。

默认 `penalty=2.0` 是配合本 toy problem 的归一化成本使用的。因为每条路线成本都除以最大路线成本，one-hot 路线的目标值落在 `[0, 1]`；`penalty=2.0` 足以让“不选路线”或“选多条路线”变得更贵。换成新实例时，应该重新用精确枚举或指标中的 `invalid_probability` 检查这个惩罚是否仍然足够。

---

## 4. Hamiltonian 又是什么

在 VQE 里，我们需要一个 Hamiltonian。你可以先把 Hamiltonian 理解为：

> 一个能量规则表。给它一个量子态，它返回这个量子态对应的能量期望值。

为了把 QUBO 交给量子线路，我们使用标准替换：

```text
x_i = (1 - Z_i) / 2
```

这里 `Z_i` 是第 i 个 qubit 上的 Pauli-Z 算符。这个替换的含义很朴素：

- qubit 测到 0 时，变量 x_i 等于 0；
- qubit 测到 1 时，变量 x_i 等于 1。

替换后，QUBO 就变成了只含 `I`、`Z`、`ZZ` 项的 Ising Hamiltonian。`task_03` 会把这些 Pauli 项写进 `hamiltonian.json`，也会画出 Pauli 项系数图。

---

## 5. VQE 在这里怎么工作

VQE 的核心循环是：

```text
经典优化器给出一组参数 theta
        ↓
量子线路 ansatz(theta) 制备一个候选量子态
        ↓
Estimator 计算该状态在 Hamiltonian 下的能量期望值
        ↓
经典优化器根据能量更新 theta
        ↓
重复，直到能量尽量低
```

在这个 VRPTW 示例中：

- 低能量 bitstring 对应低成本路线；
- 精确最低能量 bitstring 对应经典最优路线；
- VQE 希望通过参数优化，让量子态把更多概率集中到这条最优路线对应的 bitstring 上。

注意：本示例用的是本地 `StatevectorEstimator` 模拟器。它不会连接真实量子计算机，因此结果没有硬件噪声。

---

## 6. 8 个 task 在做什么

### Task 01：定义 VRPTW 实例

输入是 CLI 输出路径。输出 `instance.json` 和 `01_instance.png`。

这一步没有量子计算。它只定义仓库、客户坐标、服务时间和时间窗。

### Task 02：枚举路线

输入 `instance.json`，输出 `routes.json`、`routes.csv` 和 `02_routes.png`。

程序枚举 6 条路线，对每条路线模拟：

- 到达每个客户的时间；
- 如果太早到，等待多久；
- 如果晚于时间窗，迟到多久；
- 总距离和总代价。

这仍然是经典计算。

### Task 03：构造 QUBO 和 Hamiltonian

输入 `routes.json`，输出 `qubo.json`、`hamiltonian.json`、`hamiltonian.dill` 和图。

这是“把物流问题翻译成量子能量问题”的关键一步。它本身是经典计算，因为只是代数变换。

### Task 04：经典精确参考

输入 `qubo.json` 和 `routes.json`，输出 `reference.json` 和能量地形图。

因为只有 6 个变量，所以可以枚举 64 个 bitstring。这个参考答案用来检查 VQE 是否找到了同一条路线。

### Task 05：构建 ansatz

输入 `hamiltonian.json`，输出 `ansatz.dill`、`ansatz.json`、`initial_point.npy` 和线路图。

Ansatz 是带参数的量子线路模板。本示例使用：

- 一层 Hadamard，把初始态变成“很多路线都可能”的叠加；
- 无纠缠 `RealAmplitudes`，提供可优化参数。

这里不加纠缠层，是因为路线选择 QUBO 的精确解本来就是某个计算基态 bitstring。产品态已经足够表达“只选 R01”这类答案，也更适合新手观察概率从均匀分散到集中在某条路线上的过程。构建线路本身仍然是经典工作。

### Task 06：初始量子评估

输入 Hamiltonian、ansatz 和初始参数，输出 `initial_quantum.json` 和概率图。

这是第一次真正调用量子 primitive。它回答：

> 参数还没有优化时，量子态的平均路线能量是多少？每条路线大概有多少概率？

### Task 07：运行 VQE

输入 Hamiltonian、ansatz、初始参数和参考答案，输出 VQE 结果、trace 和收敛图。

这是混合计算核心：

- 量子侧：计算当前参数下的能量期望值；
- 经典侧：`COBYLA` 优化器决定下一组参数。

`vqe_trace.csv` 记录每次能量评估，`07_vqe_convergence.png` 展示能量如何下降。

### Task 08：最终总览

输入前面所有关键结果，输出 `metrics.json`、`08_summary_dashboard.png` 和 `09_best_route_map.png`。

这一步把问题、路线成本、VQE 收敛曲线和最终路线概率放在一张图里，方便快速检查整个流程。

---

## 7. 读结果时看什么

最重要的是 `artifacts/metrics.json`：

```json
{
  "exact_best_route_id": "R01",
  "vqe_best_probability_route_id": "R01",
  "vqe_matches_exact_best_route": true,
  "vqe_energy_lower_than_initial": true
}
```

如果 `vqe_matches_exact_best_route` 是 `true`，说明 VQE 最终概率最高的路线就是经典精确最优路线。

还建议同时看：

- `route_cost_gap`：VQE 最高概率路线与精确最优路线的原始成本差；
- `one_hot_success_probability`：最终态落在合法 one-hot 路线上的总概率；
- `invalid_probability`：最终态落在非法 bitstring 上的概率。

再看 `artifacts/figures/07_vqe_convergence.png`：曲线越往下，表示经典优化器找到了能量更低的量子态。

最后看 `artifacts/figures/09_best_route_map.png`：它展示最终选出的路线在地图上怎么走。

---

## 8. 为什么这个例子适合入门

这个例子故意很小，不是为了证明量子计算已经超过经典计算，而是为了展示完整工作流：

1. 一个普通优化问题如何写成 QUBO；
2. QUBO 如何变成 qubit Hamiltonian；
3. VQE 如何把“找最小值”变成“找最低能量量子态”；
4. 哪些步骤是经典计算，哪些步骤是真正调用量子模拟器，哪些步骤是量子-经典混合。

当你理解这个小例子后，可以继续尝试：

- 增加客户数量；
- 改变时间窗，让不同路线成为最优；
- 改 `--late-weight`，观察迟到惩罚如何影响路线选择；
- 改 `--optimizer` 或 `--ansatz-reps`，观察 VQE 收敛差异；
- 把 `pipeline_utils.create_estimator()` 换成真实量子硬件的 Estimator。
