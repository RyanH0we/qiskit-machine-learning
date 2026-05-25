# 用 VQE 求解 VRPTW：零基础讲解

这份文档用尽量直白的方式解释 `vqe_vrptw_solver_demo` 在做什么。你不需要先懂量子计算，也可以先从“问题如何被翻译成二进制变量”开始看。

---

## 1. VRPTW 是什么

VRPTW 全称是 Vehicle Routing Problem with Time Windows，中文通常叫“带时间窗车辆路径规划”。

它要回答的问题是：

```text
有多辆车、一个仓库 depot、多个客户。
每个客户有需求量和可服务时间窗。
每辆车有容量限制，并且要从 depot 出发、最终回到 depot。
怎样安排每辆车访问哪些客户、按什么顺序访问、什么时候开始服务，使总距离尽量短？
```

本示例默认问题很小：

```text
2 辆车：V0, V1
4 个客户：A, B, C, D
每辆车容量：2
每个客户需求：1
每辆车最多服务 2 个客户
```

这个规模足够小，可以完整展示建模、量子采样、VQE 优化、解码和可视化。

---

## 2. 为什么不是路线选择 toy problem

旧的 `vqe_vrptw_pipeline_demo` 做的是：

```text
先枚举所有完整路线
再用 QUBO 从这些路线里选一条
```

这种方式适合解释 QUBO 和 VQE，但它没有真正把 VRPTW 的核心约束建进二进制模型里。客户数一多，完整路线数量会阶乘级增长。

本示例换成直接建模：

```text
车辆是谁
访问位置是第几个 stop
服务哪个客户
服务开始时间是多少
```

这些都变成二进制变量，然后用 QUBO 惩罚项约束它们。

---

## 3. 二进制变量长什么样

核心变量是：

```text
x_{vehicle, position, customer, start_time}
```

它的含义是：

```text
如果这个变量等于 1：
  某辆车在某个访问序位，以某个服务开始时间服务某个客户。
如果等于 0：
  不做这件事。
```

例如：

```text
x_{V0, 0, A, 1} = 1
```

表示：

```text
车辆 V0 的第 0 个 stop 是客户 A，服务开始时间是 1。
```

还有一个辅助变量：

```text
y_{vehicle, position}
```

它表示某辆车的某个访问位置是否被占用。它的作用是帮助表达“不能有空洞路线”：

```text
允许：V0 的 position 0 有客户，position 1 有客户
允许：V0 的 position 0 有客户，position 1 为空
不允许：V0 的 position 0 为空，position 1 却有客户
```

---

## 4. QUBO 如何表达约束

QUBO 是一个只含 0/1 变量的一元和二元多项式：

```text
E(x) = constant + sum a_i x_i + sum b_ij x_i x_j
```

我们让合法、距离短的路径能量低；违法路径能量高。

本示例主要加入这些约束：

```text
每个客户恰好服务一次
每个车辆位置最多选择一个客户
slot 占用变量 y 必须和 x 变量一致
车辆路径不能有空洞
车辆容量不能超过 capacity
从 depot 出发必须来得及到达第一个客户
相邻客户之间必须满足 travel time 和 service time
服务完成后必须能在 depot 时间窗内返回
```

目标函数是总行驶距离。为了让数值更稳定，代码会把距离除以一个 `distance_scale` 做归一化；解码和指标里仍会保留原始总距离。

---

## 5. QUBO 如何变成 Hamiltonian

量子线路最终测出来的是 bitstring。为了让 VQE 能优化 QUBO，代码把每个二进制变量映射到一个 qubit：

```text
x_i = (1 - Z_i) / 2
```

这里的 `Z_i` 是第 i 个 qubit 上的 Pauli-Z 算符。

因此：

```text
QUBO 能量函数
```

可以变成：

```text
只包含 I、Z、ZZ 项的 Ising Hamiltonian
```

这个 Hamiltonian 是对角的，所以每个 bitstring 都有一个明确的 QUBO 能量。

---

## 6. VQE 在这里做什么

VQE 是量子-经典混合算法：

```text
经典优化器给出参数 theta
        ↓
量子线路 ansatz(theta) 生成 bitstring 分布
        ↓
根据 QUBO/Hamiltonian 计算能量
        ↓
经典优化器根据能量更新 theta
        ↓
重复
```

本示例使用本地 `StatevectorSampler` 做量子采样，不连接真实量子计算机。

为了让 20 qubit 教学示例能在普通电脑上快速跑完，`task_06_run_vqe.py` 对当前无纠缠 RY product ansatz 和对角 QUBO 使用了等价的快速期望值计算；最终候选 bitstring 仍由 `StatevectorSampler` 采样得到。这保留了量子-经典混合流程的结构，也让新手可以在几十秒内跑完整个 pipeline。

---

## 7. 为什么要有经典参考解

默认实例只有 20 个变量，可以枚举：

```text
2^20 = 1,048,576
```

个 bitstring。这样可以得到一个可靠的经典参考答案。

这个参考答案有两个作用：

```text
检查 QUBO 建模是否正确
检查 VQE 采样中是否出现可行或最优候选
```

真实大问题不能靠暴力枚举；这里是为了教学和验收。

---

## 8. 每个 task 的直觉

```text
Task 01：准备 VRPTW 实例
Task 02：把 VRPTW 翻译成 QUBO 和 Hamiltonian
Task 03：暴力枚举得到经典参考答案
Task 04：构建参数化量子线路 ansatz
Task 05：用本地量子模拟器采样初始线路
Task 06：运行 VQE 混合优化
Task 07：把 bitstring 解码回车辆路径
Task 08：汇总指标
Task 09：画最终 dashboard
```

其中：

```text
Task 01/02/03/04/07/08/09 主要是经典计算
Task 05 是量子模拟采样
Task 06 是量子-经典混合计算
```

---

## 9. 如何理解结果

你应该重点看：

```text
artifacts/metrics.json
artifacts/results/decoded_solution.json
artifacts/results/samples.csv
artifacts/figures/09_summary_dashboard.png
```

如果默认运行成功，你会看到：

```text
best_feasible_found = true
all_customers_served_once = true
capacity_feasible = true
time_window_feasible = true
returns_to_depot = true
```

这表示 VQE 最终样本里出现了一个满足 VRPTW 约束的车辆路径。

---

## 10. 重要限制

这个示例不是工业级 VRPTW 求解器。

原因很简单：

```text
客户更多时，时间索引变量会快速增长
qubit 数会快速增长
VQE 优化不保证每次都找到全局最优
真实硬件还有噪声、连通性和 shot 数限制
```

所以它的价值是教学：展示如何把一个有容量、路径连续性和时间窗的组合优化问题拆成 QUBO，再用量子-经典混合流程跑完整个闭环。
