# VQE 求解 JSSP 的 Flyte 编排版

这个示例把 [`examples/vqe_jssp_pipeline_demo/`](../vqe_jssp_pipeline_demo/) 的 9 步量子-经典混合流水线改写成 Flyte `@task` + `@workflow` 形式。每个任务都能作为独立 Kubernetes pod 调度运行，任务之间通过 `FlyteFile` 传递 JSON、CSV、DILL 和 PNG artifact。

默认仍然使用 Qiskit 本地 `StatevectorSampler` 模拟器，不连接真实量子计算机。

本示例面向新手：你可以先用 `python workflow.py` 在本地进程里快速跑通，再用 `pyflyte run --remote` 提交到 Flyte sandbox，在浏览器 Console UI 中观察 DAG、每个 task 的日志和输出文件。

> 说明：`flytectl demo` 启动的是本机 sandbox，适合学习、验证和复现生产形态；真正生产环境需要独立 Flyte 集群、Kubernetes、对象存储和镜像仓。

---

## 1. 本示例做了什么

JSSP 是车间调度问题：多道工序要在多台机器上排时间，目标是让总完成时间尽量短。这里的默认实例是 `2 jobs x 2 machines`，经典最优 makespan 是 `4`。

VQE 是量子-经典混合算法：量子侧采样参数化线路，经典侧根据能量更新参数。本示例把 JSSP 转成 QUBO，再把 QUBO 映射成 Ising Hamiltonian，最后用 `SamplingVQE` 在本地模拟器上搜索低能量 bitstring。

Flyte 负责编排任务：它不改变 VQE 数学逻辑，只负责让每个任务以可观测、可缓存、可重试、可迁移到集群的方式运行。

---

## 2. 数据流 DAG

```mermaid
flowchart LR
    t01["t01_define_instance<br/>CLASSICAL"] -->|instance.json| t02["t02_build_qubo<br/>CLASSICAL"]
    t02 -->|qubo.json| t03["t03_exact_reference<br/>CLASSICAL"]
    t01 --> t03
    t02 -->|hamiltonian.json| t04["t04_build_ansatz<br/>CLASSICAL"]
    t01 --> t05["t05_initial_energy<br/>QUANTUM"]
    t02 --> t05
    t04 --> t05
    t01 --> t06["t06_run_vqe<br/>HYBRID"]
    t02 --> t06
    t03 --> t06
    t04 --> t06
    t01 --> t07["t07_decode_solution<br/>CLASSICAL"]
    t02 --> t07
    t06 --> t07
    t03 --> t08["t08_evaluate<br/>CLASSICAL"]
    t05 --> t08
    t06 --> t08
    t07 --> t08
    t01 --> t09["t09_visualize_summary<br/>CLASSICAL"]
    t02 --> t09
    t03 --> t09
    t04 --> t09
    t05 --> t09
    t06 --> t09
    t07 --> t09
    t08 --> t09
```

Flyte 会根据数据依赖自动并行无冲突分支，例如 `t03` 和 `t04` 可并行，`t05` 和 `t06` 也可并行。

---

## 3. 目录结构

```text
examples/vqe_jssp_flyte_pipeline_demo/
├── README.md
├── environment.yml
├── environment.lock.yml
├── requirements-flyte.txt
├── pipeline_lib.py
├── workflow.py
├── tasks/
│   ├── t01_define_instance.py
│   ├── t02_build_qubo.py
│   ├── t03_exact_reference.py
│   ├── t04_build_ansatz.py
│   ├── t05_initial_energy.py
│   ├── t06_run_vqe.py
│   ├── t07_decode_solution.py
│   ├── t08_evaluate.py
│   └── t09_visualize_summary.py
└── artifacts/
    └── local_run/        # python workflow.py 后自动归档
```

---

## 4. 环境准备

### 4.1 Docker

远程 Flyte sandbox 需要 Docker。当前机器已验证 Docker 与 `flyte-sandbox` 容器正在运行。如果你的机器没有 Docker，可以先安装：

```bash
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
```

重新登录后检查：

```bash
docker ps
```

### 4.2 flytectl 与 Flyte sandbox

当前机器的 `flytectl` 位于 `~/.local/bin/flytectl`。如果你的 shell 找不到它，把路径加入 PATH：

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
flytectl version
```

如果还没有 sandbox：

```bash
flytectl demo start
```

启动后设置 Flyte 配置：

```bash
export FLYTECTL_CONFIG=/home/ryan/.flyte/config-sandbox.yaml
```

打开浏览器：

```text
http://localhost:30080/console
```

sandbox 常用端口：

| 端口 | 用途 |
|---|---|
| 30080 | Flyte Console UI / Admin |
| 30000 | sandbox 本地 Docker registry |
| 30002 | MinIO S3 API |
| 6443 | 内置 k3s API server |

### 4.3 创建 conda 环境

从仓库根目录运行：

```bash
conda env create -f examples/vqe_jssp_flyte_pipeline_demo/environment.yml
conda activate qml-vqe-jssp-flyte
pip install -e .
```

如果想复现本文测通时的完整版本锁：

```bash
conda env create -f examples/vqe_jssp_flyte_pipeline_demo/environment.lock.yml
conda activate qml-vqe-jssp-flyte
pip install -e .
```

检查关键依赖：

```bash
python -c "import flytekit, qiskit, qiskit_algorithms; print(flytekit.__version__, qiskit.__version__, qiskit_algorithms.__version__)"
```

---

## 5. 本地 sanity check

本地模式不走 Kubernetes，而是在当前 Python 进程中执行 Flyte workflow，适合先验证业务逻辑。

```bash
cd examples/vqe_jssp_flyte_pipeline_demo
python workflow.py
```

运行完成后会自动把所有输出归档到：

```text
examples/vqe_jssp_flyte_pipeline_demo/artifacts/local_run/
```

重点查看：

```text
artifacts/local_run/metrics.json
artifacts/local_run/09_summary_dashboard.png
artifacts/local_run/vqe_trace.csv
artifacts/local_run/samples.csv
```

本机已测通的关键结果：

```text
reference.optimal_makespan = 4
reference.optimal_qubo_energy = 4.0
acceptance.constraint_satisfied = true
acceptance.vqe_top_candidates_include_makespan_4 = true
acceptance.vqe_best_measurement_reaches_optimal_qubo_energy = true
```

---

## 6. 提交到 Flyte sandbox

确认 Docker、sandbox、conda 环境都准备好后：

```bash
cd examples/vqe_jssp_flyte_pipeline_demo
export FLYTECTL_CONFIG=/home/ryan/.flyte/config-sandbox.yaml
pyflyte run --remote --wait --copy auto workflow.py vqe_jssp_workflow \
  --horizon 5 \
  --penalty 10.0 \
  --ansatz_reps 2 \
  --optimizer COBYLA \
  --maxiter 120 \
  --shots 4096 \
  --seed 42
```

注意：`pyflyte run` 使用 Python 参数名，所以这里是 `--ansatz_reps`，不是 `--ansatz-reps`。

第一次远程运行会构建并推送镜像：

```text
localhost:30000/vqe-jssp-flyte:<hash>
```

本机已测通的一次远程 execution：

```text
execution id: al7hsrhrc6mp8bxfmkkg
status: SUCCEEDED
elapsed: 31.65s
console: http://localhost:30080/console/projects/flytesnacks/domains/development/executions/al7hsrhrc6mp8bxfmkkg
```

也可以在 sandbox 内检查 pod：

```bash
docker exec flyte-sandbox kubectl get pods -n flytesnacks-development
```

应看到本次 execution 的 `n0` 到 `n8` 共 9 个 pod 均为 `Completed`。

---

## 7. 任务输入输出表

| # | Flyte task | 类型 | 主要输入 | 主要输出 |
|---|---|---|---|---|
| 1 | `t01_define_instance` | CLASSICAL | `horizon` | `instance.json`, `01_instance.png` |
| 2 | `t02_build_qubo` | CLASSICAL | `instance.json`, `penalty` | `qubo.json`, `hamiltonian.dill`, `hamiltonian.json`, `02_qubo_matrix.png` |
| 3 | `t03_exact_reference` | CLASSICAL | `instance.json`, `qubo.json` | `reference.json`, `03_reference_gantt.png` |
| 4 | `t04_build_ansatz` | CLASSICAL | `hamiltonian.json`, `ansatz_reps`, `seed` | `ansatz.dill`, `ansatz.json`, `initial_point.npy`, `04_ansatz_circuit.png` |
| 5 | `t05_initial_energy` | QUANTUM | Hamiltonian、ansatz、初始参数 | `initial_energy.json`, `05_initial_energy.png` |
| 6 | `t06_run_vqe` | HYBRID | Hamiltonian、ansatz、reference | `vqe_result.json`, `vqe_result.dill`, `vqe_trace.csv`, `06_vqe_convergence.png` |
| 7 | `t07_decode_solution` | CLASSICAL | `vqe_result.json`, `qubo.json`, `instance.json` | `decoded_solution.json`, `samples.csv`, `07_solution_probabilities.png`, `08_vqe_gantt.png` |
| 8 | `t08_evaluate` | CLASSICAL | reference、initial、VQE、decoded | `metrics.json` |
| 9 | `t09_visualize_summary` | CLASSICAL | 全部关键 JSON/CSV | `09_summary_dashboard.png` |

---

## 8. 迁移到正式 Flyte 集群

本示例默认使用：

```python
registry=os.environ.get("FLYTE_IMAGE_REGISTRY", "localhost:30000")
```

迁移到正式集群时，不改代码，只改环境：

```bash
export FLYTE_IMAGE_REGISTRY=<你的镜像仓>
export FLYTECTL_CONFIG=<你的生产 Flyte Admin 配置>
pyflyte run --remote --wait --copy auto workflow.py vqe_jssp_workflow
```

生产环境通常需要：

- Kubernetes 集群。
- Flyte Admin / Propeller / Console。
- 对象存储：S3、GCS、MinIO 或兼容服务。
- 容器镜像仓。
- 网络与权限配置，让 task pod 能拉镜像、读写对象存储。

---

## 9. 常见问题

**`flytectl: command not found`**

把 `~/.local/bin` 加入 PATH，或直接使用 `/home/ryan/.local/bin/flytectl`。

**Console UI 打不开**

确认 sandbox 容器正在运行：

```bash
docker ps | grep flyte-sandbox
```

确认端口 `30080` 没被其他服务占用。

**`pyflyte run` 参数报错**

使用 workflow 的 Python 参数名，例如 `--ansatz_reps`，不要写成 `--ansatz-reps`。

**第一次远程运行很慢**

首次会构建并推送 ImageSpec 镜像，后续相同依赖会复用缓存。

**想强制重跑**

可以改 `pipeline_lib.py` 中的 `CACHE_VERSION`，或在 `pyflyte run` 增加：

```bash
--overwrite-cache
```

---

## 10. 清理

停止 Flyte sandbox：

```bash
flytectl demo teardown
```

删除 conda 环境：

```bash
conda env remove -n qml-vqe-jssp-flyte
```
