"""VQE-VRPTW solver 示例的共用工具。

这里集中放路径常量、JSON 工具、QUBO 能量计算、bitstring 解码、可视化
助手和本地量子 sampler 创建函数。后续如果要切到真实量子硬件，优先改
``create_sampler()``。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import defaultdict
from enum import Enum
from itertools import combinations
from pathlib import Path
from typing import Any


DEMO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = DEMO_ROOT / "artifacts"
INSTANCE_DIR = ARTIFACTS_DIR / "instance"
QUBO_DIR = ARTIFACTS_DIR / "qubo"
ANSATZ_DIR = ARTIFACTS_DIR / "ansatz"
RESULTS_DIR = ARTIFACTS_DIR / "results"
FIGURES_DIR = ARTIFACTS_DIR / "figures"

INSTANCE_JSON = INSTANCE_DIR / "instance.json"
QUBO_JSON = QUBO_DIR / "qubo.json"
HAMILTONIAN_JSON = QUBO_DIR / "hamiltonian.json"
HAMILTONIAN_DILL = QUBO_DIR / "hamiltonian.dill"
REFERENCE_JSON = RESULTS_DIR / "reference.json"
ANSATZ_DILL = ANSATZ_DIR / "ansatz.dill"
ANSATZ_JSON = ANSATZ_DIR / "ansatz.json"
INITIAL_POINT_NPY = ANSATZ_DIR / "initial_point.npy"
INITIAL_ENERGY_JSON = RESULTS_DIR / "initial_energy.json"
VQE_RESULT_JSON = RESULTS_DIR / "vqe_result.json"
VQE_RESULT_DILL = RESULTS_DIR / "vqe_result.dill"
VQE_TRACE_CSV = RESULTS_DIR / "vqe_trace.csv"
DECODED_SOLUTION_JSON = RESULTS_DIR / "decoded_solution.json"
SAMPLES_CSV = RESULTS_DIR / "samples.csv"
SOLUTION_ROUTES_CSV = RESULTS_DIR / "solution_routes.csv"
METRICS_JSON = ARTIFACTS_DIR / "metrics.json"


class TaskKind(str, Enum):
    """任务类型：经典、量子、混合。"""

    CLASSICAL = "CLASSICAL"
    QUANTUM = "QUANTUM"
    HYBRID = "HYBRID"


_COLORS = {
    TaskKind.CLASSICAL: "\033[94m",
    TaskKind.QUANTUM: "\033[95m",
    TaskKind.HYBRID: "\033[93m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def print_banner(kind: TaskKind, name: str) -> None:
    """打印任务类型 banner，帮助新手区分当前步骤。"""

    use_color = _supports_color()
    color = _COLORS[kind] if use_color else ""
    bold = _BOLD if use_color else ""
    reset = _RESET if use_color else ""
    bar = "=" * 72
    print(f"\n{color}{bold}{bar}{reset}")
    print(f"{color}{bold}  [{kind.value:9s}]  {name}{reset}")
    print(f"{color}{bold}{bar}{reset}")


class Timer:
    """简单计时器。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        print(f"  开始: {self.label}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._t0
        status = "完成" if exc_type is None else "失败"
        print(f"  [{status}: {self.label}, 用时 {self.elapsed:.2f}s]")


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | os.PathLike, payload: dict[str, Any]) -> Path:
    out = ensure_parent_dir(path)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def create_sampler(shots: int = 4096, seed: int | None = None):
    """创建本地量子采样器。

    默认使用 ``StatevectorSampler``。真实硬件迁移时，可把这里替换为 IBM
    Runtime 的 SamplerV2，并从 CLI 透传 backend/session/shots。
    """

    from qiskit.primitives import StatevectorSampler

    return StatevectorSampler(default_shots=shots, seed=seed)


def complex_to_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def bits_to_bitstring(bits: list[int]) -> str:
    """变量顺序 bits -> Qiskit 风格 bitstring。"""

    return "".join(str(int(b)) for b in reversed(bits))


def bitstring_to_bits(bitstring: str, num_variables: int | None = None) -> list[int]:
    """Qiskit 风格 bitstring -> 变量顺序 bits。"""

    bits = [int(ch) for ch in reversed(bitstring.strip())]
    if num_variables is not None and len(bits) != num_variables:
        raise ValueError(f"bitstring 长度 {len(bits)} 与变量数 {num_variables} 不一致")
    return bits


def int_to_bits(value: int, num_variables: int) -> list[int]:
    return [(value >> i) & 1 for i in range(num_variables)]


class QuboBuilder:
    """小型 QUBO 构造器，统一处理二进制变量 ``x_i^2 = x_i``。"""

    def __init__(self) -> None:
        self.offset = 0.0
        self.linear: defaultdict[int, float] = defaultdict(float)
        self.quadratic: defaultdict[tuple[int, int], float] = defaultdict(float)

    def add_linear(self, i: int, coeff: float) -> None:
        self.linear[int(i)] += float(coeff)

    def add_quadratic(self, i: int, j: int, coeff: float) -> None:
        i = int(i)
        j = int(j)
        coeff = float(coeff)
        if i == j:
            self.add_linear(i, coeff)
            return
        if i > j:
            i, j = j, i
        self.quadratic[(i, j)] += coeff

    def add_square_constraint(self, expr: dict[int, float], rhs: float, weight: float) -> None:
        """加入 ``weight * (sum_i a_i x_i - rhs)^2``。"""

        rhs = float(rhs)
        weight = float(weight)
        self.offset += weight * rhs * rhs
        for i, a_i in expr.items():
            self.add_linear(i, weight * (a_i * a_i - 2.0 * rhs * a_i))
        for (i, a_i), (j, a_j) in combinations(expr.items(), 2):
            self.add_quadratic(i, j, weight * 2.0 * a_i * a_j)

    def add_implication_not(self, i: int, j: int, weight: float) -> None:
        """加入 ``weight * x_i * (1 - x_j)``，即 i=1 时要求 j=1。"""

        self.add_linear(i, weight)
        self.add_quadratic(i, j, -weight)


def qubo_energy(qubo: dict[str, Any], bits: list[int]) -> float:
    """计算 QUBO 目标值 ``offset + linear + quadratic``。"""

    energy = float(qubo.get("offset", 0.0))
    for term in qubo.get("linear", []):
        energy += float(term["coefficient"]) * bits[int(term["index"])]
    for term in qubo.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        energy += float(term["coefficient"]) * bits[i] * bits[j]
    return float(energy)


def qubo_to_ising(qubo: dict[str, Any]):
    """把 QUBO 映射为 Pauli-Z Ising Hamiltonian。"""

    from qiskit.quantum_info import SparsePauliOp

    n = int(qubo["num_variables"])
    offset = float(qubo.get("offset", 0.0))
    z_terms: defaultdict[int, float] = defaultdict(float)
    zz_terms: defaultdict[tuple[int, int], float] = defaultdict(float)

    for term in qubo.get("linear", []):
        i = int(term["index"])
        coeff = float(term["coefficient"])
        offset += coeff / 2.0
        z_terms[i] += -coeff / 2.0

    for term in qubo.get("quadratic", []):
        i = int(term["i"])
        j = int(term["j"])
        coeff = float(term["coefficient"])
        offset += coeff / 4.0
        z_terms[i] += -coeff / 4.0
        z_terms[j] += -coeff / 4.0
        if i > j:
            i, j = j, i
        zz_terms[(i, j)] += coeff / 4.0

    pauli_terms: list[tuple[str, complex]] = [("I" * n, complex(offset))]
    for i, coeff in sorted(z_terms.items()):
        if abs(coeff) <= 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        pauli_terms.append(("".join(label), complex(coeff)))
    for (i, j), coeff in sorted(zz_terms.items()):
        if abs(coeff) <= 1e-12:
            continue
        label = ["I"] * n
        label[n - 1 - i] = "Z"
        label[n - 1 - j] = "Z"
        pauli_terms.append(("".join(label), complex(coeff)))

    return SparsePauliOp.from_list(pauli_terms).simplify(atol=1e-12)


def pauli_records(operator, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, coeff in zip(operator.paulis.to_labels(), operator.coeffs):
        rows.append(
            {
                "pauli": label,
                "coefficient": complex_to_record(complex(coeff)),
                "abs_coefficient": float(abs(coeff)),
            }
        )
    rows.sort(key=lambda row: row["abs_coefficient"], reverse=True)
    return rows if limit is None else rows[:limit]


def node_ids(instance: dict[str, Any]) -> list[str]:
    return [instance["depot"]["id"], *[customer["id"] for customer in instance["customers"]]]


def customer_by_id(instance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {customer["id"]: customer for customer in instance["customers"]}


def vehicle_by_index(instance: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {i: vehicle for i, vehicle in enumerate(instance["vehicles"])}


def matrix_value(instance: dict[str, Any], matrix_name: str, src: str, dst: str) -> float:
    return float(instance[matrix_name][src][dst])


def decode_vrptw_solution(
    instance: dict[str, Any],
    variables: list[dict[str, Any]],
    bits: list[int],
) -> dict[str, Any]:
    """把 bitstring 解码为 VRPTW 车辆路径并检查约束。"""

    depot = instance["depot"]
    depot_id = depot["id"]
    customers = customer_by_id(instance)
    vehicles = vehicle_by_index(instance)
    max_positions = int(instance["solver_config"]["max_stops_per_vehicle"])
    x_selected = [v for v in variables if v["kind"] == "x" and bits[int(v["index"])] == 1]
    y_selected = {
        (int(v["vehicle"]), int(v["position"])): bits[int(v["index"])]
        for v in variables
        if v["kind"] == "y"
    }
    by_slot: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    customer_counts: defaultdict[str, int] = defaultdict(int)
    for var in x_selected:
        by_slot[(int(var["vehicle"]), int(var["position"]))].append(var)
        customer_counts[str(var["customer_id"])] += 1

    violations: list[str] = []
    for customer in instance["customers"]:
        cid = customer["id"]
        if customer_counts[cid] != 1:
            violations.append(f"客户 {cid} 被服务 {customer_counts[cid]} 次")

    decoded_routes: list[dict[str, Any]] = []
    total_distance = 0.0
    served_customers: list[str] = []

    for vehicle_index, vehicle in vehicles.items():
        vehicle_id = vehicle["id"]
        load = 0.0
        route_distance = 0.0
        events: list[dict[str, Any]] = []
        previous_node = depot_id
        previous_departure = float(vehicle.get("start_time", 0.0))
        last_event: dict[str, Any] | None = None

        for position in range(max_positions):
            slot_key = (vehicle_index, position)
            selected = sorted(by_slot.get(slot_key, []), key=lambda row: (row["customer_id"], row["start_time"]))
            y_bit = int(y_selected.get(slot_key, 0))
            if len(selected) != y_bit:
                violations.append(
                    f"车辆 {vehicle_id} 的位置 {position} 选择了 {len(selected)} 个客户，但 y={y_bit}"
                )
            if len(selected) > 1:
                violations.append(f"车辆 {vehicle_id} 的位置 {position} 同时选择了多个客户")
            if position > 0 and y_bit == 1 and int(y_selected.get((vehicle_index, position - 1), 0)) == 0:
                violations.append(f"车辆 {vehicle_id} 的位置 {position} 有访问，但前一位置为空")
            if not selected:
                continue

            var = selected[0]
            cid = str(var["customer_id"])
            customer = customers[cid]
            start_time = float(var["start_time"])
            service_time = float(customer["service_time"])
            travel_time = matrix_value(instance, "travel_time_matrix", previous_node, cid)
            arrival_time = previous_departure + travel_time
            distance = matrix_value(instance, "distance_matrix", previous_node, cid)
            ready, due = [float(x) for x in customer["time_window"]]
            departure_time = start_time + service_time
            if arrival_time > start_time + 1e-9:
                violations.append(
                    f"车辆 {vehicle_id} 到客户 {cid} 的到达时间 {arrival_time:.1f} 晚于服务开始 {start_time:.1f}"
                )
            if start_time < ready - 1e-9 or start_time > due + 1e-9:
                violations.append(f"客户 {cid} 服务开始 {start_time:.1f} 不在时间窗 [{ready:.1f}, {due:.1f}]")
            route_distance += distance
            load += float(customer.get("demand", 0.0))
            served_customers.append(cid)
            event = {
                "vehicle_id": vehicle_id,
                "position": position,
                "customer_id": cid,
                "arrival_time": arrival_time,
                "service_start_time": start_time,
                "departure_time": departure_time,
                "demand": float(customer.get("demand", 0.0)),
                "leg_from": previous_node,
                "leg_distance": distance,
                "leg_travel_time": travel_time,
                "time_window": [ready, due],
            }
            events.append(event)
            last_event = event
            previous_node = cid
            previous_departure = departure_time

        return_distance = 0.0
        return_arrival = float(vehicle.get("start_time", 0.0))
        if last_event is not None:
            return_distance = matrix_value(instance, "distance_matrix", previous_node, depot_id)
            return_travel_time = matrix_value(instance, "travel_time_matrix", previous_node, depot_id)
            return_arrival = previous_departure + return_travel_time
            route_distance += return_distance
            depot_due = float(depot["time_window"][1])
            if return_arrival > depot_due + 1e-9:
                violations.append(
                    f"车辆 {vehicle_id} 回到 depot 时间 {return_arrival:.1f} 超过 depot 截止 {depot_due:.1f}"
                )

        if load > float(vehicle["capacity"]) + 1e-9:
            violations.append(f"车辆 {vehicle_id} 载重 {load:.1f} 超过容量 {vehicle['capacity']}")
        total_distance += route_distance
        decoded_routes.append(
            {
                "vehicle_id": vehicle_id,
                "load": load,
                "capacity": float(vehicle["capacity"]),
                "num_stops": len(events),
                "route_distance": route_distance,
                "return_distance": return_distance,
                "return_arrival_time": return_arrival,
                "events": events,
            }
        )

    unique_served = sorted(set(served_customers))
    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "routes": decoded_routes,
        "served_customers": served_customers,
        "unique_served_customers": unique_served,
        "num_customers_served": len(unique_served),
        "num_customer_visits": len(served_customers),
        "num_vehicles_used": sum(1 for route in decoded_routes if route["num_stops"] > 0),
        "all_customers_served_once": all(customer_counts[c["id"]] == 1 for c in instance["customers"]),
        "capacity_feasible": all(route["load"] <= route["capacity"] + 1e-9 for route in decoded_routes),
        "total_distance": total_distance,
    }


def euclidean(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def draw_vrptw_routes(
    instance: dict[str, Any],
    decoded: dict[str, Any],
    out_path: str | os.PathLike,
    title: str,
) -> None:
    """绘制车辆路线地图和服务时间轴。"""

    import matplotlib.pyplot as plt

    out = ensure_parent_dir(out_path)
    depot = instance["depot"]
    customers = customer_by_id(instance)
    colors = ["#3182bd", "#31a354", "#de2d26", "#756bb1"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ax_map, ax_time = axes

    ax_map.scatter([depot["x"]], [depot["y"]], s=180, marker="s", color="#525252", edgecolor="black")
    ax_map.text(depot["x"] + 0.05, depot["y"] + 0.05, depot["id"], fontsize=10, weight="bold")
    for cid, customer in customers.items():
        ax_map.scatter([customer["x"]], [customer["y"]], s=130, color="#9ecae1", edgecolor="black")
        ax_map.text(customer["x"] + 0.05, customer["y"] + 0.05, cid, fontsize=10, weight="bold")

    y_ticks: list[int] = []
    y_labels: list[str] = []
    for vehicle_i, route in enumerate(decoded.get("routes", [])):
        color = colors[vehicle_i % len(colors)]
        points = [depot]
        points += [customers[event["customer_id"]] for event in route["events"]]
        if route["events"]:
            points.append(depot)
        if len(points) > 1:
            ax_map.plot(
                [p["x"] for p in points],
                [p["y"] for p in points],
                color=color,
                linewidth=2.2,
                marker="o",
                label=f"{route['vehicle_id']} route",
            )
        y_ticks.append(vehicle_i)
        y_labels.append(route["vehicle_id"])
        for event in route["events"]:
            ready, due = event["time_window"]
            ax_time.hlines(vehicle_i, ready, due, color="#bdbdbd", linewidth=6, alpha=0.7)
            ax_time.scatter([event["service_start_time"]], [vehicle_i], s=90, color=color, edgecolor="black", zorder=4)
            ax_time.text(
                event["service_start_time"] + 0.05,
                vehicle_i + 0.08,
                event["customer_id"],
                fontsize=9,
                weight="bold",
            )
        if route["events"]:
            ax_time.scatter([route["return_arrival_time"]], [vehicle_i], s=70, marker="s", color="#525252", zorder=4)

    ax_map.set_title("Vehicle routes")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.grid(True, alpha=0.25)
    ax_map.set_aspect("equal")
    if any(route["events"] for route in decoded.get("routes", [])):
        ax_map.legend(fontsize=9)

    ax_time.set_title("Service starts and time windows")
    ax_time.set_xlabel("time")
    ax_time.set_yticks(y_ticks)
    ax_time.set_yticklabels(y_labels)
    ax_time.set_xlim(float(depot["time_window"][0]), float(depot["time_window"][1]) + 0.8)
    ax_time.grid(True, axis="x", alpha=0.25)

    status = "feasible" if decoded.get("feasible") else "check"
    fig.suptitle(f"{title} / {status} / total distance={decoded.get('total_distance', 0.0):.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
