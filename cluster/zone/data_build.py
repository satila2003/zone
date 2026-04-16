"""
按簇聚合的 Starlink 550 流量矩阵与超节点拓扑构建工具。
功能：
1. 读取底层的卫星分域结果（Cluster）和原始流量矩阵（TM）。
2. 将数千颗卫星的微观拓扑，聚合为 18 个“超节点(Super-Nodes)”的宏观拓扑。
3. 补齐路由路径至固定维度（8条），并输出符合 GNN 模型要求的标准格式 PKL 文件。

依赖库：networkx、matplotlib、numpy、pandas、pickle
Python 版本要求：Python >= 3.8
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pickle
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 尝试导入可选的网络图和数据处理库
try:
    import networkx as nx
except Exception:
    nx = None

try:
    import pandas as pd
except Exception:
    pd = None

# ================= 全局参数配置 =================
# 预期的超节点(域/簇)总数量
N_CLUSTERS = 18
# 每对超节点之间保留的最大并行路径数量（不足将自动复制补齐）
MAX_PARALLEL_LINKS = 8

def parse_args() -> argparse.Namespace:
    """初始化命令行参数"""
    base_dir = Path(__file__).resolve().parent
    default_cluster = base_dir / "outputs" / "starlink550_data_1200_domain_result_18_raan_orbital_angle.txt"
    default_tm = base_dir / "outputs" / "tms" / "data_2500_all_time_slices.pkl"
    out_pkl = base_dir / "outputs" / "tms" / "data_2500_all_time_slices_cluster.pkl"

    p = argparse.ArgumentParser()
    p.add_argument("--cluster_pkl", type=str, default=str(default_cluster))
    p.add_argument("--tm_pkl", type=str, default=str(default_tm))
    p.add_argument("--out_pkl", type=str, default=str(out_pkl))
    p.add_argument("--plot", action="store_true", default=True)
    return p.parse_args()


@dataclass(frozen=True)
class ClusterResult:
    """
    存储聚类/分域结果的数据类
    sat_to_cluster_by_slice: 按时间片划分的 卫星ID -> 簇ID 的映射列表
    centers_xy: 簇中心点的二维坐标（用于可视化）
    radii: 簇的覆盖半径
    """
    sat_to_cluster_by_slice: Optional[List[Dict[int, int]]]
    sat_to_cluster_global: Optional[Dict[int, int]]
    sat_to_plane_global: Optional[Dict[int, int]]
    sat_to_angle_global: Optional[Dict[int, float]]
    centers_xy: Optional[np.ndarray]
    radii: Optional[np.ndarray]


def _parse_int(s: str) -> int:
    """安全地将字符串解析为整数，去除首尾空白"""
    return int(str(s).strip())


def _parse_tm_key(key: str) -> Tuple[int, int]:
    """
    解析流量矩阵的键。
    将诸如 '50185, 53726' 的字符串解析为源和目的节点的整型元组 (50185, 53726)。
    """
    parts = str(key).split(",")
    if len(parts) < 2:
        raise ValueError(f"Invalid tm key: {key!r}")
    return _parse_int(parts[0]), _parse_int(parts[1])


def _parse_cluster_txt(path: Path) -> Dict[int, int]:
    """
    解析单个时间片的分域结果 TXT 文件。
    提取 卫星ID(sat_id) 到 域ID(domain_id) 的映射。
    """
    sat_to_domain: Dict[int, int] = {}
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if "sat_id" not in header or "domain_id" not in header:
            raise ValueError(f"Unexpected header in {str(path)!r}: {header!r}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            # 确保至少包含足够的列数以提取 domain_id
            if len(parts) < 7:
                continue
            sat_id = _parse_int(parts[0])
            domain_id = _parse_int(parts[-1])
            sat_to_domain[sat_id] = domain_id
    return sat_to_domain


def _sorted_step_indices_from_names(names: Iterable[str]) -> List[int]:
    """从文件名列表中通过正则提取所有的 step 序号并进行排序"""
    indices = []
    for name in names:
        m = re.search(r"step_(\d+)", name)
        if not m:
            continue
        indices.append(int(m.group(1)))
    return sorted(set(indices))

def _parse_cluster_plane_angle_txt(path: Path) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, float]]:
    sat_to_cluster: Dict[int, int] = {}
    sat_to_plane: Dict[int, int] = {}
    sat_to_angle: Dict[int, float] = {}

    with path.open("r", encoding="utf-8") as f:
        header = [h.strip() for h in f.readline().strip().split(",")]

        sat_idx = header.index("sat_id")
        plane_idx = header.index("plane_id")
        angle_idx = header.index("orbital_angle_deg")
        domain_idx = header.index("domain_id")

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split(",")]
            sat_id = int(parts[sat_idx])
            plane_id = int(parts[plane_idx])
            angle = float(parts[angle_idx])
            domain_id = int(parts[domain_idx])

            sat_to_cluster[sat_id] = domain_id
            sat_to_plane[sat_id] = plane_id
            sat_to_angle[sat_id] = angle

    return sat_to_cluster, sat_to_plane, sat_to_angle

def _load_cluster_from_dir(cluster_dir: Path) -> ClusterResult:
    """从包含多个 TXT 文件的目录中批量加载所有时间片的分域结果"""
    files = sorted(cluster_dir.glob("step_*_result_*_domains.txt"))
    if not files:
        raise FileNotFoundError(f"No cluster result txt found in: {str(cluster_dir)}")

    step_idx_to_path: Dict[int, Path] = {}
    for p in files:
        m = re.search(r"step_(\d+)_result_", p.name)
        if not m:
            continue
        step_idx_to_path[int(m.group(1))] = p

    # 按照 step 序号严格排序
    ordered_steps = sorted(step_idx_to_path.keys())
    sat_to_cluster_by_slice = [_parse_cluster_txt(step_idx_to_path[i]) for i in ordered_steps]
    return ClusterResult(
        sat_to_cluster_by_slice=sat_to_cluster_by_slice, 
        sat_to_cluster_global=None, 
        sat_to_plane_global=None, 
        sat_to_angle_global=None,
        centers_xy=None, 
        radii=None)


def load_cluster_result(cluster_pkl: str, n_slices: Optional[int] = None) -> ClusterResult:
    """
    加载聚类结果（支持目录、txt、pkl 和 json 格式）。
    将非结构化的存储反序列化为统一的 ClusterResult 对象。
    如果输入是单个 txt，则认为所有时间片共用同一份分域结果。
    """
    p = Path(cluster_pkl)
    if p.is_dir():
        return _load_cluster_from_dir(p)

    if not p.exists():
        raise FileNotFoundError(str(p))

    suffix = p.suffix.lower()
    
    if suffix == ".txt":
        print(f"Loading cluster and plane info from TXT: {str(p)}")
        sat_to_cluster, sat_to_plane, sat_to_angle = _parse_cluster_plane_angle_txt(p)
        return ClusterResult(
            sat_to_cluster_by_slice=None,
            sat_to_cluster_global=sat_to_cluster,
            sat_to_plane_global=sat_to_plane,
            sat_to_angle_global=sat_to_angle,
            centers_xy=None,
            radii=None,
        )

    if suffix in {".pkl", ".pickle"}:
        print(f"Loading cluster and plane info from PKL: {str(p)}")
        with p.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "sat_to_cluster_by_slice" in obj:
            centers = obj.get("centers_xy")
            radii = obj.get("radii")
            return ClusterResult(
                sat_to_cluster_by_slice=list(obj["sat_to_cluster_by_slice"]),
                sat_to_cluster_global=None,
                sat_to_plane_global=None,
                sat_to_angle_global=None,
                centers_xy=None if centers is None else np.asarray(centers, dtype=np.float64),
                radii=None if radii is None else np.asarray(radii, dtype=np.float64),
            )
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return ClusterResult(
                sat_to_cluster_by_slice=list(obj), 
                sat_to_cluster_global=None, 
                sat_to_plane_global=None, 
                sat_to_angle_global=None,
                centers_xy=None, 
                radii=None)
        raise ValueError(f"Unsupported cluster pickle structure: {type(obj)}")

    if suffix == ".json":
        print(f"Loading cluster and plane info from JSON: {str(p)}")
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "sat_to_cluster_by_slice" in obj:
            centers = obj.get("centers_xy")
            radii = obj.get("radii")
            return ClusterResult(
                sat_to_cluster_by_slice=[{int(k): int(v) for k, v in m.items()} for m in obj["sat_to_cluster_by_slice"]],
                sat_to_cluster_global=None,
                sat_to_plane_global=None,
                sat_to_angle_global=None,
                centers_xy=None if centers is None else np.asarray(centers, dtype=np.float64),
                radii=None if radii is None else np.asarray(radii, dtype=np.float64),
            )
        raise ValueError("Unsupported cluster json structure")

    raise ValueError(f"Unsupported cluster file type: {suffix}")


def load_original_tm(tm_pkl: str) -> List[dict]:
    """加载底层生成的原始卫星级流量矩阵 PKL 文件"""
    p = Path(tm_pkl)
    if not p.exists():
        raise FileNotFoundError(str(p))
    with p.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise ValueError("Original TM PKL must be a list[dict]")
    # 只抽取前几个进行格式校验，保证数据安全性
    for i, item in enumerate(data[:3]):
        if not isinstance(item, dict):
            raise ValueError(f"Original TM item {i} is not dict: {type(item)}")
        for k in ("graph", "tm", "path", "data_idx"):
            if k not in item:
                raise ValueError(f"Original TM item {i} missing key: {k}")
    return data

def get_sat_to_cluster(cluster: ClusterResult, data_idx: int) -> Dict[int, int]:
    if cluster.sat_to_cluster_global is not None:
        return cluster.sat_to_cluster_global

    if cluster.sat_to_cluster_by_slice is None:
        raise ValueError("No cluster mapping available.")

    if data_idx >= len(cluster.sat_to_cluster_by_slice):
        raise IndexError(
            f"cluster slice count={len(cluster.sat_to_cluster_by_slice)} < data_idx={data_idx}"
        )

    return cluster.sat_to_cluster_by_slice[data_idx]

def angular_distance_deg(a: float, b: float) -> float:
    """
    计算两个轨道面内角的最小环形角距离。
    例如 359° 和 1° 的距离是 2°，不是 358°。
    """
    diff = abs(a - b) % 360.0
    return min(diff, 360.0 - diff)

def forward_angle_gap_deg(a: float, b: float) -> float:
    """
    从角度 a 沿正方向走到角度 b 的环形角距离。
    例如 a=359, b=1，则距离为 2。
    """
    return (b - a) % 360.0


def backward_angle_gap_deg(a: float, b: float) -> float:
    """
    从角度 a 沿反方向走到角度 b 的环形角距离。
    例如 a=1, b=359，则距离为 2。
    """
    return (a - b) % 360.0

def write_missing_satellite_info_report(
    report_path: Path,
    missing_records: Dict[int, Dict[str, object]],
) -> None:
    """
    输出缺失 plane_id / orbital_angle_deg 信息的卫星总结文档。
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Missing Satellite Plane/Angle Info Report\n")
        f.write("=" * 60 + "\n\n")

        if not missing_records:
            f.write("No missing satellite plane_id or orbital_angle_deg information found.\n")
            return

        f.write(f"Total missing satellites: {len(missing_records)}\n\n")
        f.write("sat_id,missing_plane_id,missing_orbital_angle_deg,first_data_idx,appear_count\n")

        for sat_id in sorted(missing_records.keys()):
            rec = missing_records[sat_id]
            missing_plane = rec.get("missing_plane_id", False)
            missing_angle = rec.get("missing_orbital_angle_deg", False)
            first_data_idx = rec.get("first_data_idx", "")
            appear_count = rec.get("appear_count", 0)

            f.write(
                f"{sat_id},"
                f"{missing_plane},"
                f"{missing_angle},"
                f"{first_data_idx},"
                f"{appear_count}\n"
            )

def _filter_graph_4_neighbor_by_plane(
    graph: Iterable[Sequence[int]],
    sat_to_plane: Dict[int, int],
    sat_to_angle: Dict[int, float],
    data_idx: Optional[int] = None,
    missing_records: Optional[Dict[int, Dict[str, object]]] = None,
) -> List[List[int]]:
    """
    过滤原始卫星拓扑，尽量构造每颗卫星：
    - 同轨无向连接 2 条
    - 左侧相邻轨道面无向连接 1 条
    - 右侧相邻轨道面无向连接 1 条
    """

    neighbors: Dict[int, set[int]] = {}

    def record_missing_satellite(sat_id: int) -> None:
        if missing_records is None:
            return

        if sat_id not in missing_records:
            missing_records[sat_id] = {
                "missing_plane_id": sat_id not in sat_to_plane,
                "missing_orbital_angle_deg": sat_id not in sat_to_angle,
                "first_data_idx": data_idx,
                "appear_count": 0,
            }
        appear_count = missing_records[sat_id].get("appear_count", 0) if sat_id in missing_records else 0
        if not isinstance(appear_count, int):
            appear_count = 0
        missing_records[sat_id]["appear_count"] = appear_count + 1

        # 如果同一颗卫星在不同边中暴露出不同类型缺失，做一次合并更新
        missing_records[sat_id]["missing_plane_id"] = (
            bool(missing_records[sat_id]["missing_plane_id"]) or sat_id not in sat_to_plane
        )
        missing_records[sat_id]["missing_orbital_angle_deg"] = (
            bool(missing_records[sat_id]["missing_orbital_angle_deg"]) or sat_id not in sat_to_angle
        )

    # 第一轮扫描：统计每颗卫星的邻居，并记录缺失 plane_id 或 orbital_angle_deg 信息的卫星
    for e in graph:
        u, v = int(e[0]), int(e[1])

        u_missing = u not in sat_to_plane or u not in sat_to_angle
        v_missing = v not in sat_to_plane or v not in sat_to_angle

        if u_missing:
            record_missing_satellite(u)

        if v_missing:
            record_missing_satellite(v)

        if u_missing or v_missing:
            continue

        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    num_planes = max(sat_to_plane.values()) + 1

    # 第二轮扫描：基于轨道面关系和角度信息，优先保留满足同轨、左侧、右侧关系的边，构造每颗卫星最多4条边的子图。
    degree: Dict[int, int] = {u: 0 for u in neighbors}
    quota: Dict[int, Dict[str, int]] = {
        u: {"same": 0, "left": 0, "right": 0}
        for u in neighbors
    }
    filtered_edges = set()

    def relation_type(u: int, v: int) -> Optional[str]:
        pu = sat_to_plane[u]
        pv = sat_to_plane[v]

        left_plane_id = (pu - 1) % num_planes
        right_plane_id = (pu + 1) % num_planes

        if pv == pu:
            return "same"
        if pv == left_plane_id:
            return "left"
        if pv == right_plane_id:
            return "right"
        return None

    def type_limit(t: str) -> int:
        if t == "same":
            return 2
        if t in {"left", "right"}:
            return 1
        return 0

    def add_edge(u: int, v: int) -> bool:
        if u == v:
            return False

        type_u = relation_type(u, v)
        type_v = relation_type(v, u)

        if type_u is None or type_v is None:
            return False

        a, b = sorted((u, v))
        edge = (a, b)

        if edge in filtered_edges:
            return False

        if degree[u] >= 4 or degree[v] >= 4:
            return False

        if quota[u][type_u] >= type_limit(type_u):
            return False

        if quota[v][type_v] >= type_limit(type_v):
            return False

        filtered_edges.add(edge)

        degree[u] += 1
        degree[v] += 1

        quota[u][type_u] += 1
        quota[v][type_v] += 1

        return True

    candidates = []

    for u, nbrs in neighbors.items():
        pu = sat_to_plane[u]
        au = sat_to_angle[u]

        left_plane_id = (pu - 1) % num_planes
        right_plane_id = (pu + 1) % num_planes

        same_plane = []
        left_plane = []
        right_plane = []

        for v in nbrs:
            pv = sat_to_plane[v]

            if pv == pu:
                same_plane.append(v)
            elif pv == left_plane_id:
                left_plane.append(v)
            elif pv == right_plane_id:
                right_plane.append(v)

        forward = [
            v for v in same_plane
            if forward_angle_gap_deg(au, sat_to_angle[v]) > 0
        ]

        backward = [
            v for v in same_plane
            if backward_angle_gap_deg(au, sat_to_angle[v]) > 0
        ]

        if forward:
            v = min(forward, key=lambda x: forward_angle_gap_deg(au, sat_to_angle[x]))
            candidates.append((0, forward_angle_gap_deg(au, sat_to_angle[v]), u, v))

        if backward:
            v = min(backward, key=lambda x: backward_angle_gap_deg(au, sat_to_angle[x]))
            candidates.append((0, backward_angle_gap_deg(au, sat_to_angle[v]), u, v))

        if left_plane:
            v = min(left_plane, key=lambda x: angular_distance_deg(au, sat_to_angle[x]))
            candidates.append((1, angular_distance_deg(au, sat_to_angle[v]), u, v))

        if right_plane:
            v = min(right_plane, key=lambda x: angular_distance_deg(au, sat_to_angle[x]))
            candidates.append((1, angular_distance_deg(au, sat_to_angle[v]), u, v))

    candidates.sort(key=lambda x: (x[0], x[1]))

    for _priority, _dist, u, v in candidates:
        add_edge(u, v)
    
    # 第三轮扫描：对于仍未满足边数要求的卫星，尝试从剩余邻居中补齐边，优先级为：同轨 > 左侧 > 右侧，直到所有卫星的度数达到4或没有更多候选边。
    supplement = []

    for u, nbrs in neighbors.items():
        au = sat_to_angle[u]

        for v in nbrs:
            if u == v:
                continue

            edge_type = relation_type(u, v)
            if edge_type is None:
                continue

            av = sat_to_angle[v]

            if edge_type == "same":
                priority = 2
            else:
                priority = 3

            supplement.append(
                (priority, angular_distance_deg(au, av), u, v)
            )

    supplement.sort(key=lambda x: (x[0], x[1]))

    changed = True
    while changed:
        changed = False

        for _priority, _dist, u, v in supplement:
            if degree.get(u, 0) < 4 and degree.get(v, 0) < 4:
                if add_edge(u, v):
                    changed = True

        if all(d >= 4 for d in degree.values()):
            break
    
    # 最终检查：确保所有卫星满足同轨2条、左侧1条、右侧1条的边数要求，否则输出异常报告并抛出错误。
    bad_nodes = [
        u for u in neighbors
        if not (
            quota[u]["same"] == 2
            and quota[u]["left"] == 1
            and quota[u]["right"] == 1
            and degree[u] == 4
        )
    ]

    if bad_nodes:
        examples = [
            (u, degree[u], quota[u])
            for u in bad_nodes[:10]
        ]
        raise ValueError(
            f"无法保证所有卫星满足：同轨2条、左侧1条、右侧1条。"
            f"异常卫星数量: {len(bad_nodes)}，示例: {examples}"
        )

    print("[filter check] first 5 quota examples:")
    for u in list(quota.keys())[:5]:
        print(u, "degree=", degree[u], "quota=", quota[u])
        
    return [[u, v] for u, v in sorted(filtered_edges)]

def print_graph_degree_check(
    graph: Iterable[Sequence[int]],
    title: str,
) -> None:
    degree: Dict[int, int] = {}

    for e in graph:
        u, v = int(e[0]), int(e[1])
        if u == v:
            continue
        a, b = sorted((u, v))
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1

    if not degree:
        print(f"[{title}] graph is empty")
        return

    values = list(degree.values())
    bad_nodes = [u for u, d in degree.items() if d != 4]

    print(f"[{title}] nodes={len(degree)}, edges={len(list(graph))}")
    print(f"[{title}] degree min={min(values)}, max={max(values)}, avg={sum(values) / len(values):.2f}")
    print(f"[{title}] degree=4 nodes={len(degree) - len(bad_nodes)}/{len(degree)}")

    if bad_nodes:
        print(f"[{title}] bad node examples: {[(u, degree[u]) for u in bad_nodes[:10]]}")

def _parse_step_nodes_edges(step_file: Path) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, int]]]:
    """解析底层网络拓扑，提取节点经纬度和物理连线"""
    text = step_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    nodes_section = re.search(
        r"\[NODES\]\nID, Name, Lat\(deg\), Lon\(deg\), Alt\(km\)\n(.*?)(?:\n\n|\Z)",
        text,
        re.S,
    )
    if not nodes_section:
        raise ValueError(f"Cannot parse [NODES] in {str(step_file)}")
    coords: Dict[int, Tuple[float, float]] = {}
    for line in nodes_section.group(1).strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        sid = int(parts[0])
        lat = float(parts[2])
        lon = float(parts[3])
        coords[sid] = (lat, lon)

    links_section = re.search(
        r"\[LINKS\]\nType, SourceID, TargetID\n(.*?)(?:\n\n|\Z)",
        text,
        re.S,
    )
    edges: List[Tuple[int, int]] = []
    if links_section:
        for line in links_section.group(1).strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            u = int(parts[1])
            v = int(parts[2])
            edges.append((u, v))
    return coords, edges


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """使用半正矢公式计算地球上两点间的大圆距离(千米)"""
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def _find_step_file(step_dir: Path, data_idx: int) -> Path:
    """根据时间片索引查找对应的输入 txt 文件"""
    matches = sorted(step_dir.glob(f"step_{data_idx:04d}_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No step file for data_idx={data_idx} under {str(step_dir)}")
    return matches[0]


def build_super_topology(
    original_graph: Iterable[Sequence[int]],
    original_paths: Dict[str, List[List[int]]],
    sat_to_cluster: Dict[int, int],
    sat_coords: Optional[Dict[int, Tuple[float, float]]] = None,
    n_clusters: int = N_CLUSTERS,
) -> Tuple[List[List[int]], Dict[str, List[List[int]]]]:
    """
    根据底层的连线和卫星分簇关系，构建宏观层面的超节点拓扑(Super Topology)。
    """
    candidates: Dict[Tuple[int, int], List[List[int]]] = {}
    
    # 遍历底层的每条边，若源宿节点分属不同簇，则认为这两个簇之间存在宏观连接
    for e in original_graph:
        u, v = int(e[0]), int(e[1])
        if u not in sat_to_cluster or v not in sat_to_cluster:
            continue
        cu = int(sat_to_cluster[u])
        cv = int(sat_to_cluster[v])
        if cu == cv:
            continue
        if not (0 <= cu < n_clusters and 0 <= cv < n_clusters):
            continue
            
        # 收集两个底层卫星之间的微观路由路径
        pkey = f"{u}, {v}"
        plist = original_paths.get(pkey)
        if not plist:
            plist = [[u, v]]
        for p in plist:
            if not p:
                continue
            pp = [int(x) for x in p]
            # 存入到超节点之间的路径候选池中
            candidates.setdefault((cu, cv), []).append(pp)

    super_edges: List[List[int]] = []
    super_paths_all: Dict[str, List[List[int]]] = {}

    # 将收集到的所有跨簇路径去重并汇总
    for (cu, cv), items in sorted(candidates.items()):
        if not items:
            continue
        super_edges.append([cu, cv])
        uniq: List[List[int]] = []
        seen = set()
        for p in items:
            t = tuple(p)
            if t in seen:
                continue
            seen.add(t)
            uniq.append(p)
        super_paths_all[f"{cu}, {cv}"] = uniq

    # 统计每个簇包含的成员
    members: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
    for sid, cid in sat_to_cluster.items():
        c = int(cid)
        if 0 <= c < n_clusters:
            members[c].append(int(sid))
    for c in range(n_clusters):
        members[c].sort()

    # 处理孤立的超节点：强制为其分配一条自环边以防 GNN 崩溃
    out_degree = np.zeros(n_clusters, dtype=np.int64)
    for a, _b in super_edges:
        out_degree[int(a)] += 1

    for c in range(n_clusters):
        if out_degree[c] >= 1:
            continue
        if not members[c]:
            continue
        super_edges.append([c, c])
        super_paths_all[f"{c}, {c}"] = [[members[c][0]]]

    return super_edges, super_paths_all


def _path_weight(path: List[int], sat_coords: Optional[Dict[int, Tuple[float, float]]]) -> Tuple[int, float]:
    """为路由路径进行打分计算。主要参考跳数 (hops)，次要参考物理距离"""
    if len(path) <= 1:
        return 0, 0.0
    hops = len(path) - 1
    if sat_coords is None:
        return hops, 0.0
    dist = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if a not in sat_coords or b not in sat_coords:
            continue
        lat1, lon1 = sat_coords[a]
        lat2, lon2 = sat_coords[b]
        dist += _haversine_km(lat1, lon1, lat2, lon2)
    return hops, dist


def _select_top_k_paths(
    paths: List[List[int]],
    k: int,
    sat_coords: Optional[Dict[int, Tuple[float, float]]],
) -> List[List[int]]:
    """
    【关键修改点】提取前 K 条最短路径。
    如果实际路径数不足 K 条，会自动复制最后一条路径进行填充，确保输出维度严格一致。
    """
    if not paths:
        return []
    # 按照跳数和距离对所有候选路径进行排序
    scored = [(_path_weight(p, sat_coords), p) for p in paths]
    scored.sort(key=lambda x: x[0])
    
    # 截取前 K 个最优路径
    top_k = [p for _, p in scored[:k]]

    # 填充逻辑：不足 8 条时持续复制最后一条
    if top_k and len(top_k) < k:
        last_path = top_k[-1]
        while len(top_k) < k:
            top_k.append(last_path)

    return top_k


def aggregate_traffic(
    original_tm: Dict[str, int],
    sat_to_cluster: Dict[int, int],
    n_clusters: int = N_CLUSTERS,
) -> Dict[str, int]:
    """
    【关键修改点】聚合底层流量需求到超节点层面。
    返回的格式严格符合：{'src, dst': demand}，长度为 18 * 17 = 306（剔除了源宿相同的流量）。
    """
    # 1. 累加流量到 18x18 的二维矩阵
    tm_mat = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for k, demand in original_tm.items():
        src, dst = _parse_tm_key(k)
        if src not in sat_to_cluster or dst not in sat_to_cluster:
            continue
        cs = int(sat_to_cluster[src])
        cd = int(sat_to_cluster[dst])
        # 排除越界异常值
        if not (0 <= cs < n_clusters and 0 <= cd < n_clusters):
            continue
        tm_mat[cs, cd] += int(demand)
        
    # 2. 将二维矩阵展平为字符串字典，严格控制长度为 18*17，排除自己到自己的流量
    tm_dict = {}
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                continue
            # 将 numpy 整型显式转为 python 原生整型以保证序列化兼容
            tm_dict[f"{i}, {j}"] = int(tm_mat[i, j])
            
    return tm_dict


def _layout_centers(
    centers_xy: Optional[np.ndarray],
    super_edges: List[List[int]],
    n_clusters: int,
) -> np.ndarray:
    """计算用于网络可视化的超节点 2D 布局坐标"""
    if centers_xy is not None:
        arr = np.asarray(centers_xy, dtype=np.float64)
        if arr.shape == (n_clusters, 2):
            return arr
    if nx is not None:
        g = nx.DiGraph()
        g.add_nodes_from(range(n_clusters))
        g.add_edges_from([(int(a), int(b)) for a, b in super_edges])
        pos = nx.spring_layout(g, seed=7)
        return np.array([pos[i] for i in range(n_clusters)], dtype=np.float64)

    angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)


def _plot_super_topology(
    out_path: Path,
    centers_xy: Optional[np.ndarray],
    super_edges: List[List[int]],
    super_paths: Dict[str, List[List[int]]],
    ts: str,
    n_clusters: int = N_CLUSTERS,
) -> None:
    """使用 matplotlib 绘制超节点之间的逻辑连接拓扑图"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    coords = _layout_centers(centers_xy, super_edges, n_clusters=n_clusters)
    xs = coords[:, 0]
    ys = coords[:, 1]

    plt.figure(figsize=(10, 7))
    cmap = cm.get_cmap("tab20")
    for cid in range(n_clusters):
        plt.scatter([xs[cid]], [ys[cid]], s=90, color=cmap(cid % 20), zorder=3)
        plt.text(xs[cid], ys[cid], f"{cid}", fontsize=10, ha="center", va="center", color="black", zorder=4)

    legend_added = False
    for a, b in super_edges:
        if int(a) == int(b):
            continue
        key = f"{a}, {b}"
        paths = super_paths.get(key, [])
        if not paths:
            continue
        count = len(paths)
        for i in range(count):
            offset = (i - (count - 1) / 2.0)
            rad = float(np.tanh(offset / 6.0) * 0.6)
            plt.annotate(
                "",
                xy=(xs[b], ys[b]),
                xytext=(xs[a], ys[a]),
                arrowprops=dict(
                    arrowstyle="-",
                    color="grey",
                    lw=1.3,
                    alpha=0.6,
                    connectionstyle=f"arc3,rad={rad}",
                ),
                zorder=1,
            )
        if not legend_added:
            legend_added = True

    plt.title(f"18-Cluster Super-Nodes Topology | {ts}")
    plt.axis("off")
    if legend_added:
        plt.legend(
            handles=[
                Line2D([0], [0], color="grey", lw=1.3, alpha=0.6, label="绘图显示全部路径（PKL保留最短8条）"),
            ],
            loc="upper right",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_super_tm(out_pkl: str, super_slices: List[dict]) -> None:
    """将最终构建完毕的数据列表进行 Pickle 序列化存盘"""
    p = Path(out_pkl)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(super_slices, f, protocol=pickle.HIGHEST_PROTOCOL)


def _validate_super_dataset(super_slices: List[dict], out_pkl: Path) -> None:
    """
    【修改了校验逻辑】
    二次校验构建出的数据集各项格式与维度是否达标，防止非法数据流入 GNN。
    """
    if not super_slices:
        raise AssertionError("super_slices is empty")

    first = super_slices[0]
    tm0 = first["tm"]
    
    # 校验流量矩阵必须是 Dict，且长度为 18 * 17 = 306
    assert isinstance(tm0, dict)
    expected_tm_len = N_CLUSTERS * (N_CLUSTERS - 1)
    assert len(tm0) == expected_tm_len, f"TM length is {len(tm0)}, expected {expected_tm_len}"

    for s in super_slices:
        out_degree = np.zeros(N_CLUSTERS, dtype=np.int64)
        for a, b in s["graph"]:
            out_degree[int(a)] += 1
            k = f"{int(a)}, {int(b)}"
            paths = s["path"].get(k)
            assert isinstance(paths, list)
            # 校验并行路径必须严格等于 MAX_PARALLEL_LINKS (8条)
            assert len(paths) == MAX_PARALLEL_LINKS, f"Paths count is {len(paths)}, expected {MAX_PARALLEL_LINKS}"
        
        # 确保图的连通性：没有孤立向外度为 0 的节点
        assert np.all(out_degree >= 1)

    # 校验落盘文件是否可以正常回读
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with out_pkl.open("rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, list)
        if w:
            raise AssertionError(f"Unexpected warnings during load: {w}")


def _build_super_dataset(
    cluster: ClusterResult,
    original_slices: List[dict],
    step_dir: Optional[Path],
    plot: bool,
    out_dir_for_plot: Path,
    ts: str,
) -> Tuple[List[dict], Optional[Path]]:
    """核心控制流程：驱动超节点的拓扑构建、路径搜索与流量聚合"""
    super_slices: List[dict] = []
    plot_done = False
    plot_path = out_dir_for_plot / "outputs" / "super_topology.png" if plot else None
    missing_satellite_records: Dict[int, Dict[str, object]] = {}

    # 循环遍历每个时间片
    for idx, s in enumerate(original_slices):
        data_idx = int(s["data_idx"])
            
        sat_to_cluster = get_sat_to_cluster(cluster, data_idx)
        graph = s["graph"]
        tm = s["tm"]
        paths = s["path"]

        if cluster.sat_to_plane_global is not None and cluster.sat_to_angle_global is not None:
            if idx == 0:
                print_graph_degree_check(graph, title=f"Original Graph (data_idx={data_idx})")

            graph = _filter_graph_4_neighbor_by_plane(graph, cluster.sat_to_plane_global, cluster.sat_to_angle_global, 
                                                      data_idx=data_idx, missing_records=missing_satellite_records)

            if idx == 0:
                print_graph_degree_check(graph, title=f"Filtered Graph (data_idx={data_idx})")
        sat_coords = None
        # 如有可能，载入物理经纬度用于计算实际衰减/跳数权重
        if step_dir is not None:
            step_file = _find_step_file(step_dir, data_idx)
            sat_coords, _ = _parse_step_nodes_edges(step_file)

        # 1. 构建宏观图拓扑与全量路径映射
        super_graph, super_paths_all = build_super_topology(
            graph,
            paths,
            sat_to_cluster,
            sat_coords=sat_coords,
        )
        
        # 2. 截取并补齐前 8 条路径
        super_paths_top8 = {
            k: _select_top_k_paths(v, k=MAX_PARALLEL_LINKS, sat_coords=sat_coords) 
            for k, v in super_paths_all.items()
        }
        
        # 3. 聚合流量并转为目标字典格式
        super_tm = aggregate_traffic(tm, sat_to_cluster)

        # 装配单时间片数据字典
        super_slice = {"graph": super_graph, "tm": super_tm, "path": super_paths_top8, "data_idx": data_idx}
        super_slices.append(super_slice)

        # 首次构建完毕后绘制基准测试图
        if plot and (not plot_done) and plot_path is not None:
            _plot_super_topology(
                plot_path,
                centers_xy=cluster.centers_xy,
                super_edges=super_graph,
                super_paths=super_paths_all,
                ts=ts,
            )
            plot_done = True

        if (idx + 1) % 25 == 0:
            print(f"已聚合完成 {idx + 1}/{len(original_slices)} 个时间片")
        
    missing_report_path = out_dir_for_plot / "outputs" / f"missing_satellite_info_{ts}.txt"
    write_missing_satellite_info_report(
        missing_report_path,
        missing_satellite_records,
    )
    print(f"✅ 已输出缺失卫星信息报告: {str(missing_report_path)}")

    return super_slices, plot_path





def main() -> int:
    args = parse_args()
    ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_dir = Path(__file__).resolve().parent
    default_out_dir = base_dir / "outputs" / "tms"
    out_pkl = Path(args.out_pkl) if args.out_pkl else (default_out_dir / f"data_1200_super_tm_18cluster_{ts}.pkl")

    # 装载底层簇与流量数据
    cluster = load_cluster_result(args.cluster_pkl)
    original_slices = load_original_tm(args.tm_pkl)

    step_dir = base_dir / "inputs" / "starlink550_data" / "data_1200"
    if not step_dir.exists():
        step_dir = None

    # 执行超节点映射
    super_slices, plot_path = _build_super_dataset(
        cluster=cluster,
        original_slices=original_slices,
        step_dir=step_dir,
        plot=bool(args.plot),
        out_dir_for_plot=base_dir,
        ts=ts,
    )

    # 保存与二次校验
    save_super_tm(str(out_pkl), super_slices)
    _validate_super_dataset(super_slices, out_pkl=out_pkl)

    print(f"✅ 已输出 super TM PKL: {str(out_pkl)}")
    if plot_path is not None:
        print(f"✅ 已输出 super_topology.png: {str(plot_path)}")
    return 0


if __name__ == "__main__":
    code = main()
    assert code == 0
    raise SystemExit(code)



# """
# 按簇聚合的 Starlink 550 流量矩阵与超节点拓扑构建工具。

# 依赖库：networkx、matplotlib、numpy、pandas、pickle
# Python 版本要求：Python >= 3.8
# """

# from __future__ import annotations

# import argparse
# import datetime as _dt
# import json
# import os
# import pickle
# import re
# import warnings
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Iterable, List, Optional, Tuple

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.lines import Line2D
# try:
#     import networkx as nx
# except Exception:
#     nx = None

# try:
#     import pandas as pd
# except Exception:
#     pd = None


# N_CLUSTERS = 18
# MAX_PARALLEL_LINKS = 8


# @dataclass(frozen=True)
# class ClusterResult:
#     sat_to_cluster_by_slice: List[Dict[int, int]]
#     centers_xy: Optional[np.ndarray]
#     radii: Optional[np.ndarray]


# def _parse_int(s: str) -> int:
#     return int(str(s).strip())


# def _parse_tm_key(key: str) -> Tuple[int, int]:
#     parts = str(key).split(",")
#     if len(parts) < 2:
#         raise ValueError(f"Invalid tm key: {key!r}")
#     return _parse_int(parts[0]), _parse_int(parts[1])


# def _parse_cluster_txt(path: Path) -> Dict[int, int]:
#     sat_to_domain: Dict[int, int] = {}
#     with path.open("r", encoding="utf-8") as f:
#         header = f.readline()
#         if "sat_id" not in header or "domain_id" not in header:
#             raise ValueError(f"Unexpected header in {str(path)!r}: {header!r}")
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = [p.strip() for p in line.split(",")]
#             if len(parts) < 7:
#                 continue
#             sat_id = _parse_int(parts[0])
#             domain_id = _parse_int(parts[-1])
#             sat_to_domain[sat_id] = domain_id
#     return sat_to_domain


# def _sorted_step_indices_from_names(names: Iterable[str]) -> List[int]:
#     indices = []
#     for name in names:
#         m = re.search(r"step_(\d+)", name)
#         if not m:
#             continue
#         indices.append(int(m.group(1)))
#     return sorted(set(indices))


# def _load_cluster_from_dir(cluster_dir: Path) -> ClusterResult:
#     files = sorted(cluster_dir.glob("step_*_result_*_domains.txt"))
#     if not files:
#         raise FileNotFoundError(f"No cluster result txt found in: {str(cluster_dir)}")

#     step_idx_to_path: Dict[int, Path] = {}
#     for p in files:
#         m = re.search(r"step_(\d+)_result_", p.name)
#         if not m:
#             continue
#         step_idx_to_path[int(m.group(1))] = p

#     ordered_steps = sorted(step_idx_to_path.keys())
#     sat_to_cluster_by_slice = [_parse_cluster_txt(step_idx_to_path[i]) for i in ordered_steps]
#     return ClusterResult(sat_to_cluster_by_slice=sat_to_cluster_by_slice, centers_xy=None, radii=None)


# def load_cluster_result(cluster_pkl: str) -> ClusterResult:
#     p = Path(cluster_pkl)
#     if p.is_dir():
#         return _load_cluster_from_dir(p)

#     if not p.exists():
#         raise FileNotFoundError(str(p))

#     suffix = p.suffix.lower()
#     if suffix in {".pkl", ".pickle"}:
#         with p.open("rb") as f:
#             obj = pickle.load(f)
#         if isinstance(obj, dict) and "sat_to_cluster_by_slice" in obj:
#             centers = obj.get("centers_xy")
#             radii = obj.get("radii")
#             return ClusterResult(
#                 sat_to_cluster_by_slice=list(obj["sat_to_cluster_by_slice"]),
#                 centers_xy=None if centers is None else np.asarray(centers, dtype=np.float64),
#                 radii=None if radii is None else np.asarray(radii, dtype=np.float64),
#             )
#         if isinstance(obj, list) and obj and isinstance(obj[0], dict):
#             return ClusterResult(sat_to_cluster_by_slice=list(obj), centers_xy=None, radii=None)
#         raise ValueError(f"Unsupported cluster pickle structure: {type(obj)}")

#     if suffix == ".json":
#         with p.open("r", encoding="utf-8") as f:
#             obj = json.load(f)
#         if isinstance(obj, dict) and "sat_to_cluster_by_slice" in obj:
#             centers = obj.get("centers_xy")
#             radii = obj.get("radii")
#             return ClusterResult(
#                 sat_to_cluster_by_slice=[{int(k): int(v) for k, v in m.items()} for m in obj["sat_to_cluster_by_slice"]],
#                 centers_xy=None if centers is None else np.asarray(centers, dtype=np.float64),
#                 radii=None if radii is None else np.asarray(radii, dtype=np.float64),
#             )
#         raise ValueError("Unsupported cluster json structure")

#     raise ValueError(f"Unsupported cluster file type: {suffix}")


# def load_original_tm(tm_pkl: str) -> List[dict]:
#     p = Path(tm_pkl)
#     if not p.exists():
#         raise FileNotFoundError(str(p))
#     with p.open("rb") as f:
#         data = pickle.load(f)
#     if not isinstance(data, list):
#         raise ValueError("Original TM PKL must be a list[dict]")
#     for i, item in enumerate(data[:3]):
#         if not isinstance(item, dict):
#             raise ValueError(f"Original TM item {i} is not dict: {type(item)}")
#         for k in ("graph", "tm", "path", "data_idx"):
#             if k not in item:
#                 raise ValueError(f"Original TM item {i} missing key: {k}")
#     return data


# def _parse_step_nodes_edges(step_file: Path) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, int]]]:
#     text = step_file.read_text(encoding="utf-8").replace("\r\n", "\n")
#     nodes_section = re.search(
#         r"\[NODES\]\nID, Name, Lat\(deg\), Lon\(deg\), Alt\(km\)\n(.*?)(?:\n\n|\Z)",
#         text,
#         re.S,
#     )
#     if not nodes_section:
#         raise ValueError(f"Cannot parse [NODES] in {str(step_file)}")
#     coords: Dict[int, Tuple[float, float]] = {}
#     for line in nodes_section.group(1).strip().split("\n"):
#         if not line.strip():
#             continue
#         parts = [p.strip() for p in line.split(",")]
#         if len(parts) < 5:
#             continue
#         sid = int(parts[0])
#         lat = float(parts[2])
#         lon = float(parts[3])
#         coords[sid] = (lat, lon)

#     links_section = re.search(
#         r"\[LINKS\]\nType, SourceID, TargetID\n(.*?)(?:\n\n|\Z)",
#         text,
#         re.S,
#     )
#     edges: List[Tuple[int, int]] = []
#     if links_section:
#         for line in links_section.group(1).strip().split("\n"):
#             if not line.strip():
#                 continue
#             parts = [p.strip() for p in line.split(",")]
#             if len(parts) < 3:
#                 continue
#             u = int(parts[1])
#             v = int(parts[2])
#             edges.append((u, v))
#     return coords, edges


# def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
#     r = 6371.0
#     p1 = np.radians(lat1)
#     p2 = np.radians(lat2)
#     dlat = p2 - p1
#     dlon = np.radians(lon2 - lon1)
#     a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
#     return float(2 * r * np.arcsin(np.sqrt(a)))


# def _find_step_file(step_dir: Path, data_idx: int) -> Path:
#     matches = sorted(step_dir.glob(f"step_{data_idx:04d}_*.txt"))
#     if not matches:
#         raise FileNotFoundError(f"No step file for data_idx={data_idx} under {str(step_dir)}")
#     return matches[0]


# def build_super_topology(
#     original_graph: Iterable[Iterable[int]],
#     original_paths: Dict[str, List[List[int]]],
#     sat_to_cluster: Dict[int, int],
#     sat_coords: Optional[Dict[int, Tuple[float, float]]] = None,
#     n_clusters: int = N_CLUSTERS,
# ) -> Tuple[List[List[int]], Dict[str, List[List[int]]]]:
#     candidates: Dict[Tuple[int, int], List[List[int]]] = {}
#     for e in original_graph:
#         u, v = int(e[0]), int(e[1])
#         if u not in sat_to_cluster or v not in sat_to_cluster:
#             continue
#         cu = int(sat_to_cluster[u])
#         cv = int(sat_to_cluster[v])
#         if cu == cv:
#             continue
#         if not (0 <= cu < n_clusters and 0 <= cv < n_clusters):
#             continue
#         pkey = f"{u}, {v}"
#         plist = original_paths.get(pkey)
#         if not plist:
#             plist = [[u, v]]
#         for p in plist:
#             if not p:
#                 continue
#             pp = [int(x) for x in p]
#             candidates.setdefault((cu, cv), []).append(pp)

#     super_edges: List[List[int]] = []
#     super_paths_all: Dict[str, List[List[int]]] = {}

#     for (cu, cv), items in sorted(candidates.items()):
#         if not items:
#             continue
#         super_edges.append([cu, cv])
#         uniq: List[List[int]] = []
#         seen = set()
#         for p in items:
#             t = tuple(p)
#             if t in seen:
#                 continue
#             seen.add(t)
#             uniq.append(p)
#         super_paths_all[f"{cu}, {cv}"] = uniq

#     members: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
#     for sid, cid in sat_to_cluster.items():
#         c = int(cid)
#         if 0 <= c < n_clusters:
#             members[c].append(int(sid))
#     for c in range(n_clusters):
#         members[c].sort()

#     out_degree = np.zeros(n_clusters, dtype=np.int64)
#     for a, _b in super_edges:
#         out_degree[int(a)] += 1

#     for c in range(n_clusters):
#         if out_degree[c] >= 1:
#             continue
#         if not members[c]:
#             continue
#         super_edges.append([c, c])
#         super_paths_all[f"{c}, {c}"] = [[members[c][0]]]

#     return super_edges, super_paths_all


# def _path_weight(path: List[int], sat_coords: Optional[Dict[int, Tuple[float, float]]]) -> Tuple[int, float]:
#     if len(path) <= 1:
#         return 0, 0.0
#     hops = len(path) - 1
#     if sat_coords is None:
#         return hops, 0.0
#     dist = 0.0
#     for a, b in zip(path[:-1], path[1:]):
#         if a not in sat_coords or b not in sat_coords:
#             continue
#         lat1, lon1 = sat_coords[a]
#         lat2, lon2 = sat_coords[b]
#         dist += _haversine_km(lat1, lon1, lat2, lon2)
#     return hops, dist


# def _select_top_k_paths(
#     paths: List[List[int]],
#     k: int,
#     sat_coords: Optional[Dict[int, Tuple[float, float]]],
# ) -> List[List[int]]:
#     if not paths:
#         return []
#     scored = [(_path_weight(p, sat_coords), p) for p in paths]
#     scored.sort(key=lambda x: x[0])
#     return [p for _, p in scored[:k]]


# def aggregate_traffic(
#     original_tm: Dict[str, int],
#     sat_to_cluster: Dict[int, int],
#     n_clusters: int = N_CLUSTERS,
# ) -> np.ndarray:
#     tm_mat = np.zeros((n_clusters, n_clusters), dtype=np.int64)
#     for k, demand in original_tm.items():
#         src, dst = _parse_tm_key(k)
#         if src not in sat_to_cluster or dst not in sat_to_cluster:
#             continue
#         cs = int(sat_to_cluster[src])
#         cd = int(sat_to_cluster[dst])
#         if not (0 <= cs < n_clusters and 0 <= cd < n_clusters):
#             continue
#         tm_mat[cs, cd] += int(demand)
#     return tm_mat


# def _layout_centers(
#     centers_xy: Optional[np.ndarray],
#     super_edges: List[List[int]],
#     n_clusters: int,
# ) -> np.ndarray:
#     if centers_xy is not None:
#         arr = np.asarray(centers_xy, dtype=np.float64)
#         if arr.shape == (n_clusters, 2):
#             return arr
#     if nx is not None:
#         g = nx.DiGraph()
#         g.add_nodes_from(range(n_clusters))
#         g.add_edges_from([(int(a), int(b)) for a, b in super_edges])
#         pos = nx.spring_layout(g, seed=7)
#         return np.array([pos[i] for i in range(n_clusters)], dtype=np.float64)

#     angles = np.linspace(0.0, 2.0 * np.pi, n_clusters, endpoint=False)
#     return np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float64)


# def _plot_super_topology(
#     out_path: Path,
#     centers_xy: Optional[np.ndarray],
#     super_edges: List[List[int]],
#     super_paths: Dict[str, List[List[int]]],
#     ts: str,
#     n_clusters: int = N_CLUSTERS,
# ) -> None:
#     plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
#     plt.rcParams["axes.unicode_minus"] = False
#     coords = _layout_centers(centers_xy, super_edges, n_clusters=n_clusters)
#     xs = coords[:, 0]
#     ys = coords[:, 1]

#     plt.figure(figsize=(10, 7))
#     cmap = plt.get_cmap("tab20")
#     for cid in range(n_clusters):
#         plt.scatter([xs[cid]], [ys[cid]], s=90, color=cmap(cid % 20), zorder=3)
#         plt.text(xs[cid], ys[cid], f"{cid}", fontsize=10, ha="center", va="center", color="black", zorder=4)

#     legend_added = False
#     for a, b in super_edges:
#         if int(a) == int(b):
#             continue
#         key = f"{a}, {b}"
#         paths = super_paths.get(key, [])
#         if not paths:
#             continue
#         count = len(paths)
#         for i in range(count):
#             offset = (i - (count - 1) / 2.0)
#             rad = float(np.tanh(offset / 6.0) * 0.6)
#             plt.annotate(
#                 "",
#                 xy=(xs[b], ys[b]),
#                 xytext=(xs[a], ys[a]),
#                 arrowprops=dict(
#                     arrowstyle="-",
#                     color="grey",
#                     lw=1.3,
#                     alpha=0.6,
#                     connectionstyle=f"arc3,rad={rad}",
#                 ),
#                 zorder=1,
#             )
#         if not legend_added:
#             legend_added = True

#     plt.title(f"18-Cluster Super-Nodes Topology | {ts}")
#     plt.axis("off")
#     if legend_added:
#         plt.legend(
#             handles=[
#                 Line2D([0], [0], color="grey", lw=1.3, alpha=0.6, label="绘图显示全部路径（PKL保留最短8条）"),
#             ],
#             loc="upper right",
#         )
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close()


# def save_super_tm(out_pkl: str, super_slices: List[dict]) -> None:
#     p = Path(out_pkl)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     with p.open("wb") as f:
#         pickle.dump(super_slices, f, protocol=pickle.HIGHEST_PROTOCOL)


# def _validate_super_dataset(super_slices: List[dict], out_pkl: Path) -> None:
#     if not super_slices:
#         raise AssertionError("super_slices is empty")

#     first = super_slices[0]
#     tm0 = first["tm"]
#     assert isinstance(tm0, list)
#     assert len(tm0) == N_CLUSTERS
#     assert all(isinstance(r, list) and len(r) == N_CLUSTERS for r in tm0)

#     for s in super_slices:
#         out_degree = np.zeros(N_CLUSTERS, dtype=np.int64)
#         for a, b in s["graph"]:
#             out_degree[int(a)] += 1
#             k = f"{int(a)}, {int(b)}"
#             paths = s["path"].get(k)
#             assert isinstance(paths, list)
#             assert 1 <= len(paths) <= MAX_PARALLEL_LINKS
#         assert np.all(out_degree >= 1)

#     with warnings.catch_warnings(record=True) as w:
#         warnings.simplefilter("always")
#         with out_pkl.open("rb") as f:
#             obj = pickle.load(f)
#         assert isinstance(obj, list)
#         if w:
#             raise AssertionError(f"Unexpected warnings during load: {w}")


# def _build_super_dataset(
#     cluster: ClusterResult,
#     original_slices: List[dict],
#     step_dir: Optional[Path],
#     plot: bool,
#     out_dir_for_plot: Path,
#     ts: str,
# ) -> Tuple[List[dict], Optional[Path]]:
#     super_slices: List[dict] = []
#     plot_done = False
#     plot_path = out_dir_for_plot / "super_topology.png" if plot else None

#     for idx, s in enumerate(original_slices):
#         data_idx = int(s["data_idx"])
#         if data_idx >= len(cluster.sat_to_cluster_by_slice):
#             raise IndexError(f"cluster slice count={len(cluster.sat_to_cluster_by_slice)} < data_idx={data_idx}")
#         sat_to_cluster = cluster.sat_to_cluster_by_slice[data_idx]
#         graph = s["graph"]
#         tm = s["tm"]
#         paths = s["path"]

#         sat_coords = None
#         if step_dir is not None:
#             step_file = _find_step_file(step_dir, data_idx)
#             sat_coords, _ = _parse_step_nodes_edges(step_file)

#         super_graph, super_paths_all = build_super_topology(
#             graph,
#             paths,
#             sat_to_cluster,
#             sat_coords=sat_coords,
#         )
#         super_paths_top8 = {
#             k: _select_top_k_paths(v, k=MAX_PARALLEL_LINKS, sat_coords=sat_coords) for k, v in super_paths_all.items()
#         }
#         super_tm = aggregate_traffic(tm, sat_to_cluster)

#         super_slice = {"graph": super_graph, "tm": super_tm.tolist(), "path": super_paths_top8, "data_idx": data_idx}
#         super_slices.append(super_slice)

#         if plot and (not plot_done) and plot_path is not None:
#             _plot_super_topology(
#                 plot_path,
#                 centers_xy=cluster.centers_xy,
#                 super_edges=super_graph,
#                 super_paths=super_paths_all,
#                 ts=ts,
#             )
#             plot_done = True

#         if (idx + 1) % 25 == 0:
#             print(f"已聚合完成 {idx + 1}/{len(original_slices)} 个时间片")

#     return super_slices, plot_path


# def parse_args() -> argparse.Namespace:
#     base_dir = Path(__file__).resolve().parent
#     default_cluster = base_dir / "outputs_all_steps"
#     default_tm = base_dir / "outputs" / "tms" / "all_time_slices.pkl"

#     p = argparse.ArgumentParser()
#     p.add_argument("--cluster_pkl", type=str, default=str(default_cluster))
#     p.add_argument("--tm_pkl", type=str, default=str(default_tm))
#     p.add_argument("--out_pkl", type=str, default="")
#     p.add_argument("--plot", action="store_true", default=True)
#     return p.parse_args()


# def main() -> int:
#     args = parse_args()
#     ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#     base_dir = Path(__file__).resolve().parent
#     default_out_dir = base_dir / "outputs" / "tms"
#     out_pkl = Path(args.out_pkl) if args.out_pkl else (default_out_dir / f"super_tm_18cluster_{ts}.pkl")

#     cluster = load_cluster_result(args.cluster_pkl)
#     original_slices = load_original_tm(args.tm_pkl)

#     step_dir = base_dir / "inputs" / "starlink550_data" / "data_1100"
#     if not step_dir.exists():
#         step_dir = None

#     super_slices, plot_path = _build_super_dataset(
#         cluster=cluster,
#         original_slices=original_slices,
#         step_dir=step_dir,
#         plot=bool(args.plot),
#         out_dir_for_plot=base_dir,
#         ts=ts,
#     )

#     save_super_tm(str(out_pkl), super_slices)
#     _validate_super_dataset(super_slices, out_pkl=out_pkl)

#     print(f"✅ 已输出 super TM PKL: {str(out_pkl)}")
#     if plot_path is not None:
#         print(f"✅ 已输出 super_topology.png: {str(plot_path)}")
#     return 0


# if __name__ == "__main__":
#     code = main()
#     assert code == 0
#     raise SystemExit(code)
