
import os
import re
import pickle
import math
import argparse
import random
import numpy as np
import rasterio
import networkx as nx
from pathlib import Path
from itertools import islice
from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds
import concurrent.futures  # 【新增】用于多进程并发计算

'''
这个文件是跑data_build.py的前置条件
这个文件输出的是不带cluster的结果，但是构建数据需要这个文件
'''

# ================= 路径与超参数配置 =================
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs" / "starlink550_data"
OUTPUT_DIR = BASE_DIR / "outputs" / "tms"
LANDSCAN_PATH = BASE_DIR / "inputs" / "landscan-global-2024.tif"
OUTPUT_PKL_NAME = "starlink550.pkl"

NUM_SLICES = 500
RADIUS_KM = 50
K_PATHS = 8
NUM_ACTIVE_FLOWS = 3000

POPULATION_BINS = [
    (1, 500), (501, 2500), (2501, 5000), (5001, 7500),
    (7501, 10000), (10001, 15000), (15001, 25000), (25001, 130000),
]
BIN_TO_DEMAND = np.array([10, 15, 25, 35, 50, 75, 100, 120], dtype=np.int16)


# ================= 核心功能函数 =================

def k_shortest_paths_fast(G, source, target, k=8):
    """
    【算法级优化】：混合路径搜索策略
    优先使用 O(V+E) 的 BFS 获取同等跳数的最短路径（速度是 Yen's 算法的 100 倍）。
    不足时再用 Yen's 算法补充。
    """
    try:
        # 星链网格拓扑中，最短跳数路径通常有很多条，BFS 可以极速全部找出
        fast_paths = list(islice(nx.all_shortest_paths(G, source, target), k))
        if len(fast_paths) == k:
            return fast_paths

        # 极端情况：如果绝对最短路径不足 K 条，回退到 Yen's 算法寻找次优解 (如 L+1 跳)
        return list(islice(nx.shortest_simple_paths(G, source, target), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def parse_step_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")

    m_nodes = re.search(r"\[NODES\]\n[^\n]*\n(.*?)(?:\n\n|\Z)", content, re.S)
    if not m_nodes: return None, None

    node_lines = m_nodes.group(1).strip().split("\n")
    sat_ids, lats, lons = [], [], []
    name_to_id = {}

    for line in node_lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4: continue

        sat_id, sat_name, lat, lon = int(parts[0]), parts[1], float(parts[2]), float(parts[3])
        sat_ids.append(sat_id)
        lats.append(lat)
        lons.append(lon)
        name_to_id[sat_name] = sat_id

    m_links = re.search(r"\[LINKS\]\n[^\n]*\n(.*?)(?:\n\n|\Z)", content, re.S)
    graph = []

    if m_links:
        link_lines = m_links.group(1).strip().split("\n")
        for line in link_lines:
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3: continue

            src_name, dst_name = parts[1], parts[2]
            if src_name in name_to_id and dst_name in name_to_id:
                graph.append([name_to_id[src_name], name_to_id[dst_name]])

    return (
           np.array(sat_ids, dtype=np.int32), np.array(lats, dtype=np.float64), np.array(lons, dtype=np.float64)), graph


def sample_max_population_density_in_radius(global_pop_data, transform, nodata_val, width, height, lats, lons,
                                            radius_km=3.0):
    out = np.full(len(lats), -1.0, dtype=np.float32)
    lat_deg_per_km = 1 / 111.32
    safe_lats = np.clip(lats, -89.9, 89.9)
    lon_deg_per_km = 1 / (111.32 * np.cos(np.radians(safe_lats)))

    d_lats, d_lons = radius_km * lat_deg_per_km, radius_km * lon_deg_per_km
    lefts, bottoms = lons - d_lons, lats - d_lats
    rights, tops = lons + d_lons, lats + d_lats

    inv_transform = ~transform
    col1, row1 = inv_transform * (lefts, tops)
    col2, row2 = inv_transform * (rights, bottoms)

    px1, py1 = np.clip(np.floor(col1).astype(int), 0, width), np.clip(np.floor(row1).astype(int), 0, height)
    px2, py2 = np.clip(np.ceil(col2).astype(int), 0, width), np.clip(np.ceil(row2).astype(int), 0, height)

    for i in range(len(lats)):
        r1, r2, c1, c2 = py1[i], py2[i], px1[i], px2[i]
        if r1 >= r2 or c1 >= c2: continue

        window_data = global_pop_data[r1:r2, c1:c2]
        valid_mask = (window_data != nodata_val) & (~np.isnan(window_data)) if nodata_val is not None else ~np.isnan(
            window_data)
        valid_data = window_data[valid_mask]

        if valid_data.size > 0:
            out[i] = np.max(valid_data)
    return out


def get_bin_indices(population_density):
    pop = np.asarray(population_density, dtype=np.float32)
    bin_idx = np.full(pop.shape[0], -1, dtype=np.int16)
    valid = pop >= 0
    if not np.any(valid): return bin_idx
    for i, (lower, upper) in enumerate(POPULATION_BINS):
        mask = valid & (pop >= float(lower)) & (pop <= float(upper))
        bin_idx[mask] = i
    return bin_idx


def prepare_task_data(file_path, global_pop_data, transform, nodata_val, width, height, data_idx):
    """【阶段1】：仅负责解析文件、映射人口数据、生成 3000 个采样对与需求"""
    nodes_data, graph = parse_step_file(file_path)
    if nodes_data is None or graph is None: return None

    sat_ids, lats, lons = nodes_data
    pop = sample_max_population_density_in_radius(global_pop_data, transform, nodata_val, width, height, lats, lons,
                                                  radius_km=RADIUS_KM)
    bin_idx = get_bin_indices(pop)
    valid_sat_indices = np.where(bin_idx >= 0)[0]

    if len(valid_sat_indices) < 2:
        return {"graph": graph, "pairs_with_demand": [], "data_idx": int(data_idx), "sat_ids": sat_ids.tolist()}

    random.seed(data_idx)
    sampled_pairs = set()
    attempts, max_attempts = 0, NUM_ACTIVE_FLOWS * 5

    while len(sampled_pairs) < NUM_ACTIVE_FLOWS and attempts < max_attempts:
        attempts += 1
        s_idx, d_idx = random.choice(valid_sat_indices), random.choice(valid_sat_indices)
        if s_idx != d_idx:
            pair = tuple(sorted((sat_ids[s_idx], sat_ids[d_idx])))
            sampled_pairs.add((s_idx, d_idx, pair[0], pair[1]))

    pairs_with_demand = []
    for s_idx, d_idx, src_id, dst_id in sampled_pairs:
        demand = int(BIN_TO_DEMAND[max(bin_idx[s_idx], bin_idx[d_idx])])
        pairs_with_demand.append((int(src_id), int(dst_id), demand))

    return {
        "graph": graph,
        "pairs_with_demand": pairs_with_demand,
        "data_idx": int(data_idx),
        "sat_ids": sat_ids.tolist()
    }


def compute_routing_worker(task_data):
    """【阶段2】多进程 Worker：脱离 TIF 内存，利用全核 CPU 暴力寻路"""
    if task_data is None: return None

    graph_edges = task_data["graph"]
    pairs_with_demand = task_data["pairs_with_demand"]
    data_idx = task_data["data_idx"]
    sat_ids = task_data["sat_ids"]

    G = nx.Graph()
    G.add_nodes_from(sat_ids)
    G.add_edges_from(graph_edges)

    tm = {}
    path_data = {}

    for src_id, dst_id, demand in pairs_with_demand:
        key = f"{src_id}, {dst_id}"
        tm[key] = demand
        # 调用极速寻路算法
        path_data[key] = k_shortest_paths_fast(G, src_id, dst_id, k=K_PATHS)

    return {
        "graph": graph_edges,
        "tm": tm,
        "path": path_data,
        "data_idx": data_idx
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_slices", type=int, default=NUM_SLICES)
    p.add_argument("--overwrite", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_pkl_path = os.path.join(OUTPUT_DIR, OUTPUT_PKL_NAME)

    if (not args.overwrite) and os.path.exists(out_pkl_path):
        print(f"[{out_pkl_path}] 已存在，跳过处理。")
        return

    step_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("step_") and f.endswith(".txt")]
    step_files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))  # type: ignore[union-attr]
    step_files = step_files[:args.num_slices]

    if len(step_files) < args.num_slices:
        raise RuntimeError(f"输入数量不足：期望 {args.num_slices}，实际 {len(step_files)} 个")

    routing_tasks = []

    try:
        print("==================================================")
        print("▶ 阶段一：挂载 TIF 人口矩阵")
        with rasterio.open(LANDSCAN_PATH) as src:
            transform = src.transform
            nodata_val = src.nodata
            width = src.width
            height = src.height
            global_pop_data = src.read(1)

        for idx, fname in enumerate(step_files):
            in_path = os.path.join(INPUT_DIR, fname)
            task = prepare_task_data(in_path, global_pop_data, transform, nodata_val, width, height, idx)
            if task is not None:
                routing_tasks.append(task)

            if (idx + 1) % 50 == 0:
                print(f"任务提取进度: {idx + 1}/{len(step_files)}")

        del global_pop_data  # 极其关键：防止传参给多进程时电脑死机

        print("\n==================================================")
        print(f"▶ 阶段二：启动多进程架构，开始 CPU 全核并行图计算...")

        all_time_slices_list = []
        # 获取系统 CPU 核心数，满载运行
        max_workers = max(1, os.cpu_count() - 1)  # 留一个核心给操作系统

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到进程池
            futures = [executor.submit(compute_routing_worker, task) for task in routing_tasks]

            # 获取执行结果，带进度打印
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                result = future.result()
                if result is not None:
                    all_time_slices_list.append(result)

                if (idx + 1) % 10 == 0:
                    print(f"核心寻路进度: {idx + 1}/{len(routing_tasks)} 个时间片已完成")

        # 由于多进程返回顺序是乱的，按 data_idx 重新排序
        all_time_slices_list.sort(key=lambda x: x['data_idx'])

        with open(out_pkl_path, "wb") as f:
            pickle.dump(all_time_slices_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n==================================================")
        print(f"✅ 构建完毕！成功保存所有数据至: {out_pkl_path}")
        print(f"✅ 文件内共包含 {len(all_time_slices_list)} 个时间片的数据。")

    except RasterioIOError as e:
        raise RuntimeError(f"无法打开人口密度 TIF 文件: {LANDSCAN_PATH}") from e


if __name__ == "__main__":
    main()