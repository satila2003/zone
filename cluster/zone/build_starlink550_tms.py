import os
import re
import pickle
import math
import argparse
import numpy as np
import rasterio
import networkx as nx  # 新增：用于构建图结构
from itertools import islice  # 新增：用于提取前K条路径
from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds
'''
这个文件是跑data_build.py的前置条件
这个文件输出的是不带cluster的结果，但是构建数据需要这个文件
'''


# ================= 路径与超参数配置 =================
INPUT_DIR = r"F:\Py_Project\always\cluster\zone\inputs\starlink550_data\data_1100"
OUTPUT_DIR = r"F:\Py_Project\always\cluster\zone\outputs\tms"
LANDSCAN_PATH = r"f:\Py_Project\always\cluster\zone\landscan-global-2024.tif"
OUTPUT_PKL_NAME = "all_time_slices.pkl"  # 输出的统一 PKL 文件名

NUM_SLICES = 500
RADIUS_KM = 50  # 查询卫星附近最大人口密度的半径(公里)
K_PATHS = 8  # 新增：定义 K-shortest paths 的 K 值

# 全新人口等级阈值
POPULATION_BINS = [
    (1, 500),
    (501, 2500),
    (2501, 5000),
    (5001, 7500),
    (7501, 10000),
    (10001, 15000),
    (15001, 25000),
    (25001, 130000),
]
# 对应的流量需求缩放值
BIN_TO_DEMAND = np.array([10, 15, 25, 35, 50, 75, 100, 120], dtype=np.int16)


# ================= 核心功能函数 =================

def k_shortest_paths(G, source, target, k=5):
    """【新增】计算源节点到目的节点的前 K 条最短路径 (基于跳数)"""
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def parse_step_file(file_path):
    """同时解析 txt 中的 卫星节点坐标 与 网络拓扑链接"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().replace("\r\n", "\n")

    # 1. 解析节点 [NODES]
    m_nodes = re.search(
        r"\[NODES\]\nID, Name, Lat\(deg\), Lon\(deg\), Alt\(km\)\n(.*?)(?:\n\n|\Z)",
        content, re.S
    )
    if not m_nodes:
        return None, None

    node_lines = m_nodes.group(1).strip().split("\n")
    sat_ids, lats, lons = [], [], []
    for line in node_lines:
        if not line.strip(): continue
        parts = [p.strip() for p in line.split(",")]
        sat_ids.append(int(parts[0]))
        lats.append(float(parts[2]))
        lons.append(float(parts[3]))

    # 2. 解析链接 [LINKS]
    m_links = re.search(
        r"\[LINKS\]\nType, SourceID, TargetID\n(.*?)(?:\n\n|\Z)",
        content, re.S
    )
    graph = []
    if m_links:
        link_lines = m_links.group(1).strip().split("\n")
        for line in link_lines:
            if not line.strip(): continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                graph.append([int(parts[1]), int(parts[2])])

    return (
               np.array(sat_ids, dtype=np.int32), np.array(lats, dtype=np.float64),
               np.array(lons, dtype=np.float64)), graph


def sample_max_population_density_in_radius(global_pop_data, transform, nodata_val, width, height, lats, lons,
                                            radius_km=3.0):
    """
    【极致优化版本】
    利用全局内存 Numpy 数组 + 向量化数学运算 进行范围最大值提取。
    """
    out = np.full(len(lats), -1.0, dtype=np.float32)

    # 1. 向量化计算经纬度偏移量
    lat_deg_per_km = 1 / 111.32
    safe_lats = np.clip(lats, -89.9, 89.9)  # 避免极点除以0
    lon_deg_per_km = 1 / (111.32 * np.cos(np.radians(safe_lats)))

    d_lats = radius_km * lat_deg_per_km
    d_lons = radius_km * lon_deg_per_km

    # 2. 向量化得出所有卫星的包围盒地理坐标
    lefts = lons - d_lons
    bottoms = lats - d_lats
    rights = lons + d_lons
    tops = lats + d_lats

    # 3. 向量化将地理坐标转为像素索引 (使用仿射变换矩阵的逆矩阵)
    inv_transform = ~transform
    col1, row1 = inv_transform * (lefts, tops)  # 左上角
    col2, row2 = inv_transform * (rights, bottoms)  # 右下角

    px1 = np.floor(col1).astype(int)
    py1 = np.floor(row1).astype(int)
    px2 = np.ceil(col2).astype(int)
    py2 = np.ceil(row2).astype(int)

    # 安全裁剪，防止索引越界
    px1 = np.clip(px1, 0, width)
    px2 = np.clip(px2, 0, width)
    py1 = np.clip(py1, 0, height)
    py2 = np.clip(py2, 0, height)

    # 4. 在内存中进行高速 Numpy 切片与最大值计算
    for i in range(len(lats)):
        r1, r2 = py1[i], py2[i]
        c1, c2 = px1[i], px2[i]

        # 如果盒子大小异常则跳过
        if r1 >= r2 or c1 >= c2:
            continue

        # 神级优化：直接在内存里通过原生 Numpy 切片拿数据，耗时接近 0
        window_data = global_pop_data[r1:r2, c1:c2]

        # 提取有效数据
        if nodata_val is not None:
            valid_mask = (window_data != nodata_val) & (~np.isnan(window_data))
        else:
            valid_mask = ~np.isnan(window_data)

        valid_data = window_data[valid_mask]

        if valid_data.size > 0:
            out[i] = np.max(valid_data)

    return out


def get_bin_indices(population_density):
    """根据人口密度查询 Bin Index。如果是海洋或无人区返回 -1"""
    pop = np.asarray(population_density, dtype=np.float32)
    n = pop.shape[0]
    bin_idx = np.full(n, -1, dtype=np.int16)

    valid = pop >= 0
    if not np.any(valid):
        return bin_idx

    for i, (lower, upper) in enumerate(POPULATION_BINS):
        mask = valid & (pop >= float(lower)) & (pop <= float(upper))
        bin_idx[mask] = i

    return bin_idx


def build_one_slice(file_path, global_pop_data, transform, nodata_val, width, height, data_idx):
    """构建单一时间片的图和流量矩阵"""
    nodes_data, graph = parse_step_file(file_path)
    if nodes_data is None:
        return None

    sat_ids, lats, lons = nodes_data
    sat_id_to_idx = {sid: i for i, sid in enumerate(sat_ids)}

    # 【新增】构建 NetworkX 无向图用于计算路径
    G = nx.Graph()
    G.add_nodes_from(sat_ids)
    G.add_edges_from(graph)

    # 传入内存中的地图数据进行提取
    pop = sample_max_population_density_in_radius(
        global_pop_data, transform, nodata_val, width, height, lats, lons, radius_km=RADIUS_KM
    )
    bin_idx = get_bin_indices(pop)

    tm = {}
    path_data = {}  # 【新增】存储路径字典

    for src, dst in graph:
        if src not in sat_id_to_idx or dst not in sat_id_to_idx:
            continue

        s_idx = sat_id_to_idx[src]
        d_idx = sat_id_to_idx[dst]

        s_bin = bin_idx[s_idx]
        d_bin = bin_idx[d_idx]

        # 海洋或无人区筛除
        if s_bin < 0 or d_bin < 0:
            continue

        higher_bin = max(s_bin, d_bin)
        demand = int(BIN_TO_DEMAND[higher_bin])

        key = f"{int(src)}, {int(dst)}"
        tm[key] = demand

        # 【新增】只为有 demand 的 SD 对生成前 K 条最短路径
        path_data[key] = k_shortest_paths(G, int(src), int(dst), k=K_PATHS)

    return {
        "graph": graph,
        "tm": tm,
        "path": path_data,  # 【新增】将计算结果添加到返回字典中
        "data_idx": int(data_idx)
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
    step_files.sort(key=lambda x: int(re.search(r'step_(\d+)', x).group(1)))
    step_files = step_files[:args.num_slices]

    if len(step_files) < args.num_slices:
        raise RuntimeError(f"输入数量不足：期望 {args.num_slices}，实际 {len(step_files)} 个")

    all_time_slices_list = []

    try:
        print("==================================================")
        print("正在将全球高精度人口数据加载到内存中 (约需几秒钟)...")
        # 1. 核心优化点：在进入时间片循环前，一次性将全地图读入内存
        with rasterio.open(LANDSCAN_PATH) as src:
            transform = src.transform
            nodata_val = src.nodata
            width = src.width
            height = src.height
            global_pop_data = src.read(1)  # 这一步会消耗约 3GB 内存
        print("==================================================")

        for idx, fname in enumerate(step_files):
            in_path = os.path.join(INPUT_DIR, fname)

            slice_dict = build_one_slice(
                in_path, global_pop_data, transform, nodata_val, width, height, idx
            )
            if slice_dict is not None:
                all_time_slices_list.append(slice_dict)

            # 由于加入了图路径计算稍微变慢，将打印频率调回为每处理 10 个打印一次
            if (idx + 1) % 10 == 0:
                print(f"已完成 {idx + 1}/{len(step_files)} 个时间片的构建")

        with open(out_pkl_path, "wb") as f:
            pickle.dump(all_time_slices_list, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"\n构建完毕！成功保存所有数据至 PKL 文件: {out_pkl_path}")
        print(f"文件内共包含 {len(all_time_slices_list)} 个时间片的数据。")

    except RasterioIOError as e:
        raise RuntimeError(f"无法打开人口密度 TIF 文件: {LANDSCAN_PATH}") from e


if __name__ == "__main__":
    main()

