import os
import pickle
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt

# 这个脚本用于从域内数据集中提取指定时间片和域的数据，并将其保存为新的 pickle 文件，同时绘制该域的拓扑图。

# ================= 路径配置 =================
BASE_DIR = Path(__file__).parent
INTRA_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550_intra.pkl"

# ================= 选择参数（直接在这里改） =================
# 注意：TARGET_SLICE_IDX 是列表索引（第几个时间片），不是 data_idx 值
TARGET_SLICE_IDX = 0
# 指定域编号；如果为 None 则自动选择该时间片中最小的域编号
TARGET_DOMAIN_ID = None

# 输出目录（可按需修改）
OUTPUT_PKL_DIR = BASE_DIR / "outputs" / "tms"
OUTPUT_IMG_DIR = BASE_DIR / "outputs"


def main():
    if not os.path.exists(INTRA_PKL_PATH):
        raise FileNotFoundError(f"找不到域内数据集: {INTRA_PKL_PATH}")

    with open(INTRA_PKL_PATH, "rb") as f:
        slices = pickle.load(f)

    if not slices:
        raise RuntimeError("域内数据集为空，无法提取。")

    if TARGET_SLICE_IDX < 0 or TARGET_SLICE_IDX >= len(slices):
        raise IndexError(f"时间片索引越界: {TARGET_SLICE_IDX}，当前共有 {len(slices)} 个时间片。")

    target_slice = slices[TARGET_SLICE_IDX]
    domains = target_slice.get("domains", {})
    if not domains:
        raise RuntimeError(f"时间片 {TARGET_SLICE_IDX} 没有 domains 数据。")

    if TARGET_DOMAIN_ID is None:
        domain_id = sorted(domains.keys())[0]
    else:
        if TARGET_DOMAIN_ID not in domains:
            available = sorted(domains.keys())
            raise KeyError(f"时间片 {TARGET_SLICE_IDX} 不包含域 {TARGET_DOMAIN_ID}，可用域: {available}")
        domain_id = TARGET_DOMAIN_ID

    domain_data = domains[domain_id]

    os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    output_pkl_path = OUTPUT_PKL_DIR / f"slice_{TARGET_SLICE_IDX}_domain_{domain_id}.pkl"
    output_img_path = OUTPUT_IMG_DIR / f"slice_{TARGET_SLICE_IDX}_domain_{domain_id}_topology.png"

    export_data = {
        "data_idx": target_slice.get("data_idx"),
        "domain_id": domain_id,
        **domain_data
    }

    with open(output_pkl_path, "wb") as f:
        pickle.dump(export_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # -------- 绘图 --------
    edges = domain_data.get("graph", [])
    active_nodes = domain_data.get("active_sat_ids", [])

    G = nx.Graph()
    if active_nodes:
        G.add_nodes_from(active_nodes)
    G.add_edges_from(edges)

    if G.number_of_nodes() == 0:
        raise RuntimeError("该域内没有节点，无法绘图。")

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_edges(G, pos, edge_color="slategray", alpha=0.6, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="#7ec8e3", edgecolors="black", linewidths=0.5)

    plt.title(f"Slice {TARGET_SLICE_IDX} - Domain {domain_id} Topology", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    plt.close()

    print(f"✅ 已导出时间片 {TARGET_SLICE_IDX} 的域 {domain_id} 数据: {output_pkl_path}")
    print(f"✅ 已保存拓扑图: {output_img_path}")
    print(f"节点数: {G.number_of_nodes()} | 边数: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
