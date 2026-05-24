import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import itertools
 # 基于单时间片的微观物理图，构建宏观超节点图（保留全量物理链路），并基于最小跳数路径提取物理端点组合。

# ================= 路径配置 =================
BASE_DIR = Path(__file__).parent
# 输入：分域结果 txt
CLUSTER_TXT_PATH = BASE_DIR / "outputs" / "starlink550_data_v1.5_domain_result_18_raan_orbital_angle.txt"
# 输入：微观 PKL 数据集
MICRO_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550.pkl"
# 输出：宏观（分域后）PKL 数据集
MACRO_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550_cluster.pkl"
# 输出：拓扑可视化图片
TOPO_IMG_PATH = BASE_DIR / "outputs" / "super_node_topo_multigraph.png"

# 超参数
MAX_PARALLEL_LINKS = 8  # 宏观超节点之间的最大并行路径数
N_DOMAINS = 18  # 预期的域总数


# ================= 核心工具函数 =================

def load_satellite_to_domain_mapping(txt_path):
    """解析分域TXT文件，构建 卫星ID -> 域ID 的映射"""
    sat_to_domain = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            sat_id = int(parts[0])
            domain_id = int(parts[6])
            sat_to_domain[sat_id] = domain_id
    print(f"[*] 成功加载聚类映射，共 {len(sat_to_domain)} 颗卫星。")
    return sat_to_domain


def draw_macro_topology(macro_edges, output_path):
    """
    基于多重边绘制超节点拓扑图（保留所有跨域链路，画出平行线）
    """
    G_vis = nx.MultiGraph()
    G_vis.add_nodes_from(range(N_DOMAINS))

    for u, v in macro_edges:
        if u < v:
            G_vis.add_edge(u, v)

    plt.figure(figsize=(14, 14))
    pos = nx.circular_layout(G_vis)

    nx.draw_networkx_nodes(G_vis, pos, node_size=1500, node_color='skyblue', edgecolors='black', linewidths=2)
    nx.draw_networkx_labels(G_vis, pos, font_size=15, font_weight="bold")

    ax = plt.gca()
    for u, v, key, data in G_vis.edges(keys=True, data=True):
        rad = 0.08 * (key + 1) if key % 2 == 0 else -0.08 * key
        if key == 0: rad = 0
        ax.annotate("",
                    xy=pos[v], xycoords='data',
                    xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="-", color="slategray",
                                    alpha=0.6, linewidth=1.2,
                                    connectionstyle=f"arc3,rad={rad}"))

    plt.title(f"Starlink 550 - Super Node Multigraph Topology\n(Each line represents a physical inter-domain ISL)",
              fontsize=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] 成功绘制超节点多重拓扑图并保存至: {output_path}")


# ================= 主流程 =================

def main():
    os.makedirs(os.path.dirname(MACRO_PKL_PATH), exist_ok=True)

    sat_to_domain = load_satellite_to_domain_mapping(CLUSTER_TXT_PATH)

    if not os.path.exists(MICRO_PKL_PATH):
        raise FileNotFoundError(f"找不到微观数据集: {MICRO_PKL_PATH}")

    print(f"[*] 正在读取微观数据集 {MICRO_PKL_PATH} ...")
    with open(MICRO_PKL_PATH, "rb") as f:
        micro_slices = pickle.load(f)

    macro_slices = []
    print("[*] 开始执行底层到宏观的升维聚合 (Graph, TM, Path) ...")

    for slice_idx, slice_data in enumerate(micro_slices):
        micro_graph = slice_data["graph"]
        micro_tm = slice_data["tm"]
        data_idx = slice_data["data_idx"]

        # ---------------------------------------------------------
        # 初始化当前时间片的映射字典和宏观简单图
        # ---------------------------------------------------------
        macro_graph_list = []
        G_macro_simple = nx.Graph()  # 用于计算最少跳数的宏观路径
        inter_domain_links = defaultdict(set)  # 记录每一对域之间的所有微观物理连边

        # 提取跨域信息
        for u, v in micro_graph:
            if u in sat_to_domain and v in sat_to_domain:
                du = sat_to_domain[u]
                dv = sat_to_domain[v]
                if du != dv:
                    # 保留原汁原味的多重图 Graph 字段 (如你所愿不作修改)
                    macro_graph_list.append([du, dv])
                    macro_graph_list.append([dv, du])

                    # 构建简单的宏观拓扑用于寻路
                    G_macro_simple.add_edge(du, dv)

                    # 记录具体的微观边界链路。注意方向：确保 tuple 的第一个元素属于 du，第二个属于 dv
                    inter_domain_links[(du, dv)].add((u, v))
                    inter_domain_links[(dv, du)].add((v, u))

        macro_graph_list = sorted(macro_graph_list)

        # ---------------------------------------------------------
        # Step B: 聚合宏观流量矩阵 (Macro TM - 如你所愿不作修改)
        # ---------------------------------------------------------
        macro_tm = defaultdict(int)
        for key_str, demand in micro_tm.items():
            u_str, v_str = key_str.split(',')
            u, v = int(u_str.strip()), int(v_str.strip())
            if u in sat_to_domain and v in sat_to_domain:
                du = sat_to_domain[u]
                dv = sat_to_domain[v]
                if du != dv:
                    macro_key = f"{du}, {dv}"
                    macro_tm[macro_key] += demand
        macro_tm_dict = dict(macro_tm)

        # ---------------------------------------------------------
        # Step C: 【重构】基于最小跳数的物理端点路径组合
        # ---------------------------------------------------------
        macro_paths_final = {}

        for macro_key in macro_tm_dict.keys():
            du_str, dv_str = macro_key.split(',')
            du, dv = int(du_str.strip()), int(dv_str.strip())

            paths_for_this_pair = []

            try:
                # 1. 在超节点图上寻找宏观层面的简单最短路径 (例如 [0, 1, 2, 3])
                # shortest_simple_paths 天生就是按跳数从小到大排序的！
                macro_paths_generator = nx.shortest_simple_paths(G_macro_simple, du, dv)

                for m_path in macro_paths_generator:
                    # 2. 提取这条宏观路径上，每一个跨界面的物理微观边列表
                    # 如果 m_path 是 [0, 1, 3]，提取出的就是 0->1 的连边集合，和 1->3 的连边集合
                    hop_links = []
                    for i in range(len(m_path) - 1):
                        hop_links.append(list(inter_domain_links[(m_path[i], m_path[i + 1])]))

                    # 3. 笛卡尔积魔法：排列组合出所有可能的微观端点序列
                    for combo in itertools.product(*hop_links):
                        # 拍平组合，变成 [u1, v1, u2, v2, ...] 格式
                        flat_seq = [sat for link in combo for sat in link]
                        paths_for_this_pair.append(flat_seq)

                        # 凑够 8 条直接跳出循环
                        if len(paths_for_this_pair) == MAX_PARALLEL_LINKS:
                            break

                    if len(paths_for_this_pair) == MAX_PARALLEL_LINKS:
                        break

            except nx.NetworkXNoPath:
                # 如果某两个域之间在物理上彻底断开（孤岛），保留空列表
                pass

            macro_paths_final[macro_key] = paths_for_this_pair

        # 组装当前时间片
        macro_slices.append({
            "graph": macro_graph_list,
            "tm": macro_tm_dict,
            "path": macro_paths_final,
            "data_idx": data_idx
        })

        if (slice_idx + 1) % 50 == 0:
            print(f"    进度: {slice_idx + 1}/{len(micro_slices)} 个时间片已聚合。")

    # 3. 保存输出的宏观 PKL
    with open(MACRO_PKL_PATH, "wb") as f:
        pickle.dump(macro_slices, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\n✅ 成功！最小跳数宏观数据集 (保留全量物理图) 已保存至: {MACRO_PKL_PATH}")

    # 4. 绘制拓扑图 (使用第一个时间片的边拓扑)
    if len(macro_slices) > 0:
        print("\n[*] 正在渲染超节点 Multigraph 拓扑图像...")
        first_slice_edges = macro_slices[0]["graph"]
        draw_macro_topology(first_slice_edges, TOPO_IMG_PATH)


if __name__ == "__main__":
    main()

# import os
# import pickle
# import networkx as nx
# import matplotlib.pyplot as plt
# from pathlib import Path
# from collections import defaultdict

# # ================= 路径配置 =================
# BASE_DIR = Path(__file__).parent
# # 输入：聚类结果 txt
# CLUSTER_TXT_PATH = BASE_DIR / "outputs" / "starlink550_data_v1.5_domain_result_18_raan_orbital_angle.txt"
# # 输入：微观 PKL 数据集
# MICRO_PKL_PATH = BASE_DIR / "outputs" / "tms" / "all_time_slices.pkl"
# # 输出：宏观（分域后）PKL 数据集
# MACRO_PKL_PATH = BASE_DIR / "outputs" / "tms" / "macro_domain_time_slices.pkl"
# # 输出：拓扑可视化图片
# TOPO_IMG_PATH = BASE_DIR / "outputs" / "super_node_topo_multigraph.png"

# # 超参数
# MAX_PARALLEL_LINKS = 8  # 宏观超节点之间的最大并行路径数
# N_DOMAINS = 18          # 预期的域总数


# # ================= 核心工具函数 =================

# def load_satellite_to_domain_mapping(txt_path):
#     """解析分域TXT文件，构建 卫星ID -> 域ID 的映射"""
#     sat_to_domain = {}
#     with open(txt_path, "r", encoding="utf-8") as f:
#         next(f) # 跳过表头
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             parts = line.split(',')
#             sat_id = int(parts[0])
#             domain_id = int(parts[6])
#             sat_to_domain[sat_id] = domain_id
#     print(f"[*] 成功加载聚类映射，共 {len(sat_to_domain)} 颗卫星。")
#     return sat_to_domain

# def draw_macro_topology(macro_edges, output_path):
#     """
#     基于多重边（Multigraph）绘制超节点拓扑图。
#     有多少条底层物理链路，就绘制多少条平行的连边。
#     """
#     G_vis = nx.MultiGraph()
#     G_vis.add_nodes_from(range(N_DOMAINS))

#     # 因为 macro_edges 里包含了双向边 [0,1] 和 [1,0]，
#     # 为了不在视觉上画出两倍数量的线，这里只取单向组合 u < v 进行绘制
#     for u, v in macro_edges:
#         if u < v:
#             G_vis.add_edge(u, v)

#     plt.figure(figsize=(14, 14))
#     pos = nx.circular_layout(G_vis)

#     # 绘制超节点
#     nx.draw_networkx_nodes(G_vis, pos, node_size=1500, node_color='skyblue', edgecolors='black', linewidths=2)
#     nx.draw_networkx_labels(G_vis, pos, font_size=15, font_weight="bold")

#     # 绘制平行多重边
#     ax = plt.gca()
#     for u, v, key, data in G_vis.edges(keys=True, data=True):
#         # 动态计算曲线弧度，使多条边错开显示
#         rad = 0.08 * (key + 1) if key % 2 == 0 else -0.08 * key
#         if key == 0: rad = 0 # 第一条边画直线

#         ax.annotate("",
#                     xy=pos[v], xycoords='data',
#                     xytext=pos[u], textcoords='data',
#                     arrowprops=dict(arrowstyle="-", color="slategray",
#                                     alpha=0.6, linewidth=1.2,
#                                     connectionstyle=f"arc3,rad={rad}"))

#     plt.title(f"Starlink 550 - Super Node Multigraph Topology\n(Each line represents a physical inter-domain ISL)", fontsize=20)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300)
#     plt.close()
#     print(f"[*] 成功绘制超节点多重拓扑图并保存至: {output_path}")


# # ================= 主流程 =================

# def main():
#     os.makedirs(os.path.dirname(MACRO_PKL_PATH), exist_ok=True)

#     sat_to_domain = load_satellite_to_domain_mapping(CLUSTER_TXT_PATH)

#     if not os.path.exists(MICRO_PKL_PATH):
#         raise FileNotFoundError(f"找不到微观数据集: {MICRO_PKL_PATH}")

#     print(f"[*] 正在读取微观数据集 {MICRO_PKL_PATH} ...")
#     with open(MICRO_PKL_PATH, "rb") as f:
#         micro_slices = pickle.load(f)

#     macro_slices = []
#     print("[*] 开始执行底层到宏观的升维聚合 (Graph, TM, Path) ...")

#     for slice_idx, slice_data in enumerate(micro_slices):
#         micro_graph = slice_data["graph"]
#         micro_tm = slice_data["tm"]
#         micro_paths = slice_data["path"]
#         data_idx = slice_data["data_idx"]

#         # ---------------------------------------------------------
#         # Step A: 提取宏观图边 (Super Node Multigraph Edges)
#         # 每一条底层跨域物理链路，都在宏观图产生一条数据
#         # ---------------------------------------------------------
#         macro_graph_list = []
#         for u, v in micro_graph:
#             if u in sat_to_domain and v in sat_to_domain:
#                 du = sat_to_domain[u]
#                 dv = sat_to_domain[v]
#                 if du != dv:
#                     macro_graph_list.append([du, dv])
#                     macro_graph_list.append([dv, du]) # 记录双向

#         macro_graph_list = sorted(macro_graph_list)

#         # ---------------------------------------------------------
#         # Step B: 聚合宏观流量矩阵 (Macro TM)
#         # ---------------------------------------------------------
#         macro_tm = defaultdict(int)
#         for key_str, demand in micro_tm.items():
#             u_str, v_str = key_str.split(',')
#             u, v = int(u_str.strip()), int(v_str.strip())

#             if u in sat_to_domain and v in sat_to_domain:
#                 du = sat_to_domain[u]
#                 dv = sat_to_domain[v]
#                 # 过滤掉域内通信，跨域通信流量累加
#                 if du != dv:
#                     macro_key = f"{du}, {dv}"
#                     macro_tm[macro_key] += demand

#         macro_tm_dict = dict(macro_tm)

#         # ---------------------------------------------------------
#         # Step C: 精确提取物理端口跨域路径 (Macro Paths by Port IDs)
#         # ---------------------------------------------------------
#         macro_paths_dict = defaultdict(list)

#         # 遍历底层算好的每一条微观路径
#         for key_str, paths_list in micro_paths.items():
#             for p in paths_list:
#                 if len(p) < 2: continue

#                 d_src = sat_to_domain.get(p[0])
#                 d_dst = sat_to_domain.get(p[-1])

#                 # 只关心发生跨域的路由路径
#                 if d_src is None or d_dst is None or d_src == d_dst:
#                     continue

#                 boundary_seq = []
#                 # 沿微观路径滑动，侦测域切换动作
#                 for i in range(len(p) - 1):
#                     node_u = p[i]
#                     node_v = p[i+1]
#                     du = sat_to_domain.get(node_u)
#                     dv = sat_to_domain.get(node_v)

#                     # 发现跨域链路！记录物理端点卫星 ID
#                     if du is not None and dv is not None and du != dv:
#                         boundary_seq.extend([node_u, node_v])

#                 if boundary_seq:
#                     macro_key = f"{d_src}, {d_dst}"
#                     # 路径去重 (同一对超节点，可能通过不同微观路径但走了相同的域间物理端口)
#                     if boundary_seq not in macro_paths_dict[macro_key]:
#                         macro_paths_dict[macro_key].append(boundary_seq)

#         # 整理最终路径字段：截断到 MAX_PARALLEL_LINKS
#         macro_paths_final = {}
#         for k, v in macro_paths_dict.items():
#             macro_paths_final[k] = v[:MAX_PARALLEL_LINKS]

#         # 组装当前时间片
#         macro_slices.append({
#             "graph": macro_graph_list,
#             "tm": macro_tm_dict,
#             "path": macro_paths_final,
#             "data_idx": data_idx
#         })

#         if (slice_idx + 1) % 50 == 0:
#             print(f"    进度: {slice_idx + 1}/{len(micro_slices)} 个时间片已聚合。")

#     # 3. 保存输出的宏观 PKL
#     with open(MACRO_PKL_PATH, "wb") as f:
#         pickle.dump(macro_slices, f, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"\n✅ 成功！宏观数据集 (保留物理端口状态) 已保存至: {MACRO_PKL_PATH}")

#     # 4. 绘制拓扑图 (使用第一个时间片的边拓扑)
#     if len(macro_slices) > 0:
#         print("\n[*] 正在渲染超节点 Multigraph 拓扑图像...")
#         first_slice_edges = macro_slices[0]["graph"]
#         draw_macro_topology(first_slice_edges, TOPO_IMG_PATH)


# if __name__ == "__main__":
#     main()

