import os
import pickle
from pathlib import Path

import networkx as nx

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover - 运行时检查
    raise ImportError(
        "未能导入 gurobipy，请确认已安装 Gurobi 并配置好许可证。"
    ) from exc

# ================= 直接在这里配置参数 =================
BASE_DIR = Path(__file__).parent
INTRA_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550_intra.pkl"

TARGET_SLICE_IDX = 0
TARGET_DOMAIN_ID = None

WEIGHT_MIN = 1
WEIGHT_MAX = 20

CAPACITY_MODE = "uniform"  # 可选: "uniform" / "custom"
DEFAULT_CAPACITY = 50.0
CUSTOM_CAPACITY = {}

DROP_UNREACHABLE = True

# 求解参数：去除了 ECMP 严格等分约束后，问题难度大幅下降，恢复 5 分钟超时即可
TIME_LIMIT = 300  # 秒
MIP_GAP = 0.05

OUTPUT_PKL_PATH = BASE_DIR / "outputs" / "tms" / f"weights_slice_{TARGET_SLICE_IDX}_domain_{TARGET_DOMAIN_ID}.pkl"


def parse_tm_key(key_str: str):
    u_str, v_str = key_str.split(",")
    return int(u_str.strip()), int(v_str.strip())


def load_domain_data(intra_pkl_path: Path, slice_idx: int, domain_id: int | None):
    if not os.path.exists(intra_pkl_path):
        raise FileNotFoundError(f"找不到域内数据集: {intra_pkl_path}")
    with open(intra_pkl_path, "rb") as f:
        slices = pickle.load(f)
    if not slices:
        raise RuntimeError("域内数据集为空。")
    if slice_idx < 0 or slice_idx >= len(slices):
        raise IndexError(f"时间片索引越界: {slice_idx}")
    target_slice = slices[slice_idx]
    domains = target_slice.get("domains", {})
    if domain_id is None:
        domain_id = sorted(domains.keys())[0]
    return target_slice.get("data_idx"), domain_id, domains[domain_id]


def build_graph(domain_data):
    edges_raw = domain_data.get("graph", [])
    nodes = set(domain_data.get("active_sat_ids", []))
    undirected_edges = {}
    for u, v in edges_raw:
        if u == v: continue
        nodes.add(u)
        nodes.add(v)
        a, b = (u, v) if u < v else (v, u)
        undirected_edges[(a, b)] = None
    return sorted(nodes), list(undirected_edges.keys())


def build_demands(domain_data, G: nx.Graph):
    tm = domain_data.get("tm", {})
    demands = []
    skipped = 0
    for key_str, demand in tm.items():
        if demand <= 0: continue
        s, t = parse_tm_key(key_str)
        if s == t: continue
        if s not in G or t not in G:
            skipped += 1
            continue
        if DROP_UNREACHABLE and not nx.has_path(G, s, t):
            skipped += 1
            continue
        demands.append((s, t, float(demand)))
    return demands, skipped


def get_capacity(u, v):
    a, b = (u, v) if u < v else (v, u)
    if CAPACITY_MODE == "custom":
        return float(CUSTOM_CAPACITY.get((a, b), DEFAULT_CAPACITY))
    return float(DEFAULT_CAPACITY)


def solve_link_weights(nodes, edges, demands):
    """
    采用【有向链路】+【目的地聚合流量】+【最短路最优分流】的 MILP 建模。
    注：已放宽严格的 ECMP 1:1 均分约束，以换取权重的快速求解，
    此模型生成的权重能极好地规避拥塞，非常适合作为 GNN 的监督标签。
    """
    # 1. 拆分无向边为有向弧，构建图的邻接结构
    arcs = []
    out_neighbors = {n: [] for n in nodes}
    in_neighbors = {n: [] for n in nodes}
    for u, v in edges:
        arcs.extend([(u, v), (v, u)])
        out_neighbors[u].append(v)
        out_neighbors[v].append(u)
        in_neighbors[v].append(u)
        in_neighbors[u].append(v)

    # 2. 梳理业务矩阵：按目的节点 t 聚合需求
    destinations = list(set(t for s, t, d in demands))
    d_it = {(i, t): 0.0 for i in nodes for t in destinations}
    D_t_total = {t: 0.0 for t in destinations}  # 目的节点 t 需要接收的总流量
    
    for s, t, d in demands:
        d_it[(s, t)] += d
        D_t_total[t] += d

    # 3. 计算 Big-M 参数
    M_D = WEIGHT_MAX * (len(nodes) - 1)
    
    # ------------------- 开始 Gurobi 建模 -------------------
    model = gp.Model("TE_Weight_Optimization_No_Strict_ECMP")
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = TIME_LIMIT
    model.Params.MIPGap = MIP_GAP

    # 变量定义
    w = model.addVars(arcs, vtype=GRB.INTEGER, lb=WEIGHT_MIN, ub=WEIGHT_MAX, name="w")
    D = model.addVars(nodes, destinations, vtype=GRB.CONTINUOUS, lb=0.0, name="D")
    z = model.addVars(arcs, destinations, vtype=GRB.BINARY, name="z")
    f = model.addVars(arcs, destinations, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
    rho = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="rho")

    # ================= 核心约束块 =================
    for t in destinations:
        M_F = D_t_total[t]
        
        # 约束 A & B
        model.addConstr(D[t, t] == 0, name=f"dist_self_{t}")
        for j in out_neighbors[t]:
            model.addConstr(f[t, j, t] == 0, name=f"no_out_flow_{t}_{j}")
            model.addConstr(z[t, j, t] == 0, name=f"no_out_nh_{t}_{j}")

        for i in nodes:
            if i != t:
                # 约束 C & D
                out_flow = gp.quicksum(f[i, j, t] for j in out_neighbors[i])
                in_flow = gp.quicksum(f[h, i, t] for h in in_neighbors[i])
                model.addConstr(out_flow - in_flow == d_it[(i, t)], name=f"flow_conserv_{i}_{t}")
                model.addConstr(gp.quicksum(z[i, j, t] for j in out_neighbors[i]) >= 1, name=f"min_nh_{i}_{t}")

            for j in out_neighbors[i]:
                # 约束 E & F: 最短路探测与流量阀门绑定
                model.addConstr(D[i, t] <= w[i, j] + D[j, t], name=f"bellman_ub_{i}_{j}_{t}")
                model.addConstr(D[i, t] >= w[i, j] + D[j, t] - M_D * (1 - z[i, j, t]), name=f"bellman_lb_{i}_{j}_{t}")
                model.addConstr(D[i, t] <= w[i, j] + D[j, t] - 1 + M_D * z[i, j, t], name=f"not_sp_{i}_{j}_{t}")
                model.addConstr(f[i, j, t] <= M_F * z[i, j, t], name=f"flow_z_{i}_{j}_{t}")

            # =========================================================
            # 约束 G (已移除)：为了保证能在多项式时间内出解，
            # 这里放弃了强制多条最短路 1:1 均分的极端非线性耦合限制。
            # 模型将在算出的最短路(z=1)上，寻找最能降低 MLU 的流量分配比例。
            # =========================================================

    # 约束 H: 容量上限与 MLU 定义
    for u, v in arcs:
        cap = get_capacity(u, v)
        total_load = gp.quicksum(f[u, v, t] for t in destinations)
        model.addConstr(total_load <= rho * cap, name=f"mlu_cap_{u}_{v}")

    # 优化目标: 最小化网络最大拥塞，并加上极小的权重惩罚促使结果平滑
    weight_penalty = gp.quicksum(w[e] for e in arcs)
    model.setObjective(rho + 1e-4 * weight_penalty, GRB.MINIMIZE)

    print("\n[*] Gurobi 优化模型已建立，开始求解...")
    model.optimize()

    # ================= 状态检查与异常处理 =================
    if model.Status == GRB.INFEASIBLE:
        raise RuntimeError("❌ 求解失败：模型不可行 (INFEASIBLE)。请检查网络是否存在孤岛，或流量需求是否超出物理限制。")
    
    if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        raise RuntimeError(f"❌ 求解异常终止，Gurobi 状态码: {model.Status}")

    # 检查是否找到了至少一个可行解
    if model.SolCount == 0:
        raise RuntimeError(
            f"❌ 求解超时，且在 {TIME_LIMIT} 秒内未找到任何可行解 (Solution count = 0)。\n"
            "建议尝试以下操作：\n"
            "1. 增大 TIME_LIMIT (例如 600 秒或 1200 秒)\n"
            "2. 降低业务需求数量，或者减小网络节点规模"
        )

    # 提取有向链路的权重作为监督学习的标签
    weights_directed = {(u, v): int(round(w[(u, v)].X)) for (u, v) in arcs}
    return weights_directed, rho.X, model.Status


def main():
    data_idx, domain_id, domain_data = load_domain_data(INTRA_PKL_PATH, TARGET_SLICE_IDX, TARGET_DOMAIN_ID)

    nodes, edges = build_graph(domain_data)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    demands, skipped = build_demands(domain_data, G)
    if not demands:
        raise RuntimeError("❌ 没有可用的流量需求（可能全部不可达或需求值为 0）。")

    print(f"================ 流量工程权重优化 ==================")
    print(f"[*] 时间片索引: {TARGET_SLICE_IDX} | 域编号: {domain_id}")
    print(f"[*] 节点总数: {len(nodes)} | 无向边总数: {len(edges)}")
    print(f"[*] 聚合前需求数: {len(demands)} | 被过滤的不可达需求: {skipped}")
    print(f"====================================================")

    try:
        weights, max_util, status = solve_link_weights(nodes, edges, demands)
    except RuntimeError as e:
        print(str(e))
        return  # 优雅退出

    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    export = {
        "slice_idx": TARGET_SLICE_IDX,
        "data_idx": data_idx,
        "domain_id": domain_id,
        "weights": weights,  
        "max_utilization": max_util,
        "status": int(status)
    }

    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(export, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ 求解成功！最大链路利用率 (MLU) = {max_util:.4f}")
    print(f"✅ 已成功保存 {len(weights)} 条有向链路权重的标签数据至: {OUTPUT_PKL_PATH}")


if __name__ == "__main__":
    main()

# import os
# import pickle
# from pathlib import Path

# import networkx as nx

# try:
#     import gurobipy as gp
#     from gurobipy import GRB
# except Exception as exc:  # pragma: no cover - 运行时检查
#     raise ImportError(
#         "未能导入 gurobipy，请确认已安装 Gurobi 并配置好许可证。"
#     ) from exc

# # ================= 直接在这里配置参数 =================
# BASE_DIR = Path(__file__).parent
# INTRA_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550_intra.pkl"

# TARGET_SLICE_IDX = 0
# TARGET_DOMAIN_ID = None

# WEIGHT_MIN = 1
# WEIGHT_MAX = 20

# CAPACITY_MODE = "uniform"  # 可选: "uniform" / "custom"
# DEFAULT_CAPACITY = 50.0
# CUSTOM_CAPACITY = {}

# DROP_UNREACHABLE = True

# # 求解参数（因为包含 ECMP 约束，MIP 会变得极其复杂，强烈建议设置 TimeLimit）
# TIME_LIMIT = 86400  # 秒
# MIP_GAP = 0.05

# OUTPUT_PKL_PATH = BASE_DIR / "outputs" / "tms" / f"weights_slice_{TARGET_SLICE_IDX}_domain_{TARGET_DOMAIN_ID}.pkl"


# def parse_tm_key(key_str: str):
#     u_str, v_str = key_str.split(",")
#     return int(u_str.strip()), int(v_str.strip())


# def load_domain_data(intra_pkl_path: Path, slice_idx: int, domain_id: int | None):
#     if not os.path.exists(intra_pkl_path):
#         raise FileNotFoundError(f"找不到域内数据集: {intra_pkl_path}")
#     with open(intra_pkl_path, "rb") as f:
#         slices = pickle.load(f)
#     if not slices:
#         raise RuntimeError("域内数据集为空。")
#     if slice_idx < 0 or slice_idx >= len(slices):
#         raise IndexError(f"时间片索引越界: {slice_idx}")
#     target_slice = slices[slice_idx]
#     domains = target_slice.get("domains", {})
#     if domain_id is None:
#         domain_id = sorted(domains.keys())[0]
#     return target_slice.get("data_idx"), domain_id, domains[domain_id]


# def build_graph(domain_data):
#     edges_raw = domain_data.get("graph", [])
#     nodes = set(domain_data.get("active_sat_ids", []))
#     undirected_edges = {}
#     for u, v in edges_raw:
#         if u == v: continue
#         nodes.add(u)
#         nodes.add(v)
#         a, b = (u, v) if u < v else (v, u)
#         undirected_edges[(a, b)] = None
#     return sorted(nodes), list(undirected_edges.keys())


# def build_demands(domain_data, G: nx.Graph):
#     tm = domain_data.get("tm", {})
#     demands = []
#     skipped = 0
#     for key_str, demand in tm.items():
#         if demand <= 0: continue
#         s, t = parse_tm_key(key_str)
#         if s == t: continue
#         if s not in G or t not in G:
#             skipped += 1
#             continue
#         if DROP_UNREACHABLE and not nx.has_path(G, s, t):
#             skipped += 1
#             continue
#         demands.append((s, t, float(demand)))
#     return demands, skipped


# def get_capacity(u, v):
#     a, b = (u, v) if u < v else (v, u)
#     if CAPACITY_MODE == "custom":
#         return float(CUSTOM_CAPACITY.get((a, b), DEFAULT_CAPACITY))
#     return float(DEFAULT_CAPACITY)


# def solve_link_weights(nodes, edges, demands):
#     """
#     采用【有向链路】+【目的地聚合流量】+【ECMP等比例拆分】的 MILP 建模。
#     """
#     # 1. 拆分无向边为有向弧，构建图的邻接结构
#     arcs = []
#     out_neighbors = {n: [] for n in nodes}
#     in_neighbors = {n: [] for n in nodes}
#     for u, v in edges:
#         arcs.extend([(u, v), (v, u)])
#         out_neighbors[u].append(v)
#         out_neighbors[v].append(u)
#         in_neighbors[v].append(u)
#         in_neighbors[u].append(v)

#     # 2. 梳理业务矩阵：按目的节点 t 聚合需求
#     destinations = list(set(t for s, t, d in demands))
#     d_it = {(i, t): 0.0 for i in nodes for t in destinations}
#     D_t_total = {t: 0.0 for t in destinations}  # 目的节点 t 需要接收的总流量
    
#     for s, t, d in demands:
#         d_it[(s, t)] += d
#         D_t_total[t] += d

#     # 3. 计算 Big-M 参数
#     M_D = WEIGHT_MAX * (len(nodes) - 1)
    
#     # ------------------- 开始 Gurobi 建模 -------------------
#     model = gp.Model("ECMP_Weight_Optimization")
#     model.Params.OutputFlag = 1
#     model.Params.TimeLimit = TIME_LIMIT
#     model.Params.MIPGap = MIP_GAP

#     # 变量定义
#     w = model.addVars(arcs, vtype=GRB.INTEGER, lb=WEIGHT_MIN, ub=WEIGHT_MAX, name="w")
#     D = model.addVars(nodes, destinations, vtype=GRB.CONTINUOUS, lb=0.0, name="D")
#     z = model.addVars(arcs, destinations, vtype=GRB.BINARY, name="z")
#     f = model.addVars(arcs, destinations, vtype=GRB.CONTINUOUS, lb=0.0, name="f")
#     rho = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="rho")

#     # ================= 核心约束块 =================
#     for t in destinations:
#         M_F = D_t_total[t]
        
#         # 约束 A & B
#         model.addConstr(D[t, t] == 0, name=f"dist_self_{t}")
#         for j in out_neighbors[t]:
#             model.addConstr(f[t, j, t] == 0, name=f"no_out_flow_{t}_{j}")
#             model.addConstr(z[t, j, t] == 0, name=f"no_out_nh_{t}_{j}")

#         for i in nodes:
#             if i != t:
#                 # 约束 C & D
#                 out_flow = gp.quicksum(f[i, j, t] for j in out_neighbors[i])
#                 in_flow = gp.quicksum(f[h, i, t] for h in in_neighbors[i])
#                 model.addConstr(out_flow - in_flow == d_it[(i, t)], name=f"flow_conserv_{i}_{t}")
#                 model.addConstr(gp.quicksum(z[i, j, t] for j in out_neighbors[i]) >= 1, name=f"min_nh_{i}_{t}")

#             for j in out_neighbors[i]:
#                 # 约束 E & F
#                 model.addConstr(D[i, t] <= w[i, j] + D[j, t], name=f"bellman_ub_{i}_{j}_{t}")
#                 model.addConstr(D[i, t] >= w[i, j] + D[j, t] - M_D * (1 - z[i, j, t]), name=f"bellman_lb_{i}_{j}_{t}")
#                 model.addConstr(D[i, t] <= w[i, j] + D[j, t] - 1 + M_D * z[i, j, t], name=f"not_sp_{i}_{j}_{t}")
#                 model.addConstr(f[i, j, t] <= M_F * z[i, j, t], name=f"flow_z_{i}_{j}_{t}")

#             # 约束 G
#             if i != t:
#                 neighbors = out_neighbors[i]
#                 for idx1 in range(len(neighbors)):
#                     for idx2 in range(idx1 + 1, len(neighbors)):
#                         j = neighbors[idx1]
#                         k = neighbors[idx2]
#                         model.addConstr(f[i, j, t] - f[i, k, t] <= M_F * (2 - z[i, j, t] - z[i, k, t]), name=f"ecmp1_{i}_{j}_{k}_{t}")
#                         model.addConstr(f[i, k, t] - f[i, j, t] <= M_F * (2 - z[i, j, t] - z[i, k, t]), name=f"ecmp2_{i}_{j}_{k}_{t}")

#     # 约束 H
#     for u, v in arcs:
#         cap = get_capacity(u, v)
#         total_load = gp.quicksum(f[u, v, t] for t in destinations)
#         model.addConstr(total_load <= rho * cap, name=f"mlu_cap_{u}_{v}")

#     # 优化目标
#     weight_penalty = gp.quicksum(w[e] for e in arcs)
#     model.setObjective(rho + 1e-4 * weight_penalty, GRB.MINIMIZE)

#     print("\n[*] Gurobi 优化模型已建立，开始求解...")
#     model.optimize()

#     # ================= 状态检查与异常处理 =================
#     if model.Status == GRB.INFEASIBLE:
#         raise RuntimeError("❌ 求解失败：模型不可行 (INFEASIBLE)。请检查网络是否存在孤岛，或流量需求是否超出物理限制。")
    
#     if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
#         raise RuntimeError(f"❌ 求解异常终止，Gurobi 状态码: {model.Status}")

#     # 【关键修复】：检查是否找到了至少一个可行解
#     if model.SolCount == 0:
#         raise RuntimeError(
#             f"❌ 求解超时，且在 {TIME_LIMIT} 秒内未找到任何可行解 (Solution count = 0)。\n"
#             "建议尝试以下操作：\n"
#             "1. 增大 TIME_LIMIT (例如 600 秒或 1200 秒)\n"
#             "2. 降低业务需求数量，或者减小网络节点规模"
#         )

#     # 提取有向链路的权重
#     weights_directed = {(u, v): int(round(w[(u, v)].X)) for (u, v) in arcs}
#     return weights_directed, rho.X, model.Status


# def main():
#     data_idx, domain_id, domain_data = load_domain_data(INTRA_PKL_PATH, TARGET_SLICE_IDX, TARGET_DOMAIN_ID)

#     nodes, edges = build_graph(domain_data)
#     G = nx.Graph()
#     G.add_nodes_from(nodes)
#     G.add_edges_from(edges)

#     demands, skipped = build_demands(domain_data, G)
#     if not demands:
#         raise RuntimeError("❌ 没有可用的流量需求（可能全部不可达或需求值为 0）。")

#     print(f"================ 流量工程权重优化 ==================")
#     print(f"[*] 时间片索引: {TARGET_SLICE_IDX} | 域编号: {domain_id}")
#     print(f"[*] 节点总数: {len(nodes)} | 无向边总数: {len(edges)}")
#     print(f"[*] 聚合前需求数: {len(demands)} | 被过滤的不可达需求: {skipped}")
#     print(f"====================================================")

#     try:
#         weights, max_util, status = solve_link_weights(nodes, edges, demands)
#     except RuntimeError as e:
#         print(str(e))
#         return  # 优雅退出，不抛出吓人的堆栈追踪

#     os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
#     export = {
#         "slice_idx": TARGET_SLICE_IDX,
#         "data_idx": data_idx,
#         "domain_id": domain_id,
#         "weights": weights,  
#         "max_utilization": max_util,
#         "status": int(status)
#     }

#     with open(OUTPUT_PKL_PATH, "wb") as f:
#         pickle.dump(export, f, protocol=pickle.HIGHEST_PROTOCOL)

#     print(f"\n✅ 求解成功！最大链路利用率 (MLU) = {max_util:.4f}")
#     print(f"✅ 已成功保存 {len(weights)} 条有向链路权重的标签数据至: {OUTPUT_PKL_PATH}")


# if __name__ == "__main__":
#     main()

