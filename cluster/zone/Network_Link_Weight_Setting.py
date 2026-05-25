import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
import pickle
from itertools import islice

# ==========================================
# 1. 数据加载模块
# ==========================================
# 1. 数据加载与预处理模块 (完美解包 domains 版本)
# ==========================================
def load_cluster_data(filepath, time_idx=0, cluster_id=0, default_capacity=300.0):
    """
    针对 starlink550_intra.pkl 格式定制的加载函数
    自动识别并深入解包 'domains' 键中的分簇数据。
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # 1. 提取特定时间片数据
    if isinstance(data, list):
        time_slice_data = data[time_idx]
    elif isinstance(data, dict):
        time_slice_data = data.get(time_idx) or data.get(str(time_idx)) or data
    else:
        time_slice_data = data
        
    # 2. 深入解包 'domains'
    # 如果发现有 'domains' 键，说明真实的分簇数据存在于 'domains' 字典里
    if isinstance(time_slice_data, dict) and 'domains' in time_slice_data:
        domains_data = time_slice_data['domains']
    else:
        domains_data = time_slice_data
        
    # 3. 提取特定分簇数据
    cluster_data = None
    if isinstance(domains_data, dict):
        # 兼容整数键和字符串键
        if cluster_id in domains_data:
            cluster_data = domains_data[cluster_id]
        elif str(cluster_id) in domains_data:
            cluster_data = domains_data[str(cluster_id)]
        else:
            keys_sample = list(domains_data.keys())[:5]
            raise ValueError(f"分簇ID {cluster_id} 找不到！domains 中现有的键: {keys_sample}")
    elif isinstance(domains_data, list):
        # 如果分簇按列表保存，使用索引访问
        if cluster_id < len(domains_data):
            cluster_data = domains_data[cluster_id]
        else:
            raise ValueError(f"分簇ID {cluster_id} 超出 domains 列表长度 {len(domains_data)}！")
    else:
        raise TypeError(f"未知的 domains 数据格式: {type(domains_data)}")
    
    # 获取簇内具体字段
    nodes = cluster_data.get('active_sat_ids', [])
    raw_edges = cluster_data.get('graph', [])
    raw_tm = cluster_data.get('tm', {})
    
    # 4. 转换边：转换为双向有向边，赋予默认容量
    edges = []
    for edge in raw_edges:
        u, v = edge[0], edge[1]
        edges.append({'u': u, 'v': v, 'c': default_capacity})
        edges.append({'u': v, 'v': u, 'c': default_capacity})
        
    # 5. 转换流量矩阵：兼容字符串键和元组键
    traffic_matrix = {}
    for k_str, demand in raw_tm.items():
        if isinstance(k_str, str):
            s_str, d_str = k_str.split(', ')
            s, d = int(s_str), int(d_str)
        elif isinstance(k_str, tuple):
            s, d = k_str
        else:
            continue
        traffic_matrix[(s, d)] = float(demand)
        
    return nodes, edges, traffic_matrix

# ==========================================
# 2. ECMP 真实环境验证模块
# ==========================================
def evaluate_ecmp_mlu(G, traffic_matrix, int_weights):
    """在给定整数链路权重下，模拟真实 ECMP 路由，计算最大链路利用率（MLU）。

    ECMP（Equal-Cost Multi-Path）路由规则：
    - 对每一对源-目的节点，路由器只看「权重和最小」的路径。
    - 如果有多条路径的权重和并列最小，流量在这些路径上严格均分（1:1:1...）。
    - 权重和大于最小值的路径不会收到任何流量。

    参数:
        G: networkx.DiGraph，有向图，每条边已有 'c'（容量）属性
        traffic_matrix: dict, {(src, dst): demand} 流量需求矩阵
        int_weights: np.ndarray, 与 G.edges() 顺序一一对应的整数链路权重

    返回:
        float: 网络中最大链路利用率（MLU = max(链路流量 / 链路容量)）
    """

    # ---------- 步骤0：将优化出的权重写入图中 ----------
    # G.edges(data=True) 返回 [(u, v, {attr_dict}), ...]
    # int_weights 的顺序与 G.edges() 严格一致（因为图从未被重建）
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        data['weight'] = int_weights[i]   # 用整数权重替换原来的浮点权重

    # 初始化每条有向链路的累计流量为 0
    # link_flows 的 key 是 (u, v) 元组，value 是该链路上承载的总流量
    link_flows = {e: 0.0 for e in G.edges()}

    # ---------- 步骤1：逐对源-目的节点，按 ECMP 规则分配流量 ----------
    for (s, d), demand in traffic_matrix.items():
        # 跳过无需求或同节点的无效条目
        if demand <= 0 or s == d:
            continue

        try:
            # ★ ECMP 核心 ★
            # nx.all_shortest_paths 找出 s→d 之间所有权重和最小的路径
            # "权重和最小" 用的是边上 'weight' 属性的累加值
            # 如果有多条路径的权重和完全相同且都是最小值，全部返回
            # 权重和比最小值大的路径不会出现在这个列表中
            paths = list(nx.all_shortest_paths(G, s, d, weight='weight'))

            # 多少条等cost最短路径
            num_paths = len(paths)

            # ★ 流量均分 ★
            # ECMP 规则：每条等cost最短路径分得完全相同的流量
            # 例：100Mbps 需求 ÷ 3条等cost路径 = 每路径 33.33Mbps
            flow_per_path = demand / num_paths

            # 将分摊后的流量累加到路径经过的每条有向边上
            for path in paths:
                # path 是节点序列，如 [5, 12, 8, 3]
                # 每次取相邻两个节点构成一条边 (path[i], path[i+1])
                for i in range(len(path) - 1):
                    link_flows[(path[i], path[i+1])] += flow_per_path

        except nx.NetworkXNoPath:
            # s 和 d 之间没有任何路径可达（图不连通时可能出现）
            continue

    # ---------- 步骤2：计算最大链路利用率（MLU）----------
    max_utilization = 0.0
    for u, v, data in G.edges(data=True):
        cap = data['c']                            # 链路容量
        util = link_flows[(u, v)] / cap            # 利用率 = 累计流量 ÷ 容量
        if util > max_utilization:
            max_utilization = util                 # 记录最高的那条

    return max_utilization

# ==========================================
# 3. PyTorch 梯度下降核心模块
# ==========================================
class SmoothRoutingModel(nn.Module):
    def __init__(self, num_links):
        super(SmoothRoutingModel, self).__init__()
        init_w = 1.0 + 0.1 * torch.rand(num_links)
        self.w = nn.Parameter(torch.sqrt(init_w))
        
    def forward(self, P_matrix, valid_mask, demands, capacities, alpha=10.0):
        w_sq = self.w ** 2 
        path_weights = torch.sum(P_matrix * w_sq, dim=2)
        path_weights = path_weights.masked_fill(~valid_mask, float('inf'))
        flow_probs = torch.softmax(-alpha * path_weights, dim=1)
        flow_probs = torch.nan_to_num(flow_probs, nan=0.0)
        path_flows = flow_probs * demands
        link_flows = torch.sum(path_flows.unsqueeze(2) * P_matrix, dim=(0, 1))
        utilizations = link_flows / capacities
        loss = torch.max(utilizations)
        return loss

# ==========================================
# 4. 优化主流程
# ==========================================
def optimize_cluster_weights(filepath, time_idx=0, cluster_id=0, k=8, alpha=10.0, epochs=200, device='cuda'):
    print(f"\n---> 开始处理: 时间片 {time_idx} | 分簇 {cluster_id}")
    
    # 使用定制函数加载本簇数据
    nodes, edges, traffic_matrix = load_cluster_data(filepath, time_idx, cluster_id)
    
    num_links = len(edges)
    sd_pairs = [(s, d, dem) for (s, d), dem in traffic_matrix.items() if dem > 0 and s != d]
    num_sd = len(sd_pairs)
    
    print(f"节点数: {len(nodes)}, 链路数: {num_links}, 有效SD流数: {num_sd}")
    
    # 【新增保护机制】：如果该簇没有流量（例如示例中的簇1、簇5等），直接返回默认权重是s
    if num_sd == 0:
        print("💡 该簇没有有效流量需求，默认赋予全1链路权重。")
        return np.ones(num_links, dtype=int), 0.0

    G = nx.DiGraph()
    for i, edge in enumerate(edges):
        G.add_edge(edge['u'], edge['v'], c=edge['c'], id=i, weight=1.0)

    # 梯度下降优化    
    model = SmoothRoutingModel(num_links).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    capacities = torch.tensor([e['c'] for e in edges], dtype=torch.float32).to(device)
    demands = torch.tensor([dem for _, _, dem in sd_pairs], dtype=torch.float32).view(-1, 1).to(device)
    
    for epoch in range(epochs):
        model.train() # 这里其实就是梯度下降
        optimizer.zero_grad()
        
        with torch.no_grad():
            current_w_sq = (model.w ** 2).cpu().numpy()
            for i, (u, v, data) in enumerate(G.edges(data=True)):
                data['weight'] = float(current_w_sq[i])
        
        P_matrix = torch.zeros((num_sd, k, num_links), dtype=torch.float32).to(device)
        valid_mask = torch.zeros((num_sd, k), dtype=torch.bool).to(device)
        
        for i, (s, d, _) in enumerate(sd_pairs):
            try:
                k_paths = list(islice(nx.shortest_simple_paths(G, s, d, weight='weight'), k))
                for j, path in enumerate(k_paths):
                    valid_mask[i, j] = True
                    for step in range(len(path) - 1):
                        u, v = path[step], path[step+1]
                        link_id = G[u][v]['id']
                        P_matrix[i, j, link_id] = 1.0
            except nx.NetworkXNoPath:
                pass 
        
        loss = model(P_matrix, valid_mask, demands, capacities, alpha=alpha)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"  Iteration {epoch+1}/{epochs} - 预测最大链路利用率: {loss.item():.4f}")

    # 生成可行域解 (Integerization)
    best_mlu = float('inf')
    best_integer_weights = None
    
    with torch.no_grad():
        final_w_sq = (model.w ** 2).cpu().numpy()
        for s_factor in range(1, 11):
            int_weights = np.ceil(s_factor * final_w_sq).astype(int)
            int_weights = np.maximum(int_weights, 1) 
            ecmp_mlu = evaluate_ecmp_mlu(G, traffic_matrix, int_weights)
            
            if ecmp_mlu < best_mlu:
                best_mlu = ecmp_mlu
                best_integer_weights = int_weights

    print(f"✅ 分簇 {cluster_id} 优化完成! 最终 ECMP 最大链路利用率 (MLU): {best_mlu:.4f}")
    return best_integer_weights, best_mlu

# ==========================================
# 执行入口 (一键计算一个时间片内的所有簇)
# ==========================================
if __name__ == "__main__":
    dataset_file = r"F:\Py_Project\always\cluster\zone\outputs\tms\starlink550_intra.pkl"
    
    run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前运行设备: {run_device.upper()}")
    
    # 假设我们计算第 0 个时间片
    target_time_idx = 0
    # 你的数据中有 0 到 17 共 18 个簇
    num_clusters = 18 
    
    # 用于保存该时间片下所有簇的权重
    time_slice_weights = {}
    
    for c_id in range(num_clusters):
        final_weights, final_mlu = optimize_cluster_weights(
            filepath=dataset_file, 
            time_idx=target_time_idx,
            cluster_id=c_id,
            k=8, 
            alpha=10.0, 
            epochs=200, 
            device=run_device 
        )
        
        time_slice_weights[c_id] = final_weights

    print("\n==================================================")
    print("🎯 时间片 0 的所有分簇计算完毕！")
    print("==================================================")
    
    # time_slice_weights 现在包含 {0: [权重数组], 1: [权重数组], ... 17: [权重数组]}
    # 如果你想把它存下来供后续系统调用，可以使用 pickle：
    # with open('optimized_weights_t0.pkl', 'wb') as f:
    #     pickle.dump(time_slice_weights, f)