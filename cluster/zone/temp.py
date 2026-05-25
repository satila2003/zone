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
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        data['weight'] = int_weights[i]
        
    link_flows = {e: 0.0 for e in G.edges()}
    
    for (s, d), demand in traffic_matrix.items():
        if demand <= 0 or s == d:
            continue
        try:
            paths = list(nx.all_shortest_paths(G, s, d, weight='weight'))
            num_paths = len(paths)
            flow_per_path = demand / num_paths
            for path in paths:
                for i in range(len(path) - 1):
                    link_flows[(path[i], path[i+1])] += flow_per_path
        except nx.NetworkXNoPath:
            continue
            
    max_utilization = 0.0
    for u, v, data in G.edges(data=True):
        cap = data['c']
        util = link_flows[(u, v)] / cap
        if util > max_utilization:
            max_utilization = util
            
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
    
    # 【新增保护机制】：如果该簇没有流量（例如示例中的簇1、簇5等），直接返回默认权重
    if num_sd == 0:
        print("💡 该簇没有有效流量需求，默认赋予全1链路权重。")
        return np.ones(num_links, dtype=int), 0.0

    G = nx.DiGraph()
    for i, edge in enumerate(edges):
        G.add_edge(edge['u'], edge['v'], c=edge['c'], id=i, weight=1.0)
        
    model = SmoothRoutingModel(num_links).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    capacities = torch.tensor([e['c'] for e in edges], dtype=torch.float32).to(device)
    demands = torch.tensor([dem for _, _, dem in sd_pairs], dtype=torch.float32).view(-1, 1).to(device)
    
    for epoch in range(epochs):
        model.train()
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