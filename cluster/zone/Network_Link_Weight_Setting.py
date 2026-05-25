import time
from pathlib import Path

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
def optimize_cluster_weights(filepath, time_idx=0, cluster_id=0, k=8,
                             alpha_stages=None, device='cuda'):
    """多阶段温度退火优化。

    alpha_stages: list of (alpha, epochs) 元组，默认为三阶段退火。
    """
    if alpha_stages is None:
        alpha_stages = [(1.0, 80), (5.0, 60), (20.0, 60)]

    print(f"\n---> 开始处理: 时间片 {time_idx} | 分簇 {cluster_id}")
    print(f"    退火阶段: {' → '.join(f'α={a}({e}轮)' for a, e in alpha_stages)}")

    # 时间记录
    timing_info = {
        'cluster_id': cluster_id,
        'stage_times': [],
        'epoch_times': [],
        'total_time': 0.0,
    }
    cluster_start = time.time()

    # 使用定制函数加载本簇数据
    nodes, edges, traffic_matrix = load_cluster_data(filepath, time_idx, cluster_id)

    num_links = len(edges)
    sd_pairs = [(s, d, dem) for (s, d), dem in traffic_matrix.items() if dem > 0 and s != d]
    num_sd = len(sd_pairs)

    print(f"节点数: {len(nodes)}, 链路数: {num_links}, 有效SD流数: {num_sd}")

    if num_sd == 0:
        elapsed = time.time() - cluster_start
        timing_info['total_time'] = elapsed
        print(f"💡 该簇没有有效流量需求，默认赋予全1链路权重。 (耗时 {elapsed:.2f}s)")
        return np.ones(num_links, dtype=int), 0.0, timing_info

    G = nx.DiGraph()
    for i, edge in enumerate(edges):
        G.add_edge(edge['u'], edge['v'], c=edge['c'], id=i, weight=1.0)

    model = SmoothRoutingModel(num_links).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    capacities = torch.tensor([e['c'] for e in edges], dtype=torch.float32).to(device)
    demands = torch.tensor([dem for _, _, dem in sd_pairs], dtype=torch.float32).view(-1, 1).to(device)

    global_epoch = 0
    for stage_idx, (alpha, stage_epochs) in enumerate(alpha_stages):
        stage_start = time.time()
        stage_name = f"阶段{stage_idx+1} (α={alpha})"
        print(f"  [{stage_name}] 开始, 共 {stage_epochs} 轮")

        for epoch in range(stage_epochs):
            epoch_start = time.time()
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

            global_epoch += 1
            epoch_time = time.time() - epoch_start
            timing_info['epoch_times'].append(epoch_time)

            if (epoch + 1) % 20 == 0 or (epoch + 1) == stage_epochs:
                print(f"    Epoch {epoch+1:3d}/{stage_epochs} - 预测MLU: {loss.item():.4f} | 本轮耗时: {epoch_time:.2f}s")

        stage_time = time.time() - stage_start
        timing_info['stage_times'].append(stage_time)
        print(f"  [{stage_name}] 完成, 阶段耗时: {stage_time:.2f}s")

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

    total_elapsed = time.time() - cluster_start
    timing_info['total_time'] = total_elapsed
    print(f"✅ 分簇 {cluster_id} 优化完成! 最终 ECMP MLU: {best_mlu:.4f} | 总耗时: {total_elapsed:.2f}s")
    return best_integer_weights, best_mlu, timing_info

# ==========================================
# 执行入口 (一键计算一个时间片内的所有簇)
# ==========================================
if __name__ == "__main__":
    dataset_file = r"F:\Py_Project\always\cluster\zone\outputs\tms\starlink550_intra.pkl"
    output_log = Path(__file__).parent / "result.txt"

    run_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"当前运行设备: {run_device.upper()}")

    target_time_idx = 0
    num_clusters = 18

    # 三阶段温度退火: α=1(探索) → α=5(过渡) → α=20(逼近ECMP)
    alpha_stages = [(1.0, 80), (5.0, 60), (20.0, 60)]

    time_slice_weights = {}
    all_timing = {}
    total_start = time.time()

    with open(output_log, 'w', encoding='utf-8') as log:
        log.write("===== 链路权重优化 时间统计 =====\n")
        log.write(f"设备: {run_device.upper()}\n")
        log.write(f"退火阶段: {' → '.join(f'α={a}({e}轮)' for a, e in alpha_stages)}\n")
        log.write("=" * 50 + "\n\n")

        for c_id in range(num_clusters):
            final_weights, final_mlu, timing = optimize_cluster_weights(
                filepath=dataset_file,
                time_idx=target_time_idx,
                cluster_id=c_id,
                k=8,
                alpha_stages=alpha_stages,
                device=run_device
            )

            time_slice_weights[c_id] = final_weights
            all_timing[c_id] = timing

            # 写入该簇的详细时间统计
            log.write(f"--- 分簇 {c_id} ---\n")
            log.write(f"  节点/链路/流数: 见控制台输出\n")
            log.write(f"  最终ECMP MLU: {final_mlu:.4f}\n")
            for si, st in enumerate(timing['stage_times']):
                log.write(f"  阶段{si+1} (α={alpha_stages[si][0]}): {st:.2f}s\n")
            log.write(f"  总耗时: {timing['total_time']:.2f}s\n")
            avg_epoch = (sum(timing['epoch_times']) / len(timing['epoch_times'])
                         if timing['epoch_times'] else 0)
            log.write(f"  每轮平均耗时: {avg_epoch:.2f}s\n\n")

        total_elapsed = time.time() - total_start
        log.write("=" * 50 + "\n")
        log.write(f"所有18个分簇总耗时: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)\n")

        # 逐epoch耗时汇总
        all_epoch_times = []
        for c_id in range(num_clusters):
            all_epoch_times.extend(all_timing[c_id]['epoch_times'])
        if all_epoch_times:
            log.write(f"全部epoch平均耗时: {np.mean(all_epoch_times):.2f}s (min: {np.min(all_epoch_times):.2f}s, max: {np.max(all_epoch_times):.2f}s)\n")

    print(f"\n📄 时间统计已写入: {output_log}")
    print("\n==================================================")
    print("🎯 时间片 0 的所有分簇计算完毕！")
    print("==================================================")