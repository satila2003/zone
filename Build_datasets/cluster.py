import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist  # 【恢复引入】用于计算高维欧氏距离
import pandas as pd
import os
import math

# ==========================================
# 1. 配置参数
# ==========================================
TRAFFIC_FILE = 'traffic_matrix.npy'      # 流量矩阵
POS_FILE = 'sat_active_loads_detailed.csv'     # 包含经纬度/坐标的文件
MAX_VISIBLE_DIST_KM = 3000.0             # 【用户自定义】物理视距极限 (三维空间直线距离)
MIN_TRAFFIC_THRESHOLD = 0.0000001        # 忽略极微小的流量边

# --- 容量约束 ---
MAX_CLUSTER_CAPACITY = 70                # 单簇允许的最大节点数 (硬约束)
BIG_CLUSTER_THRESHOLD = MAX_CLUSTER_CAPACITY  # 大簇的节点数阈值 (用于触发分割)
SMALL_CLUSTER_THRESHOLD = 10             # 小簇合并的节点数阈值

SAT_ALTITUDE_KM = 550.0                  # 卫星轨道高度
EARTH_RADIUS_KM = 6371.0                 # 地球平均半径

# ==========================================
# 2. 数据加载与预处理
# ==========================================

def load_data():
    """加载流量矩阵和卫星位置信息，并转换为 3D 笛卡尔坐标"""
    print(f"1. 正在加载数据...")
    
    if not os.path.exists(TRAFFIC_FILE) or not os.path.exists(POS_FILE):
        raise FileNotFoundError("找不到输入文件，请确保先运行了流量生成脚本。")

    # 加载流量矩阵
    traffic_matrix = np.load(TRAFFIC_FILE)
    
    # 加载位置数据
    df = pd.read_csv(POS_FILE)
    
    # 提取经度与纬度，并转为弧度制
    lats = np.radians(df['latitude'].values)
    lons = np.radians(df['longitude'].values)
    
    # 【修改】球坐标转三维笛卡尔坐标 (x, y, z)
    R = EARTH_RADIUS_KM + SAT_ALTITUDE_KM
    x = R * np.cos(lats) * np.cos(lons)
    y = R * np.cos(lats) * np.sin(lons)
    z = R * np.sin(lats)
    coords = np.column_stack((x, y, z))
    
    return traffic_matrix, coords, df

def generate_and_save_visibility(coords):
    """
    【修改】基于三维直线欧氏距离，生成并保存可见性矩阵
    (激光沿直线传播，计算空间弦长)
    """
    print(f"2. 正在计算并保存可见性矩阵 (采用 3D 直线欧氏距离, 约束: {MAX_VISIBLE_DIST_KM} km)...")
    
    # 计算全网 3D 欧氏直线距离
    dist_matrix = cdist(coords, coords, metric='euclidean')
    
    # 生成 0/1 矩阵：距离小于阈值且不等于0 (排除自身)
    visibility_matrix = ((dist_matrix <= MAX_VISIBLE_DIST_KM) & (dist_matrix > 0)).astype(int)
    
    # 保存矩阵
    np.save('visibility_matrix.npy', visibility_matrix)
    print(f"   - 可见性矩阵已保存至: visibility_matrix.npy (Shape: {visibility_matrix.shape})")
    
    return dist_matrix  # 返回 3D 直线距离矩阵供后续建图使用

def build_constrained_graph(traffic_matrix, dist_matrix):
    """构建物理约束图"""
    print("3. 正在构建物理约束图 (Graph Construction)...")
    num_nodes = traffic_matrix.shape[0]
    
    # A. 物理掩膜 (使用 3D 直线距离)
    mask_visible = (dist_matrix <= MAX_VISIBLE_DIST_KM) & (dist_matrix > 0)
    
    # B. 流量掩膜
    mask_traffic = traffic_matrix > MIN_TRAFFIC_THRESHOLD
    
    # C. 交集
    final_mask = mask_visible & mask_traffic
    
    # D. 提取边
    sources, targets = np.where(final_mask)
    weights = traffic_matrix[final_mask]
    
    # 去重 (只保留上三角)
    upper_tri_mask = sources < targets
    sources = sources[upper_tri_mask]
    targets = targets[upper_tri_mask]
    weights = weights[upper_tri_mask]
    
    # 构建图
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edge_list = zip(sources, targets, weights)
    G.add_weighted_edges_from(edge_list)
    
    print(f"   - 节点数: {G.number_of_nodes()}")
    print(f"   - 边数量: {G.number_of_edges()}")
    
    return G

# ==========================================
# 3. 核心算法：Louvain聚类
# ==========================================

def perform_clustering(G):
    print("4. 执行 Louvain 社区发现算法...")
    partition = community_louvain.best_partition(G, weight='weight', resolution=1.0, random_state=42)
    labels = np.array([partition[i] for i in range(len(G.nodes))])
    
    # 简单的自适应分辨率调整逻辑
    unique_labels, counts = np.unique(labels, return_counts=True)
    single_node_clusters = np.sum(counts == 1)
    
    if single_node_clusters > len(G.nodes) * 0.3:
        print("   - 检测到过多单节点簇，尝试使用较低分辨率参数...")
        partition = community_louvain.best_partition(G, weight='weight', resolution=0.5, random_state=42)
        labels = np.array([partition[i] for i in range(len(G.nodes))])
    
    modularity = community_louvain.modularity(partition, G)
    print(f"   - 初始模块度: {modularity:.4f}")
    return labels, modularity

# ==========================================
# 4. 大簇二次分割 (空间约束层次聚类)
# ==========================================

def agglomerative_split(cluster_indices, sub_traffic, sub_dist, max_dist, max_capacity):
    """
    针对单个大簇执行空间约束的层次聚类，最大化流量内聚性
    """
    n_nodes = len(cluster_indices)
    target_k = math.ceil(n_nodes / max_capacity)
    if target_k < 2: target_k = 2
    
    # 1. 初始化：每个节点自成一簇
    active_clusters = {i: [i] for i in range(n_nodes)}
    
    # 预计算：节点的度（强度）和局部网络总流量
    node_strengths = np.sum(sub_traffic, axis=0) + np.sum(sub_traffic, axis=1)
    total_traffic = np.sum(sub_traffic)
    if total_traffic == 0: total_traffic = 1e-9 # 防止除零
    
    # 2. 迭代合并，直到簇数量降至 target_k
    while len(active_clusters) > target_k:
        cluster_ids = list(active_clusters.keys())
        best_pair = None
        max_delta_q = -float('inf')
        
        # 遍历所有存在的簇对，寻找最优合并对
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                c1 = cluster_ids[i]
                c2 = cluster_ids[j]
                
                nodes1 = active_clusters[c1]
                nodes2 = active_clusters[c2]

                # --- 硬约束拦截：检查容量 ---
                if len(nodes1) + len(nodes2) > max_capacity:
                    continue # 超过容量上限，直接拒绝该候选对
                
                # --- 硬约束拦截：检查物理连通性 (使用直线欧氏距离) ---
                cross_dist = sub_dist[np.ix_(nodes1, nodes2)]
                if not np.any((cross_dist <= max_dist) & (cross_dist > 0)):
                    continue # 物理不连通，直接拒绝该候选对
                
                # --- 计算流量内聚性增益 ΔQ ---
                t_12 = np.sum(sub_traffic[np.ix_(nodes1, nodes2)]) + np.sum(sub_traffic[np.ix_(nodes2, nodes1)])
                k1 = np.sum(node_strengths[nodes1])
                k2 = np.sum(node_strengths[nodes2])
                expected_traffic = (k1 * k2) / total_traffic
                
                # 模块度增益
                delta_q = t_12 - expected_traffic
                
                if delta_q > max_delta_q:
                    max_delta_q = delta_q
                    best_pair = (c1, c2)
        
        # --- 执行合并 ---
        if best_pair is None:
            print(f"     > 警告：剩余子簇受限于容量 (<= {max_capacity}) 或物理连通性，提前终止合并。")
            break
            
        merge_c1, merge_c2 = best_pair
        active_clusters[merge_c1].extend(active_clusters[merge_c2])
        del active_clusters[merge_c2]
        
    # 3. 整理输出格式：返回包含【全局】原始索引的列表
    final_sub_clusters = []
    for c_id, local_nodes in active_clusters.items():
        global_nodes = [cluster_indices[idx] for idx in local_nodes]
        final_sub_clusters.append(global_nodes)
        
    return final_sub_clusters

def split_large_clusters(labels, coords, dist_matrix, traffic_matrix, big_threshold=BIG_CLUSTER_THRESHOLD):
    """
    将超过 big_threshold 节点的大簇进行基于层次聚类的二次分割
    """
    print(f"5. 检查并分割大簇 (采用空间约束层次聚类, 阈值: {big_threshold})...")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    big_cluster_ids = unique_labels[counts > big_threshold]
    
    if len(big_cluster_ids) == 0:
        print("   - 没有发现大簇，跳过分割。")
        return labels

    print(f"   - 发现 {len(big_cluster_ids)} 个大簇需要分割")
    
    new_labels = labels.copy()
    next_available_id = max(unique_labels) + 1
    
    for cluster_id in big_cluster_ids:
        cluster_mask = (labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        n_nodes = len(cluster_indices)
        
        print(f"   - 正在分割簇 {cluster_id} (Size: {n_nodes})...")
        
        # 提取局部矩阵
        raw_local_traffic = traffic_matrix[np.ix_(cluster_indices, cluster_indices)]
        local_dist = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
        local_visibility = ((local_dist <= 2500) & (local_dist > 0)).astype(int)
        local_traffic = raw_local_traffic * local_visibility

        

        # 调用基于 ΔQ 优化的空间约束层次聚类 (传入 MAX_CLUSTER_CAPACITY)
        final_sub_clusters = agglomerative_split(
            cluster_indices, 
            local_traffic, 
            local_dist, 
            2500,
            MAX_CLUSTER_CAPACITY
        )
        
        print(f"     > 层次聚类将该大簇切分为 {len(final_sub_clusters)} 个容量合规的强连通子簇")
        
        # --- 分配新 ID ---
        for i, sub_nodes in enumerate(final_sub_clusters):
            if i == 0:
                pass # 第一个子簇沿用旧 ID
            else:
                current_new_id = next_available_id
                next_available_id += 1
                new_labels[sub_nodes] = current_new_id
    
    return new_labels

# ==========================================
# 5. 小簇合并逻辑
# ==========================================

def merge_small_clusters(labels, threshold=SMALL_CLUSTER_THRESHOLD):
    """
    将所有节点数 <= threshold 的小簇合并为一个大的'碎片簇' (通常对应无人区/非活跃区节点)
    """
    print(f"6. 执行簇合并操作 (阈值: {threshold})...")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    big_cluster_ids = unique_labels[counts > threshold]
    small_cluster_ids = unique_labels[counts <= threshold]
    
    print(f"   - 原始簇总数: {len(unique_labels)}")
    print(f"   - 大簇数量 (>{threshold}节点): {len(big_cluster_ids)}")
    print(f"   - 小簇数量 (<={threshold}节点): {len(small_cluster_ids)} -> 将被合并")

    MERGED_ID = -1
    new_labels = np.full_like(labels, MERGED_ID)
    
    for new_id, old_id in enumerate(big_cluster_ids):
        mask = (labels == old_id)
        new_labels[mask] = new_id
        
    if np.any(new_labels == MERGED_ID):
        next_id = len(big_cluster_ids)
        new_labels[new_labels == MERGED_ID] = next_id
        print(f"   - 所有小簇已统一合并为无人区碎片簇 ID: {next_id}")
    else:
        print("   - 没有需要合并的小簇。")
        
    return new_labels

# ==========================================
# 6. 可视化与保存
# ==========================================

def evaluate_and_plot(labels, coords, df_meta, modularity):
    print("7. 结果可视化...")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_clusters = len(unique_labels)
    
    print("-" * 60)
    print(f"{'Cluster ID':<12} | {'Nodes':<8} | {'Avg Lat/Lon'}")
    print("-" * 60)
    
    for label in unique_labels:
        mask = labels == label
        count = np.sum(mask)
        avg_lon = np.mean(df_meta.loc[mask, 'longitude'])
        avg_lat = np.mean(df_meta.loc[mask, 'latitude'])
        print(f"{label:<12} | {count:<8} | {avg_lat:>6.1f}, {avg_lon:>6.1f}")
    
    plt.figure(figsize=(14, 7))
    cmap = plt.get_cmap('tab20')
    
    for i, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        color = cmap(i % 20)
        plt.scatter(df_meta['longitude'][mask], 
                    df_meta['latitude'][mask], 
                    color=color, s=20, alpha=0.8, label=f'Cluster {cluster_id}')

    plt.title(f'Satellite Network Clustering (3D Euclidean Constrained)\nTotal Clusters: {num_clusters} | Modularity: {modularity:.3f}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    ncol = 1 if num_clusters < 10 else 2
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=ncol)
    plt.grid(True, alpha=0.3)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    
    output_img = 'louvain_merged_result.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"\n可视化已保存至: {output_img}")
    plt.show()

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    try:
        # 1. 加载数据
        traffic_mat, coords, df_meta = load_data()
        
        # 2. 生成并保存可见性矩阵 (返回基于 3D 欧氏距离的矩阵供后续使用)
        dist_matrix = generate_and_save_visibility(coords)
        
        # 3. 建图
        G = build_constrained_graph(traffic_mat, dist_matrix)
        
        # 4. 原始聚类 (Louvain 识别流量逻辑群)
        raw_labels, modularity = perform_clustering(G)
        
        # 5. 二次分割大簇 (基于物理连通与流量增益的层次聚类)
        labels_after_split = split_large_clusters(raw_labels, coords, dist_matrix, traffic_mat)
        
        # 6. 簇合并操作 (处理无人区节点)
        final_labels = merge_small_clusters(labels_after_split, threshold=SMALL_CLUSTER_THRESHOLD)
        
        # 7. 保存最终结果 CSV
        df_meta['cluster_id'] = final_labels
        df_meta.to_csv('satellite_louvain_result.csv', index=False)
        print("结果数据已保存至 satellite_louvain_result.csv")
        
        # 8. 绘图
        evaluate_and_plot(final_labels, coords, df_meta, modularity)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()