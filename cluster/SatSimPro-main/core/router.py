import numpy as np
import heapq

class PathFinder:
    def __init__(self):
        # 缓存上一帧的图结构 {node_id: [(neighbor_id, dist_km), ...]}
        self.adj_list = {} 
        self.c = 299792.458 # 光速 km/s

    def build_graph(self, satellites, isl_indices):
        """
        每一帧调用一次，构建邻接表
        :param satellites: 卫星对象列表 (包含 .position 属性)
        :param isl_indices: strategies 返回的平铺数组 [2, u, v, 2, x, y...]
        """
        self.adj_list = {i: [] for i in range(len(satellites))}
        
        # 1. 快速解析 PyVista 的线条格式 [2, id1, id2, ...]
        if len(isl_indices) == 0: return

        try:
            # 转换为 numpy 数组以便快速处理
            flat = np.array(isl_indices)
            # 提取索引：从第 1 个开始，每隔 3 个取一个 (id_u)，从第 2 个开始 (id_v)
            u_list = flat[1::3]
            v_list = flat[2::3]
            
            # 2. 批量计算距离作为权重
            # 获取所有 u 和 v 的坐标
            pos_u = np.array([satellites[i].position for i in u_list])
            pos_v = np.array([satellites[i].position for i in v_list])
            
            # 欧几里得距离 (km)
            dists = np.linalg.norm(pos_u - pos_v, axis=1)
            
            # 3. 填充邻接表 (无向图)
            for i in range(len(u_list)):
                u, v, d = u_list[i], v_list[i], dists[i]
                self.adj_list[u].append((v, d))
                self.adj_list[v].append((u, d))
                
        except Exception as e:
            print(f"Graph build error: {e}")

    def find_shortest_path(self, start_idx, end_idx):
        """
        Dijkstra 算法寻找最短时延路径
        :return: (total_latency_ms, path_nodes_indices)
        """
        if start_idx not in self.adj_list or end_idx not in self.adj_list:
            return None, []

        # 优先队列: (accumulated_dist, current_node, path_list)
        pq = [(0.0, start_idx, [start_idx])]
        visited = set()
        
        while pq:
            dist, u, path = heapq.heappop(pq)
            
            if u == end_idx:
                # 找到终点，计算延迟 (ms)
                latency_ms = (dist / self.c) * 1000.0
                return latency_ms, path
            
            if u in visited: continue
            visited.add(u)
            
            for v, weight in self.adj_list[u]:
                if v not in visited:
                    heapq.heappush(pq, (dist + weight, v, path + [v]))
                    
        return None, []