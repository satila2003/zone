import numpy as np
from scipy.spatial import cKDTree

class Strategy:
    def compute_links(self, satellites, gs_coords=None):
        raise NotImplementedError

class DistanceStrategy(Strategy):
    def __init__(self, max_isl_dist=2000):
        self.max_isl = max_isl_dist

    def compute_links(self, satellites, gs_coords=None):
        if not satellites: 
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
        
        # 简单距离策略：只看物理距离
        valid_idxs = [i for i, s in enumerate(satellites) if np.linalg.norm(s.position) > 100]
        if not valid_idxs: 
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
            
        coords = np.array([satellites[i].position for i in valid_idxs])
        isl = []
        
        # ISL 计算
        if len(coords) > 1:
            pairs = cKDTree(coords).query_pairs(r=self.max_isl)
            for p in pairs: 
                isl.extend([2, valid_idxs[p[0]], valid_idxs[p[1]]])
        
        # GSL 计算
        gsl = []
        if gs_coords is not None and len(gs_coords) > 0:
            sat_tree = cKDTree(coords)
            nearby = sat_tree.query_ball_point(gs_coords, r=self.max_isl)
            for gs_idx, sat_matches in enumerate(nearby):
                for local_sat_idx in sat_matches:
                    real_sat_idx = valid_idxs[local_sat_idx]
                    gsl.extend([2, real_sat_idx, gs_idx])

        return np.array(isl, dtype=np.int32), np.array(gsl, dtype=np.int32)

class StarlinkMeshStrategy(Strategy):
    def __init__(self, plane_tolerance=6.0, max_intra_dist=5000, max_inter_dist=5000, 
                 neighbor_tolerance=45.0, max_gsl_dist=2500,
                 enable_polar_cut=False, polar_cut_lat=70.0):
        """
        :param plane_tolerance: 判定两个卫星是否在同一个轨道面的 RAAN 容差 (度)
        :param max_intra_dist: 同轨链路 (前后相连) 的最大物理距离 (km)
        :param max_inter_dist: 异轨链路 (左右相连) 的最大物理距离 (km)
        :param neighbor_tolerance: 判定两个轨道面是否为邻居的 RAAN 容差 (度)
        :param max_gsl_dist: 地面站通信半径 (km)
        :param enable_polar_cut: 是否启用极地熔断
        :param polar_cut_lat: 熔断纬度阈值
        """
        self.plane_tol = plane_tolerance
        self.max_intra = max_intra_dist
        self.max_inter = max_inter_dist
        self.neighbor_tol = neighbor_tolerance
        self.max_gsl = max_gsl_dist
        self.enable_polar_cut = enable_polar_cut
        self.polar_cut_lat = polar_cut_lat

    def compute_links(self, satellites, gs_coords=None):
        if not satellites: 
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        sats_data = []
        valid_idxs = []
        
        for i, s in enumerate(satellites):
            if not hasattr(s, 'position_eci') or np.linalg.norm(s.position_eci) < 100:
                continue
            
            valid_idxs.append(i)
            
            # 使用 ECI 坐标计算几何相位
            rx, ry, rz = s.position_eci
            raan_rad = np.radians(s.raan)
            inc_rad = np.radians(s.inclination)
            
            # 旋转消除 RAAN 和 Inclination
            x_tmp = rx * np.cos(raan_rad) + ry * np.sin(raan_rad)
            y_tmp = -rx * np.sin(raan_rad) + ry * np.cos(raan_rad)
            y_plane = y_tmp * np.cos(inc_rad) + rz * np.sin(inc_rad)
            x_plane = x_tmp 
            
            u_angle = np.degrees(np.arctan2(y_plane, x_plane)) % 360.0

            # 计算纬度 (sin(lat) = z / R)
            curr_lat = np.degrees(np.arcsin(s.position[2] / np.linalg.norm(s.position)))
            
            sats_data.append({
                'idx': i,
                'raan': s.raan,
                'u': u_angle,
                'pos_eci': s.position_eci,
                'pos_ecef': s.position,
                'lat': curr_lat
            })

        if not sats_data:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        # 简单分组
        sats_data.sort(key=lambda x: x['raan'])
        
        planes_list = []
        current_plane = [sats_data[0]]
        
        for k in range(1, len(sats_data)):
            curr = sats_data[k]
            prev = sats_data[k-1]
            diff = abs(curr['raan'] - prev['raan'])
            diff = min(diff, 360.0 - diff)
            if diff > self.plane_tol:
                planes_list.append(current_plane)
                current_plane = []
            current_plane.append(curr)
        planes_list.append(current_plane)

        # 构建 ISL 拓扑
        isl_edges = set()
        def add_edge(u, v):
            if u < v: isl_edges.add((u, v))
            else: isl_edges.add((v, u))
            
        num_planes = len(planes_list)
        
        for p_idx in range(num_planes):
            plane_nodes = planes_list[p_idx]
            
            # --- A. 同轨 (Intra-plane) ---
            plane_nodes.sort(key=lambda x: x['u'])
            n_nodes = len(plane_nodes)
            
            if n_nodes > 1:
                for k in range(n_nodes):
                    u_node = plane_nodes[k]
                    v_node = plane_nodes[(k + 1) % n_nodes]
                    dist = np.linalg.norm(u_node['pos_eci'] - v_node['pos_eci'])
                    if dist <= self.max_intra:
                        add_edge(u_node['idx'], v_node['idx'])
            
            # --- B. 异轨 (Inter-plane) ---
            next_p_idx = (p_idx + 1) % num_planes
            neighbor_plane = planes_list[next_p_idx]
            
            # 检查轨道面相邻性
            raan_diff = abs(plane_nodes[0]['raan'] - neighbor_plane[0]['raan'])
            raan_diff = min(raan_diff, 360.0 - raan_diff)
            
            if raan_diff > self.neighbor_tol:
                continue
            
            neighbor_plane.sort(key=lambda x: x['u'])
            
            for u_node in plane_nodes:
                # =========================================================
                # [核心修正] 1. 先找到几何上最匹配的邻居 (Best Match)
                # 不要在找的过程中因为纬度而跳过，否则会连到错误的卫星上
                # =========================================================
                best_v = None
                min_diff = 360.0
                
                for v_node in neighbor_plane:
                    diff = abs(u_node['u'] - v_node['u'])
                    diff = min(diff, 360.0 - diff)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_v = v_node
                
                # =========================================================
                # [核心修正] 2. 找到邻居后，再根据纬度决定是“连接”还是“断开”
                # =========================================================
                if best_v:
                    # 检查是否触犯极地熔断规则
                    is_polar_violation = False
                    if self.enable_polar_cut:
                        # 如果任意一方进入极区，必须切断连接，不许找备胎
                        if abs(u_node['lat']) > self.polar_cut_lat or abs(best_v['lat']) > self.polar_cut_lat:
                            is_polar_violation = True
                    
                    # 仅在未熔断且距离合适时连接
                    if not is_polar_violation:
                        dist = np.linalg.norm(u_node['pos_eci'] - best_v['pos_eci'])
                        if dist <= self.max_inter:
                            add_edge(u_node['idx'], best_v['idx'])

        isl_lines = []
        for u, v in isl_edges:
            isl_lines.extend([2, u, v])
            
        # 4. 构建 GSL 拓扑
        gsl_lines = []
        if gs_coords is not None and len(gs_coords) > 0 and len(valid_idxs) > 0:
            sat_positions = np.array([s['pos_ecef'] for s in sats_data])
            sat_tree = cKDTree(sat_positions)
            nearby_list = sat_tree.query_ball_point(gs_coords, r=self.max_gsl)
            for gs_idx, sat_matches in enumerate(nearby_list):
                for local_idx in sat_matches:
                    real_sat_idx = sats_data[local_idx]['idx']
                    gsl_lines.extend([2, real_sat_idx, gs_idx])

        return np.array(isl_lines, dtype=np.int32), np.array(gsl_lines, dtype=np.int32)