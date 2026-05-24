import numpy as np

class Strategy:
    def compute_links(self, satellites):
        raise NotImplementedError

    def _format_output(self, isl_lines, satellites):
        link_stats = []
        isl_arr = np.array(isl_lines, dtype=np.int32)
        
        if len(isl_arr) > 0:
            positions = np.array([s.position for s in satellites])
            u_list = isl_arr[1::3]
            v_list = isl_arr[2::3]
            
            pos_u = positions[u_list]
            pos_v = positions[v_list]
            dists = np.linalg.norm(pos_u - pos_v, axis=1)
            latencies = (dists / 299792.458) * 1000.0 
            
            for i in range(len(u_list)):
                link_stats.append({
                    'id': i + 1, 
                    'src': u_list[i], 
                    'tgt': v_list[i], 
                    'src_name': satellites[u_list[i]].name, 
                    'tgt_name': satellites[v_list[i]].name,
                    'latency': round(latencies[i], 4) 
                })
                
        return isl_arr, link_stats

class GridStarStrategy(Strategy):
    """ 原 Starlink Mesh 策略，现更名为 +Grid（Star） """
    def __init__(self, plane_tolerance=6.0, max_intra_dist=5000, max_inter_dist=5000, enable_polar_cut=False, polar_cut_lat=70.0):
        self.plane_tol = plane_tolerance; self.max_intra = max_intra_dist; self.max_inter = max_inter_dist
        self.enable_polar_cut = enable_polar_cut; self.polar_cut_lat = polar_cut_lat
        
    def compute_links(self, satellites):
        if not satellites: return np.array([], dtype=np.int32), []
        sats_data = []
        for i, s in enumerate(satellites):
            if not hasattr(s, 'position_eci') or np.linalg.norm(s.position_eci) < 100: continue
            rx, ry, rz = s.position_eci; raan_rad = np.radians(s.raan); inc_rad = np.radians(s.inclination)
            x_tmp = rx * np.cos(raan_rad) + ry * np.sin(raan_rad); y_tmp = -rx * np.sin(raan_rad) + ry * np.cos(raan_rad)
            y_plane = y_tmp * np.cos(inc_rad) + rz * np.sin(inc_rad); x_plane = x_tmp 
            u_angle = np.degrees(np.arctan2(y_plane, x_plane)) % 360.0
            curr_lat = np.degrees(np.arcsin(s.position[2] / np.linalg.norm(s.position)))
            sats_data.append({'idx': i, 'raan': s.raan, 'u': u_angle, 'pos_eci': s.position_eci, 'lat': curr_lat})

        if not sats_data: return np.array([], dtype=np.int32), []
        sats_data.sort(key=lambda x: x['raan'])
        planes_list = []; current_plane = [sats_data[0]]
        for k in range(1, len(sats_data)):
            curr = sats_data[k]; prev = sats_data[k-1]
            diff = min(abs(curr['raan'] - prev['raan']), 360.0 - abs(curr['raan'] - prev['raan']))
            if diff > self.plane_tol: planes_list.append(current_plane); current_plane = []
            current_plane.append(curr)
        planes_list.append(current_plane)

        isl_edges = set()
        def add_edge(u, v): isl_edges.add((u, v) if u < v else (v, u))
            
        num_planes = len(planes_list)
        for p_idx in range(num_planes):
            plane_nodes = planes_list[p_idx]; plane_nodes.sort(key=lambda x: x['u']); n_nodes = len(plane_nodes)
            if n_nodes > 1:
                for k in range(n_nodes):
                    u_node = plane_nodes[k]; v_node = plane_nodes[(k + 1) % n_nodes]
                    if np.linalg.norm(u_node['pos_eci'] - v_node['pos_eci']) <= self.max_intra: add_edge(u_node['idx'], v_node['idx'])
            
            neighbor_plane = planes_list[(p_idx + 1) % num_planes]
            if min(abs(plane_nodes[0]['raan'] - neighbor_plane[0]['raan']), 360.0 - abs(plane_nodes[0]['raan'] - neighbor_plane[0]['raan'])) > self.plane_tol * 2: continue 
            
            for u_node in plane_nodes:
                best_v = min(neighbor_plane, key=lambda v: min(abs(u_node['u'] - v['u']), 360.0 - abs(u_node['u'] - v['u'])))
                if not (self.enable_polar_cut and (abs(u_node['lat']) > self.polar_cut_lat or abs(best_v['lat']) > self.polar_cut_lat)):
                    if np.linalg.norm(u_node['pos_eci'] - best_v['pos_eci']) <= self.max_inter: add_edge(u_node['idx'], best_v['idx'])

        isl_lines = []
        for u, v in isl_edges: isl_lines.extend([2, u, v])
        return self._format_output(isl_lines, satellites)

class GridDeltaStrategy(Strategy):
    """ 原 Walker Delta 策略，现更名为 +Grid（Delta） """
    def __init__(self, turnaround_lat=51.0):
        self.turnaround_lat = turnaround_lat; self.static_edges = None 
        
    def compute_links(self, satellites):
        if not satellites or not getattr(satellites[0], 'is_walker', False): return np.array([], dtype=np.int32), []
        if self.static_edges is None:
            self.static_edges = []; P = max(s.plane_idx for s in satellites) + 1; S = max(s.node_idx for s in satellites) + 1
            get_idx = lambda p, s: (p % P) * S + (s % S)
            for p in range(P):
                for s in range(S):
                    u_idx = get_idx(p, s); self.static_edges.append(('intra', u_idx, get_idx(p, s + 1)))
                    best_v_inter, min_dist = -1, float('inf')
                    for ns in range(S):
                        dist = np.linalg.norm(satellites[u_idx].position_eci - satellites[get_idx(p + 1, ns)].position_eci)
                        if dist < min_dist: min_dist = dist; best_v_inter = get_idx(p + 1, ns)
                    self.static_edges.append(('inter', u_idx, best_v_inter))

        isl_lines = []; positions = np.array([s.position for s in satellites])
        norms = np.linalg.norm(positions, axis=1); norms[norms == 0] = 1; lats = np.degrees(np.arcsin(positions[:, 2] / norms))
        for edge_type, u, v in self.static_edges:
            if edge_type == 'intra': isl_lines.extend([2, u, v])
            elif edge_type == 'inter' and abs(lats[u]) < self.turnaround_lat and abs(lats[v]) < self.turnaround_lat: isl_lines.extend([2, u, v])
        return self._format_output(isl_lines, satellites)