import os
import numpy as np
from datetime import datetime, timedelta

class DataExporter:
    def __init__(self):
        self.is_active = False
        self.export_dir = None
        self.frames_dir = None
        self.sats_dir = None
        
        self.start_time_ref = None
        self.end_time_ref = None
        self.step_counter = 0
        
        # 时序数据缓存字典
        self.fixed_neighbors = {} # 格式: { sat_idx: [n1_idx, n2_idx, n3_idx, n4_idx] }
        self.sat_history = {}     # 格式: { sat_idx: ["0 down down 12.34 down", ...] }
        self.last_satellites = [] # 保存卫星对象的引用以供最后提取名字

    def start(self, parent_dir, current_time, duration_sec):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.export_dir = os.path.join(parent_dir, f"Export_{timestamp}")
            
            # 分别创建两个子文件夹
            self.frames_dir = os.path.join(self.export_dir, "frames")
            self.sats_dir = os.path.join(self.export_dir, "satellites")
            os.makedirs(self.frames_dir, exist_ok=True)
            os.makedirs(self.sats_dir, exist_ok=True)

            self.is_active = True
            self.step_counter = 0
            self.start_time_ref = current_time
            self.end_time_ref = current_time + timedelta(seconds=duration_sec)
            
            # 初始化缓存
            self.fixed_neighbors = {}
            self.sat_history = {}
            self.last_satellites = []
            
            return True, f"Saving to Export_{timestamp}"
        except Exception as e:
            self.stop()
            return False, str(e)

    def _ecef_to_lla(self, x, y, z):
        """ ECEF 坐标转经纬度高程 """
        a = 6378.137; b = 6356.752314245
        f = 1.0 / 298.257223563; e2 = 2*f - f*f; ep2 = (a**2 - b**2) / b**2 
        p = np.sqrt(x**2 + y**2); th = np.arctan2(a*z, b*p)
        lon_rad = np.arctan2(y, x); lat_rad = np.arctan2(z + ep2 * b * np.sin(th)**3, p - e2 * a * np.cos(th)**3)
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        return np.degrees(lat_rad), np.degrees(lon_rad), p / np.cos(lat_rad) - N

    def _build_fixed_neighbors(self, satellites):
        """ 智能推导每个卫星在 Walker 星座中固定的 4 个邻居 """
        if not satellites or not getattr(satellites[0], 'is_walker', False):
            return # 如果不是 Walker 星座，就不生成单星拓扑文件
            
        P = max(s.plane_idx for s in satellites) + 1
        S = max(s.node_idx for s in satellites) + 1
        get_idx = lambda p, s: (p % P) * S + (s % S)
        
        # 建立向右(p+1)的最短距离映射
        inter_edges = {}
        for p in range(P):
            for s in range(S):
                u = get_idx(p, s)
                min_dist = float('inf')
                best_v = -1
                for ns in range(S):
                    cand = get_idx(p+1, ns)
                    dist = np.linalg.norm(satellites[u].position_eci - satellites[cand].position_eci)
                    if dist < min_dist:
                        min_dist = dist; best_v = cand
                inter_edges[u] = best_v

        # 寻回所有方向的邻居
        for u in range(len(satellites)):
            p = satellites[u].plane_idx
            s = satellites[u].node_idx
            
            v_next = get_idx(p, s+1)  # 同轨后
            v_prev = get_idx(p, s-1)  # 同轨前
            v_right = inter_edges[u]  # 异轨右
            
            # 异轨左（在 p-1 轨道面上，其向右的节点正是 u 的那颗卫星）
            v_left = -1
            for prev_s in range(S):
                cand = get_idx(p-1, prev_s)
                if inter_edges.get(cand) == u:
                    v_left = cand
                    break
            if v_left == -1: v_left = get_idx(p-1, s) # 兜底逻辑
            
            # 保存 4 个固定的邻居
            self.fixed_neighbors[u] = [v_left, v_right, v_prev, v_next]
            self.sat_history[u] = []

    def record_frame(self, current_time, satellites, all_links_data):
        if not self.is_active or not self.export_dir: return
        self.last_satellites = satellites

        # 1. 仅在第 0 帧时，初始化单星拓扑邻居字典
        if self.step_counter == 0:
            self._build_fixed_neighbors(satellites)

        # 2. 写入旧版的逐帧全局文件 (保存至 frames 文件夹)
        try:
            fname = f"step_{self.step_counter:04d}_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}.txt"
            with open(os.path.join(self.frames_dir, fname), 'w') as f:
                f.write(f"[METADATA]\nTime: {current_time.strftime('%Y-%m-%d %H:%M:%S')}\nStep: {self.step_counter}\nTotal_Sats: {len(satellites)}\n\n")
                f.write("[NODES]\nID, Name, Lat(deg), Lon(deg), Alt(km)\n") 
                for i, s in enumerate(satellites):
                    if np.linalg.norm(s.position) > 100:
                        lat, lon, alt = self._ecef_to_lla(s.position[0], s.position[1], s.position[2])
                        f.write(f"{s.sat_id}, {s.name}, {lat:.5f}, {lon:.5f}, {alt:.3f}\n")
                
                f.write("\n[LINKS]\nType, SourceName, TargetName, Latency(ms)\n") 
                for link in all_links_data:
                    f.write(f"ISL, {link['src_name']}, {link['tgt_name']}, {link['latency']:.4f}\n")
        except Exception as e: print(f"Export frame error: {e}")

        # 3. 收集 Per-Satellite 视角下的链路状态时序数据 (存入内存)
        if self.fixed_neighbors:
            # 建立 O(1) 的已连通链路快查表
            active_links = {}
            for link in all_links_data:
                u = link['src']; v = link['tgt']; lat = link['latency']
                active_links[(u, v)] = lat
                active_links[(v, u)] = lat
                
            for u in range(len(satellites)):
                if u not in self.fixed_neighbors: continue
                
                # 开始拼凑这一行的内容
                row = [str(self.step_counter)]
                for v in self.fixed_neighbors[u]:
                    if (u, v) in active_links:
                        # 如果在这帧是连通的，记录下算好的延迟
                        row.append(f"{active_links[(u, v)]:.8f}")
                    else:
                        # 如果没有连接，记录 down
                        row.append("down")
                        
                self.sat_history[u].append(" ".join(row))

        self.step_counter += 1

    def stop(self):
        # 4. 停止导出时，将收集好的单星时序数据一次性写入对应的 txt (保存至 satellites 文件夹)
        if self.is_active and self.fixed_neighbors and self.sat_history:
            try:
                for u, history in self.sat_history.items():
                    sat = self.last_satellites[u]
                    fpath = os.path.join(self.sats_dir, f"satellite_{sat.name}.txt")
                    with open(fpath, 'w') as f:
                        f.write("Time\n")
                        # 写入该星对应的四个固定护法的名字作为表头
                        names = [f"Satellite_{self.last_satellites[v].name}" for v in self.fixed_neighbors[u]]
                        f.write(" ".join(names) + "\n")
                        f.write("-" * 60 + "\n")
                        # 写入所有的时序流
                        for line in history:
                            f.write(line + "\n")
            except Exception as e:
                print(f"Export timeseries error: {e}")
                
        # 释放内存与清理句柄
        self.is_active = False
        self.export_dir = None
        self.frames_dir = None
        self.sats_dir = None
        self.fixed_neighbors = {}
        self.sat_history = {}
        self.last_satellites = []