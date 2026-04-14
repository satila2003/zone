import os
import numpy as np
from datetime import datetime, timedelta

class DataExporter:
    def __init__(self):
        self.is_active = False
        self.export_dir = None
        self.start_time_ref = None
        self.end_time_ref = None
        self.step_counter = 0

    def start(self, parent_dir, current_time, duration_sec):
        """
        初始化导出：在 parent_dir 下创建一个新的子文件夹
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"Export_{timestamp}"
            self.export_dir = os.path.join(parent_dir, folder_name)
            
            if not os.path.exists(self.export_dir):
                os.makedirs(self.export_dir)

            self.is_active = True
            self.step_counter = 0
            self.start_time_ref = current_time
            self.end_time_ref = current_time + timedelta(seconds=duration_sec)
            
            return True, f"Saving to {folder_name}"
            
        except Exception as e:
            self.stop()
            return False, str(e)

    def _ecef_to_lla(self, x, y, z):
        """
        内部辅助函数：将 ECEF (km) 转换为 WGS84 (Lat, Lon, Alt)
        :return: (lat_deg, lon_deg, alt_km)
        """
        # WGS84 椭球参数
        a = 6378.137
        b = 6356.752314245
        f = 1.0 / 298.257223563
        e2 = 2*f - f*f  # 第一偏心率平方
        ep2 = (a**2 - b**2) / b**2 # 第二偏心率平方
        
        p = np.sqrt(x**2 + y**2)
        th = np.arctan2(a*z, b*p)
        
        # 计算经度 (弧度)
        lon_rad = np.arctan2(y, x)
        
        # 计算纬度 (弧度)
        lat_rad = np.arctan2(z + ep2 * b * np.sin(th)**3, 
                             p - e2 * a * np.cos(th)**3)
        
        # 计算高度 (km)
        N = a / np.sqrt(1 - e2 * np.sin(lat_rad)**2)
        alt_km = p / np.cos(lat_rad) - N
        
        return np.degrees(lat_rad), np.degrees(lon_rad), alt_km

    def record_frame(self, current_time, satellites, isl_indices, gsl_indices):
        """
        记录当前这一帧：创建一个新的 .txt 文件并写入所有数据
        """
        if not self.is_active or not self.export_dir: return

        try:
            t_str_safe = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            fname = f"step_{self.step_counter:04d}_{t_str_safe}.txt"
            fpath = os.path.join(self.export_dir, fname)
            
            t_str_display = current_time.strftime("%Y-%m-%d %H:%M:%S")

            with open(fpath, 'w') as f:
                # --- A. 写入元数据 ---
                f.write("[METADATA]\n")
                f.write(f"Time: {t_str_display}\n")
                f.write(f"Step: {self.step_counter}\n")
                f.write(f"Total_Sats: {len(satellites)}\n")
                f.write("\n")

                # --- B. 写入节点 (Nodes) - 修改为经纬度格式 ---
                f.write("[NODES]\n")
                # [修改] 表头变更为 Lat/Lon/Alt
                f.write("ID, Name, Lat(deg), Lon(deg), Alt(km)\n") 
                
                idx_to_id = {}
                
                for i, s in enumerate(satellites):
                    idx_to_id[i] = s.sat_id
                    
                    # 过滤无效坐标
                    if np.linalg.norm(s.position) > 100:
                        # [修改] 进行坐标转换
                        lat, lon, alt = self._ecef_to_lla(s.position[0], s.position[1], s.position[2])
                        
                        # 写入转换后的数据
                        f.write(f"{s.sat_id}, {s.name}, {lat:.5f}, {lon:.5f}, {alt:.3f}\n")
                
                f.write("\n")

                # --- C. 写入链路 (Links) - 保持不变 ---
                f.write("[LINKS]\n")
                f.write("Type, SourceID, TargetID\n") 

                # 写入 ISL
                num_isl = len(isl_indices)
                if num_isl > 0:
                    for k in range(0, num_isl, 3):
                        idx_a = isl_indices[k+1]
                        idx_b = isl_indices[k+2]
                        if idx_a in idx_to_id and idx_b in idx_to_id:
                            f.write(f"ISL, {idx_to_id[idx_a]}, {idx_to_id[idx_b]}\n")

                # 写入 GSL
                num_gsl = len(gsl_indices)
                if num_gsl > 0:
                    for k in range(0, num_gsl, 3):
                        sat_idx = gsl_indices[k+1]
                        gs_idx = gsl_indices[k+2]
                        if sat_idx in idx_to_id:
                            f.write(f"GSL, {idx_to_id[sat_idx]}, GS-{gs_idx}\n")

            self.step_counter += 1
                    
        except Exception as e:
            print(f"Export write error at step {self.step_counter}: {e}")

    def stop(self):
        self.is_active = False
        self.export_dir = None
        self.start_time_ref = None
        self.end_time_ref = None
        self.step_counter = 0