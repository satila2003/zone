from dataclasses import dataclass
import numpy as np

@dataclass
class Satellite:
    sat_id: int
    name: str
    line1: str
    line2: str
    # 运行时属性
    position: np.ndarray = np.array([0.0, 0.0, 0.0])
    # 轨道参数 (用于拓扑分组)
    inclination: float = 0.0  # 倾角
    raan: float = 0.0         # 升交点赤经 (区分轨道面)
    mean_anomaly: float = 0.0 # 平近点角 (区分前后)
    altitude: float = 0.0     # 轨道高度 (区分Shell)
    arg_perigee: float = 0.0  # 近地点幅角
    
# GroundStation 类
@dataclass
class GroundStation:
    gid: int        # 地面站 ID
    name: str       # 名称 (e.g., "Beijing")
    lat: float      # 纬度
    lon: float      # 经度
    alt: float = 0.0
    position: np.ndarray = np.array([0.0, 0.0, 0.0]) # ECEF 坐标
    
    def compute_ecef(self):
        """ 将经纬度转为 ECEF 直角坐标 """
        lat_rad = np.radians(self.lat)
        lon_rad = np.radians(self.lon)
        alt_km = self.alt / 1000.0 # 假设 alt 是米，转为 km
        
        # WGS84 椭球参数
        a = 6378.137 # 地球长半轴 (km)
        f = 1 / 298.257223563
        e2 = 2*f - f*f
        
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        N = a / np.sqrt(1 - e2 * sin_lat**2)
        
        x = (N + alt_km) * cos_lat * cos_lon
        y = (N + alt_km) * cos_lat * sin_lon
        z = (N * (1 - e2) + alt_km) * sin_lat
        
        self.position = np.array([x, y, z])