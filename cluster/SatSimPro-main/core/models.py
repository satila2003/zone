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
    position_eci: np.ndarray = np.array([0.0, 0.0, 0.0])
    
    # 轨道参数 (用于拓扑分组)
    inclination: float = 0.0  
    raan: float = 0.0         
    mean_anomaly: float = 0.0 
    altitude: float = 0.0     
    arg_perigee: float = 0.0  
    
    # Walker 星座专属属性 
    is_walker: bool = False   
    plane_idx: int = -1       
    node_idx: int = -1