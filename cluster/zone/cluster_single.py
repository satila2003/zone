import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
# =========================
# 配置区：全局参数定义，控制卫星分域全流程
# =========================
# 输入数据文件夹：Starlink550卫星轨迹step文件
INPUT_FOLDER = r"F:\Py_Project\always\cluster\zone\inputs\starlink550_data\data_1100"   # 当前工作目录是 RedTE-main
# 输出文件夹：保存分域结果、图片、文本
OUTPUT_FOLDER = r"F:\Py_Project\always\cluster\zone\outputs"
# 地球半径（单位：km），用于轨道计算
R_EARTH = 6371.0
R_EARTH_DEFAULT = 6371.0

# 计算卫星轨道法向量的最小有效样本数
MIN_NORMAL_SAMPLES = 5
# 绘图使用的时间片索引
PLOT_STEP_INDEX = 0

# 聚类方法选择：numpy向量聚类/sklearn密度聚类/RAAN分桶聚类
PROCESS_METHOD = "raan"  # "numpy", "sklearn", "raan"
# 轨道内分段方法：按经度/轨道真角划分
PHASE_SPLIT_METHOD = "orbital_angle"  # "longitude" 或 "orbital_angle"
# RAAN分桶的总轨道面数（Starlink550标准：72个轨道面）
N_planes_by_raan = 72 # RAAN 分桶法的轨道面数量，必须能被 N_BIG_PLANES 整除
# 合并后的大轨道数量（72/12=6）
N_BIG_PLANES = 6 # 合成大轨道的数量，必须能整除 N_planes_by_raan
# 每个大轨道内分段数，最终总域数=大轨道数×分段数
N_PHASES = 3 # 每个轨道内的分段数量，最终区域数量 = N_BIG_PLANES * N_PHASES

# 图像输出开关：控制是否生成不同粒度的可视化图片
OUTPUT_TOTAL_PLANES = False           # 是否输出72个原始轨道的单独图
OUTPUT_BIG_PLANES = True         # 是否输出6个合并后大轨道图
OUTPUT_DOMAINS_SEPARATE = True  # 是否输出18个最终域的单独图

# numpy向量聚类参数：法向量夹角阈值（度）
ANGLE_THRESHOLD_DEG = 3.0

# sklearn DBSCAN密度聚类参数：邻域半径、最小样本数
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 5



# =========================
# 轨道面法向量估计
# =========================
def estimate_satellite_normals(all_steps, r_earth=6371.0):
    """
    用相邻时间片位置估计每颗卫星的一组轨道面法向量

    做法:
        对同一颗卫星在相邻两个时间片的位置向量 r1, r2 做叉积
            n = r1 × r2
        得到轨道面法向量样本

    参数:
        all_steps: list[dict]
            每个元素是一个时间片的卫星字典

    返回:
        sat_normals_raw:
            dict[sat_id] = [n1, n2, ...]
    """
    sat_normals_raw = defaultdict(list)

    for k in range(len(all_steps) - 1):
        step_a = all_steps[k]
        step_b = all_steps[k + 1]

        common_ids = set(step_a.keys()) & set(step_b.keys())

        for sat_id in common_ids:
            s1 = step_a[sat_id]
            s2 = step_b[sat_id]

            r1 = lla_to_xyz(s1["lat"], s1["lon"], s1["alt"])
            r2 = lla_to_xyz(s2["lat"], s2["lon"], s2["alt"])

            n = np.cross(r1, r2)
            n = normalize(n)
            if n is None:
                continue

            # 统一法向量朝向，避免同一轨道面方向相反
            if n[2] < 0:
                n = -n

            sat_normals_raw[sat_id].append(n)

    return sat_normals_raw



def split_satellites_within_plane_by_orbital_angle(step_data, sat_to_plane, sat_normals, n_phases, r_earth=6371.0):
    """
    按轨道面内角位置划分为 n_phases 段
    """
    sat_to_metric = {}
    sat_to_phase = {}

    phase_width = 360.0 / n_phases

    for sat_id, plane_id in sat_to_plane.items():
        if sat_id not in step_data:
            continue
        if sat_id not in sat_normals:
            continue

        lat = step_data[sat_id]["lat"]
        lon = step_data[sat_id]["lon"]
        alt = step_data[sat_id]["alt"]

        r = lla_to_xyz(lat, lon, alt, r_earth=r_earth)
        n = sat_normals[sat_id]

        u_deg = argument_of_latitude_from_r_n(r, n)
        if u_deg is None:
            continue

        phase = int(u_deg // phase_width)
        if phase >= n_phases:
            phase = n_phases - 1

        sat_to_metric[sat_id] = u_deg
        sat_to_phase[sat_id] = phase

    return sat_to_metric, sat_to_phase


def average_satellite_normals(sat_normals_raw, min_samples=5):
    """
    对每颗卫星多次法向量样本求平均
    只保留样本数 >= min_samples 的卫星

    参数:
        sat_normals_raw: dict[sat_id] = [n1, n2, ...]
        min_samples: 最少样本数阈值

    返回:
        sat_normals: dict[sat_id] = mean_n
    """
    sat_normals = {}

    for sat_id, normals in sat_normals_raw.items():
        if len(normals) < min_samples:
            continue

        mean_n = np.mean(normals, axis=0)
        mean_n = normalize(mean_n)
        if mean_n is None:
            continue

        if mean_n[2] < 0:
            mean_n = -mean_n

        sat_normals[sat_id] = mean_n

    return sat_normals


# =========================
# 方法1：RAAN 分桶聚类
# =========================
def cluster_by_raan_bucket(sat_normals, n_planes):
    """
    按 RAAN 分桶

    参数:
        sat_normals: dict[sat_id] = np.array([nx, ny, nz])
        n_planes: 轨道面数量

    返回:
        sat_to_plane: dict[sat_id] = plane_id
        sat_to_raan: dict[sat_id] = raan_deg
    """
    sat_to_plane = {}
    sat_to_raan = {}

    bin_width = 360.0 / n_planes

    for sat_id, n in sat_normals.items():
        n = normalize(n)
        if n is None:
            continue

        if n[2] < 0:
            n = -n

        raan = normal_to_raan(n)
        if raan is None:
            continue

        plane_id = int(raan // bin_width)
        if plane_id >= n_planes:
            plane_id = n_planes - 1

        sat_to_plane[sat_id] = plane_id
        sat_to_raan[sat_id] = raan

    return sat_to_plane, sat_to_raan


# =========================
# 方法2：纯 numpy 角度阈值聚类
# =========================
def cluster_by_numpy(sat_normals, angle_threshold_deg=3.0):
    """
    纯 numpy 轨道面聚类

    参数:
        sat_normals: dict[sat_id] = np.array([nx, ny, nz])
        angle_threshold_deg: 法向量夹角阈值（度）

    返回:
        sat_to_plane: dict[sat_id] = plane_id
    """
    items = sorted(sat_normals.items(), key=lambda x: x[0])

    plane_centers = []
    plane_member_normals = []
    sat_to_plane = {}

    for sat_id, n in items:
        n = normalize(n)
        if n is None:
            continue

        if n[2] < 0:
            n = -n

        best_plane = None
        best_angle = 1e9

        for pid, center in enumerate(plane_centers):
            ang = angle_deg(n, center)
            if ang < best_angle:
                best_angle = ang
                best_plane = pid

        if best_plane is not None and best_angle <= angle_threshold_deg:
            sat_to_plane[sat_id] = best_plane
            plane_member_normals[best_plane].append(n)

            new_center = np.mean(plane_member_normals[best_plane], axis=0)
            new_center = normalize(new_center)
            if new_center is not None and new_center[2] < 0:
                new_center = -new_center
            plane_centers[best_plane] = new_center
        else:
            new_pid = len(plane_centers)
            sat_to_plane[sat_id] = new_pid
            plane_centers.append(n)
            plane_member_normals.append([n])

    return sat_to_plane


# =========================
# 方法3：sklearn DBSCAN 聚类
# =========================
def cluster_by_sklearn(sat_normals, eps=0.03, min_samples=5):
    """
    sklearn DBSCAN 聚类

    参数:
        sat_normals: dict[sat_id] = np.array([nx, ny, nz])
        eps: DBSCAN 邻域阈值
        min_samples: 核心点最少样本数

    返回:
        sat_to_plane: dict[sat_id] = plane_id
    """
    sat_ids = sorted(sat_normals.keys())
    features = np.array([sat_normals[sid] for sid in sat_ids])

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean"
    )

    labels = clustering.fit_predict(features)

    sat_to_plane = {}
    for sid, label in zip(sat_ids, labels):
        sat_to_plane[sid] = int(label)

    return sat_to_plane



# =========================
# 文件读取模块：加载卫星轨迹时间片数据
# =========================
def read_one_step_file(file_path):
    """
    读取单个时间片(step)文件中的卫星节点信息
    参数: file_path - 单个step文件路径
    返回: sats - 卫星字典，key=卫星ID，value=卫星经纬度/高度信息
    """
    # 初始化卫星空字典
    sats = {}

    # 以UTF-8编码读取文件内容
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    # 查找数据表头行：ID, Name, Lat(deg), Lon(deg), Alt(km)
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ID, Name, Lat(deg), Lon(deg), Alt(km)"):
            start_idx = i
            break

    # 未找到表头，打印警告并返回空字典
    if start_idx is None:
        print(f"[WARN] 未找到表头，跳过文件: {file_path}")
        return sats

    # 从表头下一行开始解析卫星数据
    for line in lines[start_idx + 1:]:
        line = line.strip()
        # 跳过空行
        if not line:
            continue

        # 遇到新的章节标记，停止读取
        if line.startswith("[") and line.endswith("]"):
            break

        # 按逗号分割数据字段
        parts = [p.strip() for p in line.split(",")]
        # 字段不足5个，跳过
        if len(parts) < 5:
            continue

        # 数据类型转换，异常数据直接跳过
        try:
            sat_id = int(parts[0])
            sat_name = parts[1]
            lat = float(parts[2])
            lon = float(parts[3])
            alt = float(parts[4])
        except ValueError:
            continue

        # 将解析后的卫星数据存入字典
        sats[sat_id] = {
            "name": sat_name,
            "lat": lat,
            "lon": lon,
            "alt": alt
        }

    return sats

def load_all_steps(folder):
    """
    批量加载文件夹下所有卫星轨迹step文件
    参数: folder - 输入文件夹路径
    返回: files - 文件名列表, all_steps - 所有时间片的卫星数据列表
    """
    # 筛选以step_开头、.txt结尾的文件并排序
    files = sorted(
        f for f in os.listdir(folder)
        if f.startswith("step_") and f.endswith(".txt")
    )

    all_steps = []
    # 遍历读取所有step文件
    for fname in files:
        path = os.path.join(folder, fname)
        sats = read_one_step_file(path)
        all_steps.append(sats)

    return files, all_steps


# =========================
# 轨道面聚类与域划分模块：核心分域逻辑
# =========================
def validate_region_params(n_planes_total, n_big_planes, n_phases):
    """
    验证分域参数合法性，确保参数满足整除关系
    参数: n_planes_total-总轨道数, n_big_planes-大轨道数, n_phases-每轨道分段数
    """
    # 大轨道数必须大于0
    if n_big_planes <= 0:
        raise ValueError("n_big_planes 必须大于 0")
    # 分段数必须大于0
    if n_phases <= 0:
        raise ValueError("n_phases 必须大于 0")
    # 总轨道数必须能被大轨道数整除（保证均匀合并）
    if n_planes_total % n_big_planes != 0:
        raise ValueError(
            f"总轨道数 {n_planes_total} 不能被大轨道数 {n_big_planes} 整除"
        )
    
def build_big_plane_groups(sat_to_plane, n_planes_total, n_big_planes):
    """
    将细粒度原始轨道面聚合为大轨道面
    参数: sat_to_plane-卫星到原始轨道的映射, n_planes_total-总轨道数, n_big_planes-大轨道数
    返回: sat_to_big_plane-卫星到大轨道的映射
    """
    # 计算每个大轨道包含的原始轨道数量
    planes_per_big_plane = n_planes_total // n_big_planes
    sat_to_big_plane = {}

    # 遍历所有卫星，计算所属大轨道ID
    for sat_id, plane_id in sat_to_plane.items():
        sat_to_big_plane[sat_id] = plane_id // planes_per_big_plane

    return sat_to_big_plane

def build_domain_id(sat_to_plane, sat_to_phase, n_planes_total, n_big_planes, n_phases):
    """
    组合大轨道ID+相位ID，生成最终唯一域ID
    最终域编号规则：domain_id = 大轨道ID×每轨道分段数 + 相位ID
    参数: sat_to_plane-卫星轨道映射, sat_to_phase-卫星相位映射
    返回: sat_to_domain-卫星到最终域的映射
    """
    # 每个大轨道包含的原始轨道数
    planes_per_big_plane = n_planes_total // n_big_planes
    sat_to_domain = {}

    # 遍历卫星，生成唯一域ID
    for sat_id, plane_id in sat_to_plane.items():
        # 无相位数据的卫星跳过
        if sat_id not in sat_to_phase:
            continue

        # 计算大轨道ID
        big_plane_id = plane_id // planes_per_big_plane
        # 获取相位ID
        phase_id = sat_to_phase[sat_id]

        # 生成最终域ID
        domain_id = big_plane_id * n_phases + phase_id
        sat_to_domain[sat_id] = domain_id

    return sat_to_domain



def split_satellites_within_plane(step_data,
                                  sat_to_plane,
                                  n_phases,
                                  method="longitude",
                                  sat_normals=None,
                                  r_earth=6371.0):
    """
    统一对外接口：根据 method 选择分段方式

    参数:
        method:
            "longitude"     -> 按经度排序后三等分
            "orbital_angle" -> 按轨道面内角位置分三段

    返回:
        sat_to_metric:
            若 method="longitude"，则为经度
            若 method="orbital_angle"，则为轨道面内角位置
        sat_to_phase:
            phase_id = 0/1/2
    """
    if method == "longitude":
        return split_satellites_within_plane_by_longitude(step_data, sat_to_plane, n_phases)

    elif method == "orbital_angle":
        if sat_normals is None:
            raise ValueError("使用 orbital_angle 分段时，必须提供 sat_normals")
        return split_satellites_within_plane_by_orbital_angle(
            step_data,
            sat_to_plane,
            sat_normals,
            n_phases,
            r_earth=r_earth
        )

    else:
        raise ValueError("method 必须是 'longitude' 或 'orbital_angle'")

# =========================
# 向量与坐标工具
# =========================
def normalize(v):
    """
    向量归一化

    参数:
        v: numpy 向量

    返回:
        单位向量；若范数过小则返回 None
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    return v / norm


def lla_to_xyz(lat_deg, lon_deg, alt_km, r_earth=R_EARTH_DEFAULT):
    """
    经纬高 -> 地心直角坐标（球地球近似）

    参数:
        lat_deg: 纬度（度）
        lon_deg: 经度（度）
        alt_km: 高度（km）
        r_earth: 地球半径（km）

    返回:
        np.array([x, y, z])
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = r_earth + alt_km

    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)


def angle_deg(a, b):
    """
    计算两个向量夹角（单位：度）

    参数:
        a, b: numpy 向量

    返回:
        夹角（度）
    """
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return 180.0

    cos_val = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.degrees(math.acos(cos_val))


# =========================
# 轨道面几何工具
# =========================
def normal_to_raan(n):
    """
    根据轨道面法向量估计 RAAN

    参数:
        n: 轨道面法向量

    返回:
        raan_deg ∈ [0, 360)
    """
    n = normalize(n)
    if n is None:
        return None

    node_x = -n[1]
    node_y = n[0]

    raan_deg = math.degrees(math.atan2(node_y, node_x))
    if raan_deg < 0:
        raan_deg += 360.0

    return raan_deg


def normal_to_inclination_raan(n):
    """
    根据轨道面法向量近似计算倾角和 RAAN

    参数:
        n: 轨道面法向量

    返回:
        (inclination_deg, raan_deg)
    """
    n = normalize(n)
    if n is None:
        return None, None

    i_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
    raan_deg = normal_to_raan(n)

    return i_deg, raan_deg


def normal_to_node_vector(n):
    """
    根据轨道面法向量求升交点方向向量

    若 n = [nx, ny, nz]
    则升交点方向可近似取:
        [-ny, nx, 0]

    参数:
        n: 轨道面法向量

    返回:
        升交点方向单位向量
    """
    n = normalize(n)
    if n is None:
        return None

    node = np.array([-n[1], n[0], 0.0], dtype=float)
    return normalize(node)


def argument_of_latitude_from_r_n(r, n):
    """
    根据位置向量 r 和轨道面法向量 n
    计算轨道面内角位置（近似 argument of latitude）

    计算步骤:
    1. 用升交点方向作为轨道面内参考方向 e1
    2. 构造 e2 = n × e1
    3. 在 (e1, e2) 坐标系中对 r 做 atan2

    参数:
        r: 卫星位置向量
        n: 轨道面法向量

    返回:
        u_deg ∈ [0, 360)
    """
    n = normalize(n)
    if n is None:
        return None

    e1 = normal_to_node_vector(n)
    if e1 is None:
        return None

    e2 = np.cross(n, e1)
    e2 = normalize(e2)
    if e2 is None:
        return None

    x = np.dot(r, e1)
    y = np.dot(r, e2)

    u_deg = math.degrees(math.atan2(y, x))
    if u_deg < 0:
        u_deg += 360.0

    return u_deg


# =========================
# 结果分析与输出模块：打印/保存分域结果
# =========================
def save_sat_to_domain(filename,
                       sat_to_plane,
                       sat_to_raan,
                       sat_to_big_plane,
                       sat_to_metric,
                       sat_to_phase,
                       sat_to_domain,
                       metric_name="orbital_angle_deg"):
    """
    保存完整分域结果到CSV文件，包含所有关键参数
    输出列：卫星ID、原始轨道ID、RAAN、大轨道ID、轨道角度、相位ID、最终域ID
    """
    # 写入文件，UTF-8编码
    with open(filename, "w", encoding="utf-8") as f:
        # 写入表头
        f.write(f"sat_id,plane_id,raan_deg,big_plane_id,{metric_name},phase_id,domain_id\n")
        # 按卫星ID排序，逐行写入数据
        for sat_id in sorted(sat_to_domain.keys()):
            plane_id = sat_to_plane.get(sat_id, "")
            raan = sat_to_raan.get(sat_id, "")
            big_plane_id = sat_to_big_plane.get(sat_id, "")
            metric_value = sat_to_metric.get(sat_id, "")
            phase_id = sat_to_phase.get(sat_id, "")
            domain_id = sat_to_domain.get(sat_id, "")

            f.write(f"{sat_id},{plane_id},{raan},{big_plane_id},{metric_value},{phase_id},{domain_id}\n")

def build_plane_members(sat_to_plane):
    """
    构建轨道面-卫星成员映射：将 卫星→轨道 转为 轨道→卫星集合
    返回: 轨道面ID对应卫星ID集合的字典
    """
    plane_members = defaultdict(set)
    for sat_id, plane_id in sat_to_plane.items():
        plane_members[plane_id].add(sat_id)
    return dict(plane_members)

def print_plane_summary(method_name, sat_to_plane, sat_normals):
    """
    打印轨道面聚类结果摘要：轨道数、卫星数、倾角、RAAN
    参数: method_name-聚类方法, sat_to_plane-卫星轨道映射, sat_normals-卫星法向量
    """
    print()
    print(f"===== {method_name} 聚类结果 =====")
    plane_members = build_plane_members(sat_to_plane)

    print(f"轨道面数量: {len(plane_members)}")
    # 遍历每个轨道面，打印参数
    for pid in sorted(plane_members.keys()):
        members = plane_members[pid]
        normals = [sat_normals[sid] for sid in members if sid in sat_normals]

        if normals:
            # 计算轨道面平均法向量
            center = np.mean(normals, axis=0)
            center = normalize(center)
            if center is not None and center[2] < 0:
                center = -center

            # 法向量转轨道倾角和RAAN
            i_deg, raan_deg = normal_to_inclination_raan(center)

            print(
                f"Plane {pid:03d}: {len(members):4d} sats, "
                f"Inclination ~ {i_deg:6.2f} deg, "
                f"RAAN ~ {raan_deg:7.2f} deg" # 平均法向量对应的轨道面参数
            )
        else:
            print(f"Plane {pid:03d}: {len(members):4d} sats")


# =========================
# 可视化模块：卫星分布绘图
# =========================
def plot_one_step_by_plane(step_data, sat_to_label, title, output_path=None, label_prefix="Plane"):
    """
    绘制单个时间片的总览图：所有类别在一张图展示
    参数: step_data-时间片数据, sat_to_label-卫星分类映射, title-图表标题
    """
    # 创建画布
    plt.figure(figsize=(16, 8))

    # 按类别分组经纬度数据
    grouped = defaultdict(lambda: {"lon": [], "lat": []})
    unclassified_lon = []
    unclassified_lat = []

    # 遍历卫星，分组数据
    for sat_id, sat_info in step_data.items():
        lon = sat_info["lon"]
        lat = sat_info["lat"]

        if sat_id in sat_to_label:
            pid = sat_to_label[sat_id]
            grouped[pid]["lon"].append(lon)
            grouped[pid]["lat"].append(lat)
        else:
            # 未分类卫星单独存储
            unclassified_lon.append(lon)
            unclassified_lat.append(lat)

    # 使用HSV配色，区分不同类别
    cmap = plt.get_cmap('hsv')
    num_groups = len(grouped)

    # 绘制每个类别的卫星点
    for idx, pid in enumerate(sorted(grouped.keys())):
        color = cmap(idx / max(1, num_groups))
        plt.scatter(
            grouped[pid]["lon"],
            grouped[pid]["lat"],
            s=15,
            color=color,
            alpha=0.8,
            label=f"{label_prefix} {pid}"
        )

    # 绘制未分类卫星（灰色叉号）
    if unclassified_lon:
        plt.scatter(
            unclassified_lon,
            unclassified_lat,
            s=10,
            color='grey',
            alpha=0.5,
            marker="x",
            label="Unclassified"
        )

    # 图表样式设置
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(title)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc='upper left', ncol=4, fontsize=8, markerscale=1.5)
    plt.tight_layout()

    # 保存图片（指定路径则保存）
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_each_plane_separately(step_data, sat_to_label, title_prefix, output_folder=None,
                               file_prefix="plane", label_prefix="Plane"):
    """
    为每个类别单独绘制一张分布图（支持轨道/大轨道/域三种粒度）
    """
    # 按类别分组数据
    grouped = defaultdict(lambda: {"lon": [], "lat": []})

    for sat_id, sat_info in step_data.items():
        if sat_id not in sat_to_label:
            continue

        pid = sat_to_label[sat_id]
        grouped[pid]["lon"].append(sat_info["lon"])
        grouped[pid]["lat"].append(sat_info["lat"])

    # 创建输出文件夹
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    # 遍历每个类别，单独绘图
    for pid in sorted(grouped.keys()):
        plt.figure(figsize=(10, 6))

        plt.scatter(
            grouped[pid]["lon"],
            grouped[pid]["lat"],
            s=12,
            alpha=0.8,
            label=f"{label_prefix} {pid}"
        )

        # 图表样式
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title(f"{title_prefix} - {label_prefix} {pid}")
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()

        # 保存单张图片
        if output_folder is not None:
            output_path = os.path.join(output_folder, f"{file_prefix}_{pid:03d}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")

        plt.close()


# =========================
# 主程序：Starlink卫星分域核心流程
# 流程：读取数据→计算法向量→72轨道聚类→合并6大轨道→每轨道分3段→生成18个域→输出结果
# =========================
def main():
    """
    主流程:
    1. 读取所有时间片
    2. 估计轨道面法向量
    3. RAAN 分桶为72轨道
    4. 每12个轨道聚合为6个大轨道
    5. 在选定时间片上，用轨道面内角位置把每轨道分为3段
    6. 构造 6 × 3 = 18 个区域
    7. 输出图像和结果表
    """
    # 创建输出文件夹（不存在则自动创建）
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 步骤1：读取卫星轨迹时间片文件
    print("1) 读取时间片文件...")
    step_files, all_steps = load_all_steps(INPUT_FOLDER)
    print(f"共读取 {len(step_files)} 个时间片")

    # 无输入文件，直接退出
    if not step_files:
        print("[ERROR] 未找到 step 文件")
        return

    # 步骤2：计算卫星轨道面法向量（基于相邻时间片位置叉乘）
    print("2) 估计每颗卫星的轨道面法向量...")
    sat_normals_raw = estimate_satellite_normals(all_steps, r_earth=R_EARTH)
    print(f"有法向量样本的卫星数: {len(sat_normals_raw)}")

    # 步骤3：对多组法向量求平均，得到稳定轨道法向量
    print("3) 计算每颗卫星的平均法向量...")
    sat_normals = average_satellite_normals(
        sat_normals_raw,
        min_samples=MIN_NORMAL_SAMPLES
    )
    print(f"可参与聚类的卫星数: {len(sat_normals)}")

    # 无可聚类卫星，退出程序
    if not sat_normals:
        print("[ERROR] 没有可用于聚类的卫星，请检查参数")
        return

    # 步骤4：选择聚类方法，将卫星分为72个原始轨道面
    print(f"4) 使用 {PROCESS_METHOD} 聚类为 72 个轨道面...")
    sat_to_raan = None

    # 方法1：Numpy向量夹角聚类
    if PROCESS_METHOD == "numpy":
        sat_to_plane = cluster_by_numpy(
            sat_normals,
            angle_threshold_deg=ANGLE_THRESHOLD_DEG
        )
        method_name = "Numpy"

    # 方法2：Sklearn DBSCAN密度聚类
    elif PROCESS_METHOD == "dbscan":
        sat_to_plane = cluster_by_sklearn(
            sat_normals,
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES
        )
        method_name = "Sklearn DBSCAN"

    # 方法3：RAAN分桶聚类（Starlink标准方法）
    elif PROCESS_METHOD == "raan":
        sat_to_plane, sat_to_raan = cluster_by_raan_bucket(
            sat_normals,
            n_planes=N_planes_by_raan
        )
        method_name = f"RAAN Bucket ({N_planes_by_raan} planes)"

    # 无效聚类方法，抛出异常
    else:
        raise ValueError("PROCESS_METHOD 必须是 'dbscan'、'numpy' 或 'raan'")

    # 验证分域参数合法性
    validate_region_params(N_planes_by_raan, N_BIG_PLANES, N_PHASES)

    # 打印聚类结果摘要
    print_plane_summary(method_name, sat_to_plane, sat_normals)

    # 选择指定时间片，用于后续分域和绘图
    plot_idx = max(0, min(PLOT_STEP_INDEX, len(all_steps) - 1))
    step_data = all_steps[plot_idx]
    step_title = f"{method_name} - {step_files[plot_idx]}"

    # ===== 核心：将72个原始轨道合并为6个大轨道 =====
    sat_to_big_plane = build_big_plane_groups(sat_to_plane, n_planes_total=N_planes_by_raan, n_big_planes=N_BIG_PLANES)

    # ===== 核心：将每个大轨道按相位分为3段 =====
    print(f"5) 使用 {PHASE_SPLIT_METHOD} 方式划分每个轨道的3个 phase...")

    # 执行轨道内相位分割，获取卫星角度和相位ID
    sat_to_metric, sat_to_phase = split_satellites_within_plane(
        step_data,
        sat_to_plane,
        method=PHASE_SPLIT_METHOD,
        sat_normals=sat_normals,
        n_phases=N_PHASES,
        r_earth=R_EARTH,
    )

    # 组合大轨道ID+相位ID，生成最终18个域的ID
    sat_to_domain = build_domain_id(
        sat_to_plane, 
        sat_to_phase, 
        n_planes_total=N_planes_by_raan, 
        n_big_planes=N_BIG_PLANES, 
        n_phases=N_PHASES)

    # 计算最终总域数：6×3=18
    n_domain = N_BIG_PLANES * N_PHASES

    # =========================
    # 可视化输出1：72个原始轨道单独图（可选）
    # =========================
    plane_total_folder = None
    if OUTPUT_TOTAL_PLANES:
        print(f"6) 输出{N_planes_by_raan}个轨道的单独图像...")
        plane_total_folder = os.path.join(OUTPUT_FOLDER, f"separate_planes_{N_planes_by_raan}_by_{PROCESS_METHOD}")
        plot_each_plane_separately(
            step_data,
            sat_to_plane,
            title_prefix=f"{N_planes_by_raan} Planes - {step_title}",
            output_folder=plane_total_folder,
            file_prefix="plane",
            label_prefix="Plane"
        )

    # =========================
    # 可视化输出2：6个大轨道单独图（可选）
    # =========================
    bigplane_folder = None
    if OUTPUT_BIG_PLANES:
        print(f"7) 输出{N_BIG_PLANES}个大轨道的单独图像...")
        bigplane_folder = os.path.join(OUTPUT_FOLDER, f"separate_big_planes_{N_BIG_PLANES}_by_{PROCESS_METHOD}")
        plot_each_plane_separately(
            step_data,
            sat_to_big_plane,
            title_prefix=f"{N_BIG_PLANES} Big Planes - {step_title}",
            output_folder=bigplane_folder,
            file_prefix="big_plane",
            label_prefix="BigPlane"
        )

    # =========================
    # 可视化输出3：18个最终域单独图（可选）
    # =========================
    domain_folder = None
    if OUTPUT_DOMAINS_SEPARATE:
        print(f"8) 输出{n_domain}个区域的单独图像...")
        domain_folder = os.path.join(OUTPUT_FOLDER, f"separate_domains_{n_domain}_by_{PROCESS_METHOD}_{PHASE_SPLIT_METHOD}")
        plot_each_plane_separately(
            step_data,
            sat_to_domain,
            title_prefix=f"{n_domain} Domains - {step_title}",
            output_folder=domain_folder,
            file_prefix="domain",
            label_prefix="Domain"
        )

    # =========================
    # 可视化输出4：18个域总览图（必输出）
    # =========================
    print("9) 输出18个区域总图...")
    domain_total_path = os.path.join(OUTPUT_FOLDER, f"domains_{n_domain}_total_{PROCESS_METHOD}_{PHASE_SPLIT_METHOD}.png")
    plot_one_step_by_plane(
        step_data,
        sat_to_domain,
        title=f"{n_domain} Domains Total - {step_title}",
        output_path=domain_total_path,
        label_prefix="Domain"
    )

    # =========================
    # 保存最终分域结果文本文件
    # =========================
    print("10) 保存结果文件...")
    result_txt_path = os.path.join(OUTPUT_FOLDER, f"domain_result_{n_domain}_{PROCESS_METHOD}_{PHASE_SPLIT_METHOD}.txt")
    
    # 根据分段方法设置度量名称
    if PHASE_SPLIT_METHOD == "longitude":
        metric_name = "longitude_deg"
    else:
        metric_name = "orbital_angle_deg"

    # 保存完整分域数据
    save_sat_to_domain(
        result_txt_path,
        sat_to_plane,
        sat_to_raan if sat_to_raan is not None else {},
        sat_to_big_plane,
        sat_to_metric,
        sat_to_phase,
        sat_to_domain,
        metric_name=metric_name
    )

    # 打印输出文件路径
    print("已输出:")
    print(f"  结果表: {result_txt_path}")
    print(f"  {n_domain}区域总图: {domain_total_path}")

    if plane_total_folder is not None:
        print(f"  {N_planes_by_raan}轨道图文件夹: {plane_total_folder}")
    if bigplane_folder is not None:
        print(f"  {N_BIG_PLANES}大轨道图文件夹: {bigplane_folder}")
    if domain_folder is not None:
        print(f"  {n_domain}区域单图文件夹: {domain_folder}")


# 程序入口：执行主函数
if __name__ == "__main__":
    main()
