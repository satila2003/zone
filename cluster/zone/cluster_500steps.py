import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
# 修复缺失导入：DBSCAN聚类需要
from sklearn.cluster import DBSCAN

# =========================
# 配置区：全局参数定义，控制卫星分域全流程
# =========================
# 输入数据文件夹：Starlink550卫星轨迹step文件
INPUT_FOLDER = r"F:\Py_Project\always\cluster\zone\inputs\starlink550_data\data_1100"
# 输出文件夹：保存分域结果、图片、文本
OUTPUT_FOLDER = r"F:\Py_Project\always\cluster\zone\outputs_all_steps"
# 地球半径（单位：km），用于轨道计算
R_EARTH = 6371.0
R_EARTH_DEFAULT = 6371.0

# 计算卫星轨道法向量的最小有效样本数
MIN_NORMAL_SAMPLES = 5

# 聚类方法选择：numpy向量聚类/sklearn密度聚类/RAAN分桶聚类
PROCESS_METHOD = "raan"  # "numpy", "sklearn", "raan"
# 轨道内分段方法：按经度/轨道真角划分
PHASE_SPLIT_METHOD = "orbital_angle"  # "longitude" 或 "orbital_angle"
# RAAN分桶的总轨道面数（Starlink550标准：72个轨道面）
N_planes_by_raan = 72
# 合并后的大轨道数量（72/12=6）
N_BIG_PLANES = 6
# 每个大轨道内分段数，最终总域数=大轨道数×分段数
N_PHASES = 3

# 图像输出开关：批量处理时，关闭单独图（避免生成500*18张图），只保留总图
OUTPUT_TOTAL_PLANES = False
OUTPUT_BIG_PLANES = False
OUTPUT_DOMAINS_SEPARATE = False
OUTPUT_DOMAIN_TOTAL = True  # 只保留每个时间片的18域总图

# numpy向量聚类参数：法向量夹角阈值（度）
ANGLE_THRESHOLD_DEG = 3.0
# sklearn DBSCAN密度聚类参数
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 5

# =========================
# 修复原代码缺失函数：按经度分段
# =========================
def split_satellites_within_plane_by_longitude(step_data, sat_to_plane, n_phases):
    sat_to_metric = {}
    sat_to_phase = {}
    plane_sats = defaultdict(list)
    for sat_id, pid in sat_to_plane.items():
        if sat_id in step_data:
            plane_sats[pid].append(sat_id)

    for pid, sats_in_plane in plane_sats.items():
        sats_in_plane.sort(key=lambda x: step_data[x]["lon"])
        total = len(sats_in_plane)
        if total == 0:
            continue
        phase_size = total / n_phases
        for idx, sat_id in enumerate(sats_in_plane):
            phase = int(idx // phase_size)
            if phase >= n_phases:
                phase = n_phases - 1
            sat_to_metric[sat_id] = step_data[sat_id]["lon"]
            sat_to_phase[sat_id] = phase
    return sat_to_metric, sat_to_phase

# =========================
# 轨道面法向量估计
# =========================
def estimate_satellite_normals(all_steps, r_earth=6371.0):
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
            if n[2] < 0:
                n = -n
            sat_normals_raw[sat_id].append(n)
    return sat_normals_raw

def split_satellites_within_plane_by_orbital_angle(step_data, sat_to_plane, sat_normals, n_phases, r_earth=6371.0):
    sat_to_metric = {}
    sat_to_phase = {}
    phase_width = 360.0 / n_phases
    for sat_id, plane_id in sat_to_plane.items():
        if sat_id not in step_data or sat_id not in sat_normals:
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
# 聚类方法
# =========================
def cluster_by_raan_bucket(sat_normals, n_planes):
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

def cluster_by_numpy(sat_normals, angle_threshold_deg=3.0):
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

def cluster_by_sklearn(sat_normals, eps=0.03, min_samples=5):
    sat_ids = sorted(sat_normals.keys())
    features = np.array([sat_normals[sid] for sid in sat_ids])
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(features)
    sat_to_plane = {}
    for sid, label in zip(sat_ids, labels):
        sat_to_plane[sid] = int(label)
    return sat_to_plane

# =========================
# 文件读取
# =========================
def read_one_step_file(file_path):
    sats = {}
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ID, Name, Lat(deg), Lon(deg), Alt(km)"):
            start_idx = i
            break
    if start_idx is None:
        print(f"[WARN] 未找到表头，跳过文件: {file_path}")
        return sats
    for line in lines[start_idx + 1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            sat_id = int(parts[0])
            sat_name = parts[1]
            lat = float(parts[2])
            lon = float(parts[3])
            alt = float(parts[4])
        except ValueError:
            continue
        sats[sat_id] = {"name": sat_name, "lat": lat, "lon": lon, "alt": alt}
    return sats

def load_all_steps(folder):
    files = sorted(f for f in os.listdir(folder) if f.startswith("step_") and f.endswith(".txt"))
    all_steps = []
    for fname in files:
        path = os.path.join(folder, fname)
        sats = read_one_step_file(path)
        all_steps.append(sats)
    return files, all_steps

# =========================
# 分域核心逻辑
# =========================
def validate_region_params(n_planes_total, n_big_planes, n_phases):
    if n_big_planes <= 0:
        raise ValueError("n_big_planes 必须大于 0")
    if n_phases <= 0:
        raise ValueError("n_phases 必须大于 0")
    if n_planes_total % n_big_planes != 0:
        raise ValueError(f"总轨道数 {n_planes_total} 不能被大轨道数 {n_big_planes} 整除")

def build_big_plane_groups(sat_to_plane, n_planes_total, n_big_planes):
    planes_per_big_plane = n_planes_total // n_big_planes
    sat_to_big_plane = {}
    for sat_id, plane_id in sat_to_plane.items():
        sat_to_big_plane[sat_id] = plane_id // planes_per_big_plane
    return sat_to_big_plane

def build_domain_id(sat_to_plane, sat_to_phase, n_planes_total, n_big_planes, n_phases):
    planes_per_big_plane = n_planes_total // n_big_planes
    sat_to_domain = {}
    for sat_id, plane_id in sat_to_plane.items():
        if sat_id not in sat_to_phase:
            continue
        big_plane_id = plane_id // planes_per_big_plane
        phase_id = sat_to_phase[sat_id]
        domain_id = big_plane_id * n_phases + phase_id
        sat_to_domain[sat_id] = domain_id
    return sat_to_domain

def split_satellites_within_plane(step_data, sat_to_plane, n_phases, method="longitude", sat_normals=None, r_earth=6371.0):
    if method == "longitude":
        return split_satellites_within_plane_by_longitude(step_data, sat_to_plane, n_phases)
    elif method == "orbital_angle":
        if sat_normals is None:
            raise ValueError("使用 orbital_angle 分段时，必须提供 sat_normals")
        return split_satellites_within_plane_by_orbital_angle(step_data, sat_to_plane, sat_normals, n_phases, r_earth=r_earth)
    else:
        raise ValueError("method 必须是 'longitude' 或 'orbital_angle'")

# =========================
# 几何工具函数
# =========================
def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return None
    return v / norm

def lla_to_xyz(lat_deg, lon_deg, alt_km, r_earth=R_EARTH_DEFAULT):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = r_earth + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=float)

def angle_deg(a, b):
    a = normalize(a)
    b = normalize(b)
    if a is None or b is None:
        return 180.0
    cos_val = np.clip(np.dot(a, b), -1.0, 1.0)
    return math.degrees(math.acos(cos_val))

def normal_to_raan(n):
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
    n = normalize(n)
    if n is None:
        return None, None
    i_deg = math.degrees(math.acos(np.clip(n[2], -1.0, 1.0)))
    raan_deg = normal_to_raan(n)
    return i_deg, raan_deg

def normal_to_node_vector(n):
    n = normalize(n)
    if n is None:
        return None
    node = np.array([-n[1], n[0], 0.0], dtype=float)
    return normalize(node)

def argument_of_latitude_from_r_n(r, n):
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
# 结果保存与绘图
# =========================
def save_sat_to_domain(filename, sat_to_plane, sat_to_raan, sat_to_big_plane, sat_to_metric, sat_to_phase, sat_to_domain, metric_name="orbital_angle_deg"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"sat_id,plane_id,raan_deg,big_plane_id,{metric_name},phase_id,domain_id\n")
        for sat_id in sorted(sat_to_domain.keys()):
            plane_id = sat_to_plane.get(sat_id, "")
            raan = sat_to_raan.get(sat_id, "")
            big_plane_id = sat_to_big_plane.get(sat_id, "")
            metric_value = sat_to_metric.get(sat_id, "")
            phase_id = sat_to_phase.get(sat_id, "")
            domain_id = sat_to_domain.get(sat_id, "")
            f.write(f"{sat_id},{plane_id},{raan},{big_plane_id},{metric_value},{phase_id},{domain_id}\n")

def build_plane_members(sat_to_plane):
    plane_members = defaultdict(set)
    for sat_id, plane_id in sat_to_plane.items():
        plane_members[plane_id].add(sat_id)
    return dict(plane_members)

def print_plane_summary(method_name, sat_to_plane, sat_normals):
    print()
    print(f"===== {method_name} 聚类结果 =====")
    plane_members = build_plane_members(sat_to_plane)
    print(f"轨道面数量: {len(plane_members)}")
    for pid in sorted(plane_members.keys()):
        members = plane_members[pid]
        normals = [sat_normals[sid] for sid in members if sid in sat_normals]
        if normals:
            center = np.mean(normals, axis=0)
            center = normalize(center)
            if center is not None and center[2] < 0:
                center = -center
            i_deg, raan_deg = normal_to_inclination_raan(center)
            print(f"Plane {pid:03d}: {len(members):4d} sats, Inclination ~ {i_deg:6.2f} deg, RAAN ~ {raan_deg:7.2f} deg")
        else:
            print(f"Plane {pid:03d}: {len(members):4d} sats")

def plot_one_step_by_plane(step_data, sat_to_label, title, output_path=None, label_prefix="Plane"):
    plt.figure(figsize=(16, 8))
    grouped = defaultdict(lambda: {"lon": [], "lat": []})
    unclassified_lon = []
    unclassified_lat = []
    for sat_id, sat_info in step_data.items():
        lon = sat_info["lon"]
        lat = sat_info["lat"]
        if sat_id in sat_to_label:
            pid = sat_to_label[sat_id]
            grouped[pid]["lon"].append(lon)
            grouped[pid]["lat"].append(lat)
        else:
            unclassified_lon.append(lon)
            unclassified_lat.append(lat)
    cmap = plt.get_cmap('hsv')
    num_groups = len(grouped)
    for idx, pid in enumerate(sorted(grouped.keys())):
        color = cmap(idx / max(1, num_groups))
        plt.scatter(grouped[pid]["lon"], grouped[pid]["lat"], s=15, color=color, alpha=0.8, label=f"{label_prefix} {pid}")
    if unclassified_lon:
        plt.scatter(unclassified_lon, unclassified_lat, s=10, color='grey', alpha=0.5, marker="x", label="Unclassified")
    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(title)
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc='upper left', ncol=4, fontsize=8, markerscale=1.5)
    plt.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_each_plane_separately(step_data, sat_to_label, title_prefix, output_folder=None, file_prefix="plane", label_prefix="Plane"):
    grouped = defaultdict(lambda: {"lon": [], "lat": []})
    for sat_id, sat_info in step_data.items():
        if sat_id not in sat_to_label:
            continue
        pid = sat_to_label[sat_id]
        grouped[pid]["lon"].append(sat_info["lon"])
        grouped[pid]["lat"].append(sat_info["lat"])
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    for pid in sorted(grouped.keys()):
        plt.figure(figsize=(10, 6))
        plt.scatter(grouped[pid]["lon"], grouped[pid]["lat"], s=12, alpha=0.8, label=f"{label_prefix} {pid}")
        plt.xlabel("Longitude (deg)")
        plt.ylabel("Latitude (deg)")
        plt.title(f"{title_prefix} - {label_prefix} {pid}")
        plt.xlim(-180, 180)
        plt.ylim(-90, 90)
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        if output_folder is not None:
            output_path = os.path.join(output_folder, f"{file_prefix}_{pid:03d}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

# =========================
# 【核心修改】主程序：遍历所有时间片分域
# =========================
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("1) 读取所有时间片文件...")
    step_files, all_steps = load_all_steps(INPUT_FOLDER)
    total_steps = len(all_steps)
    print(f"共读取 {total_steps} 个时间片，开始批量分域！")

    if not step_files:
        print("[ERROR] 未找到 step 文件")
        return

    # ==============================================
    # 【全局固定计算：只执行1次】轨道面/聚类/大轨道
    # ==============================================
    print("2) 全局计算：卫星轨道面法向量...")
    sat_normals_raw = estimate_satellite_normals(all_steps, r_earth=R_EARTH)
    sat_normals = average_satellite_normals(sat_normals_raw, min_samples=MIN_NORMAL_SAMPLES)
    if not sat_normals:
        print("[ERROR] 没有可用于聚类的卫星")
        return

    print("3) 全局计算：轨道面聚类...")
    sat_to_raan = None
    if PROCESS_METHOD == "numpy":
        sat_to_plane = cluster_by_numpy(sat_normals, angle_threshold_deg=ANGLE_THRESHOLD_DEG)
        method_name = "Numpy"
    elif PROCESS_METHOD == "dbscan":
        sat_to_plane = cluster_by_sklearn(sat_normals, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        method_name = "Sklearn DBSCAN"
    elif PROCESS_METHOD == "raan":
        sat_to_plane, sat_to_raan = cluster_by_raan_bucket(sat_normals, n_planes=N_planes_by_raan)
        method_name = f"RAAN Bucket ({N_planes_by_raan} planes)"
    else:
        raise ValueError("PROCESS_METHOD 错误")

    validate_region_params(N_planes_by_raan, N_BIG_PLANES, N_PHASES)
    print_plane_summary(method_name, sat_to_plane, sat_normals)
    sat_to_big_plane = build_big_plane_groups(sat_to_plane, N_planes_by_raan, N_BIG_PLANES)
    n_domain = N_BIG_PLANES * N_PHASES

    # ==============================================
    # 【循环遍历：所有500个时间片】动态分域
    # ==============================================
    print(f"\n4) 开始遍历 {total_steps} 个时间片，逐帧分域...")
    for plot_idx in range(total_steps):
        step_data = all_steps[plot_idx]
        step_name = step_files[plot_idx]
        current_step = plot_idx + 1
        print(f"\n--- 处理第 {current_step}/{total_steps} 个时间片：{step_name} ---")

        # 1. 每个时间片独立计算相位+域ID
        sat_to_metric, sat_to_phase = split_satellites_within_plane(
            step_data, sat_to_plane, N_PHASES,
            method=PHASE_SPLIT_METHOD, sat_normals=sat_normals, r_earth=R_EARTH
        )
        sat_to_domain = build_domain_id(sat_to_plane, sat_to_phase, N_planes_by_raan, N_BIG_PLANES, N_PHASES)

        # 2. 保存当前时间片的分域结果文件
        result_txt_path = os.path.join(OUTPUT_FOLDER, f"step_{plot_idx:03d}_result_{n_domain}_domains.txt")
        metric_name = "longitude_deg" if PHASE_SPLIT_METHOD == "longitude" else "orbital_angle_deg"
        save_sat_to_domain(
            result_txt_path, sat_to_plane, sat_to_raan or {}, sat_to_big_plane,
            sat_to_metric, sat_to_phase, sat_to_domain, metric_name
        )

        # 3. 绘制当前时间片的18域总图
        if OUTPUT_DOMAIN_TOTAL:
            domain_total_path = os.path.join(OUTPUT_FOLDER, f"step_{plot_idx:03d}_domains_total.png")
            plot_one_step_by_plane(
                step_data, sat_to_domain,
                title=f"Step {plot_idx} - {n_domain} Domains",
                output_path=domain_total_path, label_prefix="Domain"
            )

    print(f"\n✅ 全部 {total_steps} 个时间片分域完成！所有结果保存在：{OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
