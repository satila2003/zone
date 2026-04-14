import rasterio
from rasterio.windows import from_bounds
import math
import numpy as np

file_path = r'F:\Py_Project\cluster\landscan-global-2024.tif'


def get_max_population_in_radius(lat, lon, radius_km=3.0):
    """
    获取指定经纬度附近指定半径(默认3km)内的最大人口密度。
    使用了窗口读取(Window)技术，大幅优化内存和读取速度。
    """
    try:
        with rasterio.open(file_path) as src:
            nodata_val = src.nodata

            # 1. 将 3km 转换为经纬度跨度 (近似计算)
            # 纬度：1度大约是 111.32 km
            lat_deg_per_km = 1 / 111.32
            d_lat = radius_km * lat_deg_per_km

            # 经度：1度的物理距离随纬度变化，约为 111.32 * cos(纬度)
            # 避免在极点附近除以0，做一个安全限制
            safe_lat = max(min(lat, 89.9), -89.9)
            lon_deg_per_km = 1 / (111.32 * math.cos(math.radians(safe_lat)))
            d_lon = radius_km * lon_deg_per_km

            # 2. 计算 3km 半径的外接矩形框 (Bounding Box)
            left = lon - d_lon
            bottom = lat - d_lat
            right = lon + d_lon
            top = lat + d_lat

            # 3. 将地理边界转换为栅格像素窗口
            window = from_bounds(left, bottom, right, top, transform=src.transform)

            # 4. 【性能关键】仅读取该窗口范围内的数据，而不是全图
            # boundless=True 允许查询超出图像边缘的范围（自动填充nodata）
            data_window = src.read(1, window=window, boundless=True)

            # 5. 过滤掉 NoData (海洋或无数据区) 并寻找最大值
            # data_window 此时是一个二维的 numpy 矩阵 (例如 6x6 的矩阵)
            if nodata_val is not None:
                # 提取出所有不是 nodata 的有效像素值
                valid_data = data_window[data_window != nodata_val]
            else:
                valid_data = data_window

            # 6. 判断该区域内是否有有效人口数据
            if valid_data.size > 0:
                max_pop = np.max(valid_data)
                return f"  [结果] 坐标 ({lat}, {lon}) 附近 {radius_km}km 范围内的最大人口估值为: {max_pop}"
            else:
                return f"  [结果] 坐标 ({lat}, {lon}) 附近 {radius_km}km 范围内全是 NoData (海洋/无数据区)。"

    except FileNotFoundError:
        return f"  [错误] 文件 '{file_path}' 未找到。请检查路径。"
    except Exception as e:
        return f"  [未知错误] 发生意外: {e}"


# --- 开始诊断 ---
print("--- 诊断开始 (查询附近 3km 最大人口密度) ---")

# 1. 原始问题点 (位于海上)
print("\n1. 测试原始问题点 (菲律宾海附近):")
result1 = get_max_population_in_radius(9.79292, 126.97115)
print(result1)

# 2. 测试一个附近的陆地点 - 菲律宾的达沃市 (Davao City)
print("\n2. 测试附近的陆地点 (菲律宾达沃市):")
result2 = get_max_population_in_radius(7.07, 125.61)
print(result2)

# 3. 测试一个已知有人口的大城市 - 中国北京
print("\n3. 测试一个已知有人口的城市 (中国北京):")
result3 = get_max_population_in_radius(39.9, 116.4)
print(result3)

# 4. 测试一个明确的海洋点 - 太平洋中心
print("\n4. 测试一个明确的海洋点 (太平洋中心):")
result4 = get_max_population_in_radius(0, 170)
print(result4)

print("\n--- 诊断结束 ---")

