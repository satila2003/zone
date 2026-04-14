# import rasterio
#
# file_path = 'landscan-global-2024.tif'
#
# def get_population(lat, lon):
#     with rasterio.open(file_path) as src:
#         # 将经纬度转换为数据矩阵的行列号
#         row, col = src.index(lon, lat)
#         # 读取该像素的值
#         data = src.read(1)
#         population = data[row, col]
#         return population
#
# # 示例：获取某地的数值
# density1 = get_population(55.01373, 27.61275)
# density2 = get_population(39.9, 116.4)
# print(f"对应位置的人口估值为: {density1}")
# print(f"对应位置的人口估值为: {density2}")
#

import rasterio

file_path = 'landscan-global-2024.tif'


def get_population_robust(lat, lon):
    """
    一个更健壮的函数，用于获取人口数据并进行详细诊断。
    - 修复了作用域错误。
    - 使用 rasterio.index 的返回值来判断坐标有效性，更可靠。
    """
    try:
        with rasterio.open(file_path) as src:
            # 1. 获取文件的NoData值
            nodata_val = src.nodata
            print(f"  [文件信息] NoData值为: {nodata_val}")

            # 2. 将经纬度转换为图像的行列号
            # 如果坐标在图像范围之外，row或col会是None
            row, col = src.index(lon, lat)

            # 3. 检查行列号是否有效
            if row is None or col is None:
                return f"  [结果] 坐标 ({lat}, {lon}) 超出文件范围。"

            # 4. 读取该像素的值
            data = src.read(1)
            population = data[row, col]

            # 5. 检查读取到的值是否为NoData
            if population == nodata_val:
                return f"  [结果] 坐标 ({lat}, {lon}) 的值是 NoData。这通常意味着该点位于海洋或数据缺失区。"
            else:
                return f"  [结果] 坐标 ({lat}, {lon}) 的人口估值为: {population}"
    except FileNotFoundError:
        return f"  [错误] 文件 '{file_path}' 未找到。请检查路径。"
    except Exception as e:
        return f"  [未知错误] 发生意外: {e}"


# --- 开始诊断 ---
print("--- 诊断开始 ---")

# 1. 你的原始问题点 (位于海上)
print("\n1. 测试原始问题点 (菲律宾海):")
result1 = get_population_robust(9.79292, 126.97115)
print(result1)

# 2. 测试一个附近的陆地点 - 菲律宾的达沃市 (Davao City)
print("\n2. 测试附近的陆地点 (菲律宾达沃市):")
result2 = get_population_robust(7.07, 125.61)  # 达沃市的大致坐标
print(result2)

# 3. 测试一个已知有人口的大城市 - 中国北京
print("\n3. 测试一个已知有人口的城市 (中国北京):")
result3 = get_population_robust(39.9, 116.4)
print(result3)

# 4. 测试一个明确的海洋点 - 太平洋中心
print("\n4. 测试一个明确的海洋点 (太平洋中心):")
result4 = get_population_robust(0, 170)  # 我们用 170 度来避免 180 度的边界问题
print(result4)

print("\n--- 诊断结束 ---")