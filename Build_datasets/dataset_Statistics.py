import pickle
import numpy as np

# ----------------------
# 配置常量（SD对总数，用户明确每个时间片4290个）
# ----------------------
SD_PAIR_TOTAL = 4290  # 固定分母，用于计算百分占比

# ----------------------
# 1. 加载 pkl 文件
# ----------------------
pkl_path = "F:\Py_Project/always\Build_datasets\Gurobi_size-66_mode-NA_intensity-75_volume-5000_solutions.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# ----------------------
# 2. 初始化结果容器
# ----------------------
results = []

# ----------------------
# 3. 遍历每个时间片，统计各路径的 0.92 占比数量 + 百分占比
# ----------------------
for time_slice_idx, tensor in enumerate(data):
    # 将 Tensor 转为 numpy 数组（兼容 GPU Tensor）
    arr = tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()

    # 统计每条路径（共 5 条）中分流比≈0.92 的数量 + 百分占比
    path_counts = {}
    for path_idx in range(5):  # 0-4 对应第 1-5 条路径
        # 浮点数比较用 np.isclose，容差设为 1e-6
        count = np.isclose(arr[:, path_idx], 0.92, atol=1e-6).sum()
        # 计算百分占比（保留整数，基于4290个SD对）
        percentage = (count / SD_PAIR_TOTAL) * 100
        # 组合成 "数值(百分比%)" 格式
        path_counts[f"path_{path_idx + 1}"] = f"{count}({percentage:.0f}%)"

    # 记录当前时间片的统计结果
    results.append({
        "time_slice": time_slice_idx + 1,  # 时间片从 1 开始编号
        **path_counts
    })

# ----------------------
# 4. 整理结果并写入 TXT 文件（调整格式对齐）
# ----------------------
output_txt = "path_counts.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    # 第一步：写入 TXT 表头（调整字段宽度为<15，适配数值+百分占比格式）
    header = f"{'时间片':<10} {'path_1':<15} {'path_2':<15} {'path_3':<15} {'path_4':<15} {'path_5':<15}\n"
    f.write(header)
    f.write("-" * 90 + "\n")  # 加长分隔线，适配新格式

    # 第二步：逐行写入每个时间片的统计结果
    for result in results:
        line = (
            f"{result['time_slice']:<10} "
            f"{result['path_1']:<15} "
            f"{result['path_2']:<15} "
            f"{result['path_3']:<15} "
            f"{result['path_4']:<15} "
            f"{result['path_5']:<15}\n"
        )
        f.write(line)

# 打印前 5 行结果预览（控制台）
print("统计结果预览（前5个时间片）：")
print("-" * 90)
print(f"{'时间片':<10} {'path_1':<15} {'path_2':<15} {'path_3':<15} {'path_4':<15} {'path_5':<15}")
print("-" * 90)
for result in results[:5]:
    print(
        f"{result['time_slice']:<10} "
        f"{result['path_1']:<15} "
        f"{result['path_2']:<15} "
        f"{result['path_3']:<15} "
        f"{result['path_4']:<15} "
        f"{result['path_5']:<15}"
    )

print(f"\n完整结果已保存到 {output_txt}")