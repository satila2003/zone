import pickle
import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def generate_labels_min_mlu(input_pkl, output_label_pkl, capacity, k=8, smoothing=0.1):
    """
    以「最小化最大链路利用率 (MLU)」为目标生成分流比标签，并应用标签平滑处理。
    【核心修复】：支持宏观需求(超节点)在微观路径(真实卫星)上的路由映射。
    """
    print(f"正在从 {input_pkl} 加载数据集...")
    with open(input_pkl, 'rb') as f:
        dataset = pickle.load(f)

    final_label_list = []
    start_time = time.time()

    for ts_idx, data in enumerate(dataset):
        print(f"\n正在处理时间片 {ts_idx + 1}/{len(dataset)}...")
        ts_start_time = time.time()

        tm = data['tm']
        path_dict = data['path']

        # =====================================================================
        # 核心修复：不再使用宏观的 data['graph']，而是直接从微观路径中提取真实的物理链路
        # =====================================================================
        unique_links = set()
        for sd_key, paths in path_dict.items():
            for path in paths[:k]:  # 遍历每条候选路径
                if len(path) < 2:
                    continue
                # 遍历路径中的每一跳
                for step in range(len(path) - 1):
                    # 确保是整型，防止数据类型不匹配
                    u, v = int(path[step]), int(path[step + 1])
                    unique_links.add((u, v))

        unique_links = list(unique_links)
        link_to_idx = {link: i for i, link in enumerate(unique_links)}
        num_links = len(unique_links)
        print(f"  时间片 {ts_idx}: 提取到 {num_links} 条实际使用的微观星间物理链路(ISL)。")
        # =====================================================================

        # 固定需求对的顺序
        sd_keys = list(tm.keys())
        num_demands = len(sd_keys)

        # --- Gurobi 优化模型（最小化 MLU 版本）---
        m = gp.Model(f"Min_MLU_Optimization_ts_{ts_idx}")
        m.setParam('OutputFlag', 0)  # 关闭求解器输出
        m.setParam('Threads', 4)  # 使用4个线程加速
        m.setParam('OptimalityTol', 1e-6)  # 优化精度设置

        # 定义变量
        # 1. U: 全局最大链路利用率 (MLU)，0 ≤ U
        U = m.addVar(lb=0.0, name="MLU")

        # 2. x[i, j]: 第i个OD对在第j条路径上的分流比（0≤x[i,j]≤1）
        x = m.addVars(num_demands, k, lb=0.0, ub=1.0, name="path_flow_ratio")

        # 定义优化目标：最小化全局最大链路利用率 U
        m.setObjective(U, GRB.MINIMIZE)

        # 约束 1：每个OD对的分流比之和 = 1 (所有需求必须被 100% 满足)
        for i in range(num_demands):
            m.addConstr(x.sum(i, '*') == 1.0, name=f"flow_conservation_{i}")

        # 约束 2：链路容量约束（每条物理链路的总负载 ≤ U * 链路容量）
        # 初始化一个线性表达式列表，用于累加每条微观链路上的总负载
        link_load = [gp.LinExpr() for _ in range(num_links)]

        for i, sd_key in enumerate(sd_keys):
            demand_value = tm[sd_key]
            paths = path_dict.get(sd_key, [])

            if len(paths) < k:
                # 这种情况下由于我们要求分流比和为1，不足的路径Gurobi会自动分配0
                # print(f"  提示: 宏观需求对 {sd_key} 只有 {len(paths)} 条微观路径。")
                pass

            for j, path in enumerate(paths[:k]):
                if len(path) < 2:
                    continue
                # 将该条路径分配到的流量，累加到经过的每一条真实卫星物理链路上
                for step in range(len(path) - 1):
                    u, v = int(path[step]), int(path[step + 1])
                    link_tuple = (u, v)
                    if link_tuple in link_to_idx:
                        l_idx = link_to_idx[link_tuple]
                        # 该微观链路上的流量 = 分流比 × 宏观需求值
                        link_load[l_idx] += x[i, j] * demand_value

        # 添加 MLU 约束：每条微观链路的总负载 ≤ U * 微观链路容量
        for l_idx in range(num_links):
            m.addConstr(link_load[l_idx] <= U * capacity, name=f"mlu_constraint_{l_idx}")

        # 求解模型
        m.optimize()

        # 处理求解结果（生成分流比标签）
        if m.status == GRB.OPTIMAL:
            print(f"  时间片 {ts_idx}: 求解成功，最优 MLU = {U.X:.4f} (真实最大负载: {U.X * capacity:.2f} Mbps)")

            ratios_np = np.zeros((num_demands, k), dtype=np.float64)
            for i in range(num_demands):
                for j in range(k):
                    ratios_np[i, j] = x[i, j].X

            # 转换为 torch.tensor
            current_ts_tensor = torch.from_numpy(ratios_np).to(torch.float32)
            if smoothing > 0:
                current_ts_tensor = current_ts_tensor * (1 - smoothing) + (smoothing / k)

        else:
            print(f"  警告：时间片 {ts_idx} 未能找到最优解 (状态码: {m.status})。填充均匀分布。")
            current_ts_tensor = torch.full((num_demands, k), 1.0 / k, dtype=torch.float32)

        final_label_list.append(current_ts_tensor)
        print(f"  时间片 {ts_idx} 处理耗时: {time.time() - ts_start_time:.2f} 秒。")

    # 2. 保存最终的 pkl 文件
    with open(output_label_pkl, 'wb') as f:
        pickle.dump(final_label_list, f)

    print(f"\n所有时间片处理完成！总耗时: {time.time() - start_time:.2f} 秒。")
    print(f"结果已保存至: {output_label_pkl}")

    return final_label_list


if __name__ == "__main__":
    input_data_pkl = r"F:\Py_Project\always\cluster\zone\outputs\tms\starlink550_cluster.pkl"
    output_label_pkl = r"F:\Py_Project\always\cluster\zone\outputs\tms\Labels_starlink550_cluster.pkl"

    # capacity 此时代表的是微观真实星间链路(ISL)的固定带宽容量
    # labels = generate_labels_min_mlu(input_data_pkl, output_label_pkl, capacity=250, smoothing=0)
    labels = generate_labels_min_mlu(input_data_pkl, output_label_pkl, capacity=250, smoothing=0.1)
