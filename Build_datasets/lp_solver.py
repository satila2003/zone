import pickle
import torch
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time


def generate_labels_as_tensors(input_pkl, output_label_pkl, capacity, k=5, smoothing=0.1):
    """
    以「最大化吞吐量」为目标生成分流比标签，并应用标签平滑处理。
    核心变更：
    1. 优化目标从「最小化MLU」改为「最大化总满足流量（吞吐量）」。
    2. 新增需求满足比例变量d[i]，不再强制分流比和为1（允许需求部分满足）。
    3. 约束调整为「链路负载≤固定容量」，移除MLU相关依赖。
    """
    # 1. 加载之前构建的拓扑和路径数据集
    print(f"正在从 {input_pkl} 加载数据集...")
    with open(input_pkl, 'rb') as f:
        dataset = pickle.load(f)

    final_label_list = []
    start_time = time.time()

    for ts_idx, data in enumerate(dataset):
        print(f"\n正在处理时间片 {ts_idx + 1}/{len(dataset)}...")
        ts_start_time = time.time()

        edges = data['graph']
        tm = data['tm']
        path_dict = data['path']

        # 构建有向链路索引 (直接使用输入的双向边)
        link_list = []
        for e in edges:
            # e 已经是 [u, v] 格式，这里转为元组以便作为字典的键
            link_list.append(tuple(e))
        # 使用 list(set()) 来去重，确保每条有向边只出现一次
        unique_links = list(set(link_list))
        link_to_idx = {link: i for i, link in enumerate(unique_links)}
        num_links = len(unique_links)
        print(f"  时间片 {ts_idx}: 共有 {num_links} 条唯一的有向链路。")

        # 固定需求对的顺序
        sd_keys = list(tm.keys())
        num_demands = len(sd_keys)
        # print(f"  时间片 {ts_idx}: 共有 {num_demands} 个OD需求对。")

        # --- Gurobi 优化模型（最大化吞吐量版本）---
        m = gp.Model(f"Max_Throughput_Optimization_ts_{ts_idx}")
        m.setParam('OutputFlag', 0)  # 关闭求解器输出
        m.setParam('Threads', 4)  # 使用4个线程加速
        m.setParam('OptimalityTol', 1e-6)  # 优化精度设置

        # 定义变量
        # 1. d[i]: 第i个OD对的需求满足比例（0≤d[i]≤1），表示该OD对被成功承载的流量占总需求的比例
        d = m.addVars(num_demands, lb=0.0, ub=1.0, name="demand_satisfaction_ratio")
        # 2. x[i, j]: 第i个OD对在第j条路径上的分流比（0≤x[i,j]≤1）
        # 注：分流比之和等于d[i]（而非1），即仅对「被满足的流量部分」进行路径分配
        x = m.addVars(num_demands, k, lb=0.0, ub=1.0, name="path_flow_ratio")

        # 定义优化目标：最大化总吞吐量（所有OD对被满足的流量总和）
        total_throughput = gp.LinExpr()
        for i, sd_key in enumerate(sd_keys):
            demand_value = tm[sd_key]
            # 单个OD对的满足流量 = 需求值 × 满足比例
            total_throughput += d[i] * demand_value
        # Gurobi默认是「最小化」，因此最大化总吞吐量等价于最小化「负的总吞吐量」
        m.setObjective(-total_throughput, GRB.MINIMIZE)

        # 约束 1：每个OD对的分流比之和 = 其需求满足比例（仅分配被满足的流量）
        m.addConstrs((x.sum(i, '*') == d[i] for i in range(num_demands)), name="flow_conservation_throughput")

        # 约束 2：链路容量约束（每条链路的总负载 ≤ 固定链路容量）
        # 初始化一个线性表达式列表，用于累加每条链路上的总负载
        link_load = [gp.LinExpr() for _ in range(num_links)]
        for i, sd_key in enumerate(sd_keys):
            demand_value = tm[sd_key]
            paths = path_dict.get(sd_key, [])

            # 如果某个OD对没有找到路径，则跳过
            if len(paths) < k:
                print(f"  警告: OD对 {sd_key} 只有 {len(paths)} 条路径，不足 {k} 条，已跳过。")
                continue

            for j, path in enumerate(paths[:k]):  # 确保只使用前k条路径
                for step in range(len(path) - 1):
                    u, v = path[step], path[step + 1]
                    link_tuple = (u, v)
                    if link_tuple in link_to_idx:
                        l_idx = link_to_idx[link_tuple]
                        # 该路径上的流量 = 分流比 × 需求值（仅承载被满足的部分）
                        link_load[l_idx] += x[i, j] * demand_value

        # 添加容量约束：每条链路的总负载 ≤ 链路容量
        for l_idx in range(num_links):
            m.addConstr(link_load[l_idx] <= capacity, name=f"fixed_cap_constraint_{l_idx}")

        # 求解模型
        m.optimize()

        # 处理求解结果（生成分流比标签）
        if m.status == GRB.OPTIMAL:
            total_satisfied_flow = -m.ObjVal  # 还原最大化的总吞吐量（消除目标函数的负号）
            print(f"  时间片 {ts_idx}: 求解成功，最优总吞吐量 = {total_satisfied_flow:.2f}")

            ratios_np = np.zeros((num_demands, k), dtype=np.float64)
            for i in range(num_demands):
                demand_satisfaction = d[i].X
                # 处理极端情况：若需求满足比例为0，填充均匀分流比（避免除零错误）
                if demand_satisfaction < 1e-8:
                    ratios_np[i, :] = 1.0 / k
                else:
                    # 分流比归一化：将「绝对分流比」转为「相对分流比」（和为1，符合模型训练要求）
                    for j in range(k):
                        ratios_np[i, j] = x[i, j].X / demand_satisfaction

            # 转换为 torch.tensor
            current_ts_tensor = torch.from_numpy(ratios_np).to(torch.float32)  # 使用 float32 节省空间
            if smoothing > 0:  # 保留标签平滑功能，不变
                current_ts_tensor = current_ts_tensor * (1 - smoothing) + (smoothing / k)
        else:
            print(f"  警告：时间片 {ts_idx} 未能找到最优解 (状态码: {m.status})，填充均匀分布。")
            current_ts_tensor = torch.full((num_demands, k), 1.0 / k, dtype=torch.float32)

        final_label_list.append(current_ts_tensor)
        print(f"  时间片 {ts_idx} 处理耗时: {time.time() - ts_start_time:.2f} 秒。")

    # 2. 保存最终的 pkl 文件
    with open(output_label_pkl, 'wb') as f:
        pickle.dump(final_label_list, f)

    print(f"\n所有时间片处理完成！总耗时: {time.time() - start_time:.2f} 秒。")
    print(f"吞吐量最大化标签已保存至: {output_label_pkl}")
    print(f"单个 Tensor 维度: {final_label_list[0].shape}")

    return final_label_list


if __name__ == "__main__":
    input_data_pkl = "Iridium_DataSetForAgent_75_60480.pkl"
    output_label_pkl = "Labels_Iridium_Max_Throughput.pkl"
    labels = generate_labels_as_tensors(input_data_pkl, output_label_pkl, capacity=2500, smoothing=0.1)

# # generate_labels.py
#
# import pickle
# import torch
# import gurobipy as gp
# from gurobipy import GRB
# import numpy as np
# import time
#
#
# def generate_labels_as_tensors(input_pkl, output_label_pkl, capacity, k=5, smoothing=0.1):
#     """
#     【已修复】生成分流比标签，并应用标签平滑处理。
#     核心修复：
#     1. capacity 从 6.25 修正为 25.0。
#     2. link_to_idx 的构建逻辑更加健壮，直接使用输入的双向边。
#     """
#     # 1. 加载之前构建的拓扑和路径数据集
#     print(f"正在从 {input_pkl} 加载数据集...")
#     with open(input_pkl, 'rb') as f:
#         dataset = pickle.load(f)
#
#     final_label_list = []
#     start_time = time.time()
#
#     for ts_idx, data in enumerate(dataset):
#         print(f"\n正在处理时间片 {ts_idx + 1}/{len(dataset)}...")
#         ts_start_time = time.time()
#
#         edges = data['graph']
#         tm = data['tm']
#         path_dict = data['path']
#
#         # 构建有向链路索引 (直接使用输入的双向边)
#         link_list = []
#         for e in edges:
#             # e 已经是 [u, v] 格式，这里转为元组以便作为字典的键
#             link_list.append(tuple(e))
#         # 使用 list(set()) 来去重，确保每条有向边只出现一次
#         unique_links = list(set(link_list))
#         link_to_idx = {link: i for i, link in enumerate(unique_links)}
#         num_links = len(unique_links)
#         print(f"  时间片 {ts_idx}: 共有 {num_links} 条唯一的有向链路。")
#
#         # 固定需求对的顺序
#         sd_keys = list(tm.keys())
#         num_demands = len(sd_keys)
#         print(f"  时间片 {ts_idx}: 共有 {num_demands} 个OD需求对。")
#
#         # --- Gurobi 优化模型 ---
#         m = gp.Model(f"MLU_Optimization_ts_{ts_idx}")
#         m.setParam('OutputFlag', 0)  # 关闭求解器输出
#         m.setParam('Threads', 4)  # 使用4个线程加速
#
#         # 定义变量
#         # mlu: 我们要最小化的最大链路利用率
#         mlu = m.addVar(lb=0.0, obj=1.0, name="MLU")
#         # x[i, j]: 第i个OD对在第j条路径上的分流比
#         x = m.addVars(num_demands, k, lb=0.0, ub=1.0, name="x")
#
#         # 约束 1：分流比之和等于 1
#         m.addConstrs((x.sum(i, '*') == 1.0 for i in range(num_demands)), name="flow_conservation")
#
#         # 约束 2：链路容量约束
#         # 初始化一个线性表达式列表，用于累加每条链路上的总负载
#         link_load = [gp.LinExpr() for _ in range(num_links)]
#         for i, sd_key in enumerate(sd_keys):
#             demand_value = tm[sd_key]
#             paths = path_dict.get(sd_key, [])
#
#             # 如果某个OD对没有找到路径，则跳过
#             if len(paths) < k:
#                 print(f"  警告: OD对 {sd_key} 只有 {len(paths)} 条路径，不足 {k} 条，已跳过。")
#                 continue
#
#             for j, path in enumerate(paths[:k]):  # 确保只使用前k条路径
#                 for step in range(len(path) - 1):
#                     u, v = path[step], path[step + 1]
#                     link_tuple = (u, v)
#                     if link_tuple in link_to_idx:
#                         l_idx = link_to_idx[link_tuple]
#                         # 将该路径上的流量（分流比 * 需求值）累加到对应链路上
#                         link_load[l_idx] += x[i, j] * demand_value
#
#         # 添加容量约束：每条链路的负载 <= mlu * 链路容量
#         for l_idx in range(num_links):
#             m.addConstr(link_load[l_idx] <= mlu * capacity, name=f"cap_{l_idx}")
#
#         # 求解模型
#         m.optimize()
#
#         # 处理求解结果
#         if m.status == GRB.OPTIMAL:
#             print(f"  时间片 {ts_idx}: 求解成功，最优MLU = {mlu.X:.4f}")
#             ratios_np = np.zeros((num_demands, k), dtype=np.float64)
#             for i in range(num_demands):
#                 for j in range(k):
#                     ratios_np[i, j] = x[i, j].X
#
#             # 转换为 torch.tensor
#             current_ts_tensor = torch.from_numpy(ratios_np).to(torch.float32)  # 使用 float32 节省空间
#             if smoothing > 0:  # 应用标签平滑
#                 current_ts_tensor = current_ts_tensor * (1 - smoothing) + (smoothing / k)
#         else:
#             print(f"  警告：时间片 {ts_idx} 未能找到最优解 (状态码: {m.status})，填充均匀分布。")
#             current_ts_tensor = torch.full((num_demands, k), 1.0 / k, dtype=torch.float32)
#
#         final_label_list.append(current_ts_tensor)
#         print(f"  时间片 {ts_idx} 处理耗时: {time.time() - ts_start_time:.2f} 秒。")
#
#     # 2. 保存最终的 pkl 文件
#     with open(output_label_pkl, 'wb') as f:
#         pickle.dump(final_label_list, f)
#
#     print(f"\n所有时间片处理完成！总耗时: {time.time() - start_time:.2f} 秒。")
#     print(f"标签已保存至: {output_label_pkl}")
#     print(f"单个 Tensor 维度: {final_label_list[0].shape}")
#
#     return final_label_list
#
#
# if __name__ == "__main__":
#     # --- 配置项 ---
#     input_data_pkl = "Iridium_DataSetForAgent_75_60480.pkl"
#     output_label_pkl = "Labels_Iridium_Optimized.pkl"
#
#     labels = generate_labels_as_tensors(input_data_pkl, output_label_pkl, capacity=2500, smoothing=0.1)
#
