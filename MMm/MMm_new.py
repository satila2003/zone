import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
# lambda_rate = 9  # 到达率
# mu_rate = 31  # 服务率
# ser_window = 4  # 服务窗口数
# sim_times = 2000  # 仿真次数
# P_obser = 12  # 观察参数
# Total_time = 60  # 总仿真时间
# N = 10 ** 10  # 队列最大长度

lambda_rate = 12  # 到达率
mu_rate = 10  # 服务率
ser_window = 2  # 服务窗口数
sim_times = 2000  # 仿真次数
P_obser = 12  # 观察参数
Total_time = 60  # 总仿真时间
N = 10 ** 10  # 队列最大长度
# 基础参数计算
arr_mean = 1 / lambda_rate
ser_mean = 1 / mu_rate
arr_num = round(Total_time * lambda_rate * 5)  # 生成充足的到达事件

# 初始化存储数组
customer_cur = np.zeros(sim_times)
customer_wait = np.zeros(sim_times)
Ls_poisson = np.zeros(sim_times)
Lq_poisson = np.zeros(sim_times)
Ls_random = np.zeros(sim_times)
Lq_random = np.zeros(sim_times)
Ws_sim = np.zeros(sim_times)
Wq_sim = np.zeros(sim_times)
total_poisson = []
total_random = []

# 理论分布参数
n_max = 50
p = np.zeros((2, n_max + 1))
p[0, :] = np.arange(n_max + 1)

# 多次仿真循环
for count in range(sim_times):
    # 初始化事件矩阵（11行：到达时刻/服务时间/等待时间/离开时刻等）
    events = np.zeros((11, arr_num))

    # 生成到达时刻（负指数分布）
    events[0, :] = np.cumsum(np.random.exponential(arr_mean, arr_num))
    # 生成服务时间（负指数分布）
    events[1, :] = np.random.exponential(ser_mean, arr_num)

    # 生成泊松观察时刻
    poisson_intervals = np.random.exponential(1 / P_obser, arr_num)
    events[5, :] = np.cumsum(poisson_intervals)
    # 生成随机观察时刻（均匀分布间隔）
    random_intervals = np.random.uniform(0, 2 * arr_mean, arr_num)
    events[8, :] = np.cumsum(random_intervals)

    # 有效仿真顾客数（到达时刻在总时间内）
    len_sim = np.sum(events[0, :] <= Total_time)
    if len_sim == 0:
        continue  # 无顾客到达则跳过

    # 第一个顾客处理
    events[2, 0] = 0  # 等待时间
    events[3, 0] = events[0, 0] + events[1, 0]  # 离开时刻
    events[4, 0] = 1  # 系统内顾客数

    # 处理后续顾客
    for i in range(1, len_sim):
        if events[0, i] > Total_time:
            break

        # 当前系统内顾客数（不含当前顾客）
        number = np.sum(events[3, 0:i] > events[0, i])

        if number >= N:  # 系统满员，拒绝服务
            events[4, i] = 0
            events[2, i] = -1
            events[3, i] = -1
        elif number < ser_window:  # 直接服务
            events[2, i] = 0
            events[3, i] = events[0, i] + events[1, i]
            events[4, i] = number + 1
        else:  # 排队等待
            in_system = events[3, 0:i][events[3, 0:i] > events[0, i]]
            sorted_leave = np.sort(in_system)
            fastest_free = sorted_leave[ser_window - 1]
            events[2, i] = fastest_free - events[0, i]
            events[3, i] = fastest_free + events[1, i]
            events[4, i] = number + 1

    # 泊松观察处理
    len_poisson = np.sum(events[5, :] <= Total_time)
    start_poisson = round(len_poisson * 0.7)  # 取后30%稳态数据
    for j in range(len_poisson):
        t = events[5, j]
        arrived = np.sum(events[0, 0:len_sim] <= t)
        left = np.sum(events[3, 0:len_sim][events[3, 0:len_sim] >= 0] <= t)
        events[6, j] = arrived - left
        if j >= start_poisson:
            total_poisson.append(events[6, j])
        events[7, j] = max(0, events[6, j] - ser_window)

    # 随机观察处理
    len_random = np.sum(events[8, :] <= Total_time)
    start_random = round(len_random * 0.7)  # 取后30%稳态数据
    for r in range(len_random):
        t = events[8, r]
        arrived = np.sum(events[0, 0:len_sim] < t)
        left = np.sum(events[3, 0:len_sim][events[3, 0:len_sim] >= 0] < t)
        events[9, r] = arrived - left
        if r >= start_random:
            total_random.append(events[9, r])
        events[10, r] = max(0, events[9, r] - ser_window)

    # 记录本次仿真结果
    arrived_end = np.sum(events[0, 0:len_sim] < Total_time)
    left_end = np.sum(events[3, 0:len_sim][events[3, 0:len_sim] >= 0] < Total_time)
    customer_cur[count] = arrived_end - left_end
    customer_wait[count] = max(0, customer_cur[count] - ser_window)

    # 计算观察指标平均值
    valid_p = events[6, start_poisson:len_poisson]
    Ls_poisson[count] = np.mean(valid_p) if len(valid_p) > 0 else 0
    Lq_poisson[count] = np.mean(events[7, start_poisson:len_poisson]) if len(valid_p) > 0 else 0

    valid_r = events[9, start_random:len_random]
    Ls_random[count] = np.mean(valid_r) if len(valid_r) > 0 else 0
    Lq_random[count] = np.mean(events[10, start_random:len_random]) if len(valid_r) > 0 else 0

    # 计算逗留时间和等待时间
    valid_idx = events[3, start_random:len_sim] >= 0
    if np.any(valid_idx):
        Ws_sim[count] = np.mean(events[3, start_random:len_sim][valid_idx] - events[0, start_random:len_sim][valid_idx])
        Wq_sim[count] = np.mean(events[2, start_random:len_sim][valid_idx])
    else:
        Ws_sim[count] = 0
        Wq_sim[count] = 0

# 理论指标计算
rho = lambda_rate / mu_rate
rho_m = lambda_rate / (ser_window * mu_rate)
sigma = 1 + sum((rho ** n) / math.factorial(n) for n in range(1, ser_window))
p0 = 1 / (sigma + (rho ** ser_window) / (math.factorial(ser_window) * (1 - rho_m)))
p[1, 0] = p0

for s in range(1, ser_window + 1):
    p[1, s] = (rho ** s) * p0 / math.factorial(s)
for s in range(ser_window + 1, n_max + 1):
    p[1, s] = (rho ** s) * p0 / (math.factorial(ser_window) * (ser_window ** (s - ser_window)))

Lq = (p0 * rho * (rho ** ser_window)) / (math.factorial(ser_window) * (1 - rho_m) ** 2)
Ls = Lq + rho
Ws = Ls / lambda_rate
Wq = Lq / lambda_rate

# 仿真结果平均值
Ls_Sim = np.mean(customer_cur)
Lq_Sim = np.mean(customer_wait)
Ls_Poisson = np.mean(Ls_poisson) if sim_times > 0 else 0
Lq_Poisson = np.mean(Lq_poisson) if sim_times > 0 else 0
Ls_Random = np.mean(Ls_random) if sim_times > 0 else 0
Lq_Random = np.mean(Lq_random) if sim_times > 0 else 0
Ws_Sim = np.mean(Ws_sim) if sim_times > 0 else 0
Wq_Sim = np.mean(Wq_sim) if sim_times > 0 else 0

# 频率分布计算（补全0~n_max）
def calculate_frequency(data, n_max):
    counts = Counter(data)
    total = len(data) if data else 1
    freq = np.zeros(n_max + 1)
    for n in range(n_max + 1):
        freq[n] = (counts.get(n, 0) / total) * 100
    return freq

pro_customer_cur = calculate_frequency(customer_cur.astype(int).tolist(), n_max)
pro_Ls_Poisson = calculate_frequency(total_poisson, n_max)
pro_Ls_Random = calculate_frequency(total_random, n_max)

# 输出结果
print('--- 理论结果 ---')
print(f'理论上的平均队长 (Ls) 为: {Ls:.4f}')
print(f'理论上的平均排队队长 (Lq) 为: {Lq:.4f}')
print(f'理论上的平均逗留时间 (Ws) 为: {Ws:.4f}')
print(f'理论上的平均排队时间 (Wq) 为: {Wq:.4f}\n')

print('--- 仿真结果 ---')
print(f'仿真的平均队长 (Ls_Sim) 为: {Ls_Sim:.4f}')
print(f'仿真的平均排队队长 (Lq_Sim) 为: {Lq_Sim:.4f}')
print(f'仿真的平均逗留时间 (Ws_Sim) 为: {Ws_Sim:.4f}')
print(f'仿真的平均排队时间 (Wq_Sim) 为: {Wq_Sim:.4f}\n')

print('--- 观察结果 ---')
print(f'泊松观察的平均队长 (Ls_Poisson) 为: {Ls_Poisson:.4f}')
print(f'泊松观察的平均排队队长 (Lq_Poisson) 为: {Lq_Poisson:.4f}')
print(f'随机观察的平均队长 (Ls_Random) 为: {Ls_Random:.4f}')
print(f'随机观察的平均排队队长 (Lq_Random) 为: {Lq_Random:.4f}')

# 绘图1：平均队长和排队队长对比
plt.figure(1, figsize=(10, 6))
labels = ['理论', '仿真', '泊松', '随机']
ls_values = [Ls, Ls_Sim, Ls_Poisson, Ls_Random]
lq_values = [Lq, Lq_Sim, Lq_Poisson, Lq_Random]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width / 2, ls_values, width, label='平均队长',
        color=[0.5, 0.5, 1], edgecolor='b', linewidth=1.5, alpha=0.7)
plt.bar(x + width / 2, lq_values, width, label='平均排队队长',
        color=[1, 0.8, 0.6], edgecolor='r', linewidth=1.5, alpha=0.7)
plt.ylabel('队长/排队队长')
plt.title('平均队长和平均排队队长对比')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('平均队长和平均排队队长对比.jpg', dpi=300, bbox_inches='tight')

# 绘图2：平均逗留时间和等待时间对比
plt.figure(2, figsize=(8, 6))
labels = ['理论', '仿真']
ws_values = [Ws, Ws_Sim]
wq_values = [Wq, Wq_Sim]
x = np.arange(len(labels))

plt.bar(x - width / 2, ws_values, width, label='平均逗留时间',
        color=[0.2, 0.8, 0.2], edgecolor='g', linewidth=1.5, alpha=0.7)
plt.bar(x + width / 2, wq_values, width, label='平均等待时间',
        color=[1, 1, 0], edgecolor='y', linewidth=1.5, alpha=0.7)
plt.ylabel('时间')
plt.title('平均逗留时间和等待时间对比')
plt.xticks(x, labels)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('平均逗留时间和等待时间对比.jpg', dpi=300, bbox_inches='tight')

# 绘图3：队长频率分布（修正尺寸和x轴范围）
plt.figure(3, figsize=(10, 8))  # 缩小整体尺寸
plt.suptitle('队长频率分布对比', fontsize=16)

# 理论分布
plt.subplot(2, 2, 1)
plt.plot(p[0, :], p[1, :], '-s', color=[0.8, 0.2, 0.2], linewidth=1.5, markersize=6)
plt.xlim(0, 10)  # 聚焦低顾客数范围（0-10）
plt.xlabel('系统中的顾客数')
plt.ylabel('概率')
plt.title('理论分布')
plt.grid(True, alpha=0.5)

# 仿真分布
plt.subplot(2, 2, 2)
plt.plot(np.arange(n_max + 1), pro_customer_cur / 100, '-^',
         color=[0.2, 0.6, 0.8], linewidth=1.5, markersize=6)
plt.xlim(0, 10)  # 同上
plt.xlabel('系统中的顾客数')
plt.ylabel('频率')
plt.title('仿真分布')
plt.grid(True, alpha=0.5)

# 泊松观察分布
plt.subplot(2, 2, 3)
plt.plot(np.arange(n_max + 1), pro_Ls_Poisson / 100, '-d',
         color=[0.5, 0.2, 0.8], linewidth=1.5, markersize=6)
plt.xlim(0, 10)  # 同上
plt.xlabel('系统中的顾客数')
plt.ylabel('频率')
plt.title('泊松观察分布')
plt.grid(True, alpha=0.5)

# 随机观察分布
plt.subplot(2, 2, 4)
plt.plot(np.arange(n_max + 1), pro_Ls_Random / 100, '-o',
         color=[0.2, 0.8, 0.5], linewidth=1.5, markersize=6)
plt.xlim(0, 10)  # 同上
plt.xlabel('系统中的顾客数')
plt.ylabel('频率')
plt.title('随机观察分布')
plt.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('队长频率分布对比.jpg', dpi=300, bbox_inches='tight')

# 绘图4：平均排队时间对比（修正文本偏移）
plt.figure(4, figsize=(7, 5))  # 适当缩小尺寸
waiting_times = [Wq, Wq_Sim]
labels = ['理论', '仿真']
bars = plt.bar(labels, waiting_times, color=[0.2, 0.6, 0.8], alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.0001,  # 减小文本偏移量
             f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('类型')
plt.ylabel('平均排队时间')
plt.title('理论与仿真平均排队时间对比')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('理论与仿真平均排队时间对比.jpg', dpi=300, bbox_inches='tight')

# plt.show()