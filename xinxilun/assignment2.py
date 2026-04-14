import numpy as np
import random

# ==========================================
# ① 构造校验矩阵 H 和 生成矩阵 G
# ==========================================

# 构造 P 子矩阵 (4x11)
# 包含 2^4 - 1 = 15 种非零组合中，除了4个重量为1的向量外的其余 11 个向量
P = np.array([
    [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1]
])

# 构造 4x4 单位矩阵 I_4
I4 = np.eye(4, dtype=int)

# 拼接得到校验矩阵 H = [P | I_4] (4x15)
H = np.hstack((P, I4))

# 构造 11x11 单位矩阵 I_11
I11 = np.eye(11, dtype=int)

# 拼接得到生成矩阵 G = [I_11 | P^T] (11x15)
G = np.hstack((I11, P.T))

print("=================== 矩阵构造 ===================")
print("1. 校验矩阵 H (4x15) 结构为 [P | I_4]:")
print(H)
print("\n2. 生成矩阵 G (11x15) 结构为 [I_11 | P^T]:")
print(G)

# ==========================================
# ② 编、译码与信道模拟过程
# ==========================================
print("\n=================== 编/译码测试 ===================")

# 1. 输入消息 x (随机生成 11 位二进制数据)
x = np.random.randint(0, 2, 11)
print(f"原始消息 (x):        {x}")

# 2. 编码生成码字 c = x * G (模2加法)
c = np.dot(x, G) % 2
print(f"编码后码字 (c):      {c}")

# 3. 模拟二进制对称信道 (BSC) 传输
r = c.copy()
# 随机决定是否出错 (这里强制引入1位错误以便演示译码过程)
error_idx = random.randint(0, 14)
r[error_idx] ^= 1  # 异或 1 实现位翻转

print(f"接收到的码字 (r):    {r}")
print(f"  -> [信道警报] 索引 {error_idx} 处发生了1位随机错误！")

# 4. 译码：计算伴随式 s = r * H^T (模2加法)
s = np.dot(r, H.T) % 2
print(f"\n译码后伴随式 (s):    {s}")

# 5. 错误图案估计 e_hat
e_hat = np.zeros(15, dtype=int)
if np.any(s):  # 如果伴随式不为全0，说明有错误
    # 在 H 的列向量中寻找与伴随式 s 相同的列
    for i in range(15):
        if np.array_equal(H[:, i], s):
            e_hat[i] = 1
            print(f"  -> [综合诊断] 伴随式匹配 H 矩阵的第 {i} 列。")
            break

print(f"错误图案估计 (e_hat): {e_hat}")

# 6. 码字估计 c_hat = r + e_hat (模2加法)
c_hat = (r ^ e_hat)
print(f"码字估计 (c_hat):    {c_hat}")

# 提取恢复的消息 (前11位)
x_hat = c_hat[:11]

print("\n=================== 最终结论 ===================")
success = np.array_equal(x, x_hat)
print(f"恢复的消息 (x_hat):  {x_hat}")
print(f"译码是否成功?        {'✅ 是' if success else '❌ 否'}")

# 下面是豆包
# import numpy as np
#
# # ===================== 汉明码参数 =====================
# n, k, r = 15, 11, 4  # 码长、信息位、校验位
# np.set_printoptions(linewidth=100, suppress=True)  # 格式化打印矩阵
#
# # ===================== 1. 构造校验矩阵 H = [P | I₄] =====================
# # 生成所有4位非零二进制列向量
# h_columns = [[int(b) for b in f"{i:04b}"] for i in range(1, 16)]
# # 拆分：最后4列=单位矩阵I₄，前11列=P子矩阵
# unit_cols = [[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
# p_cols = [col for col in h_columns if col not in unit_cols]
# # 拼接并转置 → 4×15 校验矩阵
# H = np.array(p_cols + unit_cols).T
# P = H[:, :k]  # 4×11 子矩阵P
#
# # ===================== 2. 构造生成矩阵 G = [I₁₁ | Pᵀ] =====================
# I_k = np.eye(k, dtype=int)
# P_T = P.T
# G = np.hstack((I_k, P_T))
#
# # ===================== 核心函数 =====================
# def encode(x):
#     """编码：11位消息 → 15位码字"""
#     return np.mod(x @ G, 2)
#
# def random_error():
#     """随机生成 ≤1位错误的向量（满足题目要求）"""
#     e = np.zeros(n, dtype=int)
#     error_pos = np.random.choice([-1, *range(n)])  # -1=无错，0-14=单比特错
#     if error_pos != -1:
#         e[error_pos] = 1
#     return e, error_pos
#
# def decode(r):
#     """译码：返回伴随式s、错误估计ê、码字估计ĉ"""
#     s = np.mod(r @ H.T, 2)
#     e_hat = np.zeros(n, dtype=int)
#     if not np.all(s == 0):
#         err_idx = np.where((H.T == s).all(axis=1))[0][0]
#         e_hat[err_idx] = 1
#     c_hat = np.mod(r + e_hat, 2)
#     return s, e_hat, c_hat
#
# # ===================== 格式化打印函数 =====================
# def print_matrix(name, mat, desc):
#     print(f"\n{name} {desc}")
#     print("-" * 80)
#     print(mat)
#
# # ===================== 主程序：完整流程输出 =====================
# if __name__ == "__main__":
#     print("="*100)
#     print("                    (15,11)汉明码 编译码全流程输出")
#     print("="*100)
#
#     # 1. 打印核心矩阵（替代截图，格式化输出）
#     print_matrix("【校验矩阵 H】", H, "(4×15) | 分块：H = [P(4×11) | I₄(4×4)]")
#     print_matrix("【生成矩阵 G】", G, "(11×15) | 分块：G = [I₁₁(11×11) | Pᵀ(11×4)]")
#
#     # 2. 输入11位原始二进制消息（可自定义修改）
#     x_str = "10110100110"
#     x = np.array([int(bit) for bit in x_str], dtype=int)
#     print("\n" + "="*80)
#     print("【1】编码前原始消息 x")
#     print(f"11位二进制数据：x = {x}")
#
#     # 3. 编码得到码字c
#     c = encode(x)
#     print("\n【2】编码输出码字 c")
#     print(f"15位编码码字：c = {c}")
#
#     # 4. 随机生成≤1位错误，得到接收码字r
#     e, error_pos = random_error()
#     r = np.mod(c + e, 2)
#     print("\n【3】信道传输（随机出错≤1位）")
#     print(f"错误位置：{'无错误' if error_pos == -1 else error_pos}")
#     print(f"错误图案 e：e = {e}")
#     print(f"接收码字 r：r = {r}")
#
#     # 5. 译码：伴随式、错误估计、码字估计
#     s, e_hat, c_hat = decode(r)
#     print("\n【4】译码结果")
#     print(f"伴随式 s：s = {s}")
#     print(f"错误图案估计 ê：ê = {e_hat}")
#     print(f"码字估计 ĉ：ĉ = {c_hat}")
#
#     # 6. 结果验证
#     x_hat = c_hat[:k]
#     print("\n【5】最终验证")
#     print(f"原始消息 x：   {x}")
#     print(f"译码恢复消息 x̂：{x_hat}")
#     print(f"译码成功：{np.array_equal(x, x_hat)}")
#     print("="*100)