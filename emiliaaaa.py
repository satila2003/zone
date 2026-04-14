import heapq
import math

# 1. 权威的英文字符概率分布 (26个字母 + 1个空格)
# 数据基于典型英文语料库的大样本统计，并进行了归一化处理确保总和为 1
char_probabilities = {
    ' ': 0.1830, 'e': 0.1021, 't': 0.0773, 'a': 0.0681, 'o': 0.0624,
    'n': 0.0594, 'i': 0.0583, 's': 0.0514, 'r': 0.0485, 'h': 0.0456,
    'l': 0.0321, 'd': 0.0315, 'c': 0.0223, 'u': 0.0223, 'm': 0.0201,
    'f': 0.0193, 'p': 0.0162, 'y': 0.0152, 'w': 0.0152, 'g': 0.0132,
    'b': 0.0112, 'v': 0.0081, 'k': 0.0061, 'x': 0.0020, 'j': 0.0010,
    'q': 0.0010, 'z': 0.0009
}

# 确保概率总和精确为 1 (浮点数精度处理)
total_prob = sum(char_probabilities.values())
char_probabilities = {k: v / total_prob for k, v in char_probabilities.items()}


# 赫夫曼树节点定义
class HuffmanNode:
    def __init__(self, char, prob):
        self.char = char
        self.prob = prob
        self.left = None
        self.right = None

    # 定义比较方法，用于优先队列 (小根堆)
    def __lt__(self, other):
        return self.prob < other.prob


def build_huffman_tree(probs):
    """根据信源概率分布构建赫夫曼树"""
    heap = [HuffmanNode(char, prob) for char, prob in probs.items()]
    heapq.heapify(heap)  # 构建极小堆

    # 只要堆中还有两个以上的节点，就不断合并概率最小的两个节点
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        # 新节点的概率为两个子节点概率之和
        merged = HuffmanNode(None, left.prob + right.prob)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]  # 返回根节点


def generate_codebook(node, prefix="", codebook=None):
    """递归遍历赫夫曼树，生成码表"""
    if codebook is None:
        codebook = {}

    if node is not None:
        # 如果是叶子节点，则记录字符及其对应的二进制前缀码
        if node.char is not None:
            codebook[node.char] = prefix
        # 左子树路径记为 '0'，右子树路径记为 '1'
        generate_codebook(node.left, prefix + "0", codebook)
        generate_codebook(node.right, prefix + "1", codebook)

    return codebook


def encode(text, codebook):
    """利用码表对输入的英文字符串进行信源编码"""
    encoded_bits = ""
    text = text.lower()  # 离散信源不区分大小写
    for char in text:
        if char in codebook:
            encoded_bits += codebook[char]
        else:
            # 过滤掉标点符号等不在信源符号集中的异常输入
            pass
    return encoded_bits


def decode(encoded_bits, codebook):
    """利用码表对二进制比特流进行译码"""
    # 翻转码表以供译码查询：因为赫夫曼码是前缀码，所以可以唯一匹配
    reverse_codebook = {v: k for k, v in codebook.items()}

    decoded_text = ""
    buffer = ""
    for bit in encoded_bits:
        buffer += bit
        # 由于满足前缀条件，一旦在反向码表中找到匹配项，即可立即译出 (Instantaneous Decoding)
        if buffer in reverse_codebook:
            decoded_text += reverse_codebook[buffer]
            buffer = ""

    return decoded_text


# ================= 运行与信息论指标计算 =================

# ① 得出赫夫曼码的码表
huffman_tree_root = build_huffman_tree(char_probabilities)
huffman_codebook = generate_codebook(huffman_tree_root)

print("① 赫夫曼信源编码表 (按码长排序):")
# 按码长从小到大排序输出，可以看到高频字符码长短，低频字符码长长
sorted_codebook = sorted(huffman_codebook.items(), key=lambda item: (len(item[1]), item[0]))
for char, code in sorted_codebook:
    repr_char = "Space" if char == ' ' else char.upper()
    print(f"符号: {repr_char:<5} 概率: {char_probabilities[char]:.4f}  码字: {code}")

# 信息论参数分析
# 1. 信源熵 H(X) = - Σ P(xi) * log2(P(xi))
entropy = -sum(p * math.log2(p) for p in char_probabilities.values())
# 2. 平均码长 L = Σ P(xi) * length(xi)
avg_length = sum(char_probabilities[char] * len(code) for char, code in huffman_codebook.items())
# 3. 编码效率 η = H(X) / L
efficiency = entropy / avg_length

print("\n--- 信息论指标分析 ---")
print(f"信源的香农熵 H(X): {entropy:.4f} bits/symbol")
print(f"编码的平均码长 L:   {avg_length:.4f} bits/symbol")
print(f"信源编码效率 η:     {efficiency:.2%}")
print("结论: 平均码长极其逼近信源熵，满足香农无失真信源编码定理 (L >= H(X))。")


# ② 输入一段英文字符，利用码表对其编、译码
print(f"\n② 编/译码测试")
source_text = input("\n请输入一段英文字符：")
print(f"原始信源序列: '{source_text}'")

encoded_data = encode(source_text, huffman_codebook)
print(f"编码后的比特流: {encoded_data}")
print(f"编码长度: {len(encoded_data)} bits (若是等长编码(5bits/symbol)则需 {len(source_text) * 5} bits)")

decoded_data = decode(encoded_data, huffman_codebook)
print(f"信宿译码恢复序列: '{decoded_data}'")