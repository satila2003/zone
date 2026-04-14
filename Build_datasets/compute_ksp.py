import os
import pickle
import networkx as nx
from itertools import islice
from glob import glob


def read_iridium_links_and_step(txt_path):
    """读取txt的连接关系+Step编号（用于大字典的键）"""
    links = []
    step = None
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        in_links = False
        for line in lines:
            line = line.strip()
            # 提取Step编号
            if line.startswith('Step:'):
                step = int(line.split(':')[1].strip())
            # 提取连接关系
            if line == '[LINKS]':
                in_links = True
                continue
            if in_links and line and not line.startswith('Type'):
                parts = line.split(',')
                if len(parts) >= 3:
                    src = int(parts[1].strip())
                    dst = int(parts[2].strip())
                    links.append((src, dst))
    return links, step


def find_ksp_per_pair(graph, src, dst, k=5):
    """获取单个节点对的k条最短路径"""
    paths = []
    try:
        path_generator = nx.shortest_simple_paths(graph, src, dst)
        paths = list(islice(path_generator, k))
    except nx.NetworkXNoPath:
        pass
    # 不足k条则用第一条填充
    if len(paths) < k and paths:
        paths += [paths[0]] * (k - len(paths))
    return paths


def compute_iridium_target_dict(txt_dir, output_pkl_path, k=5):
    """生成目标格式：大字典（200个Step键）→ 子字典（"src, dst"键→5条路径）"""
    # 按Step排序txt文件
    txt_files = sorted(glob(os.path.join(txt_dir, '*.txt')),
                       key=lambda x: int(os.path.basename(x).split('_')[-2]) if 'Step' in os.path.basename(x) else 0)

    if len(txt_files) != 200:
        print(f"注意：当前目录仅找到{len(txt_files)}个txt（预期200个）")

    # 最终大字典：键=Step编号（对应200个文件），值=该文件的路径子字典
    final_dict = {}

    for idx, txt_path in enumerate(txt_files):
        # 读取当前文件的连接关系+Step
        links, step = read_iridium_links_and_step(txt_path)
        if step is None:
            print(f"警告：第{idx + 1}个文件无法提取Step，跳过")
            continue
        if not links:
            print(f"警告：Step {step} 无有效连接，跳过")
            continue

        # 构建无向图
        graph = nx.Graph()
        graph.add_edges_from(links)
        nodes = list(graph.nodes())
        if not nodes:
            print(f"警告：Step {step} 无有效节点，跳过")
            continue

        # 构建当前文件的路径子字典（键："src, dst"，值：5条路径）
        path_subdict = {}
        for src in nodes:
            for dst in nodes:
                if src != dst:
                    # 键格式化为 "src, dst"（英文逗号+空格）
                    key_str = f"{src}, {dst}"
                    path_subdict[key_str] = find_ksp_per_pair(graph, src, dst, k)

        # 将当前文件的子字典存入大字典（键=Step）
        final_dict[step] = path_subdict
        print(f"已处理 Step {step}：子字典包含{len(path_subdict)}个节点对路径")

    # 自动创建输出目录
    output_dir = os.path.dirname(output_pkl_path)
    os.makedirs(output_dir, exist_ok=True)

    # 保存最终大字典到pkl
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(final_dict, f)

    print(f"\n完成！200个文件的路径已保存为大字典（键=Step）：{output_pkl_path}")
    print(f"格式示例：final_dict[Step_0]['6883, 8224'] → 5条路径的列表")

if __name__ == "__main__":
    # 配置实际路径
    TXT_DIRECTORY = "F:/Py_Project/always/Build_datasets/Export_20251218_160256"  # 200个txt所在目录
    OUTPUT_PKL = "F:/Py_Project/always/Build_datasets/iridium_ksp_paths/iridium_200steps_paths.pkl"  # 输出pkl路径
    compute_iridium_target_dict(TXT_DIRECTORY, OUTPUT_PKL, k=5)


