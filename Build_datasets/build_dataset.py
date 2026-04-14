import os
import re
import pickle
import networkx as nx
from itertools import islice
import rasterio
from rasterio.errors import RasterioIOError

# --- 1. 配置项 ---
# 定义人口密度分档 (与landscan数据对应)
POPULATION_BINS = [
    (1, 5),
    (6, 25),
    (26, 50),
    (51, 100),
    (101, 500),
    (501, 2500),
    (2501, 5000),
    (5001, 185000)
]

# 定义每个分档对应的Demand值 (从低到高)
BIN_TO_DEMAND = [10, 15, 25, 35, 50, 75, 100, 120]
# 定义海洋或无数据区域对应的最低Demand值
OCEAN_DEMAND = 5


def get_bin_index(population_density):
    """根据人口密度值返回其所属分档的索引。"""
    if population_density < 0:
        return -1  # 用-1代表海洋或无数据
    for i, (lower, upper) in enumerate(POPULATION_BINS):
        if lower <= population_density <= upper:
            return i
    # 如果密度超出最高档，也归为最高档
    return len(POPULATION_BINS) - 1


def k_shortest_paths(G, source, target, k=5):
    """计算源节点到目的节点的前 K 条最短路径 (基于跳数)"""
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def parse_txt_file_new(file_path):
    """
    【已修改】解析txt文件，返回节点列表、节点ID到经纬度的映射字典、以及【双向】链路列表。
    """
    nodes = []
    node_coords = {}  # key: node_id, value: (lat, lon)
    edges = []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

        # 1. 提取节点 ID 和经纬度
        node_section = re.search(r'\[NODES\]\nID, Name, Lat\(deg\), Lon\(deg\), Alt\(km\)\n(.*?)\n\n', content, re.S)
        if node_section:
            for line in node_section.group(1).strip().split('\n'):
                parts = line.split(',')
                if len(parts) >= 5 and parts[0].strip():
                    node_id = int(parts[0].strip())
                    lat = float(parts[2].strip())
                    lon = float(parts[3].strip())
                    nodes.append(node_id)
                    node_coords[node_id] = (lat, lon)

        # 2. 提取链路 SourceID, TargetID 并【显式添加双向边】
        link_section = re.search(r'\[LINKS\]\nType, SourceID, TargetID\n(.*)', content, re.S)
        if link_section:
            lines = link_section.group(1).strip().split('\n')
            for line in lines:
                parts = line.split(',')
                if len(parts) >= 3:
                    src = int(parts[1].strip())
                    dst = int(parts[2].strip())
                    # 【核心修改】：为每一条链路添加正向和反向两条边
                    edges.append([src, dst])
                    edges.append([dst, src])

    return nodes, node_coords, edges


def build_dataset_new(input_dir, output_filename, landsat_path, num_slices=200, k=5):
    """
    【已重构】主函数，构建数据集。
    """
    dataset = []
    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])
    file_list = file_list[:num_slices]

    try:
        with rasterio.open(landsat_path) as landsat_src:
            for idx, file_name in enumerate(file_list):
                print(f"正在处理第 {idx + 1}/{len(file_list)} 个时间片: {file_name}...")
                file_path = os.path.join(input_dir, file_name)

                nodes, node_coords, edges = parse_txt_file_new(file_path)

                if not nodes or not edges:
                    print(f"警告: {file_name} 中未找到有效的节点或链路数据，已跳过。")
                    continue

                # 构建 NetworkX 无向图用于计算路径
                # 注意：nx.Graph() 会自动处理重复的双向边，内部只存储一次
                G = nx.Graph()
                G.add_nodes_from(nodes)
                G.add_edges_from(edges)

                # 1. 记录 graph (【核心修改】直接使用包含双向边的edges列表)
                graph_data = edges

                # 2 & 3. 生成 tm 和 path
                tm_data = {}
                path_data = {}

                for i in nodes:
                    for j in nodes:
                        if i == j:
                            continue

                        key = f"{i}, {j}"
                        try:
                            src_lat, src_lon = node_coords[i]
                            dst_lat, dst_lon = node_coords[j]

                            src_row, src_col = landsat_src.index(src_lon, src_lat)
                            dst_row, dst_col = landsat_src.index(dst_lon, dst_lat)

                            if not (0 <= src_row < landsat_src.height and 0 <= src_col < landsat_src.width):
                                src_pop = -1
                            else:
                                src_pop = landsat_src.read(1, window=((src_row, src_row + 1), (src_col, src_col + 1)))[
                                    0, 0]

                            if not (0 <= dst_row < landsat_src.height and 0 <= dst_col < landsat_src.width):
                                dst_pop = -1
                            else:
                                dst_pop = landsat_src.read(1, window=((dst_row, dst_row + 1), (dst_col, dst_col + 1)))[
                                    0, 0]

                            src_bin_idx = get_bin_index(src_pop)
                            dst_bin_idx = get_bin_index(dst_pop)

                            if src_bin_idx == -1 or dst_bin_idx == -1:
                                demand = OCEAN_DEMAND
                            else:
                                higher_bin_idx = max(src_bin_idx, dst_bin_idx)
                                demand = BIN_TO_DEMAND[higher_bin_idx]

                            tm_data[key] = demand

                            paths = k_shortest_paths(G, i, j, k=k)
                            path_data[key] = paths

                        except KeyError:
                            print(f"警告: 在文件 {file_name} 中，节点对 ({i}, {j}) 的坐标信息缺失，已跳过。")
                            continue

                # 4. 构建字典
                slice_dict = {
                    'graph': graph_data,
                    'tm': tm_data,
                    'path': path_data,
                    'data_idx': idx
                }

                dataset.append(slice_dict)

    except RasterioIOError as e:
        print(f"\n错误：无法打开人口密度文件 '{landsat_path}'。请检查文件路径是否正确。")
        print(f"详细错误: {e}")
        return

    except Exception as e:
        print(f"\n处理过程中发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return

    # 保存为 pkl 文件
    with open(output_filename, 'wb') as f:
        pickle.dump(dataset, f)

    print(f"\n数据集构建完成！已保存至: {output_filename}")
    print(f"数据集总长度: {len(dataset)}")
    if dataset:
        # 【提示修改】现在打印的是双向边的总数
        print(f"第一个时间片的边总数(双向): {len(dataset[0]['graph'])}")
        print(f"第一个时间片的节点对总数: {len(dataset[0]['tm'])}")


if __name__ == "__main__":
    # --- 请根据你的实际情况配置以下路径 ---
    INPUT_FOLDER = r"F:\Py_Project\always\Build_datasets\Export_20260104_095407"
    OUTPUT_PKL = "Iridium_DataSetForAgent_75_60480.pkl"
    LANDSCAN_FILE_PATH = r"F:\Py_Project\always\Build_datasets\landscan-global-2024.tif"

    # --- 运行主函数 ---
    if os.path.exists(INPUT_FOLDER):
        build_dataset_new(
            input_dir=INPUT_FOLDER,
            output_filename=OUTPUT_PKL,
            landsat_path=LANDSCAN_FILE_PATH,
            num_slices=200,
            k=5
        )
    else:
        print(f"错误：找不到输入目录 {INPUT_FOLDER}")

