import os
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

# 这个脚本的目标是从微观数据集中提取每个域内的图、流量矩阵和路径信息，构建一个新的域内数据集。核心步骤包括：
# 1. 解析分域结果TXT文件，构建卫星ID到域ID的映射。
# 2. 读取微观PKL数据集，逐时间片处理每个域内的图、流量矩阵和路径。
# 3. 对于每个时间片，针对每个域，筛选出域内的物理链路、流量需求和路径，并构建新的数据结构。



# ================= 路径配置 =================
BASE_DIR = Path(__file__).parent
# 输入：分域结果 txt
CLUSTER_TXT_PATH = BASE_DIR / "outputs" / "starlink550_data_v1.5_domain_result_18_raan_orbital_angle.txt"
# 输入：微观 PKL 数据集 (包含完整的图、流量矩阵和路径信息) 
MICRO_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550.pkl"
# 输出：域内（分域后）PKL 数据集
INTRA_PKL_PATH = BASE_DIR / "outputs" / "tms" / "starlink550_intra.pkl"

# 超参数
MAX_PARALLEL_LINKS = 8  # 域内每对卫星保留的最大并行路径数


# ================= 核心工具函数 =================

def load_satellite_to_domain_mapping(txt_path):
    """解析分域TXT文件，构建 卫星ID -> 域ID 的映射"""
    sat_to_domain = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        next(f)  # 跳过表头
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            sat_id = int(parts[0])
            domain_id = int(parts[6])
            sat_to_domain[sat_id] = domain_id
    print(f"[*] 成功加载聚类映射，共 {len(sat_to_domain)} 颗卫星。")
    return sat_to_domain


def parse_tm_key(key_str):
    u_str, v_str = key_str.split(',')
    return int(u_str.strip()), int(v_str.strip())


def classify_path_in_domain(path, sat_to_domain, domain_id):
    for node in path:
        node_domain = sat_to_domain.get(node)
        if node_domain is None:
            return "missing_mapping"
        if node_domain != domain_id:
            return "cross_domain"
    return "ok"


def build_component_map(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    comp_id = {}
    cid = 0
    for node in adj:
        if node in comp_id:
            continue
        stack = [node]
        comp_id[node] = cid
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb not in comp_id:
                    comp_id[nb] = cid
                    stack.append(nb)
        cid += 1
    return comp_id


# ================= 主流程 =================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster_txt", type=str, default=str(CLUSTER_TXT_PATH))
    p.add_argument("--micro_pkl", type=str, default=str(MICRO_PKL_PATH))
    p.add_argument("--output_pkl", type=str, default=str(INTRA_PKL_PATH))
    p.add_argument("--max_paths", type=int, default=MAX_PARALLEL_LINKS)
    p.add_argument("--diagnose", action="store_true", default=True)
    p.add_argument("--diagnose_samples", type=int, default=20)
    p.add_argument("--overwrite", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_pkl), exist_ok=True)

    sat_to_domain = load_satellite_to_domain_mapping(args.cluster_txt)
    domain_ids = sorted(set(sat_to_domain.values()))

    if (not args.overwrite) and os.path.exists(args.output_pkl):
        print(f"[{args.output_pkl}] 已存在，跳过处理。")
        return

    if not os.path.exists(args.micro_pkl):
        raise FileNotFoundError(f"找不到微观数据集: {args.micro_pkl}")

    print(f"[*] 正在读取微观数据集 {args.micro_pkl} ...")
    with open(args.micro_pkl, "rb") as f:
        micro_slices = pickle.load(f)

    intra_slices = []
    print("[*] 开始执行域内数据集提取 (Graph, TM, Path) ...")

    diag = {
        "same_domain_pairs": 0,
        "pair_missing_mapping": 0,
        "pair_cross_domain": 0,
        "empty_in_micro_paths": 0,
        "empty_after_filter": 0,
        "kept_nonempty": 0,
        "path_missing_mapping": 0,
        "path_cross_domain": 0,
        "path_empty": 0,
        "disconnected_in_domain_graph": 0,
        "connected_but_no_domain_path_in_topk": 0
    }
    diag_samples = {
        "empty_in_micro_paths": [],
        "empty_after_filter": [],
        "connected_but_no_domain_path_in_topk": [],
        "disconnected_in_domain_graph": []
    }

    for slice_idx, slice_data in enumerate(micro_slices):
        micro_graph = slice_data["graph"]
        micro_tm = slice_data["tm"]
        micro_paths = slice_data.get("path", {})
        data_idx = slice_data["data_idx"]

        # 初始化域内结构
        domain_graph = {d: [] for d in domain_ids}
        domain_tm = {d: {} for d in domain_ids}
        domain_path = {d: defaultdict(list) for d in domain_ids}
        domain_active_sats = {d: set() for d in domain_ids}

        # Step A: 域内物理图
        for u, v in micro_graph:
            du = sat_to_domain.get(u)
            dv = sat_to_domain.get(v)
            if du is None or dv is None:
                continue
            # 只有当两颗卫星属于同一域时，才将它们之间的链路加入该域的图中
            if du == dv:
                domain_graph[du].append([u, v])
                domain_active_sats[du].add(u)
                domain_active_sats[du].add(v)

        # Step B: 域内流量矩阵 
        # 只有当流量需求的源和目的卫星都属于同一域时，才将该需求加入该域的流量矩阵中
        for key_str, demand in micro_tm.items():
            u, v = parse_tm_key(key_str)
            du = sat_to_domain.get(u)
            dv = sat_to_domain.get(v)
            if du is None or dv is None:
                continue
            if du == dv:
                domain_tm[du][key_str] = demand

        # Step C: 域内路径过滤
        # 只有当路径的所有节点都属于同一域时，才将该路径加入该域的路径列表中
        # 同时，为了控制每对卫星之间的路径数量，我们使用一个集合来跟踪已经添加的路径，避免重复，并且限制最大路径数。
        domain_components = {}
        if args.diagnose:
            for d in domain_ids:
                domain_components[d] = build_component_map(domain_graph[d])

        for key_str, paths_list in micro_paths.items():
            u, v = parse_tm_key(key_str)
            du = sat_to_domain.get(u)
            dv = sat_to_domain.get(v)
            if du is None or dv is None:
                if args.diagnose:
                    diag["pair_missing_mapping"] += 1
                continue
            if du != dv:
                if args.diagnose:
                    diag["pair_cross_domain"] += 1
                continue

            if args.diagnose:
                diag["same_domain_pairs"] += 1

            if not paths_list:
                if args.diagnose:
                    diag["empty_in_micro_paths"] += 1
                    diag["empty_after_filter"] += 1
                    if len(diag_samples["empty_in_micro_paths"]) < args.diagnose_samples:
                        diag_samples["empty_in_micro_paths"].append((du, key_str))

                    comp_map = domain_components.get(du, {})
                    u_comp = comp_map.get(u)
                    v_comp = comp_map.get(v)
                    if u_comp is None or v_comp is None or u_comp != v_comp:
                        diag["disconnected_in_domain_graph"] += 1
                        if len(diag_samples["disconnected_in_domain_graph"]) < args.diagnose_samples:
                            diag_samples["disconnected_in_domain_graph"].append((du, key_str))
                    else:
                        diag["connected_but_no_domain_path_in_topk"] += 1
                        if len(diag_samples["connected_but_no_domain_path_in_topk"]) < args.diagnose_samples:
                            diag_samples["connected_but_no_domain_path_in_topk"].append((du, key_str))
                continue

            seen = set()
            for p in paths_list:
                if len(domain_path[du][key_str]) >= args.max_paths:
                    break
                if not p:
                    if args.diagnose:
                        diag["path_empty"] += 1
                    continue
                reason = classify_path_in_domain(p, sat_to_domain, du)
                if reason != "ok":
                    if args.diagnose:
                        if reason == "missing_mapping":
                            diag["path_missing_mapping"] += 1
                        elif reason == "cross_domain":
                            diag["path_cross_domain"] += 1
                    continue
                key = tuple(p)
                if key in seen:
                    continue
                seen.add(key)
                domain_path[du][key_str].append(p)

            if args.diagnose:
                if len(domain_path[du][key_str]) == 0:
                    diag["empty_after_filter"] += 1
                    if len(diag_samples["empty_after_filter"]) < args.diagnose_samples:
                        diag_samples["empty_after_filter"].append((du, key_str))

                    comp_map = domain_components.get(du, {})
                    u_comp = comp_map.get(u)
                    v_comp = comp_map.get(v)
                    if u_comp is None or v_comp is None or u_comp != v_comp:
                        diag["disconnected_in_domain_graph"] += 1
                        if len(diag_samples["disconnected_in_domain_graph"]) < args.diagnose_samples:
                            diag_samples["disconnected_in_domain_graph"].append((du, key_str))
                    else:
                        diag["connected_but_no_domain_path_in_topk"] += 1
                        if len(diag_samples["connected_but_no_domain_path_in_topk"]) < args.diagnose_samples:
                            diag_samples["connected_but_no_domain_path_in_topk"].append((du, key_str))
                else:
                    diag["kept_nonempty"] += 1

        # 组装当前时间片
        slice_domains = {}
        for d in domain_ids:
            slice_domains[d] = {
                "graph": domain_graph[d],
                "tm": domain_tm[d],
                "path": dict(domain_path[d]),
                "active_sat_ids": sorted(domain_active_sats[d]),
                "data_idx": data_idx
            }

        intra_slices.append({
            "data_idx": data_idx,
            "domains": slice_domains
        })

        if (slice_idx + 1) % 50 == 0:
            print(f"    进度: {slice_idx + 1}/{len(micro_slices)} 个时间片已处理。")

    with open(args.output_pkl, "wb") as f:
        pickle.dump(intra_slices, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✅ 成功！域内数据集已保存至: {args.output_pkl}")
    print(f"✅ 文件内共包含 {len(intra_slices)} 个时间片的数据。")

    if args.diagnose:
        print("\n================= 诊断摘要（仅源宿同域） =================")
        print(f"同域对数量: {diag['same_domain_pairs']}")
        print(f"原始微观 path 为空: {diag['empty_in_micro_paths']}")
        print(f"过滤后为空（总计）: {diag['empty_after_filter']}")
        print(f"过滤后非空: {diag['kept_nonempty']}")
        print(f"路径含跨域节点: {diag['path_cross_domain']}")
        print(f"路径含缺失映射节点: {diag['path_missing_mapping']}")
        print(f"路径为空条目: {diag['path_empty']}")
        print(f"域内子图不连通: {diag['disconnected_in_domain_graph']}")
        print(f"域内连通但 top-k 无纯域内路径: {diag['connected_but_no_domain_path_in_topk']}")

        if diag_samples["empty_in_micro_paths"]:
            print(f"\n样例 - 原始微观 path 为空 (域, pair): {diag_samples['empty_in_micro_paths']}")
        if diag_samples["empty_after_filter"]:
            print(f"样例 - 过滤后为空 (域, pair): {diag_samples['empty_after_filter']}")
        if diag_samples["disconnected_in_domain_graph"]:
            print(f"样例 - 域内不连通 (域, pair): {diag_samples['disconnected_in_domain_graph']}")
        if diag_samples["connected_but_no_domain_path_in_topk"]:
            print(f"样例 - 域内连通但 top-k 无纯域内路径 (域, pair): {diag_samples['connected_but_no_domain_path_in_topk']}")


if __name__ == "__main__":
    main()
