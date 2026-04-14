# import networkx as nx
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# # 创建有向图
# G = nx.DiGraph()
# G.add_nodes_from(range(12))
#
# edges = [
#     (0,1,9.92), (1,0,9.92), (1,4,9.92), (1,5,2.48), (1,11,9.92),
#     (4,1,9.92), (4,6,9.92), (4,7,9.92),
#     (5,1,2.48), (5,2,9.92), (5,6,9.92),
#     (6,3,9.92), (6,4,9.92), (6,5,9.92),
#     (7,4,9.92), (7,9,9.92),
#     (8,2,9.92), (8,11,9.92),
#     (9,3,9.92), (9,7,9.92), (9,10,9.92),
#     (10,3,9.92), (10,9,9.92),
#     (11,8,9.92)
# ]
#
# for src, tgt, cap in edges:
#     G.add_edge(src, tgt, capacity=cap)
#
# # 可视化设置
# pos = nx.spring_layout(G, seed=42)
# edge_colors = ['red' if cap == 2.48 else 'green' for u, v, cap in G.edges(data='capacity')]
#
# nx.draw(G, pos, with_labels=True, node_color='lightblue',
#         edge_color=edge_colors, width=2, arrowsize=20)
#
# # 添加容量标签
# edge_labels = {(u, v): f"{G[u][v]['capacity']}G" for u, v in G.edges()}
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
#
# # 添加图例
# legend_elements = [
#     mpatches.Patch(color='red', label='Low Capacity (2.48 Gbps)'),
#     mpatches.Patch(color='green', label='High Capacity (9.92 Gbps)')
# ]
# plt.legend(handles=legend_elements, loc='upper right')
#
# plt.title("Abilene Network Topology with Capacity Labels")
# plt.savefig('abilene_topo_with_labels.png')
# plt.close()


import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 定义拓扑结构
topo = {
    "directed": True,
    "multigraph": False,
    "graph": {},
    "nodes": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}, {"id": 6},
              {"id": 7}, {"id": 8}, {"id": 9}, {"id": 10}, {"id": 11}, {"id": 12}, {"id": 13},
              {"id": 14}, {"id": 15}, {"id": 16}, {"id": 17}, {"id": 18}, {"id": 19}, {"id": 20},
              {"id": 21}],
    "links": [
        {"source": 0, "target": 2, "capacity": 10.0},
        {"source": 0, "target": 4, "capacity": 10.0},
        {"source": 0, "target": 9, "capacity": 10.0},
        {"source": 0, "target": 15, "capacity": 2.4},
        {"source": 0, "target": 19, "capacity": 2.4},
        {"source": 1, "target": 6, "capacity": 2.4},
        {"source": 1, "target": 13, "capacity": 0.155},
        {"source": 1, "target": 14, "capacity": 2.4},
        {"source": 2, "target": 0, "capacity": 10.0},
        {"source": 2, "target": 6, "capacity": 10.0},
        {"source": 2, "target": 12, "capacity": 10.0},
        {"source": 3, "target": 4, "capacity": 10.0},
        {"source": 3, "target": 16, "capacity": 2.4},
        {"source": 3, "target": 20, "capacity": 2.4},
        {"source": 4, "target": 0, "capacity": 10.0},
        {"source": 4, "target": 3, "capacity": 10.0},
        {"source": 4, "target": 6, "capacity": 10.0},
        {"source": 4, "target": 7, "capacity": 2.4},
        {"source": 4, "target": 10, "capacity": 2.4},
        {"source": 4, "target": 12, "capacity": 10.0},
        {"source": 4, "target": 14, "capacity": 10.0},
        {"source": 4, "target": 18, "capacity": 10.0},
        {"source": 5, "target": 6, "capacity": 10.0},
        {"source": 5, "target": 12, "capacity": 10.0},
        {"source": 5, "target": 17, "capacity": 2.4},
        {"source": 6, "target": 1, "capacity": 2.4},
        {"source": 6, "target": 2, "capacity": 10.0},
        {"source": 6, "target": 4, "capacity": 10.0},
        {"source": 6, "target": 5, "capacity": 10.0},
        {"source": 6, "target": 13, "capacity": 0.155},
        {"source": 6, "target": 21, "capacity": 10.0},
        {"source": 7, "target": 4, "capacity": 2.4},
        {"source": 7, "target": 12, "capacity": 2.4},
        {"source": 8, "target": 9, "capacity": 2.4},
        {"source": 8, "target": 19, "capacity": 2.4},
        {"source": 9, "target": 0, "capacity": 10.0},
        {"source": 9, "target": 8, "capacity": 2.4},
        {"source": 9, "target": 20, "capacity": 2.4},
        {"source": 10, "target": 4, "capacity": 2.4},
        {"source": 10, "target": 21, "capacity": 2.4},
        {"source": 11, "target": 12, "capacity": 0.155},
        {"source": 11, "target": 14, "capacity": 0.155},
        {"source": 12, "target": 2, "capacity": 10.0},
        {"source": 12, "target": 4, "capacity": 10.0},
        {"source": 12, "target": 5, "capacity": 10.0},
        {"source": 12, "target": 7, "capacity": 2.4},
        {"source": 12, "target": 11, "capacity": 0.155},
        {"source": 13, "target": 1, "capacity": 0.155},
        {"source": 13, "target": 6, "capacity": 0.155},
        {"source": 14, "target": 1, "capacity": 2.4},
        {"source": 14, "target": 4, "capacity": 10.0},
        {"source": 14, "target": 11, "capacity": 0.155},
        {"source": 14, "target": 21, "capacity": 10.0},
        {"source": 15, "target": 0, "capacity": 2.4},
        {"source": 15, "target": 21, "capacity": 2.4},
        {"source": 16, "target": 3, "capacity": 2.4},
        {"source": 16, "target": 18, "capacity": 10.0},
        {"source": 17, "target": 5, "capacity": 2.4},
        {"source": 17, "target": 21, "capacity": 2.4},
        {"source": 18, "target": 4, "capacity": 10.0},
        {"source": 18, "target": 16, "capacity": 10.0},
        {"source": 18, "target": 21, "capacity": 10.0},
        {"source": 19, "target": 0, "capacity": 2.4},
        {"source": 19, "target": 8, "capacity": 2.4},
        {"source": 20, "target": 3, "capacity": 2.4},
        {"source": 20, "target": 9, "capacity": 2.4},
        {"source": 21, "target": 6, "capacity": 10.0},
        {"source": 21, "target": 10, "capacity": 2.4},
        {"source": 21, "target": 14, "capacity": 10.0},
        {"source": 21, "target": 15, "capacity": 2.4},
        {"source": 21, "target": 17, "capacity": 2.4},
        {"source": 21, "target": 18, "capacity": 10.0}
    ]
}

# 创建有向图
G = nx.DiGraph()

# 添加节点
for node in topo["nodes"]:
    G.add_node(node["id"])

# 添加边和容量
for link in topo["links"]:
    G.add_edge(link["source"], link["target"], capacity=link["capacity"])

# 可视化设置
pos = nx.spring_layout(G, seed=42)
edge_colors = []
for u, v, data in G.edges(data=True):
    capacity = data["capacity"]
    if capacity == 0.155:
        edge_colors.append('blue')
    elif capacity == 2.4:
        edge_colors.append('red')
    else:
        edge_colors.append('green')

nx.draw(G, pos, with_labels=True, node_color='lightblue',
        edge_color=edge_colors, width=2, arrowsize=20)

# 添加容量标签
edge_labels = {(u, v): f"{G[u][v]['capacity']}G" for u, v in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# 添加图例
legend_elements = [
    mpatches.Patch(color='blue', label='Lowest Capacity (0.155 Gbps)'),
    mpatches.Patch(color='red', label='Low Capacity (2.4 Gbps)'),
    mpatches.Patch(color='green', label='High Capacity (10.0 Gbps)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.title("Topology Visualization with Capacity Labels")
plt.savefig('geant_topo_with_labels.png')
plt.close()
