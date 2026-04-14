# 优化 Starlink 可视化脚本计划

该计划旨在优化 `f:\Py_Project\cluster\visualize_starlink.py`，实现全中文化的提示信息，并通过轨道计算为图中的卫星节点和同轨连线进行色彩区分，直观地展示星链的轨道面结构。

## 1. 导入必要的数学与计算库
- 添加 `import math` 和 `import numpy as np`：用于卫星轨道的数学计算（升交点赤经 RAAN 推算）。
- 添加 `import matplotlib.cm as cm`：用于生成区分不同轨道的颜色映射表。

## 2. 界面与终端信息全中文化
- **配置 Matplotlib 中文字体支持**：
  ```python
  plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
  plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
  ```
- **翻译终端输出**：
  - `Loaded X nodes and Y edges.` -> `成功加载 X 个节点和 Y 条边。`
  - `Visualization successfully saved to...` -> `可视化图像已成功保存至...`
  - `Error occurred:...` -> `发生错误:...`
- **翻译图表标题及标签**：
  - 标题：`Starlink 550 Topology - Step 0000` -> `Starlink 550 卫星拓扑图 - 第 0000 步`
  - X/Y轴标签：`Longitude (deg)` -> `经度 (度)`，`Latitude (deg)` -> `纬度 (度)`
  - 图例标签：`Satellites` -> `卫星节点`

## 3. 实现轨道区分逻辑（基于 RAAN 计算）
- 依据 Starlink 550 壳层参数（倾角 ~53 度，共计 72 个轨道面，间距 5 度）：
  1. 遍历解析后的卫星节点（经度、纬度），计算每个卫星如果处于**升轨**和**降轨**状态下的候选升交点赤经（RAAN）。
  2. 收集所有节点的候选 RAAN（共计 2N 个候选值），并对其按照 5 度（$360/72$）进行取余，通过直方图分析（`np.histogram`）找出最密集的余数，从而确定整体轨道面偏移量。
  3. 再次遍历卫星节点，计算每个候选 RAAN 到理想 72 个轨道面的距离。挑选距离最近的一个作为卫星真实的 RAAN。
  4. 最终将每个卫星映射到一个唯一的轨道面编号（`plane_id`，取值范围 0~71）。

## 4. 优化图表绘制（同轨连线高亮与节点着色）
- **初始化色彩映射**：利用 `cm.get_cmap('hsv', 72)` 为每一个轨道面生成一种唯一颜色。
- **分离异轨与同轨连线**：
  - 遍历原有 `edges`，比较 `src` 与 `dst` 的 `plane_id`：
    - 若 `plane_id` 不同，则归为**异轨连线（Inter-plane）**，将其统一绘制为淡灰色且降低透明度（作为背景网格）。
    - 若 `plane_id` 相同，则归为**同轨连线（Intra-plane）**，将其加入并绘制为该轨道面专属颜色，适当加粗。
  - 保留原有的防止跨越国际日期变更线的超长连线过滤逻辑（`abs(lon1 - lon2) > 180`）。
- **绘制卫星节点**：
  - 通过生成的轨道编号为每个散点单独着色（与同轨连线颜色一致），使每一条轨道在二维平面上清晰呈现正弦曲线特征。

## 5. 代码整合与替换
- 利用 `SearchReplace` 或直接写入新内容的方法，重构整个 `try` 代码块，使其符合上述逻辑并成功保存图像文件 `starlink550_step_0000_vis.png`。
