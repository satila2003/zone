from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np

class LinkStatsChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.plot_widget = pg.PlotWidget(title="Real-time Monitor")
        self.plot_widget.setBackground('#2b2b2b') 
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend(offset=(10, 10))
        self.plot_widget.setLabel('bottom', 'Time Step')
        
        self.layout.addWidget(self.plot_widget)

        self.max_points = 50
        self.ptr = 0
        self.mode = "COUNT" 

        # 动态曲线字典: { 'label_name': {'curve': PlotDataItem, 'data': [y1, y2...], 'x': [x1, x2...]} }
        self.curves = {}
        
        # 预定义一组鲜艳的颜色，用于多路径区分
        self.colors = ['#00FF00', '#FF00FF', '#00FFFF', '#FFFF00', '#FF5555', '#5555FF', '#FFAA00', '#FFFFFF']
        self.color_idx = 0

    def set_mode(self, mode):
        """ 切换模式会清空所有曲线 """
        if self.mode == mode: return
        self.mode = mode
        self.plot_widget.clear()
        self.curves = {} # 重置曲线字典
        self.color_idx = 0
        self.ptr = 0

        if mode == "LATENCY":
            self.plot_widget.setTitle("Multi-Path Latency")
            self.plot_widget.setLabel('left', 'Latency (ms)')
        elif mode == "HOPS":
            self.plot_widget.setTitle("Path Hop Count")
            self.plot_widget.setLabel('left', 'Hops (Count)')
        elif mode == "JITTER":
            self.plot_widget.setTitle("Network Jitter")
            self.plot_widget.setLabel('left', 'Jitter (ms)')
        elif mode == "HANDOVER":
            self.plot_widget.setTitle("Path Switching Count")
            self.plot_widget.setLabel('left', 'Total Switches')
        else: # Default: COUNT
            self.plot_widget.setTitle("Network Statistics")
            self.plot_widget.setLabel('left', 'Count')
            # 预初始化统计模式的两条固定曲线
            self._get_or_create_curve("ISL Count", '#00FF00')
            self._get_or_create_curve("GSL Count", 'cyan')

    def _get_or_create_curve(self, name, color=None):
        """ 内部辅助函数：获取或创建曲线 """
        if name not in self.curves:
            if color is None:
                # 轮询颜色
                color = self.colors[self.color_idx % len(self.colors)]
                self.color_idx += 1
            
            # 创建新曲线
            pen = pg.mkPen(color=color, width=2)
            curve = self.plot_widget.plot(pen=pen, name=name)
            self.curves[name] = {
                'curve': curve,
                'data': [],
                'x': []
            }
        return self.curves[name]

    def update_dict_data(self, data_dict):
        """
        通用数据更新接口
        :param data_dict: 字典 { '曲线名称': 数值, ... }
        """
        self.ptr += 1
        
        # 1. 遍历输入数据，更新现有曲线或创建新曲线
        active_keys = set(data_dict.keys())
        
        for name, value in data_dict.items():
            c_obj = self._get_or_create_curve(name)
            
            # 追加数据
            c_obj['x'].append(self.ptr)
            c_obj['data'].append(value)
            
            # 保持窗口大小
            if len(c_obj['x']) > self.max_points:
                c_obj['x'].pop(0)
                c_obj['data'].pop(0)
                
            # 绘图
            c_obj['curve'].setData(c_obj['x'], c_obj['data'])
            
        # 2. 清理已经不存在的曲线 (例如用户取消了某条路径)
        # 注意：在 COUNT 模式下通常不需要清理，但在路由模式下需要清理掉旧路径的线
        if self.mode != "COUNT":
            existing_keys = list(self.curves.keys())
            for key in existing_keys:
                if key not in active_keys:
                    # 移除曲线
                    self.plot_widget.removeItem(self.curves[key]['curve'])
                    del self.curves[key]