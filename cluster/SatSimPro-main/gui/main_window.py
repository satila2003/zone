import os
import numpy as np
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QMainWindow, QDockWidget, QPushButton, QVBoxLayout, 
                               QWidget, QLabel, QGroupBox, QSplitter, QFileDialog, 
                               QComboBox, QMessageBox, QCheckBox, QSpinBox, 
                               QHBoxLayout, QDoubleSpinBox, QProgressBar, QTextEdit, 
                               QFrame, QScrollArea, QSizePolicy)
from PySide6.QtCore import QTimer, Qt, QSize
from PySide6.QtGui import QColor, QPalette, QFont

# 假设后端模块保持不变
from .visualizer import Visualizer
from .charts import LinkStatsChart
from core.calculator import OrbitCalculator
from core.exporter import DataExporter
from core.strategies import DistanceStrategy, StarlinkMeshStrategy
from core.router import PathFinder

# === 1. Modern Dark Pro 样式表 ===
DARK_THEME_STYLESHEET = """
QMainWindow, QWidget { background-color: #1e1e1e; color: #cccccc; font-family: "Segoe UI", "Roboto", sans-serif; font-size: 13px; }
QScrollArea { border: none; background-color: #1e1e1e; }
QDockWidget { titlebar-close-icon: url(none); titlebar-normal-icon: url(none); border: 1px solid #333333; }
QDockWidget::title { background: #252526; text-align: left; padding-left: 10px; padding-top: 6px; padding-bottom: 6px; font-weight: bold; color: #e0e0e0; }
QSplitter::handle { background-color: #333333; }
QSplitter::handle:hover { background-color: #007acc; }
QFrame#Card { background-color: #252526; border: 1px solid #333333; border-radius: 6px; margin-bottom: 8px; }
QLabel#CardTitle { font-size: 13px; font-weight: bold; color: #569cd6; padding-bottom: 4px; border-bottom: 1px solid #3e3e42; margin-bottom: 8px; }
QPushButton { background-color: #3c3c3c; border: 1px solid #3c3c3c; border-radius: 4px; color: #ffffff; padding: 6px 12px; }
QPushButton:hover { background-color: #4c4c4c; border-color: #555555; }
QPushButton:pressed { background-color: #2a2d2e; }
QPushButton:disabled { background-color: #2d2d2d; color: #666666; border: 1px solid #2d2d2d; }
QPushButton#PrimaryBtn { background-color: #388e3c; border: 1px solid #388e3c; }
QPushButton#PrimaryBtn:hover { background-color: #4caf50; }
QPushButton#PrimaryBtn:pressed { background-color: #2e7d32; }
QPushButton#DangerBtn { background-color: #d32f2f; border: 1px solid #d32f2f; }
QPushButton#DangerBtn:hover { background-color: #f44336; }
QSpinBox, QDoubleSpinBox, QComboBox { background-color: #3c3c3c; border: 1px solid #3c3c3c; border-radius: 3px; color: #f0f0f0; padding: 4px; selection-background-color: #264f78; }
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #007acc; }
QComboBox::drop-down { border: none; width: 20px; }
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { background-color: transparent; border: none; width: 16px; }
QCheckBox { spacing: 8px; color: #cccccc; }
QCheckBox::indicator { width: 16px; height: 16px; background-color: #3c3c3c; border: 1px solid #555555; border-radius: 3px; }
QCheckBox::indicator:checked { background-color: #007acc; border: 1px solid #007acc; }
QTextEdit { background-color: #1e1e1e; border: 1px solid #333333; border-radius: 4px; color: #9cdcfe; font-family: "Consolas", monospace; font-size: 12px; }
QProgressBar { background-color: #2d2d2d; border: none; border-radius: 3px; height: 6px; text-align: center; }
QProgressBar::chunk { background-color: #007acc; border-radius: 3px; }
QScrollBar:vertical { border: none; background: #1e1e1e; width: 10px; margin: 0px; }
QScrollBar::handle:vertical { background: #424242; min-height: 20px; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #686868; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
"""

# === 2. 辅助类：仪表盘卡片 ===
class DashboardCard(QFrame):
    def __init__(self, title, icon_emoji="", parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)
        
        full_title = f"{icon_emoji}  {title}" if icon_emoji else title
        self.lbl_title = QLabel(full_title)
        self.lbl_title.setObjectName("CardTitle")
        self.main_layout.addWidget(self.lbl_title)
        
        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: transparent;") 
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        self.main_layout.addWidget(self.content_widget)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)
    
    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

# === 3. 主窗口 ===
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Network Simulation Pro") 
        self.resize(1400, 950)
        self.setStyleSheet(DARK_THEME_STYLESHEET)

        self.calculator = OrbitCalculator()
        self.exporter = DataExporter()
        self.path_finder = PathFinder() 
        
        self.current_tle_content = None 
        self.is_playing = False
        self.strategy = DistanceStrategy(max_isl_dist=0)
        self.current_time = datetime.utcnow()
        self.gs_coords = np.empty((0, 3)) 
        self.active_paths = []
        self.selected_node = -1
        self.current_isl = np.array([], dtype=np.int32)
        
        # === 状态变量 ===
        self.last_latencies = {}   # { "src->tgt": latency }
        self.last_paths = {}       # { "src->tgt": [id1, id2...] }
        self.handover_counts = {}  # { "src->tgt": 0 }
        
        # === 关键修复：防抖定时器 (解决卡死问题) ===
        self.filter_debounce_timer = QTimer()
        self.filter_debounce_timer.setSingleShot(True) # 只触发一次
        self.filter_debounce_timer.setInterval(400)    # 延迟 400ms
        self.filter_debounce_timer.timeout.connect(self.reapply_filters)

        self.visualizer = Visualizer()
        self.visualizer.satellite_picked.connect(self.on_sat_picked)
        self.visualizer.routes_updated.connect(self.on_routes_updated)
        
        self.setCentralWidget(self.visualizer)
        self._init_control_panel()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)

    def _init_control_panel(self):
        dock = QDockWidget("Mission Control", self)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures) 
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        
        splitter = QSplitter(Qt.Orientation.Vertical)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        control_container = QWidget()
        scroll.setWidget(control_container)
        
        main_layout = QVBoxLayout(control_container)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # --- Card 1: Data Source ---
        card_data = DashboardCard("Data Source", "🛰️")
        
        h_load = QHBoxLayout()
        self.btn_load = QPushButton("Load TLE")
        self.btn_load.setCursor(Qt.PointingHandCursor)
        self.lbl_file_status = QLabel("No TLE Loaded")
        self.lbl_file_status.setStyleSheet("color: #777777; font-style: italic;")
        h_load.addWidget(self.btn_load)
        h_load.addWidget(self.lbl_file_status)
        card_data.addLayout(h_load)
        
        from PySide6.QtWidgets import QGridLayout
        grid_filters = QGridLayout()
        grid_filters.setSpacing(8)

        self.chk_alt = QCheckBox("Alt Filter"); self.chk_alt.setChecked(True)
        self.spin_alt = QSpinBox(); self.spin_alt.setRange(0, 50000); self.spin_alt.setValue(0); self.spin_alt.setSuffix(" km")
        self.spin_tol_alt = QSpinBox(); self.spin_tol_alt.setRange(0, 5000); self.spin_tol_alt.setValue(0); self.spin_tol_alt.setPrefix("± ")

        self.chk_inc = QCheckBox("Inc Filter"); self.chk_inc.setChecked(True)
        self.spin_inc = QDoubleSpinBox(); self.spin_inc.setRange(0, 180); self.spin_inc.setValue(0.0); self.spin_inc.setSuffix(" °")
        self.spin_tol_inc = QDoubleSpinBox(); self.spin_tol_inc.setRange(0, 180); self.spin_tol_inc.setValue(0.0); self.spin_tol_inc.setPrefix("± ")

        grid_filters.addWidget(self.chk_alt, 0, 0)
        grid_filters.addWidget(self.spin_alt, 0, 1)
        grid_filters.addWidget(self.spin_tol_alt, 0, 2)
        grid_filters.addWidget(self.chk_inc, 1, 0)
        grid_filters.addWidget(self.spin_inc, 1, 1)
        grid_filters.addWidget(self.spin_tol_inc, 1, 2)

        card_data.addLayout(grid_filters)
        
        self.btn_load.clicked.connect(self.load_tle_file)
        
        # 连接信号
        self.chk_alt.toggled.connect(self.spin_alt.setEnabled)
        self.chk_alt.toggled.connect(self.trigger_filter_update)
        self.chk_inc.toggled.connect(self.spin_inc.setEnabled)
        self.chk_inc.toggled.connect(self.trigger_filter_update)
        
        self.spin_alt.valueChanged.connect(self.trigger_filter_update)
        self.spin_inc.valueChanged.connect(self.trigger_filter_update)
        self.spin_tol_alt.valueChanged.connect(self.trigger_filter_update)
        self.spin_tol_inc.valueChanged.connect(self.trigger_filter_update)
        
        main_layout.addWidget(card_data)
        
        # --- Card 2: Strategy (Updated with Polar Cut) ---
        card_strat = DashboardCard("Topology", "🌐")
        
        self.combo_strat = QComboBox()
        self.combo_strat.addItems(["Distance Only", "+ Grid (Mesh)", "Ultra Long Range"])
        card_strat.addWidget(self.combo_strat)
        
        self.param_widget = QWidget()
        self.param_widget.setStyleSheet("background: transparent;")
        p_layout = QVBoxLayout(self.param_widget)
        p_layout.setContentsMargins(0,0,0,0)
        p_layout.setSpacing(6)

        def add_param_row(label, widget):
            h = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setMinimumWidth(90)
            lbl.setStyleSheet("color: #aaaaaa;")
            h.addWidget(lbl)
            h.addWidget(widget)
            p_layout.addLayout(h)
            return widget
        
        # [新增] 极地熔断开关
        self.chk_polar_cut = QCheckBox("Enable Polar Cut")
        self.chk_polar_cut.setChecked(False) # 默认不开启
        self.chk_polar_cut.setToolTip("断开高纬度地区的异轨链路（模拟极轨星座特性）")
        p_layout.addWidget(self.chk_polar_cut)
        
        # [新增] 熔断纬度
        self.spin_polar_lat = add_param_row("Cut Latitude:", QDoubleSpinBox())
        self.spin_polar_lat.setRange(0, 90); self.spin_polar_lat.setValue(70.0); self.spin_polar_lat.setSuffix(" °")

        self.spin_plane_tol = add_param_row("Plane Tol:", QDoubleSpinBox())
        self.spin_plane_tol.setRange(0, 180); self.spin_plane_tol.setValue(0.0); self.spin_plane_tol.setSingleStep(0.5); self.spin_plane_tol.setSuffix(" °")

        self.spin_neighbor_tol = add_param_row("Neigh. Tol:", QDoubleSpinBox())
        self.spin_neighbor_tol.setRange(0, 180); self.spin_neighbor_tol.setValue(0.0); self.spin_neighbor_tol.setSuffix(" °")

        self.spin_intra = add_param_row("Intra Dist:", QSpinBox())
        self.spin_intra.setRange(0, 50000); self.spin_intra.setValue(0); self.spin_intra.setSingleStep(100); self.spin_intra.setSuffix(" km")

        self.spin_inter = add_param_row("Inter Dist:", QSpinBox())
        self.spin_inter.setRange(0, 50000); self.spin_inter.setValue(0); self.spin_inter.setSingleStep(100); self.spin_inter.setSuffix(" km")
        
        self.spin_gsl = add_param_row("GSL Radius:", QSpinBox())
        self.spin_gsl.setRange(0, 20000); self.spin_gsl.setValue(0); self.spin_gsl.setSingleStep(100); self.spin_gsl.setSuffix(" km")
        
        card_strat.addWidget(self.param_widget)
        
        self.combo_strat.currentIndexChanged.connect(self.update_strategy_params)
        
        # [新增信号]
        self.chk_polar_cut.toggled.connect(self.update_strategy_params)
        self.chk_polar_cut.toggled.connect(self.spin_polar_lat.setEnabled)
        self.spin_polar_lat.valueChanged.connect(self.update_strategy_params)
        
        self.spin_plane_tol.valueChanged.connect(self.update_strategy_params)
        self.spin_neighbor_tol.valueChanged.connect(self.update_strategy_params)
        self.spin_intra.valueChanged.connect(self.update_strategy_params)
        self.spin_inter.valueChanged.connect(self.update_strategy_params)
        self.spin_gsl.valueChanged.connect(self.update_strategy_params)

        main_layout.addWidget(card_strat)

        # --- Card 3: Monitor (Updated) ---
        card_mon = DashboardCard("Monitor", "📊")
        
        self.combo_monitor = QComboBox()
        self.combo_monitor.addItems([
            "Global: Link Counts", 
            "Path: Latency (ms)", 
            "Path: Hop Count", 
            "Path: Jitter (ms)",
            "Path: Handover Count"
        ])
        card_mon.addWidget(self.combo_monitor)
        
        self.combo_monitor.currentIndexChanged.connect(self.on_chart_mode_changed)
        main_layout.addWidget(card_mon)

        # --- Card 4: Simulation ---
        card_sim = DashboardCard("Simulation", "⏱️")
        
        grid_sim = QGridLayout()
        grid_sim.addWidget(QLabel("Step Size:"), 0, 0)
        self.spin_step = QDoubleSpinBox(); self.spin_step.setRange(0, 3600); self.spin_step.setValue(1.0); self.spin_step.setSuffix(" s")
        grid_sim.addWidget(self.spin_step, 0, 1)
        
        grid_sim.addWidget(QLabel("Exp Duration:"), 1, 0)
        self.spin_duration = QSpinBox(); self.spin_duration.setRange(0, 36000); self.spin_duration.setValue(0); self.spin_duration.setSuffix(" s")
        grid_sim.addWidget(self.spin_duration, 1, 1)
        card_sim.addLayout(grid_sim)
        
        h_btns = QHBoxLayout()
        self.btn_run = QPushButton("Start Simulation")
        self.btn_run.setObjectName("PrimaryBtn")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self.toggle_sim)
        self.btn_run.setEnabled(False)
        
        self.btn_export = QPushButton("Export Data")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_export.clicked.connect(self.start_export)
        self.btn_export.setEnabled(False)

        h_btns.addWidget(self.btn_run)
        h_btns.addWidget(self.btn_export)
        card_sim.addLayout(h_btns)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4) 
        card_sim.addWidget(self.progress_bar)
        
        main_layout.addWidget(card_sim)

        # --- Info Log ---
        self.txt_info = QTextEdit()
        self.txt_info.setReadOnly(True)
        self.txt_info.setMaximumHeight(150)
        self.txt_info.setPlaceholderText(">>> System Ready.\n>>> Waiting for TLE data...")
        main_layout.addWidget(self.txt_info)
        
        main_layout.addStretch() 

        # --- 设置 Dock ---
        splitter.addWidget(scroll)
        
        self.chart = LinkStatsChart()
        chart_container = QWidget()
        chart_container.setStyleSheet("background-color: #1e1e1e;") 
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(5, 5, 5, 5)
        chart_layout.addWidget(self.chart)
        
        splitter.addWidget(chart_container)
        splitter.setStretchFactor(0, 5) 
        splitter.setStretchFactor(1, 2) 
        
        dock.setWidget(splitter)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        
        self.update_strategy_params()

    def load_tle_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open TLE", "", "Files (*.txt *.tle)")
        if fname:
            try:
                if self.timer.isActive(): self.toggle_sim()
                with open(fname, 'r') as f: self.current_tle_content = f.read()
                self.reapply_filters()
            except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def trigger_filter_update(self):
        """ 防抖函数：重置计时器，避免频繁计算 """
        self.filter_debounce_timer.start()

    def reapply_filters(self):
        if not self.current_tle_content: return
        try:
            f_alt = self.spin_alt.value() if self.chk_alt.isChecked() else None
            f_inc = self.spin_inc.value() if self.chk_inc.isChecked() else None
            t_alt = self.spin_tol_alt.value()
            t_inc = self.spin_tol_inc.value()
            
            # 耗时操作，现在只有在用户停止输入 400ms 后才会执行
            count = self.calculator.load_tle_data(self.current_tle_content, filter_alt=f_alt, alt_tol=t_alt, filter_inc=f_inc, inc_tol=t_inc)
            
            self.lbl_file_status.setText(f"Active: {count} Sats")
            self.lbl_file_status.setStyleSheet("color: #4caf50; font-weight: bold;") 
            has_sats = count > 0
            self.btn_run.setEnabled(has_sats); self.btn_export.setEnabled(has_sats)
            if has_sats:
                self.update_strategy_params()
                if not self.is_playing: self.loop(advance_time=False)
            else:
                self.visualizer.update_scene(np.empty((0,3)), np.empty((0,3)), np.array([]), np.array([]))
        except Exception as e: print(f"Filter update error: {e}")

    def update_strategy_params(self):
        idx = self.combo_strat.currentIndex()
        
        p_tol = self.spin_plane_tol.value()
        n_tol = self.spin_neighbor_tol.value()
        d_intra = self.spin_intra.value()
        d_inter = self.spin_inter.value()
        gsl_dist = self.spin_gsl.value()
        
        # [新增] 获取熔断参数
        is_cut = self.chk_polar_cut.isChecked()
        cut_lat = self.spin_polar_lat.value()
        
        # =========================================================
        # [修复核心] 每次更新参数前，强制启用整个参数容器
        # 避免从 Ultra Long 切换回来时，容器依然是被禁用的状态
        # =========================================================
        self.param_widget.setEnabled(True) 

        if idx == 0: # Distance Only
            self.strategy = DistanceStrategy(max_isl_dist=d_intra if d_intra > 0 else 2000)
            self.spin_plane_tol.setEnabled(False); self.spin_neighbor_tol.setEnabled(False); self.spin_inter.setEnabled(False)
            self.chk_polar_cut.setEnabled(False); self.spin_polar_lat.setEnabled(False) 
            self.spin_gsl.setEnabled(True)
            
        elif idx == 1: # + Grid
            self.strategy = StarlinkMeshStrategy(
                plane_tolerance=p_tol, 
                max_intra_dist=d_intra, 
                max_inter_dist=d_inter, 
                neighbor_tolerance=n_tol, 
                max_gsl_dist=gsl_dist,
                enable_polar_cut=is_cut, 
                polar_cut_lat=cut_lat    
            )
            self.spin_plane_tol.setEnabled(True); self.spin_neighbor_tol.setEnabled(True); self.spin_inter.setEnabled(True)
            self.chk_polar_cut.setEnabled(True); self.spin_polar_lat.setEnabled(is_cut) # 启用
            self.spin_gsl.setEnabled(True)
            
        elif idx == 2: # Ultra Long
            self.strategy = DistanceStrategy(max_isl_dist=5000)
            # 在这里禁用，下一次循环会在上面被强制启用
            self.param_widget.setEnabled(False)

        if not self.is_playing and self.btn_run.isEnabled(): self.loop(advance_time=False)

    def start_export(self):
        if self.exporter.is_active: return
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if not dir_path: return
        duration_sec = self.spin_duration.value()
        success, msg = self.exporter.start(dir_path, self.current_time, duration_sec)
        if success:
            self.btn_export.setText("Stop Export"); self.btn_export.setObjectName("DangerBtn")
            self.btn_export.setStyleSheet("background-color: #d32f2f; border: 1px solid #d32f2f;") 
            self.btn_run.setEnabled(False); self.input_widgets_enabled(False)
            if not self.timer.isActive(): self.timer.start(50)
        else: QMessageBox.critical(self, "Export Error", msg)

    def stop_export(self):
        self.exporter.stop()
        self.btn_export.setText("Export Data"); self.btn_export.setObjectName("")
        self.btn_export.setStyleSheet("") 
        self.btn_run.setEnabled(True); self.input_widgets_enabled(True); self.progress_bar.setValue(0)
        QMessageBox.information(self, "Finished", "Export done.")

    def input_widgets_enabled(self, val):
        self.param_widget.setEnabled(val); self.btn_load.setEnabled(val)

    def toggle_sim(self):
        if self.is_playing: 
            self.timer.stop()
            self.btn_run.setText("Resume Simulation")
            self.btn_run.setObjectName("")
            self.btn_run.setStyleSheet("")
            self.is_playing = False
        else: 
            self.timer.start(50)
            self.btn_run.setText("Pause Simulation")
            self.btn_run.setObjectName("DangerBtn")
            self.btn_run.setStyleSheet("background-color: #d32f2f; border: 1px solid #d32f2f;")
            self.is_playing = True

    def on_chart_mode_changed(self):
        mode_text = self.combo_monitor.currentText()
        if "Link Counts" in mode_text:
            self.chart.set_mode("COUNT")
        elif "Latency" in mode_text:
            self.chart.set_mode("LATENCY")
        elif "Hop" in mode_text:
            self.chart.set_mode("HOPS")
        elif "Jitter" in mode_text:
            self.chart.set_mode("JITTER")
        elif "Handover" in mode_text:
            self.chart.set_mode("HANDOVER")
            
    def on_sat_picked(self, idx):
        self.selected_node = idx
        if not self.is_playing and self.btn_run.isEnabled():
            self.loop(advance_time=False)

    def on_routes_updated(self, paths):
        self.active_paths = paths
        # 重置历史状态
        self.last_latencies = {}
        self.last_paths = {}
        self.handover_counts = {}
        
        if not paths:
            self.txt_info.setText(">>> Monitoring Standby.\n(Click node to inspect. Ctrl+Click to route.)")
            self.chart.update_dict_data({}) 
            self.selected_node = -1 
        else:
            msg = f">>> Tracking {len(paths)} Active Paths:\n"
            for (u, v) in paths:
                msg += f" [LINK] Sat-{u} <==> Sat-{v}\n"
            self.txt_info.setText(msg)

        if not self.is_playing and self.btn_run.isEnabled():
            self.loop(advance_time=False)

    def loop(self, advance_time=True):
        step_seconds = self.spin_step.value()
        if self.is_playing and step_seconds <= 0.0 and advance_time: step_seconds = 0.1 

        if self.exporter.is_active:
            advance_time = True
            if step_seconds <= 0: step_seconds = 1.0

        if advance_time: 
            self.current_time += timedelta(seconds=step_seconds)
            
        self.calculator.propagate(self.current_time)
        sat_coords = np.array([s.position for s in self.calculator.satellites], dtype=np.float32)
        
        isl, gsl = self.strategy.compute_links(self.calculator.satellites, self.gs_coords)
        self.current_isl = isl
        
        # === 核心逻辑修改：根据下拉菜单计算指标 ===
        chart_data = {}
        highlight_segments = []
        current_mode = self.combo_monitor.currentText()
        
        if len(self.active_paths) > 0:
            self.path_finder.build_graph(self.calculator.satellites, isl)
            info_msg = f"TIME: {self.current_time.strftime('%H:%M:%S')}\n"
            info_msg += "-"*30 + "\n"
            
            for (src, tgt) in self.active_paths:
                label = f"{src}->{tgt}"
                latency, path = self.path_finder.find_shortest_path(src, tgt)
                
                if latency is not None:
                    # 1. 可视化路径高亮
                    if len(path) > 1:
                        for i in range(len(path)-1):
                            highlight_segments.extend([2, path[i], path[i+1]])

                    # 2. 数据计算分支
                    val_to_plot = 0
                    
                    if "Latency" in current_mode:
                        val_to_plot = latency
                        info_msg += f" √ {label}: {latency:.1f}ms\n"
                        
                    elif "Hop" in current_mode:
                        hops = len(path) - 1
                        val_to_plot = hops
                        info_msg += f" √ {label}: {hops} hops\n"
                        
                    elif "Jitter" in current_mode:
                        last_val = self.last_latencies.get(label, latency)
                        jitter = abs(latency - last_val)
                        self.last_latencies[label] = latency # 更新历史
                        val_to_plot = jitter
                        info_msg += f" √ {label}: Jitter {jitter:.2f}ms\n"

                    elif "Handover" in current_mode:
                        prev_path = self.last_paths.get(label, None)
                        # 检测变化: 如果之前有路径且不同，则计数
                        if prev_path is not None and path != prev_path:
                            self.handover_counts[label] = self.handover_counts.get(label, 0) + 1
                        
                        self.last_paths[label] = path # 更新
                        
                        count = self.handover_counts.get(label, 0)
                        val_to_plot = count
                        info_msg += f" √ {label}: Switches {count}\n"
                    
                    else:
                        val_to_plot = latency
                        
                    chart_data[label] = val_to_plot
                    
                else:
                    # 如果断连，保持计数不变
                    if "Handover" in current_mode:
                        chart_data[label] = self.handover_counts.get(label, 0)
                    else:
                        chart_data[label] = 0
                    info_msg += f" X {label}: UNREACHABLE\n"
            self.txt_info.setText(info_msg)
            
        elif self.selected_node != -1:
            count = 0
            for i in range(0, len(isl), 3):
                u = isl[i+1]
                v = isl[i+2]
                if u == self.selected_node or v == self.selected_node:
                    highlight_segments.extend([2, u, v])
                    count += 1
            if "Link Counts" in current_mode:
                self.txt_info.setText(f"TARGET: Sat-{self.selected_node}\nSTATUS: Online\nNEIGHBORS: {count} Active Links")

        else:
            if "Link Counts" in current_mode:
                self.txt_info.setText(f"TIME: {self.current_time.strftime('%H:%M:%S')}\nNETWORK STATUS:\n • ISL Links: {len(isl)//3}\n • GSL Links: {len(gsl)//3}")

        # 如果是全局模式，强制覆盖 chart_data
        if "Link Counts" in current_mode:
            chart_data = {"ISL Count": len(isl)//3, "GSL Count": len(gsl)//3}
        
        self.chart.update_dict_data(chart_data)
        
        hl_array = np.array(highlight_segments, dtype=np.int32) if len(highlight_segments) > 0 else None
        self.visualizer.update_scene(sat_coords, self.gs_coords, isl, gsl, highlight_lines=hl_array)

        if self.exporter.is_active:
            self.exporter.record_frame(self.current_time, self.calculator.satellites, isl, gsl)
            total_sec = (self.exporter.end_time_ref - self.exporter.start_time_ref).total_seconds()
            elapsed = (self.current_time - self.exporter.start_time_ref).total_seconds()
            if total_sec > 0: self.progress_bar.setValue(min(int((elapsed/total_sec)*100), 100))
            if self.current_time >= self.exporter.end_time_ref: self.stop_export()