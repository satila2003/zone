import os
import numpy as np
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QSplitter, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QLineEdit, QLabel, QPushButton, 
                               QFileDialog, QMessageBox, QDialog, QFormLayout, 
                               QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, 
                               QStyledItemDelegate, QStyle, QDialogButtonBox,
                               QInputDialog)
from PySide6.QtCore import QTimer, Qt, QRect
from PySide6.QtGui import QColor, QPainter, QAction, QFont

from .visualizer import Visualizer
from core.calculator import OrbitCalculator
from core.exporter import DataExporter
from core.strategies import GridStarStrategy, GridDeltaStrategy

DARK_THEME = """
QMainWindow, QWidget, QDialog { background-color: #1e1e1e; color: #cccccc; font-family: "Segoe UI", sans-serif; font-size: 13px; }
QTableWidget { background-color: #252526; border: 1px solid #333333; gridline-color: #3e3e42; outline: none; alternate-background-color: #1e1e1e;}
QHeaderView::section { background-color: #2d2d2d; border: 1px solid #3e3e42; padding: 4px; font-weight: bold;}
QTableWidget::item { border-bottom: 1px solid #333333; }
QTableWidget::item:hover { background-color: #2a2d2f; }
QTableWidget::item:selected { background-color: #094771; }
QLineEdit { background-color: #333333; border: 1px solid #444444; padding: 4px 8px; color: #fff; border-radius: 12px;}
QLineEdit:focus { border: 1px solid #007acc; background-color: #1e1e1e;}
QSpinBox, QDoubleSpinBox, QComboBox { background-color: #3c3c3c; border: 1px solid #555; padding: 3px; color: #fff; border-radius: 3px;}
QPushButton { background-color: #3c3c3c; border: 1px solid #555; padding: 5px 15px; color: #fff; border-radius: 3px;}
QPushButton:hover { background-color: #4c4c4c; }
QScrollBar:vertical { border: none; background: #1e1e1e; width: 8px; margin: 0px; }
QScrollBar::handle:vertical { background: #555555; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #777777; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { border: none; background: #1e1e1e; height: 8px; margin: 0px; }
QScrollBar::handle:horizontal { background: #555555; border-radius: 4px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #777777; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""

class LatencyDelegate(QStyledItemDelegate):
    def __init__(self, max_latency=25.0, parent=None):
        super().__init__(parent)
        self.max_latency = max_latency
        
    def paint(self, painter, option, index):
        try: latency = float(index.data(Qt.EditRole))
        except: return super().paint(painter, option, index)
        
        ratio = min(max(latency / self.max_latency, 0.0), 1.0)
        color = QColor(int(255 * ratio), int(200 * (1.0 - ratio)), 60, 200) 
        
        painter.save()
        painter.setPen(Qt.NoPen)
        if option.state & QStyle.State_Selected: painter.fillRect(option.rect, QColor("#094771"))
        else: painter.fillRect(option.rect, QColor("#252526") if index.row() % 2 == 1 else QColor("#1e1e1e"))
        
        bar_rect = QRect(option.rect.x() + 4, option.rect.y() + 4, int((option.rect.width() - 8) * ratio), option.rect.height() - 8)
        painter.setBrush(color); painter.drawRoundedRect(bar_rect, 4, 4)
        
        text = f"{latency:.4f} ms"
        painter.setPen(QColor(0, 0, 0, 150)); painter.drawText(option.rect.translated(1, 1), Qt.AlignCenter, text)
        painter.setPen(QColor("#ffffff")); painter.drawText(option.rect, Qt.AlignCenter, text)
        painter.restore()

class WalkerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Walker Constellation"); layout = QFormLayout(self)
        self.spin_t = QSpinBox(); self.spin_t.setRange(1, 10000); self.spin_t.setValue(1584)
        self.spin_p = QSpinBox(); self.spin_p.setRange(1, 1000); self.spin_p.setValue(72)
        self.spin_f = QSpinBox(); self.spin_f.setRange(0, 1000); self.spin_f.setValue(39)
        self.spin_alt = QDoubleSpinBox(); self.spin_alt.setRange(100, 20000); self.spin_alt.setValue(550.0); self.spin_alt.setSuffix(" km")
        self.spin_inc = QDoubleSpinBox(); self.spin_inc.setRange(0, 180); self.spin_inc.setValue(53.0); self.spin_inc.setSuffix(" °")
        layout.addRow("Total Satellites (T):", self.spin_t); layout.addRow("Orbital Planes (P):", self.spin_p); layout.addRow("Phase Factor (F):", self.spin_f)
        layout.addRow("Altitude:", self.spin_alt); layout.addRow("Inclination:", self.spin_inc)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

class TopologyDialog(QDialog):
    def __init__(self, current_strategy_idx, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Network Topology Settings"); self.setMinimumWidth(350); layout = QVBoxLayout(self)
        
        self.combo_strat = QComboBox()
        self.combo_strat.addItems(["+Grid（Star）", "+Grid（Delta）"])
        self.combo_strat.setCurrentIndex(current_strategy_idx)
        layout.addWidget(QLabel("Connection Strategy:")); layout.addWidget(self.combo_strat)
        
        self.panel_mesh = QWidget(); l_mesh = QFormLayout(self.panel_mesh)
        self.spin_plane_tol = QDoubleSpinBox(); self.spin_plane_tol.setValue(6.0); self.spin_plane_tol.setSuffix(" °")
        self.spin_intra = QSpinBox(); self.spin_intra.setRange(0, 10000); self.spin_intra.setValue(5000); self.spin_intra.setSuffix(" km")
        self.spin_inter = QSpinBox(); self.spin_inter.setRange(0, 10000); self.spin_inter.setValue(5000); self.spin_inter.setSuffix(" km")
        self.chk_polar = QCheckBox("Enable Polar Cut (极地熔断)"); self.chk_polar.setChecked(True)
        self.spin_polar_lat = QDoubleSpinBox(); self.spin_polar_lat.setRange(0, 90); self.spin_polar_lat.setValue(70.0); self.spin_polar_lat.setSuffix(" °")
        l_mesh.addRow("Plane Tolerance:", self.spin_plane_tol); l_mesh.addRow("Max Intra-plane Dist:", self.spin_intra); l_mesh.addRow("Max Inter-plane Dist:", self.spin_inter)
        l_mesh.addRow(self.chk_polar); l_mesh.addRow("Cutoff Latitude:", self.spin_polar_lat)
        
        self.panel_delta = QWidget(); l_delta = QFormLayout(self.panel_delta)
        self.spin_delta_lat = QDoubleSpinBox(); self.spin_delta_lat.setRange(0, 90); self.spin_delta_lat.setValue(70.0); self.spin_delta_lat.setSuffix(" °"); l_delta.addRow("Turnaround Latitude:", self.spin_delta_lat)
        
        layout.addWidget(self.panel_mesh); layout.addWidget(self.panel_delta)
        self.combo_strat.currentIndexChanged.connect(self.update_panels); self.update_panels(current_strategy_idx)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); layout.addWidget(btns)

    def update_panels(self, idx):
        self.panel_mesh.setVisible(idx == 0)
        self.panel_delta.setVisible(idx == 1)

class ExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Data Settings"); layout = QFormLayout(self)
        self.btn_path = QPushButton("Select Directory..."); self.lbl_path = QLabel("Not Selected"); self.path = ""
        self.btn_path.clicked.connect(lambda: [setattr(self, 'path', d), self.lbl_path.setText(os.path.basename(d))] if (d := QFileDialog.getExistingDirectory(self)) else None)
        self.spin_duration = QSpinBox(); self.spin_duration.setRange(10, 36000); self.spin_duration.setValue(60); self.spin_duration.setSuffix(" s")
        layout.addRow("Save To:", self.btn_path); layout.addRow("", self.lbl_path); layout.addRow("Export Duration:", self.spin_duration)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel); btns.accepted.connect(self.accept); btns.rejected.connect(self.reject); layout.addRow(btns)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Satellite Network Simulation Pro"); self.resize(1200, 900); self.setStyleSheet(DARK_THEME)
        self.calculator = OrbitCalculator(); self.exporter = DataExporter()
        self.strategy = GridStarStrategy(); self.strategy_idx = 0
        self.step_size = 1.0; self.is_playing = False; self.current_time = datetime.utcnow()
        self.all_links_data = []
        self.selected_link_pairs = set() 
        
        self._init_ui(); self._init_menu()
        self.timer = QTimer(); self.timer.timeout.connect(self.loop)

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central); layout = QVBoxLayout(central); layout.setContentsMargins(0,0,0,0)
        splitter = QSplitter(Qt.Vertical)
        self.visualizer = Visualizer()
        splitter.addWidget(self.visualizer)
        
        bottom = QWidget(); b_layout = QVBoxLayout(bottom)
        tool_layout = QHBoxLayout()
        self.txt_search = QLineEdit(); self.txt_search.setPlaceholderText("Filter Name (e.g. '1203' or '1203-1204')..."); self.txt_search.setFixedWidth(300)
        self.txt_search.textChanged.connect(self.refresh_table_view)
        self.lbl_stats = QLabel("Active Links: 0")
        tool_layout.addWidget(self.txt_search); tool_layout.addStretch(); tool_layout.addWidget(self.lbl_stats); b_layout.addLayout(tool_layout)

        self.table = QTableWidget(0, 4)
        table_font = QFont("Consolas", 10); table_font.setStyleHint(QFont.Monospace); self.table.setFont(table_font)
        self.table.setAlternatingRowColors(True)
        self.table.setHorizontalHeaderLabels(["Link ID", "Source", "Target", "Latency (ms)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setItemDelegateForColumn(3, LatencyDelegate(25.0)); self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection) 
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        b_layout.addWidget(self.table)
        
        self.page_size = 15; self.current_page = 1
        page_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◄ Prev"); self.btn_prev.clicked.connect(lambda: self.change_page(-1))
        self.lbl_page = QLabel("Page 1/1")
        self.btn_next = QPushButton("Next ►"); self.btn_next.clicked.connect(lambda: self.change_page(1))
        page_layout.addStretch(); page_layout.addWidget(self.btn_prev); page_layout.addWidget(self.lbl_page); page_layout.addWidget(self.btn_next); page_layout.addStretch()
        b_layout.addLayout(page_layout)
        splitter.addWidget(bottom); splitter.setStretchFactor(0, 6); splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

    def _init_menu(self):
        mb = self.menuBar(); m_data = mb.addMenu("Data")
        m_data.addAction("Load TLE File...", self.load_tle_file); m_data.addAction("Generate Walker...", self.open_walker_gen)
        m_topo = mb.addMenu("Topology"); m_topo.addAction("Settings...", self.open_topology_settings)
        m_sim = mb.addMenu("Simulation")
        self.act_play = QAction("Start", self); self.act_play.triggered.connect(self.toggle_sim); self.act_play.setEnabled(False)
        m_sim.addAction(self.act_play); m_sim.addAction("Set Step Size...", self.open_step_settings)
        m_sim.addSeparator(); m_sim.addAction("Export Simulation Data...", self.open_export_settings)

    def on_table_selection(self):
        self.selected_link_pairs = set()
        selected_rows = set(item.row() for item in self.table.selectedItems())
        for row in selected_rows:
            try:
                src = self.table.item(row, 1).data(Qt.UserRole)
                tgt = self.table.item(row, 2).data(Qt.UserRole)
                if src is not None and tgt is not None: self.selected_link_pairs.add((src, tgt))
            except: pass
        if not self.is_playing: self.loop(advance=False)

    def open_walker_gen(self):
        dlg = WalkerDialog(self)
        if dlg.exec() == QDialog.Accepted and (c := self.calculator.generate_walker(dlg.spin_t.value(), dlg.spin_p.value(), dlg.spin_f.value(), dlg.spin_alt.value(), dlg.spin_inc.value(), self.current_time)):
            self.act_play.setEnabled(True); self.loop(False)

    def open_topology_settings(self):
        dlg = TopologyDialog(self.strategy_idx, self)
        if dlg.exec() == QDialog.Accepted:
            idx = dlg.combo_strat.currentIndex(); self.strategy_idx = idx
            if idx == 0: self.strategy = GridStarStrategy(plane_tolerance=dlg.spin_plane_tol.value(), max_intra_dist=dlg.spin_intra.value(), max_inter_dist=dlg.spin_inter.value(), enable_polar_cut=dlg.chk_polar.isChecked(), polar_cut_lat=dlg.spin_polar_lat.value())
            elif idx == 1: self.strategy = GridDeltaStrategy(turnaround_lat=dlg.spin_delta_lat.value())
            self.loop(False)

    def open_step_settings(self):
        val, ok = QInputDialog.getDouble(self, "Set Step Size", "Enter simulation step size in seconds:", self.step_size, 0.1, 3600.0, 1)
        if ok: self.step_size = val

    def open_export_settings(self):
        dlg = ExportDialog(self)
        if dlg.exec() == QDialog.Accepted and dlg.path:
            if self.exporter.start(dlg.path, self.current_time, dlg.spin_duration.value())[0]: QMessageBox.information(self, "Export", "Export started.")

    def load_tle_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Open TLE", "", "Files (*.txt *.tle)")
        if f:
            with open(f, 'r') as file:
                if self.calculator.load_tle_data(file.read()): self.act_play.setEnabled(True); self.loop(False)

    def toggle_sim(self):
        self.is_playing = not self.is_playing
        self.act_play.setText("Pause" if self.is_playing else "Start")
        if self.is_playing: self.timer.start(100)
        else: self.timer.stop()

    def change_page(self, d):
        self.current_page += d; self.refresh_table_view()

    def refresh_table_view(self):
        search = self.txt_search.text().strip()
        filtered = [r for r in self.all_links_data if not search or (("-" in search and search in [f"{r['src_name']}-{r['tgt_name']}", f"{r['tgt_name']}-{r['src_name']}"]) or search in (r['src_name'], r['tgt_name']))]
        
        total_p = max(1, (len(filtered) + self.page_size - 1) // self.page_size)
        self.current_page = max(1, min(self.current_page, total_p))
        self.lbl_page.setText(f"Page {self.current_page} / {total_p}"); self.lbl_stats.setText(f"Total Active ISLs: {len(self.all_links_data)}")

        self.table.blockSignals(True); self.table.setUpdatesEnabled(False); self.table.setSortingEnabled(False)
        self.table.setRowCount(len(page_data := filtered[(self.current_page - 1) * self.page_size : self.current_page * self.page_size]))
        
        for i, d in enumerate(page_data):
            it_id = QTableWidgetItem(); it_id.setData(Qt.EditRole, d['id'])
            it_src = QTableWidgetItem(d['src_name']); it_src.setData(Qt.UserRole, d['src'])
            it_tgt = QTableWidgetItem(d['tgt_name']); it_tgt.setData(Qt.UserRole, d['tgt'])
            it_lat = QTableWidgetItem(); it_lat.setData(Qt.EditRole, d['latency'])
            
            for it in (it_id, it_src, it_tgt, it_lat): it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, it_id); self.table.setItem(i, 1, it_src); self.table.setItem(i, 2, it_tgt); self.table.setItem(i, 3, it_lat)
            if (d['src'], d['tgt']) in self.selected_link_pairs or (d['tgt'], d['src']) in self.selected_link_pairs: self.table.selectRow(i)
            
        self.table.setSortingEnabled(True); self.table.setUpdatesEnabled(True); self.table.blockSignals(False)

    def loop(self, advance=True):
        if advance: self.current_time += timedelta(seconds=self.step_size)
        self.calculator.propagate(self.current_time)
        sats = np.array([s.position for s in self.calculator.satellites], dtype=np.float32)
        
        isl, self.all_links_data = self.strategy.compute_links(self.calculator.satellites)
        
        if self.exporter.is_active:
            self.exporter.record_frame(self.current_time, self.calculator.satellites, self.all_links_data)
            if self.current_time >= self.exporter.end_time_ref: self.exporter.stop()

        self.refresh_table_view()
        
        hl_lines = []
        for src, tgt in self.selected_link_pairs: hl_lines.extend([2, src, tgt])
        hl_array = np.array(hl_lines, dtype=np.int32) if hl_lines else np.empty((0,), dtype=np.int32)
        
        self.visualizer.update_scene(sats, isl, highlight_lines=hl_array)