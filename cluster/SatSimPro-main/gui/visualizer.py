import os
import time
from PySide6.QtWidgets import QWidget, QVBoxLayout, QApplication
from PySide6.QtCore import Signal, Qt
from pyvistaqt import QtInteractor
import pyvista as pv
import numpy as np
import vtk

class Visualizer(QWidget):
    satellite_picked = Signal(int) 
    routes_updated = Signal(list) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        # 设置焦点策略
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter)
        
        self.cached_sat_pos = np.array([])
        self.cached_gs_pos = np.array([])
        self.cached_isl = np.array([])
        self.cached_gsl = np.array([])
        self.cached_path_lines = np.array([])
        
        self.sat_actor = None 
        self.gs_actor = None
        self.isl_actor = None
        self.gsl_actor = None
        self.path_actor = None
        
        self.confirmed_paths = [] 
        self.pending_source = -1
        
        self.last_click_time = 0 
        self._init_scene()
        
        self.plotter.interactor.AddObserver("LeftButtonPressEvent", self._on_left_click)
        self.plotter.interactor.AddObserver("RightButtonPressEvent", self._on_right_click)

    def _init_scene(self):
        self.plotter.set_background('black')
        tex_path = "assets/earth.jpg"
        sphere = pv.Sphere(radius=6371, theta_resolution=120, phi_resolution=120)
        try: sphere = sphere.texture_map_to_sphere()
        except: pass
        texture_loaded = False
        if os.path.exists(tex_path):
            try:
                texture = pv.read_texture(tex_path)
                self.plotter.add_mesh(sphere, texture=texture, smooth_shading=True, pickable=False)
                texture_loaded = True
            except: pass
        if not texture_loaded:
            self.plotter.add_mesh(sphere, color='blue', style='wireframe', opacity=0.3, pickable=False)
        self.plotter.add_axes()
        self.plotter.camera.position = (20000, 0, 0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            if self.confirmed_paths:
                print("Delete Key Pressed: Clearing ALL paths")
                self.confirmed_paths = []
                self.pending_source = -1
                self.routes_updated.emit([])
                self._render_frame()
        else:
            super().keyPressEvent(event)

    def _pick_satellite(self, x, y):
        if len(self.cached_sat_pos) == 0: return -1, float('inf')

        renderer = self.plotter.renderer
        threshold = 25.0 
        closest_idx = -1
        min_dist = float('inf')

        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        
        for i, pos in enumerate(self.cached_sat_pos):
            coordinate.SetValue(pos)
            display_coord = coordinate.GetComputedDisplayValue(renderer)
            dx = display_coord[0] - x
            dy = display_coord[1] - y
            if abs(dx) > threshold or abs(dy) > threshold: continue
            dist = (dx*dx + dy*dy)**0.5
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        if min_dist < threshold:
            return closest_idx, min_dist
        return -1, float('inf')

    def _on_left_click(self, obj, event):
        curr_time = time.time()
        if curr_time - self.last_click_time < 0.2: return
        self.last_click_time = curr_time

        click_pos = self.plotter.interactor.GetEventPosition()
        idx, _ = self._pick_satellite(click_pos[0], click_pos[1])
        
        modifiers = QApplication.queryKeyboardModifiers()
        is_ctrl_pressed = (modifiers & Qt.ControlModifier)

        if idx != -1:
            print(f"Left Click: Sat {idx}")
            self.satellite_picked.emit(idx)

            if is_ctrl_pressed:
                if self.pending_source != -1:
                    if idx != self.pending_source:
                        new_pair = (self.pending_source, idx)
                        if new_pair not in self.confirmed_paths:
                            self.confirmed_paths.append(new_pair)
                            print(f"Path Added: {new_pair}")
                            self.routes_updated.emit(self.confirmed_paths)
                        self.pending_source = -1 
                    else:
                        print("Cannot route to self")
                else:
                    print("Select a source node first (Click), then Ctrl+Click target.")
            else:
                self.pending_source = idx
                print(f"Pending Source: {idx}")
        else:
            pass

        self._render_frame()

    def _on_right_click(self, obj, event):
        click_pos = self.plotter.interactor.GetEventPosition()
        idx, _ = self._pick_satellite(click_pos[0], click_pos[1])
        
        if idx != -1:
            to_remove = []
            for pair in self.confirmed_paths:
                if idx in pair:
                    to_remove.append(pair)
            
            if to_remove:
                for p in to_remove:
                    self.confirmed_paths.remove(p)
                    print(f"Right Click: Removed path {p}")
                self.routes_updated.emit(self.confirmed_paths)
            
            if self.pending_source == idx:
                self.pending_source = -1
                print("Right Click: Cleared pending source")
                
            self._render_frame()

    def update_scene(self, sat_positions, gs_positions, isl_lines, gsl_lines, highlight_lines=None):
        self.cached_sat_pos = sat_positions
        self.cached_gs_pos = gs_positions
        self.cached_isl = isl_lines
        self.cached_gsl = gsl_lines
        if highlight_lines is not None:
            self.cached_path_lines = highlight_lines
        else:
            self.cached_path_lines = np.array([])
            
        self._render_frame()

    def _render_frame(self):
        if len(self.cached_sat_pos) == 0: return
        sats = self.cached_sat_pos
        gss = self.cached_gs_pos
        
        # 1. 颜色渲染
        colors = np.tile([1.0, 1.0, 1.0], (len(sats), 1))
        
        for (src, tgt) in self.confirmed_paths:
            if src < len(sats): colors[src] = [0.0, 1.0, 0.0]   
            if tgt < len(sats): colors[tgt] = [1.0, 0.0, 1.0]   
        
        if self.pending_source != -1 and self.pending_source < len(sats):
             colors[self.pending_source] = [1.0, 1.0, 0.0]      

        if self.sat_actor:
            self.sat_actor.mapper.dataset.points = sats
            self.sat_actor.mapper.dataset.point_data['colors'] = colors
        else:
            cloud = pv.PolyData(sats)
            cloud.point_data['colors'] = colors
            self.sat_actor = self.plotter.add_mesh(
                cloud, scalars='colors', rgb=True, point_size=12, 
                render_points_as_spheres=True, pickable=True
            )

        # 2. 渲染高亮路径
        if len(self.cached_path_lines) > 0:
            if self.path_actor:
                self.path_actor.mapper.dataset.points = sats 
                self.path_actor.mapper.dataset.lines = self.cached_path_lines
                self.path_actor.SetVisibility(True)
            else:
                mesh_path = pv.PolyData(sats)
                mesh_path.lines = self.cached_path_lines
                self.path_actor = self.plotter.add_mesh(
                    mesh_path, color='#FFD700', style='wireframe', 
                    line_width=4, render_lines_as_tubes=True, pickable=False
                )
        else:
            if self.path_actor: self.path_actor.SetVisibility(False)

        # 3. 渲染普通 ISL 
        if len(self.cached_isl) > 0:
            if len(gss) > 0: merged = np.vstack((sats, gss))
            else: merged = sats

            if self.isl_actor:
                self.isl_actor.mapper.dataset.points = merged
                self.isl_actor.mapper.dataset.lines = self.cached_isl
                self.isl_actor.SetVisibility(True)
            else:
                mesh = pv.PolyData(merged)
                mesh.lines = self.cached_isl
                self.isl_actor = self.plotter.add_mesh(
                    mesh, 
                    color='#00FF00',      
                    style='wireframe', 
                    line_width=2,         
                    opacity=0.6,          
                    pickable=False 
                )
        else:
            if self.isl_actor: self.isl_actor.SetVisibility(False)

        # 4. GSL
        if len(gss) > 0: merged = np.vstack((sats, gss))
        else: merged = sats

        if len(gss) > 0 and not self.gs_actor: 
            self.gs_actor = self.plotter.add_mesh(pv.PolyData(gss), color='#00FFFF', point_size=15, render_points_as_spheres=True, pickable=False)

        if len(self.cached_gsl) > 0:
            if self.gsl_actor:
                self.gsl_actor.mapper.dataset.points = merged
                self.gsl_actor.mapper.dataset.lines = self.cached_gsl
                self.gsl_actor.SetVisibility(True)
            else:
                mesh_g = pv.PolyData(merged)
                mesh_g.lines = self.cached_gsl
                self.gsl_actor = self.plotter.add_mesh(mesh_g, color='cyan', style='wireframe', line_width=1.5, opacity=0.8, pickable=False)
        else:
            if self.gsl_actor: self.gsl_actor.SetVisibility(False)

        self.plotter.render()