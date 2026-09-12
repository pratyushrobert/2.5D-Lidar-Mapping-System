"""
app.py
======
Real-Time Interactive Dashboard for Adaptive 2.5D LiDAR Mapping.

Features:
- Top-down 2.5D elevation/semantic map view
- Resolution zone boundary visualization
- Side-by-side analytics HUD (memory, cells, FPS, latency)
- Playback controls (play/pause, frame slider, speed)
- 3D point cloud viewer (OpenGL via VisPy / PyQtGraph GL)
"""

import sys
import time
import queue
import numpy as np
from collections import deque
from typing import Optional, List

# PyQt6 / PySide6 / PyQt5 compatibility layer
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QFormLayout,
        QSplitter, QFrame, QStatusBar, QCheckBox, QSpinBox, QDoubleSpinBox
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QColor, QPalette
    QT_VERSION = 6
    def qt_exec(app):
        return app.exec()
except ImportError:
    try:
        from PyQt5.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QFormLayout,
            QSplitter, QFrame, QStatusBar, QCheckBox, QSpinBox, QDoubleSpinBox
        )
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QFont, QColor, QPalette
        QT_VERSION = 5
        def qt_exec(app):
            return app.exec_()
    except ImportError:
        raise ImportError("PyQt6 or PyQt5 is required for the GUI dashboard")

import pyqtgraph as pg
import pyqtgraph.opengl as gl

from data_loader import SyntheticLiDARGenerator, LiDARFrame
from grid_engine import AdaptiveGridEngine, UniformGridEngine
from perception import GeometricPerception
from benchmark import BenchmarkEngine, BenchmarkResult


class LiDARDashboard(QMainWindow):
    """Main interactive dashboard window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive 2.5D LiDAR Mapping Dashboard")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; font-size: 12px; }
            QGroupBox { color: #e0e0e0; font-weight: bold; border: 1px solid #444; 
                       margin-top: 8px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #0d47a1; color: white; border-radius: 4px;
                         padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #1565c0; }
            QPushButton:pressed { background-color: #0a367c; }
            QSlider::groove:horizontal { height: 8px; background: #333; border-radius: 4px; }
            QSlider::handle:horizontal { background: #42a5f5; width: 14px; margin: -3px 0;
                                          border-radius: 7px; }
        """)
        
        # Data pipeline
        self.generator = SyntheticLiDARGenerator(seed=42)
        self.perception = GeometricPerception()
        self.benchmark = BenchmarkEngine(max_range=100.0)
        
        # State
        self.frames: List[LiDARFrame] = []
        self.current_frame_idx = 0
        self.is_playing = False
        self.playback_speed = 1.0
        
        # History for FPS/latency plots
        self.fps_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.memory_history = deque(maxlen=100)
        
        # Current result
        self.current_result: Optional[BenchmarkResult] = None
        
        # Build UI
        self._init_ui()
        
        # Timers
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._playback_step)
        
        self.ui_update_timer = QTimer()
        self.ui_update_timer.timeout.connect(self._update_ui)
        self.ui_update_timer.start(100)  # 10 Hz UI refresh
        
        # Generate initial data
        self._generate_sequence(200)
    
    def _init_ui(self):
        """Initialize the dashboard UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Left panel - visualization
        left_panel = self._build_visualization_panel()
        main_layout.addWidget(left_panel, stretch=3)
        
        # Right panel - analytics and controls
        right_panel = self._build_analytics_panel()
        main_layout.addWidget(right_panel, stretch=1)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Synthetic Demo Mode")
    
    def _build_visualization_panel(self) -> QWidget:
        """Build the main visualization area."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # 2.5D Elevation/Semantic Map
        self.map_view = pg.GraphicsLayoutWidget()
        self.map_plot = self.map_view.addPlot()
        self.map_plot.setAspectLocked(True)
        self.map_plot.setXRange(-55, 55)
        self.map_plot.setYRange(-55, 55)
        self.map_plot.setTitle("2.5D Elevation Map (Top-Down)", color='#e0e0e0', size='12pt')
        self.map_plot.showGrid(x=True, y=True, alpha=0.3)
        
        self.map_image = pg.ImageItem()
        self.map_plot.addItem(self.map_image)
        
        layout.addWidget(self.map_view, stretch=2)
        
        # 3D Point Cloud View
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setCameraPosition(distance=50, elevation=30, azimuth=45)
        self.gl_view.setBackgroundColor('#1e1e1e')
        
        self.gl_grid = gl.GLGridItem()
        self.gl_grid.setSize(100, 100)
        self.gl_grid.setSpacing(10, 10)
        self.gl_view.addItem(self.gl_grid)
        
        self.gl_scatter = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=5,
                                               color=np.zeros((0, 4)), pxMode=True)
        self.gl_view.addItem(self.gl_scatter)
        
        layout.addWidget(self.gl_view, stretch=1)
        
        return panel
    
    def _build_analytics_panel(self) -> QWidget:
        """Build the analytics HUD and controls."""
        panel = QWidget()
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        
        # Performance Metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QFormLayout(metrics_group)
        
        self.lbl_memory_saved = QLabel("--")
        self.lbl_adaptive_cells = QLabel("--")
        self.lbl_uniform_cells = QLabel("--")
        self.lbl_compression = QLabel("--")
        self.lbl_fps = QLabel("--")
        self.lbl_latency = QLabel("--")
        self.lbl_points = QLabel("--")
        
        metrics_layout.addRow("Memory Saved:", self.lbl_memory_saved)
        metrics_layout.addRow("Adaptive Cells:", self.lbl_adaptive_cells)
        metrics_layout.addRow("Uniform Cells:", self.lbl_uniform_cells)
        metrics_layout.addRow("Compression:", self.lbl_compression)
        metrics_layout.addRow("FPS:", self.lbl_fps)
        metrics_layout.addRow("Latency (ms):", self.lbl_latency)
        metrics_layout.addRow("Points:", self.lbl_points)
        
        layout.addWidget(metrics_group)
        
        # Zone Breakdown
        zone_group = QGroupBox("Zone Statistics")
        zone_layout = QFormLayout(zone_group)
        
        self.lbl_zone0_cells = QLabel("--")
        self.lbl_zone1_cells = QLabel("--")
        self.lbl_zone2_cells = QLabel("--")
        self.lbl_zone3_cells = QLabel("--")
        self.lbl_total_active = QLabel("--")
        
        zone_layout.addRow("Zone 0 (0-10m):", self.lbl_zone0_cells)
        zone_layout.addRow("Zone 1 (10-30m):", self.lbl_zone1_cells)
        zone_layout.addRow("Zone 2 (30-60m):", self.lbl_zone2_cells)
        zone_layout.addRow("Zone 3 (60-100m):", self.lbl_zone3_cells)
        zone_layout.addRow("Total Active:", self.lbl_total_active)
        
        layout.addWidget(zone_group)
        
        # Semantic Distribution
        sem_group = QGroupBox("Semantic Distribution")
        sem_layout = QFormLayout(sem_group)
        
        self.lbl_free = QLabel("--")
        self.lbl_drivable = QLabel("--")
        self.lbl_static = QLabel("--")
        self.lbl_dynamic = QLabel("--")
        
        sem_layout.addRow("Free:", self.lbl_free)
        sem_layout.addRow("Drivable:", self.lbl_drivable)
        sem_layout.addRow("Static Obstacles:", self.lbl_static)
        sem_layout.addRow("Dynamic Objects:", self.lbl_dynamic)
        
        layout.addWidget(sem_group)
        
        # Real-time Charts
        charts_group = QGroupBox("Real-Time Analytics")
        charts_layout = QVBoxLayout(charts_group)
        
        self.fps_plot = pg.PlotWidget(title="FPS (Adaptive)", background='#2d2d2d')
        self.fps_plot.showGrid(x=True, y=True, alpha=0.3)
        self.fps_curve = self.fps_plot.plot(pen=pg.mkPen('#42a5f5', width=2))
        charts_layout.addWidget(self.fps_plot)
        
        self.memory_plot = pg.PlotWidget(title="Memory (MB)", background='#2d2d2d')
        self.memory_plot.showGrid(x=True, y=True, alpha=0.3)
        self.memory_curve = self.memory_plot.plot(pen=pg.mkPen('#66bb6a', width=2))
        charts_layout.addWidget(self.memory_plot)
        
        layout.addWidget(charts_group)
        
        # Playback Controls
        controls_group = QGroupBox("Playback Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self._toggle_playback)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self._pause_playback)
        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_playback)
        
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_reset)
        controls_layout.addLayout(btn_layout)
        
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Frame:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal if QT_VERSION >= 6 else Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.valueChanged.connect(self._on_frame_slider)
        frame_layout.addWidget(self.frame_slider)
        self.lbl_frame_num = QLabel("0/0")
        frame_layout.addWidget(self.lbl_frame_num)
        controls_layout.addLayout(frame_layout)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal if QT_VERSION >= 6 else Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(1)
        self.speed_slider.valueChanged.connect(self._on_speed_slider)
        speed_layout.addWidget(self.speed_slider)
        self.lbl_speed = QLabel("1x")
        speed_layout.addWidget(self.lbl_speed)
        controls_layout.addLayout(speed_layout)
        
        layout.addWidget(controls_group)
        
        # Settings
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout(settings_group)
        
        self.chk_pointnet = QCheckBox("Enable PointNet++ (if available)")
        self.chk_pointnet.setChecked(False)
        settings_layout.addRow(self.chk_pointnet)
        
        self.spin_frames = QSpinBox()
        self.spin_frames.setRange(50, 1000)
        self.spin_frames.setValue(200)
        self.spin_frames.setSuffix(" frames")
        settings_layout.addRow("Sequence Length:", self.spin_frames)
        
        layout.addWidget(settings_group)
        
        layout.addStretch()
        
        return panel
    
    def _generate_sequence(self, n_frames: int):
        """Generate a sequence of frames."""
        self.frames = self.generator.generate_sequence(n_frames)
        self.current_frame_idx = 0
        self.frame_slider.setMaximum(n_frames - 1)
        self.frame_slider.setValue(0)
        self._update_frame(0)
    
    def _update_frame(self, idx: int):
        """Process and display a single frame."""
        if not self.frames or idx < 0 or idx >= len(self.frames):
            return
        
        self.current_frame_idx = idx
        frame = self.frames[idx]
        
        # Process through benchmark pipeline
        result = self.benchmark.process_frame(frame)
        self.current_result = result
        
        # Update 2.5D map visualization
        self._update_map_view()
        
        # Update 3D scatter
        self._update_3d_view(frame)
        
        # Update slider and label
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(idx)
        self.frame_slider.blockSignals(False)
        self.lbl_frame_num.setText(f"{idx + 1}/{len(self.frames)}")
        
        # Store metrics for plots
        self.fps_history.append(result.fps_adaptive)
        self.latency_history.append(result.adaptive_latency_ms)
        self.memory_history.append(result.adaptive_memory_mb)
    
    def _update_map_view(self):
        """Update the 2D elevation/semantic map view."""
        cells = list(self.benchmark.adaptive_grid.grid.values())
        if not cells:
            return
        
        all_i = [c.cell_i for c in cells]
        all_j = [c.cell_j for c in cells]
        
        if not all_i:
            return
        
        min_i, max_i = min(all_i), max(all_i)
        min_j, max_j = min(all_j), max(all_j)
        
        width = max_i - min_i + 1
        height = max_j - min_j + 1
        
        img = np.zeros((height, width, 4), dtype=np.uint8)
        
        color_map = {
            0: (80, 80, 80, 255),      # Free - gray
            1: (76, 175, 80, 255),     # Drivable - green
            2: (244, 67, 54, 255),     # Static obstacle - red
            3: (33, 150, 243, 255),    # Dynamic - blue
        }
        
        for cell in cells:
            gi = cell.cell_i - min_i
            gj = cell.cell_j - min_j
            
            if 0 <= gi < width and 0 <= gj < height:
                color = color_map.get(cell.semantic_class, (128, 128, 128, 255))
                img[gj, gi] = color
        
        self.map_image.setImage(img)
    
    def _update_3d_view(self, frame: LiDARFrame):
        """Update the 3D point cloud scatter plot."""
        points = frame.points
        labels = frame.labels
        n = len(points)
        
        if n == 0:
            return
        
        colors = np.zeros((n, 4), dtype=np.float32)
        color_map = {
            0: (0.5, 0.5, 0.5, 0.3),   # Free
            1: (0.3, 0.7, 0.3, 0.6),   # Drivable
            2: (0.9, 0.2, 0.2, 0.8),   # Static
            3: (0.2, 0.5, 0.9, 0.9),   # Dynamic
        }
        
        for cls, color in color_map.items():
            mask = labels == cls
            colors[mask] = color
        
        # Downsample for performance
        step = max(1, n // 5000)
        self.gl_scatter.setData(pos=points[::step], color=colors[::step], size=5)
    
    def _update_ui(self):
        """Update all HUD labels and charts."""
        if self.current_result is None:
            return
        
        r = self.current_result
        
        self.lbl_memory_saved.setText(f"{r.memory_saved_pct:.1f}%")
        self.lbl_adaptive_cells.setText(f"{r.adaptive_active_cells:,}")
        self.lbl_uniform_cells.setText(f"{r.uniform_active_cells:,}")
        self.lbl_compression.setText(f"{r.compression_ratio:.2f}x")
        self.lbl_fps.setText(f"{r.fps_adaptive:.1f}")
        self.lbl_latency.setText(f"{r.adaptive_latency_ms:.2f}")
        self.lbl_points.setText(f"{r.n_points:,}")
        
        zone_stats = self.benchmark.adaptive_grid.get_zone_statistics()
        self.lbl_zone0_cells.setText(f"{zone_stats.get(0, {}).get('cell_count', 0):,}")
        self.lbl_zone1_cells.setText(f"{zone_stats.get(1, {}).get('cell_count', 0):,}")
        self.lbl_zone2_cells.setText(f"{zone_stats.get(2, {}).get('cell_count', 0):,}")
        self.lbl_zone3_cells.setText(f"{zone_stats.get(3, {}).get('cell_count', 0):,}")
        self.lbl_total_active.setText(f"{self.benchmark.adaptive_grid.get_active_cell_count():,}")
        
        if r.semantic_class_distribution:
            total = sum(r.semantic_class_distribution.values())
            if total > 0:
                self.lbl_free.setText(f"{r.semantic_class_distribution.get(0, 0):,} "
                                     f"({r.semantic_class_distribution.get(0, 0)/total*100:.1f}%)")
                self.lbl_drivable.setText(f"{r.semantic_class_distribution.get(1, 0):,} "
                                         f"({r.semantic_class_distribution.get(1, 0)/total*100:.1f}%)")
                self.lbl_static.setText(f"{r.semantic_class_distribution.get(2, 0):,} "
                                        f"({r.semantic_class_distribution.get(2, 0)/total*100:.1f}%)")
                self.lbl_dynamic.setText(f"{r.semantic_class_distribution.get(3, 0):,} "
                                         f"({r.semantic_class_distribution.get(3, 0)/total*100:.1f}%)")
        
        # Update charts
        if self.fps_history:
            self.fps_curve.setData(list(self.fps_history))
        if self.memory_history:
            self.memory_curve.setData(list(self.memory_history))
        
        self.status_bar.showMessage(
            f"Frame {self.current_frame_idx + 1}/{len(self.frames)} | "
            f"Adaptive: {r.adaptive_active_cells:,} cells | "
            f"Uniform: {r.uniform_active_cells:,} cells | "
            f"Memory Saved: {r.memory_saved_pct:.1f}% | "
            f"FPS: {r.fps_adaptive:.1f}"
        )
    
    def _toggle_playback(self):
        """Start playback."""
        self.is_playing = True
        interval = max(16, int(100 / self.playback_speed))
        self.playback_timer.start(interval)
    
    def _pause_playback(self):
        """Pause playback."""
        self.is_playing = False
        self.playback_timer.stop()
    
    def _reset_playback(self):
        """Reset to first frame."""
        self._pause_playback()
        self.benchmark.reset()
        self.fps_history.clear()
        self.latency_history.clear()
        self.memory_history.clear()
        self.current_frame_idx = 0
        self._update_frame(0)
    
    def _playback_step(self):
        """Advance one frame during playback."""
        if not self.is_playing:
            return
        
        next_idx = self.current_frame_idx + 1
        if next_idx >= len(self.frames):
            next_idx = 0
            self.benchmark.reset()
            self.fps_history.clear()
            self.latency_history.clear()
            self.memory_history.clear()
        
        self._update_frame(next_idx)
    
    def _on_frame_slider(self, value: int):
        """Handle frame slider change."""
        self._update_frame(value)
    
    def _on_speed_slider(self, value: int):
        """Handle speed slider change."""
        self.playback_speed = float(value)
        self.lbl_speed.setText(f"{value}x")
        if self.is_playing:
            self.playback_timer.setInterval(int(100 / self.playback_speed))


def run_dashboard():
    """Launch the interactive dashboard."""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Dark palette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Text, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(224, 224, 224))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)
    
    window = LiDARDashboard()
    window.show()
    sys.exit(qt_exec(app))


if __name__ == "__main__":
    run_dashboard()
