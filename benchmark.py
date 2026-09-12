"""
benchmark.py
============
Benchmarking and Comparative Analysis Engine.

Compares Adaptive Variable-Resolution 2.5D Grid against
Traditional Fixed Uniform Grid in terms of:
- Memory footprint (MB)
- Active cell count
- Compression ratio
- Processing latency (ms) and FPS
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import defaultdict

from data_loader import SyntheticLiDARGenerator, LiDARFrame
from grid_engine import AdaptiveGridEngine, UniformGridEngine, GridCell25D
from perception import GeometricPerception, PointNetPlusStub


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""
    frame_id: int
    n_points: int
    
    adaptive_active_cells: int = 0
    uniform_active_cells: int = 0
    
    adaptive_memory_mb: float = 0.0
    uniform_memory_mb: float = 0.0
    
    compression_ratio: float = 1.0
    memory_saved_pct: float = 0.0
    
    adaptive_latency_ms: float = 0.0
    uniform_latency_ms: float = 0.0
    
    fps_adaptive: float = 0.0
    fps_uniform: float = 0.0
    
    adaptive_zone_stats: Dict = None
    uniform_zone_stats: Dict = None
    
    semantic_class_distribution: Dict[int, int] = None


class BenchmarkEngine:
    """Comparative benchmarking between Adaptive and Uniform grids."""
    
    def __init__(self, max_range: float = 100.0):
        self.max_range = max_range
        self.adaptive_grid = AdaptiveGridEngine(max_range=max_range)
        self.uniform_grid = UniformGridEngine(resolution=0.05, max_range=max_range)
        
        self.perception = GeometricPerception()
        self.pointnet = PointNetPlusStub()
        self.use_pointnet = False
        
        self.results: List[BenchmarkResult] = []
        self.frame_times_adaptive: List[float] = []
        self.frame_times_uniform: List[float] = []
        
        self.warmup_frames = 5
    
    def process_frame(self, frame: LiDARFrame) -> BenchmarkResult:
        """Process a single frame through both pipelines."""
        points = frame.points
        intensity = frame.intensity
        n_points = len(points)
        
        if n_points == 0:
            return BenchmarkResult(frame_id=frame.frame_id, n_points=0)
        
        # Perception / segmentation
        if self.use_pointnet and self.pointnet.available:
            raw_labels = self.pointnet.infer(points)
            labels = np.where(raw_labels == -1,
                            self.perception.classify(points, intensity),
                            raw_labels)
        else:
            labels = self.perception.classify(points, intensity)
        
        class_dist = {cls: int(np.sum(labels == cls)) for cls in range(4)}
        
        # Adaptive grid processing
        t0 = time.perf_counter()
        self.adaptive_grid.insert_points(points, labels)
        t1 = time.perf_counter()
        adaptive_latency = (t1 - t0) * 1000.0
        
        # Uniform grid processing
        t0 = time.perf_counter()
        self.uniform_grid.insert_points(points, labels)
        t1 = time.perf_counter()
        uniform_latency = (t1 - t0) * 1000.0
        
        adaptive_cells = self.adaptive_grid.get_active_cell_count()
        uniform_cells = self.uniform_grid.get_active_cell_count()
        adaptive_mem = self.adaptive_grid.get_memory_mb()
        uniform_mem = self.uniform_grid.get_memory_mb()
        compression = self.adaptive_grid.get_compression_ratio(uniform_cells)
        
        memory_saved = 0.0
        if uniform_mem > 0:
            memory_saved = ((uniform_mem - adaptive_mem) / uniform_mem) * 100.0
        
        self.frame_times_adaptive.append(adaptive_latency)
        self.frame_times_uniform.append(uniform_latency)
        
        if len(self.frame_times_adaptive) > 100:
            self.frame_times_adaptive.pop(0)
            self.frame_times_uniform.pop(0)
        
        avg_adaptive_latency = np.mean(self.frame_times_adaptive) if self.frame_times_adaptive else 0.0
        avg_uniform_latency = np.mean(self.frame_times_uniform) if self.frame_times_uniform else 0.0
        
        fps_adaptive = 1000.0 / avg_adaptive_latency if avg_adaptive_latency > 0 else 0.0
        fps_uniform = 1000.0 / avg_uniform_latency if avg_uniform_latency > 0 else 0.0
        
        zone_stats = self.adaptive_grid.get_zone_statistics()
        
        result = BenchmarkResult(
            frame_id=frame.frame_id,
            n_points=n_points,
            adaptive_active_cells=adaptive_cells,
            uniform_active_cells=uniform_cells,
            adaptive_memory_mb=adaptive_mem,
            uniform_memory_mb=uniform_mem,
            compression_ratio=compression,
            memory_saved_pct=memory_saved,
            adaptive_latency_ms=adaptive_latency,
            uniform_latency_ms=uniform_latency,
            fps_adaptive=fps_adaptive,
            fps_uniform=fps_uniform,
            adaptive_zone_stats=zone_stats,
            semantic_class_distribution=class_dist,
        )
        
        self.results.append(result)
        return result
    
    def process_sequence(self, frames: List[LiDARFrame]) -> List[BenchmarkResult]:
        """Process a sequence of frames."""
        results = []
        for i, frame in enumerate(frames):
            if i == self.warmup_frames:
                self.adaptive_grid.clear()
                self.uniform_grid.clear()
                self.frame_times_adaptive.clear()
                self.frame_times_uniform.clear()
            
            result = self.process_frame(frame)
            results.append(result)
        return results
    
    def get_summary_stats(self) -> Dict:
        """Return aggregate statistics across all processed frames."""
        if not self.results:
            return {}
        
        recent = self.results[-20:] if len(self.results) >= 20 else self.results
        
        return {
            'total_frames': len(self.results),
            'adaptive_avg_memory_mb': np.mean([r.adaptive_memory_mb for r in recent]),
            'uniform_avg_memory_mb': np.mean([r.uniform_memory_mb for r in recent]),
            'avg_memory_saved_pct': np.mean([r.memory_saved_pct for r in recent]),
            'adaptive_active_cells': self.adaptive_grid.get_active_cell_count(),
            'uniform_active_cells': self.uniform_grid.get_active_cell_count(),
            'compression_ratio': self.adaptive_grid.get_compression_ratio(
                self.uniform_grid.get_active_cell_count()
            ),
            'avg_adaptive_latency_ms': np.mean([r.adaptive_latency_ms for r in recent]),
            'avg_uniform_latency_ms': np.mean([r.uniform_latency_ms for r in recent]),
            'fps_adaptive': np.mean([r.fps_adaptive for r in recent]),
            'fps_uniform': np.mean([r.fps_uniform for r in recent]),
        }
    
    def reset(self):
        """Reset all grids and statistics."""
        self.adaptive_grid.clear()
        self.uniform_grid.clear()
        self.results.clear()
        self.frame_times_adaptive.clear()
        self.frame_times_uniform.clear()
