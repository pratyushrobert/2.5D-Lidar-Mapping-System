"""
grid_engine.py
==============
Adaptive Variable-Resolution 2.5D Elevation Grid Engine.

Implements a foveated, multi-scale grid with:
- Distance-adaptive cell sizes (5cm near, 50cm far)
- Morton-indexed spatial hashing for boundary alignment
- 2.5D cell statistics (z_min, z_max, z_mean, z_variance)
- Semantic majority voting per cell
- Occupancy state and traversability scoring
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import math


@dataclass
class GridCell25D:
    """2.5D Elevation Grid Cell with compressed 3D point statistics."""
    cell_i: int = 0
    cell_j: int = 0
    zone: int = 0
    resolution: float = 0.05
    
    z_min: float = float('inf')
    z_max: float = float('-inf')
    z_mean: float = 0.0
    z_variance: float = 0.0
    z_sum: float = 0.0
    z_sq_sum: float = 0.0
    point_count: int = 0
    
    semantic_counts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    semantic_class: int = 0
    
    occupancy_state: int = 0  # 0=Free, 1=Drivable, 2=Hazard, 3=Dynamic
    traversability_score: float = 1.0
    
    def update(self, z: float, semantic_label: int):
        """Update cell statistics with a new 3D point."""
        self.point_count += 1
        self.z_sum += z
        self.z_sq_sum += z * z
        
        if z < self.z_min:
            self.z_min = z
        if z > self.z_max:
            self.z_max = z
        
        self.z_mean = self.z_sum / self.point_count
        self.z_variance = (self.z_sq_sum / self.point_count) - (self.z_mean ** 2)
        
        self.semantic_counts[semantic_label] += 1
        self._update_semantic_class()
        self._update_occupancy()
    
    def _update_semantic_class(self):
        """Majority vote for semantic class."""
        if self.semantic_counts:
            self.semantic_class = max(self.semantic_counts, key=self.semantic_counts.get)
    
    def _update_occupancy(self):
        """Compute occupancy state and traversability."""
        dz = self.z_max - self.z_min
        
        if self.point_count == 0:
            self.occupancy_state = 0
            self.traversability_score = 1.0
            return
        
        if self.semantic_class == 3:  # Dynamic
            self.occupancy_state = 3
            self.traversability_score = 0.0
        elif self.semantic_class == 2:  # Static obstacle
            if dz > 0.1:
                self.occupancy_state = 2  # Hazard
                self.traversability_score = 0.0
            else:
                self.occupancy_state = 2
                self.traversability_score = 0.0
        else:
            # Drivable or free
            if dz > 0.05:
                self.occupancy_state = 2  # Hazard (curb/pothole)
                self.traversability_score = max(0.0, 1.0 - dz / 0.3)
            else:
                self.occupancy_state = 1  # Drivable
                self.traversability_score = max(0.0, 1.0 - dz / 0.1)
    
    def memory_bytes(self) -> int:
        """Estimate memory footprint in bytes."""
        base = 64  # overhead for object
        base += 8 * 9  # numeric fields
        base += 32  # semantic counts dict estimate
        return base


class MortonCode:
    """Morton (Z-order) curve encoding for spatial hashing."""
    
    @staticmethod
    def encode(x: int, y: int) -> int:
        """Encode 2D integer coordinates to Morton code."""
        morton = 0
        for i in range(32):
            morton |= ((x >> i) & 1) << (2 * i)
            morton |= ((y >> i) & 1) << (2 * i + 1)
        return morton
    
    @staticmethod
    def decode(morton: int) -> Tuple[int, int]:
        """Decode Morton code to 2D integer coordinates."""
        x, y = 0, 0
        for i in range(32):
            x |= ((morton >> (2 * i)) & 1) << i
            y |= ((morton >> (2 * i + 1)) & 1) << i
        return x, y


class AdaptiveGridEngine:
    """
    Adaptive Variable-Resolution 2.5D Elevation Grid.
    
    Uses multi-scale nested hash grid with foveated resolution:
    - Zone 0 (0-10m): 0.05m cells
    - Zone 1 (10-30m): 0.10m cells
    - Zone 2 (30-60m): 0.25m cells
    - Zone 3 (60-100m): 0.50m cells
    """
    
    ZONE_RANGES = [
        (0.0, 10.0, 0.05),
        (10.0, 30.0, 0.10),
        (30.0, 60.0, 0.25),
        (60.0, 100.0, 0.50),
    ]
    
    LABEL_FREE = 0
    LABEL_DRIVABLE = 1
    LABEL_STATIC = 2
    LABEL_DYNAMIC = 3
    
    def __init__(self, max_range: float = 100.0):
        self.max_range = max_range
        self.grid: Dict[int, GridCell25D] = {}
        self.cell_map: Dict[Tuple[int, int], int] = {}
        
        self.total_points_processed = 0
        self.total_active_cells = 0
        self.memory_bytes = 0
    
    def get_zone_and_resolution(self, distance: float) -> Tuple[int, float]:
        """Determine zone and resolution based on radial distance from origin."""
        for zone_idx, (d_min, d_max, res) in enumerate(self.ZONE_RANGES):
            if d_min <= distance < d_max:
                return zone_idx, res
        if distance >= self.ZONE_RANGES[-1][1]:
            return len(self.ZONE_RANGES) - 1, self.ZONE_RANGES[-1][2]
        return 0, self.ZONE_RANGES[0][2]
    
    def world_to_cell(self, x: float, y: float, resolution: float) -> Tuple[int, int]:
        """Convert world coordinates to cell indices."""
        return int(np.floor(x / resolution)), int(np.floor(y / resolution))
    
    def distance_to_origin(self, x: float, y: float) -> float:
        """Compute Euclidean distance from vehicle origin."""
        return math.sqrt(x * x + y * y)
    
    def insert_points(self, points: np.ndarray, labels: np.ndarray):
        """Insert a batch of 3D points into the adaptive grid."""
        n_points = len(points)
        if n_points == 0:
            return
        
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        
        for idx in range(n_points):
            x, y, z = points[idx]
            label = labels[idx] if idx < len(labels) else 0
            dist = distances[idx]
            
            zone, resolution = self.get_zone_and_resolution(dist)
            cell_i, cell_j = self.world_to_cell(x, y, resolution)
            
            morton_key = MortonCode.encode(cell_i, cell_j)
            zone_key = (zone, morton_key)
            
            if zone_key not in self.cell_map:
                new_cell = GridCell25D(
                    cell_i=cell_i,
                    cell_j=cell_j,
                    zone=zone,
                    resolution=resolution,
                )
                cell_idx = len(self.grid)
                self.grid[cell_idx] = new_cell
                self.cell_map[zone_key] = cell_idx
            
            cell_idx = self.cell_map[zone_key]
            self.grid[cell_idx].update(z, label)
        
        self.total_points_processed += n_points
        self._update_memory()
    
    def _update_memory(self):
        """Recalculate total memory footprint."""
        self.total_active_cells = len(self.grid)
        self.memory_bytes = sum(cell.memory_bytes() for cell in self.grid.values())
    
    def get_active_cell_count(self) -> int:
        return self.total_active_cells
    
    def get_memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)
    
    def get_compression_ratio(self, uniform_cell_count: int) -> float:
        """Compute compression ratio vs uniform grid."""
        if self.total_active_cells == 0:
            return 1.0
        return uniform_cell_count / self.total_active_cells
    
    def get_zone_statistics(self) -> Dict[int, dict]:
        """Return per-zone grid statistics."""
        stats = {}
        for zone in range(4):
            zone_cells = [c for c in self.grid.values() if c.zone == zone]
            stats[zone] = {
                'cell_count': len(zone_cells),
                'resolution': self.ZONE_RANGES[zone][2],
                'total_points': sum(c.point_count for c in zone_cells),
            }
        return stats
    
    def get_elevation_grid(self, zone: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract elevation, semantic, and occupancy grids for a zone."""
        resolution = self.ZONE_RANGES[zone][2]
        d_max = self.ZONE_RANGES[zone][1]
        d_min = self.ZONE_RANGES[zone][0]
        
        extent = int(np.ceil(d_max / resolution))
        origin_i = extent
        origin_j = extent
        
        shape = (2 * extent + 1, 2 * extent + 1)
        elevation = np.full(shape, np.nan, dtype=np.float32)
        semantic = np.full(shape, -1, dtype=np.int32)
        occupancy = np.full(shape, -1, dtype=np.int32)
        
        for cell in self.grid.values():
            if cell.zone != zone:
                continue
            gi = cell.cell_i + origin_i
            gj = cell.cell_j + origin_j
            if 0 <= gi < shape[0] and 0 <= gj < shape[1]:
                elevation[gi, gj] = cell.z_mean
                semantic[gi, gj] = cell.semantic_class
                occupancy[gi, gj] = cell.occupancy_state
        
        return elevation, semantic, occupancy
    
    def clear(self):
        """Reset the grid."""
        self.grid.clear()
        self.cell_map.clear()
        self.total_points_processed = 0
        self.total_active_cells = 0
        self.memory_bytes = 0


class UniformGridEngine:
    """
    Uniform Fixed-Resolution 2.5D Grid for comparison benchmarking.
    
    Uses 5cm cells everywhere over the full 100m range.
    """
    
    def __init__(self, resolution: float = 0.05, max_range: float = 100.0):
        self.resolution = resolution
        self.max_range = max_range
        self.grid: Dict[Tuple[int, int], GridCell25D] = {}
        
        self.total_points_processed = 0
        self.total_active_cells = 0
        self.memory_bytes = 0
    
    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return int(np.floor(x / self.resolution)), int(np.floor(y / self.resolution))
    
    def insert_points(self, points: np.ndarray, labels: np.ndarray):
        """Insert points into uniform grid."""
        n_points = len(points)
        if n_points == 0:
            return
        
        for idx in range(n_points):
            x, y, z = points[idx]
            label = labels[idx] if idx < len(labels) else 0
            
            ci, cj = self.world_to_cell(x, y)
            
            if (ci, cj) not in self.grid:
                self.grid[(ci, cj)] = GridCell25D(
                    cell_i=ci, cell_j=cj, zone=0, resolution=self.resolution
                )
            
            self.grid[(ci, cj)].update(z, label)
        
        self.total_points_processed += n_points
        self._update_memory()
    
    def _update_memory(self):
        self.total_active_cells = len(self.grid)
        self.memory_bytes = sum(cell.memory_bytes() for cell in self.grid.values())
    
    def get_active_cell_count(self) -> int:
        return self.total_active_cells
    
    def get_memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)
    
    def clear(self):
        self.grid.clear()
        self.total_points_processed = 0
        self.total_active_cells = 0
        self.memory_bytes = 0
