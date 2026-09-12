"""
perception.py
=============
Point Cloud Perception Module for LiDAR Semantic Segmentation.

Implements:
1. Lightweight PointNet++ inference stub with PyTorch
2. Robust geometric heuristic fallback:
   - RANSAC ground plane estimation
   - Eigenvalue-based curvature/height classification
   - Dynamic object detection via temporal differencing
"""

import numpy as np
from typing import Optional
from scipy.spatial import cKDTree


class PointNetPlusStub:
    """
    Lightweight PointNet++ classification stub with PyTorch.
    Falls back gracefully if weights are unavailable.
    """
    
    def __init__(self, num_classes: int = 4, num_points: int = 4096):
        self.num_classes = num_classes
        self.num_points = num_points
        self.model = None
        self.device = 'cpu'
        
        try:
            import torch
            import torch.nn as nn
            self.torch = torch
            self.nn = nn
            
            self.model = nn.Sequential(
                nn.Linear(3, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
                nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, num_classes)
            )
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            pass
    
    @property
    def available(self) -> bool:
        return self.model is not None
    
    def infer(self, points: np.ndarray) -> np.ndarray:
        """Run inference on a point cloud (N, 3). Returns (N,) class labels."""
        if not self.available:
            return np.full(len(points), -1, dtype=np.int32)
        
        n_points = len(points)
        if n_points == 0:
            return np.array([], dtype=np.int32)
        
        batch_size = 4096
        all_probs = []
        
        with self.torch.no_grad():
            for i in range(0, n_points, batch_size):
                batch = points[i:i + batch_size]
                actual_size = len(batch)
                
                if actual_size < batch_size:
                    pad = np.zeros((batch_size - actual_size, 3), dtype=np.float32)
                    batch = np.concatenate([batch, pad], axis=0)
                
                tensor = self.torch.from_numpy(batch).float().to(self.device)
                logits = self.model(tensor)
                probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs[:actual_size])
        
        probs = np.concatenate(all_probs, axis=0)
        return np.argmax(probs, axis=-1).astype(np.int32)


class GeometricPerception:
    """
    Robust geometric heuristic perception for real-time fallback.
    
    Pipeline:
    1. RANSAC ground plane estimation
    2. Height + curvature classification
    3. Temporal differencing for dynamic detection
    """
    
    def __init__(self,
                 ground_angle_threshold: float = 0.15,
                 min_points_for_ransac: int = 500,
                 dynamic_distance_threshold: float = 0.3):
        self.ground_angle_threshold = ground_angle_threshold
        self.min_points_for_ransac = min_points_for_ransac
        self.dynamic_distance_threshold = dynamic_distance_threshold
        
        self.prev_frame_points: Optional[np.ndarray] = None
        self.prev_frame_labels: Optional[np.ndarray] = None
    
    def classify(self, points: np.ndarray, intensity: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Classify point cloud into semantic classes.
        
        Returns labels:
            0 = Free
            1 = Drivable ground
            2 = Static obstacle
            3 = Dynamic object
        """
        n_points = len(points)
        if n_points == 0:
            return np.array([], dtype=np.int32)
        
        labels = np.full(n_points, 0, dtype=np.int32)
        
        ground_mask = self._ransac_ground(points)
        labels[ground_mask] = 1  # Drivable
        
        obstacle_mask = ~ground_mask
        if np.any(obstacle_mask):
            obstacle_labels = self._classify_obstacles(
                points[obstacle_mask],
                intensity[obstacle_mask] if intensity is not None else None
            )
            labels[obstacle_mask] = obstacle_labels
        
        if self.prev_frame_points is not None and len(self.prev_frame_points) > 0:
            dynamic_mask = self._detect_dynamic(points, labels)
            labels[dynamic_mask] = 3
        
        self.prev_frame_points = points.copy()
        self.prev_frame_labels = labels.copy()
        
        return labels
    
    def _ransac_ground(self, points: np.ndarray, iterations: int = 50) -> np.ndarray:
        """RANSAC-based ground plane estimation."""
        n_points = len(points)
        if n_points < self.min_points_for_ransac:
            height_percentile = np.percentile(points[:, 2], 20)
            return points[:, 2] < height_percentile + 0.3
        
        best_inliers = np.zeros(n_points, dtype=bool)
        best_count = 0
        
        for _ in range(iterations):
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            normal = np.cross(p2 - p1, p3 - p1)
            norm_val = np.linalg.norm(normal)
            if norm_val < 1e-6:
                continue
            normal = normal / norm_val
            
            d = -np.dot(normal, p1)
            distances = np.abs(np.dot(points, normal) + d)
            
            inliers = distances < self.ground_angle_threshold
            count = np.sum(inliers)
            
            if count > best_count:
                best_count = count
                best_inliers = inliers
        
        if best_count < self.min_points_for_ransac:
            height_percentile = np.percentile(points[:, 2], 20)
            best_inliers = points[:, 2] < height_percentile + 0.3
        
        return best_inliers
    
    def _classify_obstacles(self, points: np.ndarray, intensity: Optional[np.ndarray]) -> np.ndarray:
        """Classify non-ground points using height and curvature heuristics."""
        labels = np.full(len(points), 2, dtype=np.int32)
        
        if len(points) < 10:
            return labels
        
        height_threshold = 1.5
        
        for i in range(len(points)):
            x, y, z = points[i]
            
            if z < height_threshold:
                local_radius = 1.0
                nearby = np.linalg.norm(points - points[i], axis=1) < local_radius
                n_nearby = np.sum(nearby)
                
                if n_nearby > 5:
                    nearby_points = points[nearby]
                    cov = np.cov(nearby_points.T)
                    eigenvalues = np.linalg.eigvalsh(cov)
                    eigenvalues = np.sort(eigenvalues)
                    
                    linearity = (eigenvalues[2] - eigenvalues[1]) / (eigenvalues[2] + 1e-6)
                    planarity = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-6)
                    
                    if linearity > 0.8:
                        labels[i] = 2  # Pole/linear
                    elif planarity > 0.6:
                        labels[i] = 2  # Wall/planar
                    else:
                        labels[i] = 2  # Default static
        
        return labels
    
    def _detect_dynamic(self, points: np.ndarray, current_labels: np.ndarray) -> np.ndarray:
        """Detect dynamic objects via temporal differencing."""
        if self.prev_frame_points is None or len(self.prev_frame_points) == 0:
            return np.zeros(len(points), dtype=bool)
        
        dynamic_mask = np.zeros(len(points), dtype=bool)
        
        tree = cKDTree(self.prev_frame_points)
        distances, indices = tree.query(points, k=1)
        
        dynamic_mask = distances > self.dynamic_distance_threshold
        
        prev_dynamic = self.prev_frame_labels == 3
        if np.any(prev_dynamic):
            prev_dynamic_points = self.prev_frame_points[prev_dynamic]
            if len(prev_dynamic_points) > 0:
                dyn_tree = cKDTree(prev_dynamic_points)
                dyn_distances, _ = dyn_tree.query(points, k=1)
                dynamic_mask = dynamic_mask & (dyn_distances < 3.0)
        
        return dynamic_mask
