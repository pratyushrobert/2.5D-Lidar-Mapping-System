"""
data_loader.py
==============
Synthetic LiDAR Data Generator and SemanticKITTI Loader.

Generates realistic driving corridors with:
- Flat drivable terrain with slope and potholes
- Curbs and elevation changes near roads
- Static obstacles (poles, buildings, trees, barriers)
- Dynamic objects (walking pedestrians, moving vehicles)
- Coordinate frame transforms: p_W = R * p_L + t
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass
class LiDARFrame:
    """Container for a single LiDAR scan frame."""
    points: np.ndarray          # (N, 3) XYZ in vehicle frame
    intensity: np.ndarray       # (N,) intensity values [0, 1]
    labels: np.ndarray          # (N,) semantic labels
    timestamps: np.ndarray      # (N,) per-point timestamps
    frame_id: int = 0
    sensor_pose: np.ndarray = None  # (4, 4) homogeneous transform


class SemanticKITTILoader:
    """Loader for SemanticKITTI dataset format."""
    
    LABEL_MAP = {
        0: 0,    # unlabeled -> unknown
        1: 1,    # car -> dynamic
        2: 2,    # bicycle -> dynamic
        3: 2,    # motorcycle -> dynamic
        4: 2,    # truck -> dynamic
        5: 3,    # other-vehicle -> dynamic
        6: 2,    # person -> dynamic
        7: 2,    # bicyclist -> dynamic
        8: 2,    # motorcyclist -> dynamic
        9: 1,    # road -> drivable
        10: 1,   # parking -> drivable
        11: 1,   # sidewalk -> drivable
        12: 2,   # other-ground -> drivable
        13: 3,   # building -> static
        14: 3,   # fence -> static
        15: 3,   # other-structure -> static
        16: 3,   # pole -> static
        17: 3,   # traffic-sign -> static
        18: 2,   # vegetation -> static
        19: 3,   # trunk -> static
        20: 3,   # terrain -> static
        21: 2,   # other-object -> dynamic
        252: 0,  # unknown
    }
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.velodyne_dir = os.path.join(data_path, "velodyne")
        self.labels_dir = os.path.join(data_path, "labels")
    
    def load_frame(self, frame_id: int) -> Optional[LiDARFrame]:
        """Load a single frame by index."""
        velodyne_file = os.path.join(
            self.velodyne_dir, f"{frame_id:06d}.bin"
        )
        label_file = os.path.join(
            self.labels_dir, f"{frame_id:06d}.label"
        )
        
        if not os.path.exists(velodyne_file):
            return None
        
        points = np.fromfile(velodyne_file, dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        intensity = points[:, 3]
        
        labels = np.zeros(len(points), dtype=np.int32)
        if os.path.exists(label_file):
            raw_labels = np.fromfile(label_file, dtype=np.int32)
            for raw, mapped in self.LABEL_MAP.items():
                labels[raw_labels == raw] = mapped
        
        timestamps = np.arange(len(points), dtype=np.float32) / 10.0
        
        return LiDARFrame(
            points=xyz,
            intensity=intensity,
            labels=labels,
            timestamps=timestamps,
            frame_id=frame_id,
        )
    
    def count_frames(self) -> int:
        """Count available frames in dataset."""
        if not os.path.exists(self.velodyne_dir):
            return 0
        files = [f for f in os.listdir(self.velodyne_dir) if f.endswith('.bin')]
        return len(files)


class SyntheticLiDARGenerator:
    """
    Generates realistic synthetic LiDAR scans for driving scenarios.
    
    Creates:
    - Drivable road surface with slope, potholes, and curbs
    - Static obstacles (poles, buildings, trees, barriers)
    - Dynamic objects (pedestrians, vehicles)
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.frame_count = 0
        
        # Semantic class constants
        self.LABEL_FREE = 0
        self.LABEL_DRIVABLE = 1
        self.LABEL_STATIC = 2
        self.LABEL_DYNAMIC = 3
        
        # Environment parameters
        self.road_width = 8.0
        self.road_length = 80.0
        self.curb_height = 0.15
        self.pothole_depth = 0.08
        
        # Static objects
        self.static_poles = []
        self.static_buildings = []
        self.static_trees = []
        self._generate_static_environment()
        
        # Dynamic objects
        self.dynamic_pedestrians = []
        self.dynamic_vehicles = []
        self._generate_dynamic_templates()
    
    def _generate_static_environment(self):
        """Pre-generate static obstacle positions."""
        # Roadside poles
        for i in range(20):
            side = np.random.choice([-1, 1])
            x = np.random.uniform(-40, 40)
            y = side * (self.road_width / 2 + np.random.uniform(1.0, 3.0))
            height = np.random.uniform(3.0, 6.0)
            radius = np.random.uniform(0.1, 0.3)
            self.static_poles.append({
                'pos': np.array([x, y, 0.0]),
                'height': height,
                'radius': radius,
            })
        
        # Buildings on both sides
        for side in [-1, 1]:
            for i in range(5):
                x = np.random.uniform(-35, 35)
                y = side * (self.road_width / 2 + 4.0 + np.random.uniform(0, 10))
                width = np.random.uniform(5, 15)
                depth = np.random.uniform(3, 8)
                height = np.random.uniform(4, 12)
                self.static_buildings.append({
                    'pos': np.array([x, y, 0.0]),
                    'width': width,
                    'depth': depth,
                    'height': height,
                })
        
        # Trees
        for i in range(15):
            side = np.random.choice([-1, 1])
            x = np.random.uniform(-38, 38)
            y = side * (self.road_width / 2 + 2.0 + np.random.uniform(0, 8))
            height = np.random.uniform(2.0, 5.0)
            trunk_radius = np.random.uniform(0.15, 0.4)
            canopy_radius = np.random.uniform(1.0, 2.5)
            self.static_trees.append({
                'pos': np.array([x, y, 0.0]),
                'trunk_height': height,
                'trunk_radius': trunk_radius,
                'canopy_radius': canopy_radius,
            })
    
    def _generate_dynamic_templates(self):
        """Pre-generate dynamic object motion templates."""
        # Pedestrian walking patterns
        for i in range(5):
            path_x = np.random.uniform(-30, 30)
            path_y = np.random.choice([
                self.road_width / 2 + 1.5,
                -self.road_width / 2 - 1.5,
                0.0,
            ])
            speed = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            self.dynamic_pedestrians.append({
                'base_x': path_x,
                'base_y': path_y,
                'speed': speed,
                'phase': phase,
                'height': 1.6,
                'radius': 0.3,
            })
        
        # Moving vehicles
        for i in range(3):
            lane_y = np.random.choice([
                self.road_width / 4,
                -self.road_width / 4,
            ])
            direction = np.random.choice([1, -1])
            speed = np.random.uniform(2.0, 6.0)
            start_x = direction * 40.0
            self.dynamic_vehicles.append({
                'lane_y': lane_y,
                'direction': direction,
                'speed': speed,
                'start_x': start_x,
                'length': np.random.uniform(3.5, 5.0),
                'height': np.random.uniform(1.4, 2.0),
                'width': np.random.uniform(1.6, 2.0),
            })
    
    def generate_frame(self, time_step: float = 0.0) -> LiDARFrame:
        """Generate a single LiDAR frame at the given time step."""
        points_list = []
        intensity_list = []
        labels_list = []
        
        sensor_pose = np.eye(4)
        sensor_pose[:3, 3] = np.array([
            np.sin(time_step * 0.3) * 0.1,
            np.cos(time_step * 0.3) * 0.05,
            0.0
        ])
        sensor_pose[:3, :3] = self._random_rotation(time_step * 0.02)
        
        # 1. Ground plane
        self._add_ground_points(
            points_list, intensity_list, labels_list, sensor_pose
        )
        
        # 2. Curbs
        self._add_curb_points(points_list, intensity_list, labels_list)
        
        # 3. Potholes
        self._add_pothole_points(points_list, intensity_list, labels_list)
        
        # 4. Static poles
        for pole in self.static_poles:
            self._add_cylinder_points(
                pole['pos'], pole['height'], pole['radius'],
                points_list, intensity_list, labels_list,
                self.LABEL_STATIC
            )
        
        # 5. Buildings
        for bldg in self.static_buildings:
            self._add_box_points(
                bldg['pos'], bldg['width'], bldg['depth'], bldg['height'],
                points_list, intensity_list, labels_list,
                self.LABEL_STATIC
            )
        
        # 6. Trees
        for tree in self.static_trees:
            self._add_tree_points(
                tree, points_list, intensity_list, labels_list
            )
        
        # 7. Dynamic pedestrians
        for ped in self.dynamic_pedestrians:
            ped_x = ped['base_x'] + np.sin(time_step * ped['speed']) * 3.0
            ped_y = ped['base_y'] + np.cos(time_step * ped['speed'] * 0.7) * 1.5
            self._add_cylinder_points(
                np.array([ped_x, ped_y, 0.0]),
                ped['height'], ped['radius'],
                points_list, intensity_list, labels_list,
                self.LABEL_DYNAMIC
            )
        
        # 8. Dynamic vehicles
        for veh in self.dynamic_vehicles:
            veh_x = veh['start_x'] + time_step * veh['speed'] * veh['direction']
            veh_x = np.clip(veh_x, -45, 45)
            self._add_box_points(
                np.array([veh_x, veh['lane_y'], 0.0]),
                veh['length'], veh['width'], veh['height'],
                points_list, intensity_list, labels_list,
                self.LABEL_DYNAMIC
            )
        
        # Apply sensor pose transformation
        if len(points_list) > 0:
            points = np.array(points_list)
            points_h = np.ones((len(points), 4))
            points_h[:, :3] = points
            points_w = (sensor_pose @ points_h.T).T[:, :3]
            points = points_w
        else:
            points = np.zeros((0, 3))
        
        intensity = np.array(intensity_list, dtype=np.float32)
        labels = np.array(labels_list, dtype=np.int32)
        timestamps = np.full(len(points), time_step, dtype=np.float32)
        
        self.frame_count += 1
        
        return LiDARFrame(
            points=points,
            intensity=intensity,
            labels=labels,
            timestamps=timestamps,
            frame_id=self.frame_count,
            sensor_pose=sensor_pose,
        )
    
    def _add_ground_points(self, points_list, intensity_list, labels_list, 
                           sensor_pose):
        """Generate ground plane with subtle slope and texture."""
        n_points = 15000
        
        xs = np.random.uniform(-50, 50, n_points)
        ys = np.random.uniform(-6, 6, n_points)
        
        # Road slope
        zs = 0.02 * xs + np.random.normal(0, 0.01, n_points)
        
        # Road surface filtering
        road_mask = np.abs(ys) < self.road_width / 2
        zs[~road_mask] = -0.3 + np.random.normal(0, 0.02, np.sum(~road_mask))
        
        # Shoulder areas
        shoulder_mask = (np.abs(ys) >= self.road_width / 2) & (np.abs(ys) < self.road_width / 2 + 1.5)
        
        labels = np.full(n_points, self.LABEL_FREE)
        labels[road_mask] = self.LABEL_DRIVABLE
        labels[shoulder_mask] = self.LABEL_STATIC
        
        intensities = np.full(n_points, 0.3)
        intensities[road_mask] = np.random.uniform(0.4, 0.6, np.sum(road_mask))
        intensities[shoulder_mask] = np.random.uniform(0.2, 0.4, np.sum(shoulder_mask))
        
        points_list.extend(zip(xs, ys, zs))
        intensity_list.extend(intensities)
        labels_list.extend(labels)
    
    def _add_curb_points(self, points_list, intensity_list, labels_list):
        """Add curb elevation lines along road edges."""
        n_points_per_curb = 500
        for side in [-1, 1]:
            curb_y = side * self.road_width / 2
            xs = np.random.uniform(-45, 45, n_points_per_curb)
            zs = self.curb_height + np.random.normal(0, 0.02, n_points_per_curb)
            ys = np.full(n_points_per_curb, curb_y + np.random.normal(0, 0.1, n_points_per_curb))
            
            points_list.extend(zip(xs, ys, zs))
            intensity_list.extend(np.random.uniform(0.5, 0.7, n_points_per_curb))
            labels_list.extend([self.LABEL_STATIC] * n_points_per_curb)
    
    def _add_pothole_points(self, points_list, intensity_list, labels_list):
        """Add pothole depressions on the road."""
        n_potholes = 8
        for _ in range(n_potholes):
            cx = np.random.uniform(-35, 35)
            cy = np.random.uniform(-2, 2)
            radius = np.random.uniform(0.3, 0.8)
            depth = self.pothole_depth * np.random.uniform(0.5, 1.5)
            
            n_pts = 200
            angles = np.random.uniform(0, 2 * np.pi, n_pts)
            r = np.random.uniform(0, radius, n_pts)
            xs = cx + r * np.cos(angles)
            ys = cy + r * np.sin(angles)
            zs = -depth * (1 - r / radius) + np.random.normal(0, 0.01, n_pts)
            
            mask = np.abs(ys) < self.road_width / 2
            
            points_list.extend(zip(xs[mask], ys[mask], zs[mask]))
            intensity_list.extend(np.random.uniform(0.3, 0.5, np.sum(mask)))
            labels_list.extend([self.LABEL_STATIC] * np.sum(mask))
    
    def _add_cylinder_points(self, center, height, radius, 
                             points_list, intensity_list, labels_list, label):
        """Generate points on a vertical cylinder."""
        n_points = 300
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        heights = np.random.uniform(0, height, n_points)
        
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        zs = heights
        
        points_list.extend(zip(xs, ys, zs))
        intensity_list.extend(np.random.uniform(0.4, 0.7, n_points))
        labels_list.extend([label] * n_points)
    
    def _add_box_points(self, center, width, depth, height,
                        points_list, intensity_list, labels_list, label):
        """Generate points on the surface of a rectangular box."""
        n_points = 400
        face = np.random.choice(['top', 'bottom', 'front', 'back', 'left', 'right'], 
                                n_points, p=[0.3, 0.05, 0.2, 0.2, 0.15, 0.1])
        
        xs, ys, zs = [], [], []
        for f in face:
            if f == 'top':
                xs.append(center[0] + np.random.uniform(-width/2, width/2))
                ys.append(center[1] + np.random.uniform(-depth/2, depth/2))
                zs.append(center[2] + height + np.random.normal(0, 0.02))
            elif f == 'bottom':
                xs.append(center[0] + np.random.uniform(-width/2, width/2))
                ys.append(center[1] + np.random.uniform(-depth/2, depth/2))
                zs.append(center[2] + np.random.normal(0, 0.02))
            elif f == 'front':
                xs.append(center[0] + np.random.uniform(-width/2, width/2))
                ys.append(center[1] + depth/2 + np.random.normal(0, 0.02))
                zs.append(center[2] + np.random.uniform(0, height))
            elif f == 'back':
                xs.append(center[0] + np.random.uniform(-width/2, width/2))
                ys.append(center[1] - depth/2 + np.random.normal(0, 0.02))
                zs.append(center[2] + np.random.uniform(0, height))
            elif f == 'left':
                xs.append(center[0] - width/2 + np.random.normal(0, 0.02))
                ys.append(center[1] + np.random.uniform(-depth/2, depth/2))
                zs.append(center[2] + np.random.uniform(0, height))
            else:
                xs.append(center[0] + width/2 + np.random.normal(0, 0.02))
                ys.append(center[1] + np.random.uniform(-depth/2, depth/2))
                zs.append(center[2] + np.random.uniform(0, height))
        
        points_list.extend(zip(xs, ys, zs))
        intensity_list.extend(np.random.uniform(0.3, 0.6, n_points))
        labels_list.extend([label] * n_points)
    
    def _add_tree_points(self, tree, points_list, intensity_list, labels_list):
        """Generate points for a tree (trunk + canopy)."""
        center = tree['pos']
        trunk_h = tree['trunk_height']
        trunk_r = tree['trunk_radius']
        canopy_r = tree['canopy_radius']
        
        n_trunk = 150
        n_canopy = 300
        
        # Trunk
        trunk_angles = np.random.uniform(0, 2 * np.pi, n_trunk)
        trunk_heights = np.random.uniform(0, trunk_h, n_trunk)
        trunk_xs = center[0] + trunk_r * np.cos(trunk_angles)
        trunk_ys = center[1] + trunk_r * np.sin(trunk_angles)
        trunk_zs = trunk_heights
        
        # Canopy (spherical)
        canopy_theta = np.random.uniform(0, 2 * np.pi, n_canopy)
        canopy_phi = np.random.uniform(0, np.pi, n_canopy)
        canopy_r = canopy_r * np.cbrt(np.random.uniform(0, 1, n_canopy))
        canopy_xs = center[0] + canopy_r * np.sin(canopy_phi) * np.cos(canopy_theta)
        canopy_ys = center[1] + canopy_r * np.sin(canopy_phi) * np.sin(canopy_theta)
        canopy_zs = trunk_h + canopy_r * np.cos(canopy_phi)
        
        xs = np.concatenate([trunk_xs, canopy_xs])
        ys = np.concatenate([trunk_ys, canopy_ys])
        zs = np.concatenate([trunk_zs, canopy_zs])
        
        points_list.extend(zip(xs, ys, zs))
        intensity_list.extend(np.random.uniform(0.2, 0.5, len(xs)))
        labels_list.extend([self.LABEL_STATIC] * len(xs))
    
    def _random_rotation(self, angle_mag):
        """Generate a small random rotation matrix."""
        angle = np.random.uniform(-angle_mag, angle_mag)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    def generate_sequence(self, n_frames: int, dt: float = 0.1) -> list:
        """Generate a sequence of frames simulating vehicle motion."""
        frames = []
        for i in range(n_frames):
            t = i * dt
            frame = self.generate_frame(t)
            frames.append(frame)
        return frames
