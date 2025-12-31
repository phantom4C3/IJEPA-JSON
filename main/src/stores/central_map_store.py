 

#!/usr/bin/env python3
"""
CENTRAL MAP STORE - Single source of truth for all mapping data
Manages geometric, semantic, and spatial state with thread-safe operations
"""

import threading
import time
import json
import pickle
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import deque, defaultdict
import logging
from src.stores.frame_buffer import global_frame_buffer
import os


class MapUpdateType(Enum):
    GEOMETRIC_POINTS = "geometric_points"
    OBJECT_DETECTION = "object_detection"
    ROOM_CHANGE = "room_change"
    POSE_UPDATE = "pose_update"
    OBSTACLE_UPDATE = "obstacle_update"


@dataclass
class GeometricPoint:
    x: float
    y: float
    z: float
    timestamp: float
    source: str = "orb_slam"
    confidence: float = 1.0


@dataclass
class SemanticObject:
    name: str
    position: List[float]  # [x, y, z]
    confidence: float
    first_seen: float
    last_seen: float
    bbox: Optional[List[float]] = None
    attributes: Dict[str, Any] = None
    # ✅ ADD FRAME TRACKING FIELDS:
    first_frame_seen: int = 0
    last_frame_seen: int = 0
    observation_count: int = 1

    def __post_init__(self):
        """Initialize default values for mutable fields"""
        if self.attributes is None:
            self.attributes = {}


@dataclass
class RoomTransition:
    room_type: str
    timestamp: float
    position: List[float]  # Where transition occurred


@dataclass
class CameraPose:
    position: List[float]  # [x, y, z] - for visualization
    rotation_quat: List[float]  # [qx, qy, qz, qw] - clear format
    transform_matrix: List[List[float]]  # 4x4 matrix - for 3D projection ✅
    timestamp: float
    frame_id: int  # ✅ CRITICAL for frame matching
    tracking_quality: float = 1.0

    @classmethod
    def from_transform_matrix(
        cls,
        transform_4x4: np.ndarray,
        frame_id: int,
        timestamp: float = None,
        tracking_quality: float = 1.0,
    ):
        """Create from 4x4 transform matrix (what OWL needs)"""
        if timestamp is None:
            timestamp = time.time()

        # Extract position from translation vector
        position = transform_4x4[:3, 3].tolist()

        # Extract rotation matrix and convert to quaternion (simplified)
        rotation_matrix = transform_4x4[:3, :3]

        # Simple rotation matrix to quaternion conversion
        # For now, use identity quaternion - replace with proper conversion if needed
        trace = np.trace(rotation_matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        else:
            # Handle other cases...
            qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

        rotation_quat = [qx, qy, qz, qw]

        return cls(
            position=position,
            rotation_quat=rotation_quat,
            transform_matrix=transform_4x4.tolist(),
            timestamp=timestamp,
            frame_id=frame_id,
            tracking_quality=tracking_quality,
        )


class CentralMapStore:
    """
    Centralized store for all mapping data - thread-safe and optimized for frequent queries
    """

    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.logger = logging.getLogger("CentralMapStore")

        # GEOMETRIC DATA (from ORB-SLAM)
        self.geometric_points: List[GeometricPoint] = []
        # ✅ OPTIMIZED POSE STORAGE:
        self.current_pose: Optional[CameraPose] = None
        self.recent_poses: deque = deque(maxlen=50)  # Keep last 50 for debugging
        self.pose_by_frame: Dict[int, CameraPose] = (
            {}
        )  # Frame_id -> pose (last 1000 frames)
        self.trajectory: List[Tuple[float, float, float]] = []  # Simplified path

        # SEMANTIC DATA (from Perception)
        self.semantic_objects: Dict[str, SemanticObject] = {}  # name -> object
        self.object_categories: Dict[str, List[str]] = defaultdict(
            list
        )  # category -> [names]
        
        self.collision_points = []
        
        
        
        
        
        
        
        
        
        
        
        
        
        # ✅ ADD OCCUPANCY GRID PARAMETERS (SemExp style)
        self.grid_resolution = 0.05  # 5cm per cell
        self.map_size_cm = 1000  # 10m x 10m world (1000cm)
        self.grid_size = self.map_size_cm // int(1/self.grid_resolution)  # 1000 cells
        
        self.occupancy_grid_initialized = False
        self.occupancy_grid = None
        
        
        self.grid_origin = (-5.0, -5.0)  # World coordinates [-5m, +5m] centered at 0,0
        
        
          
          
          
          
                # ❌ MISSING THESE 5 LINES:
        self.occupancy_grid_2d = None      # Your grid array ✓ 
        self.map_initialized = False        # ❌ ADD THIS






        # SPATIAL CONTEXT (from CLIP + ORB-SLAM)
        self.room_history: List[RoomTransition] = []
        self.current_room: str = "unknown"
        self.explored_positions: Set[Tuple[float, float]] = set()  # (x,y) tuples

        self.keyframes: Dict[int, Dict] = {}
        self.keyframe_embeddings: Dict[int, np.ndarray] = {}  # For I-JEPA

        # TEMPORAL TRACKING
        self.last_update_time: float = time.time()
        self.creation_time: float = time.time()
        self.update_counters: Dict[MapUpdateType, int] = {
            update_type: 0 for update_type in MapUpdateType
        }

        # CHANGE DETECTION
        self._last_query_time: float = 0.0
        self._recent_updates: deque = deque(
            maxlen=100
        )  # Recent changes for notifications

        # PERFORMANCE OPTIMIZATION
        self._spatial_index = None  # For future spatial queries
        self._object_position_index: Dict[str, List[float]] = {}  # name -> position

        print("Central Map Store initialized")

    # ✅ ADD THIS METHOD RIGHT HERE - Global access point
    @classmethod
    def get_global(cls):
        """
        Get the single shared instance from ANYWHERE without passing instances
        Usage: CentralMapStore.get_global().get_map_summary()
        """
        if cls._instance is None:
            # Create instance if it doesn't exist
            cls._instance = CentralMapStore()
        return cls._instance

    # ==================== GEOMETRIC DATA METHODS ====================

    # Add method:
    def store_keyframe(self, frame_id: int, keyframe_data: Dict):
        self.keyframes[frame_id] = keyframe_data

    def add_geometric_points_rich(
        self, points: List[Dict], timestamp: float = None, source: str = "orb_slam"
    ):
        """Add rich format map points with metadata"""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            new_points = []
            for i, point_data in enumerate(points):  # 🎯 ADD ENUMERATE
                try:
                    if (
                        "position_3d" in point_data
                        and len(point_data["position_3d"]) >= 3
                    ):
                        pos = point_data["position_3d"]
                        geom_point = GeometricPoint(
                            x=float(pos[0]),
                            y=float(pos[1]),
                            z=float(pos[2]),
                            timestamp=timestamp
                            + (i * 0.000001),  # 🎯 FIX: Unique timestamp for each point
                            source=source,
                        )

                        # ✅ CRITICAL: Store the rich data for later saving
                        geom_point.rich_data = point_data
                        new_points.append(geom_point)

                except Exception as e:
                    continue

            self.geometric_points.extend(new_points)
            return len(new_points)

    # Add these to CentralMapStore class:

    def update_semantic_data(self, semantic_data: Dict[str, Any]):
        """Interface expected by OWL Integration - MERGES data"""
        semantic_objects = semantic_data.get("semantic_objects", {})

        objects_dict = {}
        for obj_id, obj_data in semantic_objects.items():
            existing_obj = self.semantic_objects.get(obj_id)

            if existing_obj:
                # ... existing merge logic ...
                objects_dict[obj_id] = {
                    "confidence": max(existing_obj.confidence, obj_data["confidence"]),
                    "attributes": {
                        "class_id": obj_data["class_id"],
                        "status": obj_data["status"],
                        "first_seen": existing_obj.first_seen,
                        "observation_count": existing_obj.attributes.get(
                            "observation_count", 0
                        )
                        + 1,
                        "category": obj_data["label"],  # ✅ ADD CATEGORY
                    },
                }
            else:
                objects_dict[obj_id] = {
                    "position": obj_data["position_3d"],
                    "confidence": obj_data["confidence"],
                    "attributes": {
                        "class_id": obj_data["class_id"],
                        "status": obj_data["status"],
                        "first_seen": obj_data.get("first_seen", time.time()),
                        "observation_count": 1,
                        "category": obj_data["label"],  # ✅ ADD CATEGORY
                    },
                }

        self.update_objects(objects_dict)

    def update_objects(
        self,
        objects_dict: Dict[str, Dict[str, Any]],
        timestamp: float = None,
        frame_id: int = None,  # ✅ ADD FRAME ID PARAMETER
    ):
        """FIXED: Preserves existing object data with proper frame tracking"""
        if timestamp is None:
            timestamp = time.time()

        if frame_id is None:
            frame_id = getattr(self, "current_frame_id", 0)

        with self._lock:
            new_objects = []
            updated_objects = []

            for obj_name, obj_data in objects_dict.items():
                position = obj_data.get("position", [0, 0, 0])
                confidence = obj_data.get("confidence", 0.0)
                attributes = obj_data.get("attributes", {})

                # ✅ EXTRACT OR USE CURRENT FRAME ID
                obj_frame_id = attributes.get("frame_id", frame_id)

                if obj_name in self.semantic_objects:
                    # ✅ UPDATE EXISTING OBJECT WITH FRAME TRACKING
                    existing = self.semantic_objects[obj_name]

                    # Update position if valid new data
                    if position != [0, 0, 0]:
                        existing.position = position

                    # Keep highest confidence
                    existing.confidence = max(existing.confidence, confidence)

                    # Update timestamps
                    existing.last_seen = timestamp

                    # ✅ UPDATE FRAME TRACKING
                    existing.last_frame_seen = obj_frame_id
                    existing.observation_count += 1

                    # Merge attributes
                    existing.attributes.update(attributes)
                    updated_objects.append(obj_name)

                else:
                    # ✅ CREATE NEW OBJECT WITH COMPLETE FRAME TRACKING
                    new_obj = SemanticObject(
                        name=obj_name,
                        position=position.copy(),
                        confidence=confidence,
                        first_seen=timestamp,  # Use current timestamp for new objects
                        last_seen=timestamp,
                        bbox=None,
                        attributes=attributes.copy(),
                        # ✅ INITIALIZE FRAME TRACKING
                        first_frame_seen=obj_frame_id,
                        last_frame_seen=obj_frame_id,
                        observation_count=1,
                    )
                    self.semantic_objects[obj_name] = new_obj
                    new_objects.append(obj_name)

                    # Update category indexing
                    category = attributes.get("category", "unknown")
                    self.object_categories[category].append(obj_name)

            # Log and record updates
            if new_objects or updated_objects:
                self._record_update(MapUpdateType.OBJECT_DETECTION, len(new_objects))
                print(
                    f"Objects updated: {len(new_objects)} new, {len(updated_objects)} updated"
                )

            return new_objects

    def update_transform_data(self, data: Dict[str, Any]):
        """Store ALL map data including coordinate transforms - RENAMED to avoid conflict"""
        print(f"\n🎯 MAPSTORE.update_map_data called:")
        print(f"   📍 Data keys: {list(data.keys())}")

        # Store coordinate transforms if present
        if "coordinate_transforms" in data:
            if not hasattr(self, "_coordinate_transforms"):
                self._coordinate_transforms = {}
            self._coordinate_transforms.update(data["coordinate_transforms"])
            print(
                f"   ✅ TRANSFORM STORED: {list(data['coordinate_transforms'].keys())}"
            )

            # ALSO store as public attribute for OWL
            self.coordinate_transforms = data["coordinate_transforms"].copy()
            print(f"   💾 Also stored as self.coordinate_transforms")

        # Handle semantic data
        if "semantic" in data:
            self.update_semantic_data(data["semantic"])

    @property
    def semantic_data(self):
        """Interface expected by OWL Integration - now includes frame tracking"""
        return {
            "semantic_objects": {
                name: {
                    "object_id": obj.name,
                    "label": obj.name.split("_")[0],
                    "position_3d": obj.position,
                    "confidence": obj.confidence,
                    "first_seen": obj.first_seen,
                    "last_seen": obj.last_seen,
                    # ✅ INCLUDE FRAME TRACKING IN OUTPUT
                    "first_frame_seen": obj.first_frame_seen,
                    "last_frame_seen": obj.last_frame_seen,
                    "observation_count": obj.observation_count,
                    "class_id": obj.attributes.get("class_id", 0),
                    "category": obj.attributes.get("category", "unknown"),
                    "status": obj.attributes.get("status", "active"),
                }
                for name, obj in self.semantic_objects.items()
            }
        }

    def save_semantic_map(self, semantic_objects=None):
        """Interface expected by OWL Integration"""
        self.save_map("semantic_map.pkl")  # Or use provided objects

    def update_camera_pose(
        self,
        transform_4x4: np.ndarray,
        frame_id: int,
        timestamp: float = None,
        tracking_quality: float = 1.0,
    ) -> bool:
        """Update current camera pose with 4x4 transform matrix (what OWL needs)"""
        if timestamp is None:
            timestamp = time.time()

        with self._lock:
            # Create proper CameraPose with 4x4 matrix
            self.current_pose = CameraPose.from_transform_matrix(
                transform_4x4, frame_id, timestamp, tracking_quality
            )

            # Store for frame matching
            self.pose_by_frame[frame_id] = self.current_pose

            # Keep recent poses (optional, for debugging)
            self.recent_poses.append(self.current_pose)

            # Clean up old poses (keep last 1000 frames)
            if len(self.pose_by_frame) > 1000:
                oldest_frame = min(self.pose_by_frame.keys())
                del self.pose_by_frame[oldest_frame]

            # Add to trajectory (simplified position only)
            position = self.current_pose.position
            if len(position) >= 3:
                self.trajectory.append((position[0], position[1], position[2]))

            # Track explored area
            if len(position) >= 2:
                pos_key = (round(position[0], 1), round(position[1], 1))
                self.explored_positions.add(pos_key)

            self._record_update(MapUpdateType.POSE_UPDATE, 1)
            return True

    def update_robot_position(self, position, frame_id: int, timestamp: float):
        """Store current robot position as simple [x, y, z] coordinates"""
        if not hasattr(self, "_robot_positions"):
            self._robot_positions = {}  # Initialize if not exists

        self._robot_positions[frame_id] = {
            "position": position,
            "timestamp": timestamp,
            "frame_id": frame_id,
        }
        print(f"📦 Stored complete 4x4 pose matrix for frame {frame_id}")

        # Keep only recent positions to save memory
        if len(self._robot_positions) > 100:
            oldest_key = min(self._robot_positions.keys())
            del self._robot_positions[oldest_key]

    def get_current_robot_position(self) -> list:
        """Get latest robot position as [x, y, z]"""
        if not hasattr(self, "_robot_positions") or not self._robot_positions:
            return [0, 0, 0]  # Default fallback

        latest_frame = max(self._robot_positions.keys())
        return self._robot_positions[latest_frame]["position"]

    def get_current_pose_matrix(self) -> Optional[np.ndarray]:
        """Get current pose as 4x4 matrix for OWL projection (FAST ACCESS)"""
        with self._lock:
            if self.current_pose and self.current_pose.transform_matrix:
                return np.array(self.current_pose.transform_matrix)
            return None

    def get_pose_for_frame(self, frame_id: int) -> Optional[CameraPose]:
        """Get exact pose for specific frame (if available)"""
        with self._lock:
            # Return exact frame pose if available, otherwise current pose
            return self.pose_by_frame.get(frame_id)

    def get_pose_matrix_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """Get 4x4 transform matrix for specific frame"""
        with self._lock:
            pose = self.pose_by_frame.get(frame_id)
            if pose and pose.transform_matrix:
                return np.array(pose.transform_matrix)
            return None
        
        
    def transform_all_stored_poses_before_alignment(self, umeyama_aligner):
        """🚀 Transform ALL historical poses to Habitat world coordinates"""
        if not umeyama_aligner or not umeyama_aligner.current_transform:
            print("❌ No Umeyama transform available for historical poses")
            return 0
        
        transformed_count = 0
        
        with self.lock:
            # 1️⃣ Transform pose_by_frame (ALL keyframes history)
            for frame_id, pose_obj in list(self.pose_by_frame.items()):
                if hasattr(pose_obj, 'transform_matrix') and pose_obj.transform_matrix:
                    try:
                        slam_pose = np.array(pose_obj.transform_matrix)
                        world_pose = umeyama_aligner.transform_pose(slam_pose)
                        pose_obj.transform_matrix = world_pose.tolist()
                        transformed_count += 1
                    except Exception as e:
                        print(f"⚠️ Failed to transform pose {frame_id}: {e}")
            
            # 2️⃣ Transform robot_positions (trajectory history)
            if hasattr(self, 'robot_positions') and self.robot_positions:
                for frame_id, data in list(self.robot_positions.items()):
                    if 'position' in data and len(data['position']) == 3:
                        try:
                            # Reconstruct 4x4 from position
                            pose_4x4 = np.eye(4)
                            pose_4x4[:3, 3] = data['position']
                            world_pose = umeyama_aligner.transform_pose(pose_4x4)
                            data['position'] = world_pose[:3, 3].tolist()
                            transformed_count += 1
                        except Exception as e:
                            print(f"⚠️ Failed to transform robot pos {frame_id}: {e}")
        
        print(f"✅ Fix2 COMPLETE: Transformed {transformed_count} historical poses!")
        return transformed_count
 
    def update_camera_parameters(self, params: Dict[str, Any]):
        """Store camera parameters for other modules to use"""
        self.camera_parameters = params

    # In MapStore class:
    def get_camera_parameters(self) -> Optional[Dict[str, Any]]:
        """Get camera parameters - public interface"""
        # Instead of just returning self.camera_parameters,
        # use the sophisticated method:
        if hasattr(self, "_get_camera_parameters_for_orb_slam"):
            return self._get_camera_parameters_for_orb_slam()
        else:
            # Fallback to attribute
            return getattr(self, "camera_parameters", None)

    def _get_camera_parameters_for_orb_slam(self) -> Dict[str, Any]:
        """Get camera parameters - PRIORITY: Use saved ORB-SLAM params first"""

        # 🥇 PRIORITY 1: Use parameters from saved ORB-SLAM map (most reliable)
        orb_slam_map_params = self._get_camera_params_from_saved_orb_slam_map()
        if orb_slam_map_params:
            self.camera_parameters = orb_slam_map_params
            print("✅ Using camera parameters from saved ORB-SLAM map")
            return orb_slam_map_params

        # 🥈 PRIORITY 2: Already stored parameters in MapStore
        if hasattr(self, "camera_parameters") and self.camera_parameters:
            print("✅ Using camera parameters from MapStore cache")
            return self.camera_parameters

        # # 🥉 PRIORITY 3: Extract from Habitat simulation
        # habitat_params = self._extract_from_habitat()
        # if habitat_params:
        #     self.camera_parameters = habitat_params
        #     print("✅ Using camera parameters from Habitat simulation")
        #     return habitat_params

        # 🏅 PRIORITY 4: Fallback to defaults (shouldn't happen if ORB-SLAM worked)
        fallback_params = {
            "fx": 535.4,
            "fy": 539.2,
            "cx": 320.1,
            "cy": 247.6,
            "width": 640,
            "height": 480,
            "source": "fallback_defaults_emergency",
            "config_path": "orb_slam_map_not_found",
        }
        self.camera_parameters = fallback_params
        self.logger.warning("⚠️ Using fallback camera parameters")
        return fallback_params
    
    
    
    def add_collision(self, position: List[float], timestamp: float = None):
        """Add a collision location to the map store"""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Initialize collision storage if it doesn't exist
            if not hasattr(self, 'collision_points'):
                self.collision_points = []
                print("✅ Initialized collision_points storage in MapStore")
            
            # Create a collision point
            collision_point = {
                'position': position[:3],  # Ensure [x, y, z]
                'timestamp': timestamp,
                'type': 'collision',
                'source': 'action_executor'
            }
            
            # Add to collision points
            self.collision_points.append(collision_point)
            
            # Also mark as obstacle in geometric points
            geo_point = GeometricPoint(
                x=position[0],
                y=position[1],
                z=position[2],
                timestamp=timestamp,
                source='collision_detection',
                confidence=1.0
            )
            self.geometric_points.append(geo_point)
            
            # Record the update
            self._record_update(MapUpdateType.OBSTACLE_UPDATE, 1)
            
            print(f"📍 Collision recorded at position: {position[:3]}")
            return True
    
    
    
    

    def _get_camera_params_from_saved_orb_slam_map(self) -> Optional[Dict[str, Any]]:
        """Extract camera parameters from the latest saved ORB-SLAM map"""
        try:
            # Look for the most recent ORB-SLAM map file
            orb_slam_map_path = self._find_latest_orb_slam_map()
            if orb_slam_map_path and os.path.exists(orb_slam_map_path):
                with open(orb_slam_map_path, "r") as f:
                    orb_slam_data = json.load(f)

                # Extract camera parameters from ORB-SLAM map
                if "camera_parameters" in orb_slam_data:
                    params = orb_slam_data["camera_parameters"]
                    print(
                        f"📷 Loaded camera params from ORB-SLAM map: {params['source']}"
                    )
                    return params

        except Exception as e:
            self.logger.warning(f"Could not load camera params from ORB-SLAM map: {e}")

        return None

    def _find_latest_orb_slam_map(self) -> Optional[str]:
        """Find the most recent ORB-SLAM map file"""
        try:
            # Common locations where ORB-SLAM saves maps
            possible_dirs = [
                "experiments/results/logs/",
                "projects/hybrid_zero_shot_slam_nav/main/experiments/results/logs/",
                "./output/maps/",
                "./logs/",
                "../results/logs/",
            ]

            orb_slam_maps = []
            for directory in possible_dirs:
                if os.path.exists(directory):
                    for file in os.listdir(directory):
                        if file.startswith(
                            ("orbslam_map", "orbslam_only_map")
                        ) and file.endswith(".json"):
                            filepath = os.path.join(directory, file)
                            orb_slam_maps.append(filepath)

            # Return the most recently modified file
            if orb_slam_maps:
                latest_map = max(orb_slam_maps, key=os.path.getmtime)
                print(f"📍 Found ORB-SLAM map: {latest_map}")
                return latest_map

        except Exception as e:
            self.logger.debug(f"Could not find ORB-SLAM map: {e}")

    # ==================== SPATIAL CONTEXT METHODS ====================
    def get_map_points_for_frame(self, frame_id: int) -> List[Any]:
        """
        Get map points associated with a specific frame
        """
        with self._lock:
            if frame_id in self.geometric_points:
                frame_points = self.geometric_points[frame_id]

                # Convert to rich format
                rich_points = []
                for i, geom_point in enumerate(frame_points):
                    # Same conversion logic as above...
                    pass

                return rich_points
            else:
                return []  # No points for this frame

    def get_all_map_points(self) -> List[Any]:
        """
        Get all map points regardless of frame association
        NOW HANDLES BOTH LIST AND DICT FORMATS
        """
        with self._lock:
            try:
                all_points = []

                # ✅ CHECK TYPE: Could be list OR dict!
                if isinstance(self.geometric_points, list):
                    # Format 1: Direct list of points
                    for i, point in enumerate(self.geometric_points):
                        point_dict = self._convert_point_to_dict(point, i)
                        if point_dict:
                            all_points.append(point_dict)

                elif isinstance(self.geometric_points, dict):
                    # Format 2: Dictionary of {frame_id: [points]}
                    for frame_id, frame_points in self.geometric_points.items():
                        if not frame_points:
                            continue
                        for i, point in enumerate(frame_points):
                            point_dict = self._convert_point_to_dict(point, i)
                            if point_dict:
                                point_dict["frame_id"] = (
                                    frame_id  # Add frame association
                                )
                                all_points.append(point_dict)
                else:
                    print(
                        f"⚠️ geometric_points is unexpected type: {type(self.geometric_points)}"
                    )

                print(f"🔍 get_all_map_points: Found {len(all_points)} points")
                return all_points

            except Exception as e:
                print(f"❌ get_all_map_points error: {e}")
                import traceback

                traceback.print_exc()
                return []  # Always return list

    def _convert_point_to_dict(self, point, point_id):
        """Convert any point format to dictionary"""
        point_dict = {}

        if isinstance(point, dict):
            # Already a dictionary
            point_dict = point.copy()
            point_dict["point_id"] = point_id

        elif hasattr(point, "x") and hasattr(point, "y") and hasattr(point, "z"):
            # Point object with x,y,z attributes
            point_dict = {
                "point_id": point_id,
                "position_3d": [float(point.x), float(point.y), float(point.z)],
                "source": getattr(point, "source", "orb_slam3"),
                "timestamp": getattr(point, "timestamp", time.time()),
            }

        elif isinstance(point, (list, tuple)) and len(point) >= 3:
            # Raw coordinates [x, y, z]
            point_dict = {
                "point_id": point_id,
                "position_3d": [float(point[0]), float(point[1]), float(point[2])],
                "source": "unknown",
                "timestamp": time.time(),
            }

        return point_dict if point_dict else None

    def update_room_context(
            self, room_type: str, position: List[float] = None, timestamp: float = None
        ) -> bool:
            """Update current room and track room transitions"""
            if timestamp is None:
                timestamp = time.time()

            with self._lock:
                room_changed = False

                # Check if room actually changed
                if room_type != self.current_room and room_type != "unknown":
                    room_changed = True

                    # Use current pose if position not provided
                    if position is None and self.current_pose:
                        position = self.current_pose.position

                    # Record room transition
                    transition = RoomTransition(
                        room_type=room_type,
                        timestamp=timestamp,
                        position=position.copy() if position else [0, 0, 0],
                    )
                    self.room_history.append(transition)
                    self.current_room = room_type

                    self._record_update(MapUpdateType.ROOM_CHANGE, 1)
                    print(f"Room changed to: {room_type}")

                return room_changed

    # ==================== QUERY METHODS ====================

    def get_map_summary(self) -> Dict[str, Any]:
        """Get lightweight summary of current map state"""
        with self._lock:
            # 🎯 ADD DEBUG HERE:
            print(f"🔍 CENTRAL MAP STORE DEBUG:")
            print(f"   self.current_pose type: {type(self.current_pose)}")
            print(f"   self.current_pose value: {self.current_pose}")

            # Get found targets (objects that are also in target list)
            target_objects = self.config.get("target_objects", [])
            found_targets = [
                obj for obj in self.semantic_objects.keys() if obj in target_objects
            ]

            summary = {
                # Geometric metrics
                "geometric_points_count": len(self.geometric_points),
                "trajectory_length": len(self.trajectory),
                "explored_positions_count": len(self.explored_positions),
                "stored_poses_count": len(self.pose_by_frame),  # ✅ NEW
                # Semantic metrics
                "objects_count": len(self.semantic_objects),
                "found_targets": found_targets,
                "remaining_targets": list(set(target_objects) - set(found_targets)),
                # Spatial context
                "current_room": self.current_room,
                "rooms_visited_count": len(
                    set(transition.room_type for transition in self.room_history)
                ),
                "room_history": [
                    transition.room_type for transition in self.room_history
                ],
                "current_pose": (
                    asdict(self.current_pose)
                    if self.current_pose
                    and hasattr(self.current_pose, "__dataclass_fields__")
                    else None
                ),
                "has_transform_matrix": (
                    self.current_pose.transform_matrix is not None
                    if self.current_pose
                    else False
                ),  # ✅ NEW
                "last_update_time": self.last_update_time,
                "uptime_seconds": time.time() - self.creation_time,
                # Performance
                "update_counts": self.update_counters.copy(),
            }

            # Calculate exploration progress (simplified)
            if self.explored_positions:
                summary["exploration_progress"] = min(
                    1.0, len(self.explored_positions) / 1000.0
                )
            else:
                summary["exploration_progress"] = 0.0

            return summary

    def get_objects_in_radius(
        self, position: List[float], radius: float
    ) -> List[SemanticObject]:
        """Get objects within radius of position (simplified)"""
        with self._lock:
            nearby_objects = []
            if len(position) < 3:
                return nearby_objects

            for obj in self.semantic_objects.values():
                if len(obj.position) >= 3:
                    distance = np.linalg.norm(
                        np.array(position) - np.array(obj.position)
                    )
                    if distance <= radius:
                        nearby_objects.append(obj)

            return nearby_objects

    def get_recent_changes(self, since_timestamp: float) -> Dict[str, Any]:
        """Get changes since specified timestamp"""
        with self._lock:
            changes = {
                "new_objects": [],
                "room_changes": [],
                "geometric_updates": 0,
                "pose_updates": 0,
            }

            # Find new objects
            for obj_name, obj in self.semantic_objects.items():
                if obj.first_seen > since_timestamp:
                    changes["new_objects"].append(obj_name)

            # Find room changes
            for transition in self.room_history:
                if transition.timestamp > since_timestamp:
                    changes["room_changes"].append(transition.room_type)

            # Count other updates from recent updates log
            for update in self._recent_updates:
                if update["timestamp"] > since_timestamp:
                    if update["type"] == MapUpdateType.GEOMETRIC_POINTS:
                        changes["geometric_updates"] += update["count"]
                    elif update["type"] == MapUpdateType.POSE_UPDATE:
                        changes["pose_updates"] += update["count"]

            return changes

    def save_map(self, save_dir: str = "./output/maps") -> str:
        """Use the OLD WORKING version that treats geometric_points as LIST"""
        try:
            with self._lock:
                current_time = time.time()

                # ✅ Use OLD LOGIC that works with LIST
                rich_map_points = []
                if hasattr(self, "geometric_points") and self.geometric_points:
                    # This assumes self.geometric_points is a LIST
                    for i, geometric_point in enumerate(self.geometric_points):
                        try:
                            # Start with basic point data
                            point_data = {
                                "point_id": i,
                                "is_tracked": True,
                                "position_3d": [
                                    float(geometric_point.x),
                                    float(geometric_point.y),
                                    float(geometric_point.z),
                                ],
                                "source": getattr(
                                    geometric_point, "source", "orb_slam3"
                                ),
                                "point_timestamp": getattr(
                                    geometric_point, "timestamp", current_time
                                ),
                            }

                            # Add rich data if available
                            if (
                                hasattr(geometric_point, "rich_data")
                                and geometric_point.rich_data
                            ):
                                point_data.update(geometric_point.rich_data)
                            else:
                                point_data["raw_data"] = {
                                    "id": i,
                                    "x": float(geometric_point.x),
                                    "y": float(geometric_point.y),
                                    "z": float(geometric_point.z),
                                }

                            rich_map_points.append(point_data)
                        except Exception as point_error:
                            print(f"⚠️ Point {i} conversion error: {point_error}")
                            continue
 

                # ✅ BUILD MAP DATA WITH FALLBACKS FOR ALL FIELDS
                map_data = {
                    # Basic frame info
                    "latest_frame_id": getattr(self, "latest_frame_id", 0),

                    "map_points_count": len(rich_map_points),
                    "map_points": rich_map_points,  # Could be empty list
                    
                    "timestamp": current_time,
                    "tracking_state": "COMPLETED",
                    # System stats
                    "system_stats": {
                        "tracking_state": "COMPLETED",
                        "map_points_count": len(rich_map_points),
                        "keyframes_count": len(getattr(self, "keyframes", [])),

                        "is_initialized": getattr(self, "is_initialized", True),
                    },
                    # Camera parameters
                    "camera_parameters": getattr(
                        self, "_get_camera_parameters_for_orb_slam", lambda: {}
                    )(),
                    # Coordinate system
                    "coordinate_system": {
                        "slam_to_world_transform": None,
                        "world_to_slam_transform": None,
                        "alignment_status": "not_aligned",
                        "source": "orb_slam3_monocular",
                    },
                    # Processing info
                    "processing_info": {
                        "is_initialized": True,
                        "is_running": False,
                        "config_path": "orb_slam3_config.yaml",
                        "vocabulary_path": "ORBvoc.txt",
                    },
                    # File info
                    "file_info": {
                        "format_version": "1.1_enhanced",
                        "saved_from": "ORB_SLAM3_Integration",
                        "coordinate_system": "orb_slam3_world",
                        "save_timestamp": current_time,
                    },
                    # Habitat data (with fallbacks)
                    "habitat_data": {
                        "semantic_objects_count": len(
                            getattr(self, "semantic_objects", {})
                        ),
                        "rooms_visited": len(getattr(self, "room_history", [])),
                        "current_room": getattr(self, "current_room", "unknown"),
                        "explored_positions_count": len(
                            getattr(self, "explored_positions", [])
                        ),
                        "mission_duration": current_time
                        - getattr(self, "creation_time", current_time),
                        "save_status": "COMPLETE",
                    },
                    # Semantic objects (with fallback)
                    "semantic_objects": {
                        name: {
                            "object_id": obj.name,
                            "label": obj.name.split("_")[0],
                            "position_3d": obj.position,
                            "confidence": obj.confidence,
                            "first_seen": obj.first_seen,
                            "last_seen": obj.last_seen,
                            "observation_count": getattr(obj, "attributes", {}).get(
                                "observation_count", 1
                            ),
                            "category": getattr(obj, "attributes", {}).get(
                                "category", "unknown"
                            ),
                        }
                        for name, obj in getattr(self, "semantic_objects", {}).items()
                    },
                    "collision_data": {
    "collision_points_count": len(getattr(self, 'collision_points', [])),
    "collision_points": getattr(self, 'collision_points', []),
    "collision_density_per_meter": len(getattr(self, 'collision_points', [])) / max(1, len(getattr(self, 'trajectory', [])))
},
                    # Additional data from memory store
                    "pose_history": list(
                        getattr(self, "camera_pose_history", {}).values()
                    ),
                    "robot_trajectory": list(getattr(self, "robot_trajectory", [])),
                }

                # ✅ ALWAYS SAVE THE FILE REGARDLESS OF DATA
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                filename = f"orb_slam_map_{timestamp_str}.json"
                filepath = os.path.join(save_dir, filename)

                os.makedirs(save_dir, exist_ok=True)
                with open(filepath, "w") as f:
                    json.dump(map_data, f, indent=2)

                # ✅ ALSO SAVE NPZ FOR PYTHON USE
                if rich_map_points:
                    npz_filepath = filepath.replace(".json", ".npz")
                    positions = np.array(
                        [p["position_3d"] for p in rich_map_points], dtype=np.float32
                    )
                    np.savez_compressed(
                        npz_filepath,
                        positions=positions,
                        metadata={
                            "total_points": len(positions),
                            "format_version": "1.0",
                            "coordinate_system": "orb_slam3_world",
                        },
                    )
                    print(f"💾 NPZ saved: {npz_filepath}")

                    # ✅ ADD THIS: SAVE ORB-SLAM BIN FORMAT
                try:
                    if hasattr(self, "slam_system") and self.slam_system:
                        bin_filepath = filepath.replace(".json", ".bin")
                        # Check if SaveMap method exists
                        if hasattr(self.slam_system, "SaveMap"):
                            print(f"🔧 Attempting to save ORB-SLAM BIN format...")
                            self.slam_system.SaveMap(bin_filepath)
                            print(f"💾 BIN saved: {bin_filepath}")
                        else:
                            print(f"⚠️ ORB-SLAM system has no SaveMap method")

                        # Also try to save trajectory
                        traj_filepath = filepath.replace(".json", "_trajectory.txt")
                        if hasattr(self.slam_system, "SaveKeyFrameTrajectoryTUM"):
                            self.slam_system.SaveKeyFrameTrajectoryTUM(traj_filepath)
                            print(f"💾 Trajectory saved: {traj_filepath}")
                except Exception as bin_error:
                    print(f"⚠️ BIN save failed: {bin_error}")
 
                return filepath

        except Exception as e:
            # ✅ EVEN IF EXCEPTION, TRY TO SAVE MINIMAL JSON
            try:
                print(f"❌ Main save failed, attempting emergency save: {e}")
                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                emergency_filename = f"orb_slam_map_EMERGENCY_{timestamp_str}.json"
                emergency_filepath = os.path.join(save_dir, emergency_filename)

                emergency_data = {
                    "file_info": {
                        "format_version": "1.1_emergency",
                        "saved_from": "ORB_SLAM3_Integration",
                        "coordinate_system": "orb_slam3_world",
                        "save_timestamp": time.time(),
                        "save_status": "EMERGENCY_BACKUP",
                        "error": str(e),
                    },
                    "timestamp": time.time(),
                    "tracking_state": "ERROR",
                    "system_stats": {
                        "tracking_state": "ERROR",
                        "error_message": str(e),
                    },
                    "map_points": [],
                    "frames_data": [],
                }

                os.makedirs(save_dir, exist_ok=True)
                with open(emergency_filepath, "w") as f:
                    json.dump(emergency_data, f, indent=2)

                print(f"⚠️ EMERGENCY MAP SAVED: {emergency_filepath}")
                return emergency_filepath

            except Exception as emergency_e:
                print(f"🚨 CRITICAL: Even emergency save failed: {emergency_e}")
                return None

    # YES, but you need to ADD those data sources! Your current save_map() only saves:
    # ✅ Geometric points (ORB-SLAM map points)
    # ✅ Pose history (camera poses)
    # ✅ Robot trajectory
    # ✅ Semantic objects (OWL-ViT detections)
    # ❌ Missing: Collision data
    # ❌ Missing: Prediction data
    # ❌ Missing: Habitat state
    # ❌ Missing: OWL-ViT detection history
    
    
    
    
    def get_collision_points(self, radius: float = None, position: List[float] = None):
        """Get collision points, optionally filtered by radius"""
        with self._lock:
            if not hasattr(self, 'collision_points'):
                return []
            
            collision_points = self.collision_points
            
            # Filter by radius if specified
            if radius is not None and position is not None and len(position) >= 3:
                filtered_points = []
                for point in collision_points:
                    if len(point['position']) >= 3:
                        distance = np.linalg.norm(
                            np.array(position[:3]) - np.array(point['position'][:3])
                        )
                        if distance <= radius:
                            filtered_points.append(point)
                return filtered_points
            
            return collision_points

    def clear_collisions(self):
        """Clear all collision data"""
        with self._lock:
            if hasattr(self, 'collision_points'):
                self.collision_points.clear()
                print("🧹 Cleared all collision points")
                
                
                

    def load_map(self, filepath: str) -> bool:
        """Load map state from file"""
        try:
            with self._lock:
                with open(filepath, "rb") as f:
                    load_data = pickle.load(f)

                # Reconstruct geometric points
                self.geometric_points = [
                    GeometricPoint(**point_data)
                    for point_data in load_data["geometric_points"]
                ]

                # Reconstruct semantic objects
                self.semantic_objects = {
                    name: SemanticObject(**obj_data)
                    for name, obj_data in load_data["semantic_objects"].items()
                }

                # Reconstruct room history
                self.room_history = [
                    RoomTransition(**transition_data)
                    for transition_data in load_data["room_history"]
                ]

                # Load other data
                self.trajectory = load_data["trajectory"]
                self.explored_positions = set(load_data["explored_positions"])
                self.current_room = load_data["current_room"]

                # Update metadata
                if "metadata" in load_data:
                    self.creation_time = load_data["metadata"]["creation_time"]
                    self.last_update_time = load_data["metadata"]["last_update_time"]
                    self.update_counters = load_data["metadata"]["update_counts"]

                    # ✅ ADD KEYFRAME LOADING:
                if "keyframes" in load_data:
                    self.keyframes = load_data["keyframes"]
                if "keyframe_embeddings" in load_data:
                    self.keyframe_embeddings = {
                        k: np.array(v)
                        for k, v in load_data["keyframe_embeddings"].items()
                    }

                print(f"Map loaded from {filepath}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            return False

    # ==================== INTERNAL METHODS ====================

    def _record_update(self, update_type: MapUpdateType, count: int = 1):
        """Internal method to record updates for change tracking"""
        self.last_update_time = time.time()
        self.update_counters[update_type] += count

        # Record for recent changes
        self._recent_updates.append(
            {"type": update_type, "count": count, "timestamp": self.last_update_time}
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about map store"""
        with self._lock:
            return {
                "total_geometric_points": len(self.geometric_points),
                "total_semantic_objects": len(self.semantic_objects),
                "total_room_transitions": len(self.room_history),
                "total_trajectory_points": len(self.trajectory),
                "current_room": self.current_room,
                "explored_area": len(self.explored_positions),
                "uptime_hours": (time.time() - self.creation_time) / 3600,
                "update_frequency": {
                    update_type.value: count
                    for update_type, count in self.update_counters.items()
                },
            }

    def store_orb_slam_pose(
        self, pose_matrix: np.ndarray, frame_id: int, tracking_status: dict = None
    ):
        """Store ORB-SLAM pose with debugging - this is what's being called"""
        print(f"🎯 ORB-SLAM POSE DEBUG - Frame {frame_id}:")
        print(f"   🤖 Pose matrix shape: {pose_matrix.shape}")
        print(f"   📍 Translation values: {pose_matrix[:3, 3]}")
        print(f"   🎯 Is identity matrix: {np.allclose(pose_matrix, np.eye(4))}")

        # Check if we should update the position
        if np.allclose(pose_matrix, np.eye(4)):
            print("⚠️ ORB-SLAM returned identity matrix - tracking may be lost")
            # Don't update position if it's identity
            return False

        # Extract translation (last column)
        translation = pose_matrix[:3, 3].tolist()

        # Update using your existing method
        success = self.update_camera_pose(pose_matrix, frame_id, time.time())

        if success:
            print(f"🤖 ORB-SLAM: Robot at position {translation}")
        else:
            print(f"❌ ORB-SLAM: Failed to store pose")

        return success

    def shutdown_and_save(self, save_dir: str = "experiments/results/final_maps"):
        """
        Call this during system shutdown to save final map
        """
        try:
            print("🔄 MapStore shutdown - saving final map...")

            # Save complete map in ORB-SLAM compatible format
            map_path = self.save_map(save_dir)
            
            print(f"✅ MapStore shutdown complete. Maps saved:")
            print(f"   📍 Complete: {map_path}")

            return map_path

        except Exception as e:
            self.logger.error(f"❌ MapStore shutdown save failed: {e}")
            return None

    def clear_map(self):
        """Clear all map data (for reset)"""
        with self._lock:
            self.geometric_points.clear()
            self.semantic_objects.clear()
            self.room_history.clear()
            self.trajectory.clear()
            self.explored_positions.clear()
            self.current_room = "unknown"
            self.current_pose = None

            # Reset counters but keep creation time
            self.update_counters = {update_type: 0 for update_type in MapUpdateType}
            self.last_update_time = time.time()

            print("Map store cleared")

    # ==================== CONFIGURATION ====================

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration (would be injected from main system)"""
        # In real implementation, this would come from main config
        return {
            "target_objects": ["chair", "table", "door", "sofa", "bed", "window"],
            "exploration_threshold": 1000,
            "auto_save_interval": 300,  # 5 minutes
        }


if __name__ == "__main__":
    # Test the central map store
    logging.basicConfig(level=logging.INFO)

    store = CentralMapStore()

    # Test geometric data
    store.add_geometric_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    store.update_camera_pose([1.0, 2.0, 3.0], [0, 0, 0, 1])

    # Test semantic data
    store.update_objects(
        {
            "chair": {
                "position": [1.5, 2.5, 0],
                "confidence": 0.9,
                "attributes": {"category": "furniture"},
            }
        }
    )

    # Test room context
    store.update_room_context("kitchen", [1.0, 2.0, 0])

    # Get summary
    summary = store.get_map_summary()
    print("Map Summary:", json.dumps(summary, indent=2, default=str))

    # Get stats
    stats = store.get_stats()
    print("Store Stats:", json.dumps(stats, indent=2))
