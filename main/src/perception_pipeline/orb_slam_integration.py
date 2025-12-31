#!/usr/bin/env python3
"""
ORB-SLAM3 Integration Module - Using Built-in Tools
"""

# 🚨 ADD THESE LINES AT THE VERY TOP - FORCE HEADLESS MODE
import os

os.environ["DISPLAY"] = ""  # Force headless mode
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # Use software rendering
os.environ["EGL_PLATFORM"] = "surfaceless"  # For EGL offscreen

# Disable any GUI attempts
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend

import time
import threading
import numpy as np
import cv2
from typing import Dict, Any, List
from orbslam3 import System
import os
import json
import numpy as np
from scipy.ndimage import binary_dilation

# ✅ CORRECT IMPORTS - ORB-SLAM needs BOTH for map fusion
from src.stores.frame_buffer import global_frame_buffer  # For frames
from src.stores.habitat_store import (
    global_habitat_store,
)  # ✅ ADD THIS - For map fusion & coordinate alignment

# ✅ KEEP library path setup for ORB-SLAM3
current_dir = os.path.dirname(os.path.abspath(__file__))
orb_lib_path = os.path.join(current_dir, "ORB_SLAM3", "lib")
if os.path.exists(orb_lib_path):
    os.environ["LD_LIBRARY_PATH"] = (
        orb_lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    try:
        from ctypes import cdll

        cdll.LoadLibrary("libORB_SLAM3.so")
    except:
        pass


class ORBSLAMIntegration:
    """
    ORB-SLAM3 Integration - Uses built-in tools instead of manual implementations
    """

    def __init__(
        self,
        map_store=None,
        task_store=None,
        prediction_store=None,
        config: Dict[str, Any] = None,
    ):
        # ❌ REMOVE habitat_store parameter (we import it directly)
        self.slam_system = None
        self.is_initialized = False
        self.is_running = False
        
        # Add to __init__():
        self.pose_correspondences = []  # List of (slam_pose, habitat_pose) pairs
        self.min_correspondences = 3
        
        self.coordinates_aligned = False  # 🆕 ADD THIS LINE
        self.apply_transform_to_new_points = False  # 🆕 ADD THIS LINE
        
        
        # # ✅ ADD OCCUPANCY GRID PARAMETERS (SemExp style)
        # self.grid_resolution = 0.05  # 5cm per cell
        # self.map_size_cm = 1000  # 10m x 10m world (1000cm)
        # self.grid_size = self.map_size_cm // int(1/self.grid_resolution)  # 1000 cells
        
        # self.occupancy_grid_initialized = False
        # self.occupancy_grid = None
        # self.grid_origin = (-5.0, -5.0)  # World coordinates [-5m, +5m] centered at 0,0
        

        # ✅ ONLY business stores via DI
        self.map_store = map_store
        self.task_store = task_store
        self.prediction_store = prediction_store

        # Robot footprint (like SemExp's collision_map)
        self.robot_radius_cells = int(0.3 / self.map_store.grid_resolution)  # 30cm robot
        
        print(f"Occupancy grid: {self.map_store.grid_size}x{self.map_store.grid_size} cells, {self.map_store.grid_resolution}m resolution")

        self.config = config or {}

        self.last_pose = None  # 🚨 ADD THIS LINE FOR MOVEMENT TRACKING

        self.last_processed_habitat_position = None  # For motion checking
        self.last_processed_habitat_rotation = None  # ✅ ADD THIS - CRITICAL!
        self.last_processed_time = None  # For time-based fallback
        self.processed_frames = 0  # Actually processed frames
        self.skipped_frames = 0  # Skipped due to motion check
        self.consecutive_tracking_failures = 0  # Track if SLAM is struggling

        # Motion thresholds (TUNE THESE)
        self.TRANSLATION_THRESHOLD = 0.20  # 30cm minimum movement
        self.ROTATION_THRESHOLD = 0.10  # ~6 degrees (0.1 rad)
        self.TIME_THRESHOLD = 1.0  # 3 seconds fallback
        self.MAX_SKIPPED_FRAMES = 50  # Force process after N skips

        # Coordinate transformation state
        self.slam_to_world_transform = None
        self.world_to_slam_transform = None

        # Configuration paths
        current_dir = os.path.dirname(os.path.abspath(__file__))

        default_config_path = os.path.join(
            current_dir, "../../configs/slam/orb_slam3_config.yaml"
        )

        default_vocab_path = os.path.join(
            current_dir, "ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )

        self.config_path = self.config.get("orb_slam_config", default_config_path)
        self.vocabulary_path = self.config.get("vocabulary_path", default_vocab_path)

        print("ORBSLAMIntegration instance created")

    def initialize(self):
        """Initialize ORB-SLAM3 system and start continuous processing"""
        print("Initializing ORB-SLAM3 system...")
        print("🔄 ORB-SLAM: Starting initialization...")
        print(f"📁 ORB-SLAM: Config path: {os.path.abspath(self.config_path)}")
        print(f"📁 ORB-SLAM: Vocab path: {os.path.abspath(self.vocabulary_path)}")

        # ADD DEBUG INFO
        print(f"Looking for config at: {os.path.abspath(self.config_path)}")
        print(f"Looking for vocab at: {os.path.abspath(self.vocabulary_path)}")

        # Check if files exist and are readable
        if not os.path.exists(self.config_path):
            print(f"Config file does not exist: {self.config_path}")
            self.is_initialized = False
            return False

        if not os.path.exists(self.vocabulary_path):
            print(f"Vocabulary file does not exist: {self.vocabulary_path}")
            self.is_initialized = False
            return False 

        print(f"Config file size: {os.path.getsize(self.config_path)} bytes")
        print(f"Vocab file size: {os.path.getsize(self.vocabulary_path)} bytes")

        try:
            # ✅ CHANGE TO RGB-D SYSTEM
            print("Creating ORB-SLAM3 System instance for RGB-D...")
            self.slam_system = System(
                self.vocabulary_path,
                self.config_path,
            )

            self.is_initialized = True

            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._processing_loop, daemon=True
            )
            self.processing_thread.start()

            print("ORB-SLAM3 RGB-D initialized and processing started")
            return True

        except Exception as e:
            print(f"ORB-SLAM3 RGB-D initialization failed: {e}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            self.is_initializedz = False
            return False

    def start_processing(self):
        """Start ORB-SLAM processing - call this AFTER main orchestration begins"""
        if self.is_initialized and not self.is_running:
            self.is_running = True
            print("🎯 ORB-SLAM: Processing started (synced with main loop)")

    def _processing_loop(self):
        """Processing loop with proper timing"""
        last_process_time = time.time()

        while self.is_running:
            current_time = time.time()
            # Process at ~10 FPS for SLAM
            if current_time - last_process_time >= 0.5:  # 10 FPS
                self.process_frame()
                last_process_time = current_time

            time.sleep(0.05)  # Small sleep to prevent CPU overload

    def _prepare_depth_frame(self, depth_frame: np.ndarray) -> np.ndarray:
        """Prepare depth frame for ORB-SLAM3 - WITH MM TO METERS CONVERSION"""
        try:
            # 🚨 CHECK IF DEPTH IS IN MILLIMETERS (uint16 with large values)
            if depth_frame.dtype == np.uint16 and depth_frame.max() > 100:
                print(f"🔄 CONVERTING: Depth appears to be in millimeters")
                print(
                    f"   Before: {depth_frame.min():.1f} to {depth_frame.max():.1f} (uint16)"
                )

                # Convert mm to meters
                depth_frame_prepared = depth_frame.astype(np.float32) / 1000.0

                print(
                    f"   After: {depth_frame_prepared.min():.3f}m to {depth_frame_prepared.max():.3f}m (float32)"
                )
                return depth_frame_prepared

            # If already float32, just return copy
            elif depth_frame.dtype == np.float32:
                depth_frame_prepared = depth_frame.copy()
                # Check if values are reasonable for meters
                if depth_frame_prepared.max() > 100:  # Still too large for meters
                    print(
                        f"⚠️ WARNING: Depth values too large for meters: {depth_frame_prepared.max():.1f}"
                    )
                    print(f"   Might need additional scaling")
                return depth_frame_prepared

            # Convert other types to float32
            else:
                depth_frame_prepared = depth_frame.astype(np.float32)
                print(
                    f"🔧 CONVERTED: to float32, range: {depth_frame_prepared.min():.3f} to {depth_frame_prepared.max():.3f}"
                )
                return depth_frame_prepared

        except Exception as e:
            print(f"❌ ORB-SLAM: Depth frame preparation failed: {e}")
            return None

    def process_frame(self):
        """Process RGB-D frame using ORB-SLAM3 with both color and depth"""
        if not self.is_initialized:
            print(f"⚠️ ORB-SLAM: SKIPPING - NOT INITIALIZED")
            return

        try:

            sim = global_habitat_store.get_simulator()

            if sim is None:
                print("❌ ORB-SLAM: No simulator available")
                return

            agent_state = sim.get_agent(0).get_state()
            current_rotation = agent_state.rotation  # Quaternion
            current_time = time.time()
            current_position = agent_state.position
            print(f"🎯 HABITAT ACTUAL POSITION: {current_position}")

            # 🎯 MOTION-BASED CONDITIONING: Should we process this frame?
            should_process, reason = self._should_process_frame(
                current_time, current_position, current_rotation
            )

            if not should_process:
                self.skipped_frames += 1
                if self.skipped_frames % 10 == 0:
                    print(f"⏭️ ORB-SLAM: SKIP - {reason}")
                    print(f"   Skipped {self.skipped_frames} consecutive frames")
                return

            # Get latest frame from buffer
            latest_frame_id = global_frame_buffer.get_latest_frame_id()
            result = global_frame_buffer.read_frame(latest_frame_id)

            print(f"🎯 BUFFER DEBUG: Frame ID requested: {latest_frame_id}")
            print(f"🎯 BUFFER DEBUG: Result type: {type(result)}")

            if result is None:
                print("🚨 CRITICAL: Buffer returned None!")
                print("   This means ORB-SLAM will get NO FRAMES!")
                return  # ⬅️ EXIT EARLY, DON'T CONTINUE!

            frame_dict, metadata = result

            print(f"✅ BUFFER DEBUG: Got frame_dict type: {type (frame_dict)}")
            print(f"✅ BUFFER DEBUG: Got metadata type: {type(metadata)}")

            frame_id = metadata["frame_id"]
            timestamp = metadata["timestamp"]

            # Extract RGB and depth from dictionary
            rgb_frame = frame_dict["rgb"]
            depth_frame = frame_dict["depth"]

            rgb_frame, depth_frame = np.array(rgb_frame), np.array(depth_frame)

            print(
                f"🎯 DEPTH SCALE CHECK: range=({depth_frame.min():.1f}, {depth_frame.max():.1f}) dtype={depth_frame.dtype}"
            )
            
                    # 🔥 GET HABITAT POSE FROM BUFFER (NOT FROM SIMULATOR)
            if "habitat_pose_matrix" in frame_dict:
                habitat_pose = frame_dict["habitat_pose_matrix"]
            else:
                # Fallback: convert position+rotation to matrix
                habitat_position = frame_dict.get("habitat_position")
                habitat_rotation = frame_dict.get("habitat_rotation")
                if habitat_position and habitat_rotation:
                    habitat_pose = self._habitat_pose_to_matrix(habitat_position, habitat_rotation)
                else:
                    # Get from simulator as last resort
                    sim = global_habitat_store.get_simulator()
                    agent_state = sim.get_agent(0).get_state()
                    habitat_pose = self._habitat_pose_to_matrix(
                        agent_state.position, 
                        agent_state.rotation
                    )

            self.processed_frames += 1
            self.skipped_frames = 0

            print(
                f"✅ ORB-SLAM: PROCESSED FRAME #{self.processed_frames} (buffer ID: {frame_id}) - {reason}"
            )

            print(f"   Current position: {current_position}")

            if self.last_processed_habitat_position is not None:
                movement = np.linalg.norm(
                    np.array(current_position)
                    - np.array(self.last_processed_habitat_position)
                )
                print(f"   Actual movement: {movement:.3f}m")

            self.last_processed_habitat_position = current_position
            self.last_processed_habitat_rotation = current_rotation
            self.last_processed_time = current_time

            # 🚨 DEPTH PREPARATION
            depth_frame_prepared = self._prepare_depth_frame(depth_frame)
            if depth_frame_prepared is not None:
                depth_frame = depth_frame_prepared
            else:
                print("❌ ORB-SLAM: Depth preparation failed - skipping frame")
                return

            # ✅ VALIDATE DEPTH DATA
            valid_depth_pixels = np.sum(depth_frame > 0)
            print(f"🔍 DEPTH VALIDATION: {valid_depth_pixels} valid pixels")

            if valid_depth_pixels < 1000:
                print("❌ ORB-SLAM: SKIPPING - INSUFFICIENT DEPTH DATA")
                return

            print(f"🔍 RGB-D Frames: RGB={rgb_frame.shape}, Depth={depth_frame.shape}")
            print(
                f"🔍 Depth info: dtype={depth_frame.dtype}, range=({depth_frame.min():.3f}, {depth_frame.max():.3f})"
            )

            if self.slam_system is None:
                print("❌ ORB-SLAM: SLAM system not initialized")
                return

            if depth_frame is None or depth_frame.size == 0:
                print("❌ ORB-SLAM: No depth frame available")
                return

            print(f"🔍 DEBUG process_frame parameters:")
            print(f"   RGB: dtype={rgb_frame.dtype}, shape={rgb_frame.shape}")
            print(f"   Depth: dtype={depth_frame.dtype}, shape={depth_frame.shape}")
            print(f"   Timestamp: {timestamp}")

            # 🚨 PROCESS WITH ORB-SLAM
            try:

                result_dict = self.slam_system.process_frame(
                    rgb_frame, depth_frame, timestamp
                )

                print(f"🔍 ORB-SLAM RAW RESULT KEYS: {list(result_dict.keys())}")

                # Extract data from result dict
                current_pose = result_dict.get("current_pose")
                tracking_status = result_dict.get("tracking_status", {})
                visible_points = result_dict.get("visible_points", [])

                print(f"Current result dict pose : {current_pose}")

                tracking_info = tracking_status

                print(f"🔍 ORB-SLAM INTERNAL DEBUG:")
                print(f"   Raw tracking_info keys: {list(tracking_info.keys())}")

                if "tracking_ok" in tracking_info:
                    print(f"   📍 tracking_ok: {tracking_info['tracking_ok']}")
                if "tracking_lost" in tracking_info:
                    print(f"   📍 tracking_lost: {tracking_info['tracking_lost']}")
                if "system_shutdown" in tracking_info:
                    print(f"   📍 system_shutdown: {tracking_info['system_shutdown']}")

                is_tracking = tracking_info.get("tracking_ok", False)
                is_lost = tracking_info.get("tracking_lost", False)
                print(
                    f"   🎯 TRACKING STATUS: {'✅ TRACKING' if is_tracking else '❌ NOT TRACKING'}"
                )
                print(f"   🎯 LOST STATUS: {'🚨 LOST' if is_lost else '✅ NOT LOST'}")

                print(f"🔍 FEATURE DEBUG:")
                print(f"   👁️ Visible points: {len(visible_points)}")

                if len(visible_points) == 0:
                    print(f"🚨 CRITICAL: ORB-SLAM sees ZERO features!")
                    print(f"   This means tracking CANNOT work!")

                    rgb_mean = rgb_frame.mean()
                    print(f"   RGB brightness: {rgb_mean:.1f} (should be > 50)")

                    depth_coverage = (valid_depth_pixels / (480 * 640)) * 100
                    print(f"   Depth coverage: {depth_coverage:.1f}% (should be > 20%)")

            except Exception as direct_error:
                print(f"❌ ORB-SLAM: Processing failed: {direct_error}")
                print("💡 Only 'process_frame' method is available in bindings")
                return

            # ✅ PROCESS POSE MATRIX DIRECTLY (NO EXTRACTION)
            if current_pose is not None:
                try:
                    # Convert pose to numpy array if needed
                    if hasattr(current_pose, "shape"):
                        pose_matrix = current_pose
                    else:
                        pose_matrix = np.array(current_pose)

                    if pose_matrix.shape == (4, 4):
                        print(f"🤖 ORB-SLAM: Received valid 4x4 pose matrix")

                        # ✅ FIX 1: Store ORIGINAL SLAM pose BEFORE transforming it
                        original_slam_pose = pose_matrix.copy()

                        # ✅ FIX 2: APPLY TRANSFORM TO POSE BEFORE STORING
                        if self.slam_to_world_transform is not None:
                            # Transform SLAM pose to world coordinates
                            # Convert ORB-SLAM format [[R|0], [t|1]] to standard [[R|t], [0|1]]
                            standard_pose = np.eye(4)
                            standard_pose[:3, :3] = pose_matrix[:3, :3]  # Copy rotation
                            standard_pose[:3, 3] = pose_matrix[3, :3]    # Move translation from bottom row to last column

                            # Transform to world coordinates
                            world_pose = self.slam_to_world_transform @ standard_pose
                            print(f"🌍 TRANSFORMED: SLAM → World coordinates")
                            print(f"   Before: {pose_matrix}")
                            print(f"   After:  {world_pose}")
                            
                                                        # 🔥🔥 COMPLETE MATRIX DEBUG - NO [:3,3] EXTRACTION 🔥🔥
                            print("🔥🔥 COMPLETE MATRIX MULTIPLICATION VERIFICATION 🔥🔥")
                            print(f"   SLAM POSE MATRIX (Before):\n{pose_matrix}")
                            print(f"   SLAM_TO_WORLD TRANSFORM:\n{self.slam_to_world_transform}")
                            print(f"   WORLD POSE MATRIX (After):\n{world_pose}")
                            print(f"   ---")
                            print(f"   SLAM MATRIX SHAPE: {pose_matrix.shape}")
                            print(f"   TRANSFORM SHAPE: {self.slam_to_world_transform.shape}")
                            print(f"   RESULT SHAPE: {world_pose.shape}")
                            pose_matrix = world_pose  # Use transformed pose
                            print(f"⏳ After pose_matrix = world_pose  # Use transformed pose , pose matrix = {pose_matrix}")
                            
                        else:
                            # Not yet aligned - store for future alignment
                            print(f"⏳ Coordinates from orbslam Not aligned yet, storing SLAM-local pose")

                        # Store for next frame movement check
                        self.last_pose = pose_matrix.copy()

                    else:
                        print(f"⚠️ ORB-SLAM: Unexpected pose shape: {pose_matrix.shape}")
                        return
                    
                    tracking_ok = tracking_status.get("tracking_ok", False)

                    # 🎯 STORE POSE PAIRS FOR LATER ALIGNMENT
                    if (self.slam_to_world_transform is None and tracking_ok and 
                        len(visible_points) > 20 and habitat_pose is not None):
                        
                        # Store BOTH poses together
                        self.pose_correspondences.append({
                            "slam": original_slam_pose.copy(),
                            "habitat": habitat_pose.copy()  # From buffer, not simulator
                        })
                        
                        print(f"📦 Stored pose pair #{len(self.pose_correspondences)}")
                        print(f"   SLAM position: {original_slam_pose}")
                        print(f"   Habitat position: {habitat_pose}")

                        # When we have enough, call alignment with ALL pairs
                        if len(self.pose_correspondences) >= self.min_correspondences:
                            print(f"✅ Have {len(self.pose_correspondences)} pairs - computing alignment")
                            print(f"✅ Have {(self.pose_correspondences)} pairs - for pose correspondences")
                            # Pass ALL correspondences to existing method
                            self._initialize_coordinate_alignment(self.pose_correspondences)

                    # ✅ PASS TRANSFORMED POSE to store function
                    current_pose = pose_matrix  # This will be passed to _update_store_with_orb_slam_data

                except Exception as pose_error:
                    print(f"⚠️ ORB-SLAM: Pose processing failed: {pose_error}")

            else:
                print("⚠️ ORB-SLAM: No pose estimated (normal during initialization)")

            # ✅ USE SLAM-TO-WORLD TRANSFORM ONLY (COMMENT OUT Umeyama)
            final_pose_to_store = current_pose
            # ✅ Keep Umeyama for backward compatibility but don't use it
            if (hasattr(self, '_umeyama_aligner') and 
                self._umeyama_aligner is not None and 
                self._umeyama_aligner.current_transform is not None):
                
                # ❌ COMMENT OUT Umeyama transformation - use SLAM-to-World only
                # world_pose = self._umeyama_aligner.transform_pose(current_pose)
                # final_pose_to_store = world_pose
                # print(f"🔄 SLAM→World: {current_pose[0,3]:.2f}→{world_pose[0,3]:.2f}")
                
                # ✅ Instead, just log that Umeyama is available but not being used
                print(f"📝 Umeyama available but using SLAM-to-World transform only")

            # 🎯 STORE TRANSFORMED DATA TO MAP STORE
            try:
                self._update_store_with_orb_slam_data(
                    frame_id,
                    timestamp,
                    {
                        "current_pose": final_pose_to_store,  # ✅ WORLD COORDS
                        "tracking_status": tracking_status,
                        "visible_points": visible_points,
                        "method_used": "rgbd_processing",
                    },
                    final_pose_to_store,  # ✅ WORLD COORDS
                    depth_frame=depth_frame,
                )
                print(f"💾 ORB-SLAM: Frame {frame_id} WORLD-ALIGNED data stored")
            except Exception as store_error:
                print(f"❌ ORB-SLAM: Data storage failed: {store_error}")
                
                
                
                
                
                
                
                
                
                
                
                
                
                
            #             # 🔥 NEW: BUILD DEPTH OCCUPANCY GRID (BACKWARD COMPATIBLE!)
            # # 🔥 NEW: BUILD DEPTH OCCUPANCY GRID (BACKWARD COMPATIBLE!)
            # print(f"🗺️ Building DEPTH occupancy grid from frame {frame_id}...")

            # # ✅ REUSE EXISTING depth_frame (NO re-fetch!)
            # grid_data = self._build_2d_occupancy_grid(
            #     depth_frame=depth_frame,           # ✅ From Line ~150
            #     world_points_3d=visible_points,    # ✅ Fallback
            #     current_pose=final_pose_to_store,  # ✅ World pose
            #     frame_id=frame_id
            # )

            # if grid_data is not None:
            #     self.map_store.occupancy_grid_2d = grid_data['grid']
            #     self.map_store.occupancy_grid_metadata = grid_data['metadata']
                
                
                
                
            #     # 🔥 ADD THESE 3
            #     self.map_store.grid_origin = np.array(grid_data['metadata']['origin'])
            #     self.map_store.grid_resolution = grid_data['metadata']['resolution']
            #     self.map_store.grid_size = grid_data['metadata']['size']


                
                
                
                
                
            #     print(f"💾 DEPTH GRID stored! Source: {grid_data['metadata']['source']}")   

            # else:
            #     print("⚠️ Grid build failed")
                
                
                    
                    

            # 🎯 TRACKING QUALITY MONITORING
            tracking_ok = tracking_status.get("tracking_ok", False)
            if tracking_ok:
                self.consecutive_tracking_failures = 0
            else:
                self.consecutive_tracking_failures += 1
                if self.consecutive_tracking_failures > 5:
                    print(
                        f"🚨 ORB-SLAM: {self.consecutive_tracking_failures} consecutive tracking failures!"
                    )

            # 🎯 PRINT SUMMARY
            print(f"📊 ORB-SLAM FRAME {frame_id} SUMMARY:")
            print(f"   Method: rgbd_processing")
            print(f"   Pose estimated: {current_pose is not None}")
            print(f"   Visible points: {len(visible_points)}")
            print(f"   Tracking: {'OK' if tracking_ok else 'FAILED'}")

        except Exception as e:
            print(f"💥 ORB-SLAM: RGB-D processing error: {e}")
            import traceback
            traceback.print_exc()

    def _initialize_coordinate_alignment(self, pose_correspondences):
        """Use UmeyamaCoordinateAligner for robust alignment"""
        print(f"🎯 Starting coordinate alignment with {len(pose_correspondences)} pose pairs")
        
        # Import the aligner (lazy import to avoid circular dependencies)
        from src.perception_pipeline.umeyama_coordinate_alignment import UmeyamaCoordinateAligner
        
        print(f"🔍 DEBUG: UmeyamaCoordinateAligner imported successfully: {UmeyamaCoordinateAligner}")
        
        # Initialize aligner if not exists
        if not hasattr(self, '_umeyama_aligner'):
            self._umeyama_aligner = UmeyamaCoordinateAligner(
                camera_type="rgbd",  # Change to "monocular" if needed
                map_store=self.map_store, force_sim3=True
            )
            print(f"✅ UMEYAMA: New aligner created")
        else:
            print(f"✅ UMEYAMA: Using existing aligner")
        
        print(f"🔍 DEBUG: Aligner has {len(self._umeyama_aligner.correspondences)} existing correspondences")
        
        # Add all correspondences to aligner
        for i, corr in enumerate(pose_correspondences):
            success = self._umeyama_aligner.add_correspondence(
                slam_pose=corr["slam"],
                world_pose=corr["habitat"],
                confidence=1.0
            )
            if success:
                print(f"✅ Correspondence {i} added successfully")
            else:
                print(f"⚠️ Failed to add correspondence {i}")
        
        # Compute alignment (trigger only on initial start)
        print(f"🎯 Calling compute_alignment()...")
        transform = self._umeyama_aligner.compute_alignment(min_correspondences=3)
        
        if transform is not None:
            print(f"✅ UMEYAMA: Alignment successful! Transform shape: {transform.shape}")

            # ✅ STORE transform for SLAM-to-World use
            self.slam_to_world_transform = transform
            
            print(f"🔄 PHASE 1: Transforming poses using SLAM-to-World...")
            # Apply transform to ENTIRE system using SLAM-to-World only
            self._apply_transform_to_all(transform)
            
            # ❌ COMMENT OUT Umeyama point update - points will be transformed via SLAM-to-World
            """
            # Update map store with transformed points
            points_transformed = self._umeyama_aligner.update_map_store_points()
            print(f"✅ Transformed {points_transformed} points in map store")
            """
            
            # ✅ Points will be transformed when stored via SLAM→World transform
            print(f"✅ SLAM→World transform ready for point transformation")
                        
            # 🚨 STEP 3: ONLY NOW mark as aligned - AFTER ALL TRANSFORMS
            self.coordinates_aligned = True
            self.apply_transform_to_new_points = True
            print(f"🎯 COORDINATE SYSTEM ALIGNED - Using SLAM→World transform only")
            
            return transform
        else:
            print(f"⚠️ UMEYAMA: Alignment failed, not enough valid correspondences")
            return None

    def _transform_points_with_slam_to_world(self, rich_points):
        """Transform points using SLAM-to-World transform (backward compatible)"""
        if self.slam_to_world_transform is None or not rich_points:
            return rich_points
        
        transformed_points = []
        for point in rich_points:
            try:
                if 'position_3d' in point and point['position_3d'] is not None:
                    # Convert to homogeneous coordinates
                    slam_pos = np.array(point['position_3d'] + [1.0])
                    # Apply same transform as poses
                    world_pos = self.slam_to_world_transform @ slam_pos
                    
                    # Create new point
                    new_point = point.copy()
                    new_point['position_3d'] = world_pos[:3].tolist()
                    transformed_points.append(new_point)
                else:
                    transformed_points.append(point)
            except Exception as e:
                print(f"⚠️ Point transformation failed: {e}")
                transformed_points.append(point)
        
        print(f"✅ Transformed {len(transformed_points)} points using SLAM→World transform")
        return transformed_points

    def _apply_transform_to_all(self, transform):
        """Apply transform to entire ORB-SLAM system - POSE TRANSFORMATION ONLY"""
        print(f"🔄 Applying transform to poses...")
        
        # 1. Store transform for future use
        self.slam_to_world_transform = transform
        print(f"   ✅ Transform stored in slam_to_world_transform")
        
        # 2. ❌ COMMENT OUT Umeyama transformation of current pose
        # ✅ Keep for backward compatibility but don't execute
        """
        if hasattr(self, 'current_pose') and self.current_pose is not None:
            try:
                transformed_pose = self._umeyama_aligner.transform_pose(self.current_pose)
                self.current_pose = transformed_pose
                print(f"   ✅ Current pose transformed")
            except Exception as e:
                print(f"   ⚠️ Pose transformation failed: {e}")
        """
        
        # ✅ INSTEAD: Update flags only
        self.coordinates_aligned = True
        self.apply_transform_to_new_points = True
        
        # ❌ COMMENT OUT map store transformation via Umeyama
        # ✅ Keep for backward compatibility but don't execute
        """
        if hasattr(self, 'map_store') and self.map_store:
            if hasattr(self.map_store, 'transform_all_stored_poses'):
                pose_count = self.map_store.transform_all_stored_poses(self._umeyama_aligner)
                print(f"✅ Historical poses transformed: {pose_count}")
        """
        
        print(f"✅ SLAM→World transform stored for future use")
        return True

    def _align_slam_to_world_coordinates(self, slam_pose: np.ndarray) -> np.ndarray:
        """
        Convert ORB-SLAM coordinates to Habitat world coordinates
        This is CRITICAL for map fusion
        """
        if self.slam_to_world_transform is None:
            # Initialize coordinate transformation on first good tracking
            self._initialize_coordinate_alignment(slam_pose)

        if self.slam_to_world_transform is not None:
            return self.slam_to_world_transform @ slam_pose
        return slam_pose  # Fallback

    def _should_process_frame(
        self, current_time, current_position=None, current_rotation=None
    ):
        """
        Check if we should process this frame using REAL Habitat data
        Returns: (should_process, reason_string)

        Parameters:
        - current_time: current timestamp
        - current_position: optional, pass if already have position data
        - current_rotation: optional, pass if already have rotation data
        """

        # 🎯 ADD COMPREHENSIVE DEBUG PRINTS
        print(f"🔍 MOTION CHECK DEBUG: Starting motion check")
        print(f"🔍   last_processed_position: {self.last_processed_habitat_position}")
        print(f"🔍   last_processed_time: {self.last_processed_time}")
        print(f"🔍   skipped_frames: {self.skipped_frames}")

        # Get position/rotation if not provided
        if current_position is None or current_rotation is None:
            # Fallback to getting from HabitatStore
            sim = global_habitat_store.get_simulator()
            if sim is None:
                print(f"🔍 MOTION CHECK: ❌ No simulator")
                # Check if HabitatStore has current state
                current_state = global_habitat_store.get_current_agent_state()
                if current_state is None:
                    print(f"🔍 MOTION CHECK: ❌ No habitat data")
                    return False, "no_habitat_data"
                current_position = current_state["position"]
                current_rotation = current_state["rotation"]
            else:
                # Get agent state directly
                agent_state = sim.get_agent(0).get_state()
                current_position = agent_state.position
                current_rotation = agent_state.rotation

        print(f"🔍 MOTION CHECK: Current position = {current_position}")

        # First frame always process
        if self.last_processed_habitat_position is None:
            print(f"🔍 MOTION CHECK: ✅ First frame, will process")
            return True, "first_frame"

        # Calculate REAL translation distance
        translation = np.linalg.norm(
            np.array(current_position) - np.array(self.last_processed_habitat_position)
        )

        # Calculate REAL rotation angle
        rotation_angle = 0.0
        if self.last_processed_habitat_rotation is not None:
            try:
                rotation_angle = global_habitat_store._quaternion_angle(
                    current_rotation, self.last_processed_habitat_rotation
                )
            except Exception as e:
                print(f"🔍 MOTION CHECK: Rotation calculation failed: {e}")
                rotation_angle = 0.0

        # Time since last processed frame
        time_delta = (
            current_time - self.last_processed_time
            if self.last_processed_time
            else float("inf")
        )

        # Get DYNAMIC thresholds from HabitatStore (based on real motion)
        motion_stats = global_habitat_store.motion_stats
        trans_threshold = max(
            self.TRANSLATION_THRESHOLD, motion_stats.get("avg_translation", 0.30) * 0.8
        )
        rot_threshold = max(
            self.ROTATION_THRESHOLD, motion_stats.get("avg_rotation", 0.10) * 0.8
        )

        print(f"🔍 MOTION CALCULATIONS:")
        print(
            f"🔍   translation: {translation:.4f}m, threshold: {trans_threshold:.4f}m"
        )
        print(
            f"🔍   rotation: {rotation_angle:.4f}rad, threshold: {rot_threshold:.4f}rad"
        )
        print(f"🔍   time_delta: {time_delta:.2f}s, threshold: {self.TIME_THRESHOLD}s")
        print(
            f"🔍   motion_stats: avg_trans={motion_stats.get('avg_translation', 0.30):.3f}, "
            f"avg_rot={motion_stats.get('avg_rotation', 0.10):.3f}"
        )

        # Too many skipped frames - force processing
        if self.skipped_frames >= self.MAX_SKIPPED_FRAMES:
            print(
                f"🔍 MOTION CHECK: 🚨 Force processing after {self.skipped_frames} skipped frames"
            )
            return True, f"force_after_{self.skipped_frames}_skips"

        # Check thresholds using REAL data
        if translation > trans_threshold:
            print(
                f"🔍 MOTION CHECK: ✅ Translation exceeds threshold: {translation:.3f}m > {trans_threshold:.3f}m"
            )
            return True, f"translation_{translation:.3f}m > {trans_threshold:.3f}m"

        if rotation_angle > rot_threshold:
            print(
                f"🔍 MOTION CHECK: ✅ Rotation exceeds threshold: {rotation_angle:.3f}rad > {rot_threshold:.3f}rad"
            )
            return True, f"rotation_{rotation_angle:.3f}rad > {rot_threshold:.3f}rad"

        if time_delta > self.TIME_THRESHOLD:
            print(
                f"🔍 MOTION CHECK: ⏰ Time threshold exceeded: {time_delta:.1f}s > {self.TIME_THRESHOLD}s"
            )
            return True, f"time_{time_delta:.1f}s > {self.TIME_THRESHOLD}s"

        # Not enough motion - SKIP
        print(
            f"🔍 MOTION CHECK: ⏭️ Skipping - trans={translation:.3f}m, rot={rotation_angle:.3f}rad, time={time_delta:.1f}s"
        )
        print(
            f"🔍   All below thresholds: trans<{trans_threshold:.3f}, rot<{rot_threshold:.3f}, time<{self.TIME_THRESHOLD}"
        )
        return (
            False,
            f"skip_trans_{translation:.3f}m_rot_{rotation_angle:.3f}rad_time_{time_delta:.1f}s",
        )
  
    # def _build_2d_occupancy_grid(self, depth_frame=None, world_points_3d=None, current_pose=None, frame_id=0):
    #     """
    #     ✅ FIXED: NO REDUNDANT TRANSFORMS! Uses ALREADY transformed current_pose
    #     ✅ Backward compatible: depth_frame OR world_points_3d
    #     ✅ Single pass: Camera 3D → World 3D → Grid
        
    #     Args:
    #         depth_frame: Habitat depth (priority)
    #         world_points_3d: Legacy points (fallback)
    #         current_pose: ALREADY TRANSFORMED world pose (Line ~520)
    #         frame_id: Debug counter
    #     """
    #     try:
    #         # 🎯 DYNAMIC ORIGIN: Center grid on robot (FIXED BUG #1)
    #         # if current_pose is not None:
    #         #     robot_x, _, robot_z = current_pose[:3, 3]
    #         #     print(f"🎯 Dynamic origin: {self.grid_origin} (centered on robot: [{robot_x:.2f}, {robot_z:.2f}])")
            
            
            
            
            
            
    #         # 🚨 CRITICAL: do not build grid until SLAM is valid
    #         if current_pose is None or not self.is_initialized:
    #             print("⏭️ Skipping grid update — SLAM not initialized")
    #             return None

            
    #         # PRIORITY #1: DEPTH FRAME
    #         if depth_frame is not None and depth_frame.size > 0:
    #             print(f"🔥 DEPTH MODE: {depth_frame.shape}")
                
    #             if not self.occupancy_grid_initialized:
    #                 print(f"🆕 Initializing 3D grid: {self.grid_size}x{self.grid_size}x3")
    #                 self.occupancy_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
    #                 self.occupancy_grid_initialized = True

    #             # Camera intrinsics
    #             H, W = depth_frame.shape
    #             fx = fy = 518.0
    #             cx, cy = W / 2, H / 2
                
    #             # Valid depth
    #             valid_depth = (depth_frame > 0.1) & (depth_frame < 5.0) & np.isfinite(depth_frame)
    #             y_coords, x_coords = np.indices((H, W))
    #             x_norm = (x_coords - cx) / fx
    #             y_norm = (y_coords - cy) / fy
                
    #             points_cam_3d = np.stack([
    #                 x_norm * depth_frame,
    #                 y_norm * depth_frame, 
    #                 depth_frame
    #             ], axis=-1)[valid_depth]
                
    #             if len(points_cam_3d) == 0:
    #                 print(f"⚠️ No valid depth pixels")
    #                 return None
                
    #             # 🔥 SINGLE TRANSFORM: Use ALREADY transformed pose (NO REDUNDANCY!)
    #             if current_pose is not None:
    #                 R_world_cam = current_pose[:3, :3]
    #                 t_world_cam = current_pose[:3, 3]
    #                 points_world_3d = (R_world_cam @ points_cam_3d.T).T + t_world_cam
                    
    #                 # 🔥 Z-FILTER: Only ground-level obstacles (0.1m-1.5m height)
    #                 z_hit_mask = (points_world_3d[:, 1] >= 0.1) & (points_world_3d[:, 1] <= 1.5)
    #                 points_world_3d = points_world_3d[z_hit_mask]  # Filter to ground hits ONLY
    #                 print(f"✅ Z-filter: kept {np.sum(z_hit_mask)}/{len(z_hit_mask)} points (ground level)")

    #             else:
    #                 points_world_3d = points_cam_3d  # Fallback
                
    #             # Project to 2D grid
    #             points_2d = points_world_3d[:, [0, 2]]  # X,Z
    #             grid_x = ((points_2d[:, 0] - self.grid_origin[0]) / self.grid_resolution).astype(int)
    #             grid_z = ((points_2d[:, 1] - self.grid_origin[1]) / self.grid_resolution).astype(int)
                
    #             valid_mask = (grid_x >= 0) & (grid_x < self.grid_size) & (grid_z >= 0) & (grid_z < self.grid_size)
    #             grid_x, grid_z = grid_x[valid_mask], grid_z[valid_mask]
                
    #             if len(grid_x) > 0:
    #                 # NEW: Add to channel 1 (occupied) like MapBuilder
    #                 self.occupancy_grid[grid_z, grid_x, 1] += 1.0  # Accumulate hits
    #                 print(f"✅ Marked {len(grid_x)} depth pixels in channel 1")
            
    #         # PRIORITY #2: LEGACY POINTS (FIXED BUG #2)
    #         elif world_points_3d is not None and len(world_points_3d) > 0:
    #             print(f"📍 POINTS MODE: {len(world_points_3d)} points")
                
    #             if not self.occupancy_grid_initialized:
    #                 # FIXED: np.float32 not np.uint8
    #                 self.occupancy_grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
    #                 self.occupancy_grid_initialized = True
                
    #             points_2d = []
    #             for point in world_points_3d:
    #                 if isinstance(point, dict) and "position_3d" in point:
    #                     x, y, z = point["position_3d"]
    #                     if -2.0 < y < 2.0:
    #                         points_2d.append((x, z))
                
    #             if points_2d:
    #                 points_array = np.array(points_2d)
    #                 grid_x = ((points_array[:, 0] - self.grid_origin[0]) / self.grid_resolution).astype(int)
    #                 grid_z = ((points_array[:, 1] - self.grid_origin[1]) / self.grid_resolution).astype(int)
                    
    #                 valid_mask = (grid_x >= 0) & (grid_x < self.grid_size) & (grid_z >= 0) & (grid_z < self.grid_size)
    #                 grid_x, grid_z = grid_x[valid_mask], grid_z[valid_mask]
                    
    #                 if len(grid_x) > 0:
    #                     self.occupancy_grid[grid_z, grid_x, 1] += 1.0  
    #                     print(f"✅ Marked {len(grid_x)} points in channel 1")
            
    #         else:
    #             print(f"⚠️ No valid input")
    #             return None
            
    #         # ROBOT FREE SPACE (using SAME transformed pose)
    #         if current_pose is not None:
    #             robot_x, _, robot_z = current_pose[:3, 3]
    #             robot_grid_x = int((robot_x - self.grid_origin[0]) / self.grid_resolution)
    #             robot_grid_z = int((robot_z - self.grid_origin[1]) / self.grid_resolution)
                
    #             if (0 <= robot_grid_x < self.grid_size and 0 <= robot_grid_z < self.grid_size):
    #                 # 1️⃣ STANDARD CLEARING (always)
    #                 for dz in range(-self.robot_radius_cells, self.robot_radius_cells + 1):
    #                     for dx in range(-self.robot_radius_cells, self.robot_radius_cells + 1):
    #                         if dx*dx + dz*dz <= self.robot_radius_cells**2:
    #                             z_idx, x_idx = robot_grid_z + dz, robot_grid_x + dx
    #                             if 0 <= z_idx < self.grid_size and 0 <= x_idx < self.grid_size:
    #                                 if self.occupancy_grid[z_idx, x_idx, 1] != 1:  # Check channel 1
    #                                     self.occupancy_grid[z_idx, x_idx, 2] = 1.0  # Set channel 2 (free space)
                    
    #                 # 2️⃣ I-JEPA ENHANCED CLEARING (if available)
    #                 try:
    #                     if hasattr(self, '_get_ijepa_scores'):
    #                         ijepa_scores = self._get_ijepa_scores()
    #                         free_space_score = ijepa_scores.get('free_space', 0.0)
                            
    #                         if free_space_score > 0.7:
    #                             clear_radius_meters = 1.0 + (free_space_score - 0.7) * 5.0
    #                             clear_radius_cells = int(clear_radius_meters / self.grid_resolution)
                                
    #                             # Clear larger circle + forward cone
    #                             for dz in range(-clear_radius_cells, clear_radius_cells + 1):
    #                                 for dx in range(-clear_radius_cells, clear_radius_cells + 1):
    #                                     if dx*dx + dz*dz <= clear_radius_cells**2:
    #                                         z_idx, x_idx = robot_grid_z + dz, robot_grid_x + dx
    #                                         if (0 <= z_idx < self.grid_size and 0 <= x_idx < self.grid_size and
    #                                             # FIXED BUG #3: Check channel 1, not whole array
    #                                             self.occupancy_grid[z_idx, x_idx, 1] != 1):
    #                                             self.occupancy_grid[z_idx, x_idx, 2] = 1.0
                                
    #                             # Optional: Clear forward wedge for navigation
    #                             robot_rotation = current_pose[:3, :3]
    #                             forward_dir = robot_rotation @ np.array([1, 0, 0])
    #                             forward_angle = np.arctan2(forward_dir[2], forward_dir[0])
                                
    #                             for angle_offset in np.linspace(-np.pi/4, np.pi/4, 5):
    #                                 clear_angle = forward_angle + angle_offset
    #                                 for dist_cells in range(5, clear_radius_cells + 1, 5):
    #                                     target_x = robot_grid_x + int(dist_cells * np.cos(clear_angle))
    #                                     target_z = robot_grid_z + int(dist_cells * np.sin(clear_angle))
    #                                     if (0 <= target_z < self.grid_size and 0 <= target_x < self.grid_size and
    #                                         # FIXED BUG #4: Check channel 1, set channel 2
    #                                         self.occupancy_grid[target_z, target_x, 1] != 1):
    #                                         self.occupancy_grid[target_z, target_x, 2] = 1.0
                                
    #                             print(f"🧠 I-JEPA: Cleared {clear_radius_meters:.1f}m radius (score: {free_space_score:.2f})")
    #                 except Exception as e:
    #                     print(f"⚠️ I-JEPA clearing failed (safe): {e}")
    #                     # Continue with standard clearing only
            
    #         # INFLATE OBSTACLES
    #         from scipy.ndimage import binary_dilation
    #         occupied_mask = self.occupancy_grid[:,:,1] > 0.5
    #         if np.any(occupied_mask):
    #             structure = np.ones((self.robot_radius_cells, self.robot_radius_cells))
    #             dilated = binary_dilation(occupied_mask, structure=structure)
    #             self.occupancy_grid[:,:,1][dilated] = 1.0
            
    #         # STATS + RETURN
    #         occupied_count = np.sum(self.occupancy_grid[:,:,1] > 0.5)
    #         explored_count = np.sum(self.occupancy_grid[:,:,2] > 0.5)
    #         source = 'depth' if depth_frame is not None else 'points'
    #         pixels_used = len(grid_x) if 'grid_x' in locals() else 0
            
    #         # 🔥 DEBUG PRINT (FIXED BUG #5)
    #         print(f"🔍 GRID VERIFICATION #{frame_id}:")
    #         print(f"   Grid shape: {self.occupancy_grid.shape}")
    #         print(f"   Channel 1 (occupied) range: [{self.occupancy_grid[:,:,1].min():.1f}, {self.occupancy_grid[:,:,1].max():.1f}]")
    #         print(f"   Channel 2 (explored) range: [{self.occupancy_grid[:,:,2].min():.1f}, {self.occupancy_grid[:,:,2].max():.1f}]")
    #         print(f"   Cells >0.5 threshold: occ={occupied_count}, exp={explored_count}")
    #         print(f"   Grid origin: {self.grid_origin}")
            
    #         # 🔥 SEMANTIC INIT (once)
    #         if not hasattr(self.map_store, 'grid_semantics'):
    #             self.map_store.grid_semantics = {}

    #         # Add generic wall semantics (sample 5 cells)
    #         if 'grid_x' in locals() and len(grid_x) > 5:
    #             for i in range(5):
    #                 self.map_store.grid_semantics[(int(grid_z[i]), int(grid_x[i]))] = f"{source}_wall" 
            
    #         print(f"📊 GRID #{frame_id}: occ={occupied_count}/{self.grid_size**2}, exp={explored_count}/{self.grid_size**2}")
            
    #         # Create 2D binary for compatibility
    #         occupied_2d = self.occupancy_grid[:,:,1].copy()
    #         occupied_2d[occupied_2d >= 0.5] = 1.0
    #         occupied_2d[occupied_2d < 0.5] = 0.0
            
    #         return {
    #             'grid': occupied_2d,  # 2D binary (for compatibility)
    #             'full_map': self.occupancy_grid.copy(),  # 3D persistent map
    #             'metadata': {
    #                 'origin': self.grid_origin,
    #                 'resolution': self.grid_resolution,
    #                 'size': self.grid_size,
    #                 'frame_id': frame_id,
    #                 'occupied_cells': int(occupied_count),
    #                 'explored_cells': int(explored_count),
    #                 'total_cells': self.grid_size * self.grid_size,
    #                 'source': source,
    #                 'has_semantics': True,
    #                 'semantic_count': len(self.map_store.grid_semantics),
    #                 'depth_pixels_used': pixels_used,
    #                 'has_3d_map': True
    #             }
    #         }
        
    #     except Exception as e:
    #         print(f"❌ Grid error: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return None
    
    
    
    def _get_ijepa_scores(self):
        """Get I-JEPA scores from prediction_store (SAME as ActionExecutor)"""
        try:
            # Check if we have access to prediction_store
            if hasattr(self, 'prediction_store') and self.prediction_store is not None:
                if hasattr(self.prediction_store, 'predictions') and self.prediction_store.predictions:
                    # Get latest prediction
                    latest_id = max(self.prediction_store.predictions.keys())
                    entry = self.prediction_store.get_prediction(latest_id)
                    
                    if entry:
                        predictors = entry.get('predictors', {})
                        ijepa_data = predictors.get('structural_continuity', {})
                        data = ijepa_data.get('data', {})
                        analysis = data.get('structural_analysis', {})
                        return analysis.get('similarity_scores', {})
        except Exception as e:
            print(f"⚠️ ORB-SLAM: I-JEPA fetch failed: {e}")
        
        return {}  # Default empty dict
    
              

    def _validate_map_points_with_simulator(self, slam_map_points: List) -> List[Dict]:
        """
        Use Habitat simulator to validate and enhance SLAM map points
        """
        validated_points = []
        sim = global_habitat_store.get_simulator()

        if sim is None:
            return slam_map_points  # Can't validate without simulator

        for point in slam_map_points:
            try:
                # Convert SLAM coordinates to world coordinates
                world_point = self._align_slam_to_world_coordinates(point.position)

                # Use Habitat to check if point is valid (not inside walls, etc.)
                # This is where we'd use raycasting to validate geometry
                is_valid = self._validate_point_with_raycast(world_point)

                validated_point = {
                    "position": world_point,
                    "is_valid": is_valid,
                    "source": "orb_slam_validated",
                    "confidence": (
                        point.confidence if hasattr(point, "confidence") else 1.0
                    ),
                }
                validated_points.append(validated_point)

            except Exception as e:
                print(f"Point validation failed: {e}")
                continue

        return validated_points

    def _validate_point_with_raycast(self, world_point: np.ndarray) -> bool:
        """
        Use Habitat raycasting to validate if a SLAM map point makes sense
        """
        try:
            sim = global_habitat_store.get_simulator()
            if sim is None:
                return True  # Can't validate, assume valid

            # Simple validation: check if point is within reasonable bounds
            # In practice, you'd do more sophisticated geometric validation
            camera_center = np.array([0, 0, 0])  # Simplified
            point_distance = np.linalg.norm(world_point[:3] - camera_center)

            # Reasonable distance check (avoid points too far/close)
            return 0.1 < point_distance < 20.0

        except Exception as e:
            print(f"Raycast validation failed: {e}")
            return True
 
    def _update_store_with_orb_slam_data(
        self,
        frame_id: int,
        timestamp: float,
        result: dict,
        current_pose,
        depth_frame=None,
    ):
        """Store available map data to CentralMapStore - RICH FORMAT"""
        try:
            # 🎯 ADD THIS DEBUG PRINT:
            # 🎯 DEBUG PRINT (KEEP THIS):
            print(f"🎯 STORE DEBUG - Frame {frame_id}:")
            print(f"   📦 Current pose type: {type(current_pose)}")
            if current_pose is not None:
                if hasattr(current_pose, "shape"):
                    print(f"   📦 Pose shape: {current_pose.shape}")
                    print(
                        f"   📦 Translation: {current_pose}"
                    )  # This should show WORLD coordinates
                else:
                    print(f"   📦 Pose value: {current_pose}")
            else:
                print(f"   📦 Current pose is None!")

            # ✅ FIX: DON'T OVERWRITE current_pose - USE THE TRANSFORMED ONE!
            # current_pose should already be in world coordinates from process_frame()
            tracking_info = result.get("tracking_status", {})
            tracked_points = result.get("visible_points", [])

            print(f"🗺️ ORB-SLAM: {len(tracked_points)} tracked points this frame")

            # 🎯 SAFELY Store pose - Handle different pose formats
            if current_pose is not None and self.map_store is not None:
                safe_frame_id = int(frame_id)
                if hasattr(self.map_store, "update_camera_pose"):
                    try:

                        # Convert pose to proper 4x4 matrix if needed
                        if isinstance(current_pose, (list, tuple)):
                            # If it's a flat list, reshape to 4x4
                            if len(current_pose) == 16:
                                pose_matrix = np.array(current_pose).reshape(4, 4)
                            else:
                                print(
                                    f"⚠️ Unexpected pose format: {len(current_pose)} elements"
                                )
                                pose_matrix = np.eye(4)  # Fallback to identity
                        elif hasattr(current_pose, "shape"):
                            # If it's already a numpy array
                            pose_matrix = current_pose
                        else:
                            print(f"⚠️ Unknown pose type: {type(current_pose)}")
                            pose_matrix = np.eye(4)  # Fallback to identity

                        # Ensure it's a proper 4x4 matrix
                        if pose_matrix.shape != (4, 4):
                            print(
                                f"⚠️ Pose shape is {pose_matrix.shape}, reshaping to 4x4"
                            )
                            pose_matrix = np.eye(4)  # Fallback

                        self.map_store.update_camera_pose(
                            pose_matrix, safe_frame_id, timestamp
                        )
                        print(f"✅ ORB-SLAM: Pose stored successfully")

                    except Exception as pose_error:
                        print(f"❌ ORB-SLAM: Pose storage failed: {pose_error}")
                        # Continue with point storage even if pose fails

                    # ✅ STORE ROBOT POSITION - NO EXTRACTION, NO FALLBACKS
            if self.map_store is not None and hasattr(
                self.map_store, "update_robot_position"
            ):
                safe_frame_id = int(frame_id)

                # 🚨 CHECK TRACKING STATUS
                tracking_info = result.get("tracking_status", {})
                tracking_ok = tracking_info.get("tracking_ok", False)
                visible_points = len(result.get("visible_points", []))

                # STRICT REQUIREMENTS
                if (
                    tracking_ok
                    and visible_points > 50
                    and current_pose is not None
                    and hasattr(current_pose, "shape")
                    and current_pose.shape == (4, 4)
                ):

                    # 🎯 STORE COMPLETE POSE - NO EXTRACTION
                    self.map_store.update_robot_position(
                        current_pose, safe_frame_id, timestamp
                    )
                    print(f"🤖 ORB-SLAM: Complete pose stored directly")

                else:
                    # 🚫 DO NOT STORE ANYTHING
                    print(f"🚫 ORB-SLAM: SKIPPING position - invalid tracking")
                    if not tracking_ok:
                        print(f"   Reason: tracking_ok = {tracking_ok}")
                    if visible_points <= 20:
                        print(
                            f"   Reason: visible_points = {visible_points} (need > 50)"
                        )
                    if current_pose is None:
                        print(f"   Reason: current_pose is None")
                    elif not hasattr(current_pose, "shape"):
                        print(f"   Reason: current_pose has no shape attribute")
                    elif current_pose.shape != (4, 4):
                        print(f"   Reason: current_pose.shape = {current_pose.shape}")

            # 🎯 NEW: Store TRACKED points in RICH FORMAT (REPLACES OLD CODE)
            # ✅ FIX: Only store NEW points observed in current frame
            rich_points = []
            if tracked_points and self.map_store:
                conversion_success = 0
                new_points_count = 0

                for i, point in enumerate(tracked_points):
                    try:
                        # 🚨 FIX: Track which points are NEW to this frame
                        point_id = i + (frame_id)  # Use larger multiplier

                        # Create rich point data structure
                        point_data = {
                            "point_id": point_id,
                            "is_tracked": True,
                            "source": "orb_slam3_rgbd",
                            "frame_id": frame_id,  # 🎯 Associate with CURRENT frame
                            "timestamp": timestamp,
                            "first_observed": frame_id,  # 🆕 Track first observation
                        }

                        # 🚨 RGB-D: USE REAL DEPTH CONFIDENCE
                        if depth_frame is not None:
                            point_data["depth_confidence"] = (
                                0.95  # High confidence for real depth
                            )
                            point_data["is_dense"] = True  # RGB-D produces dense points
                            point_data["has_real_depth"] = (
                                True  # 🆕 Flag for real depth data
                            )
                        else:
                            point_data["depth_confidence"] = 0.3
                            point_data["is_dense"] = False

                        # Extract position with multiple format support
                        position = None
                        raw_data = {}

                        # Method 1: If point has x,y,z attributes
                        if (
                            hasattr(point, "x")
                            and hasattr(point, "y")
                            and hasattr(point, "z")
                        ):
                            position = [float(point.x), float(point.y), float(point.z)]
                            raw_data = {
                                "id": getattr(point, "id", i + (frame_id * 1000)),
                                "x": float(point.x),
                                "y": float(point.y),
                                "z": float(point.z),
                            }
                            # Add any additional attributes the point might have
                            if hasattr(point, "confidence"):
                                raw_data["confidence"] = float(point.confidence)
                            if hasattr(point, "observations"):
                                raw_data["observations"] = int(point.observations)

                        # Method 2: If point is a dictionary
                        elif (
                            isinstance(point, dict)
                            and "x" in point
                            and "y" in point
                            and "z" in point
                        ):
                            position = [
                                float(point["x"]),
                                float(point["y"]),
                                float(point["z"]),
                            ]
                            raw_data = {
                                "id": point.get("id", i + (frame_id * 1000)),
                                "x": float(point["x"]),
                                "y": float(point["y"]),
                                "z": float(point["z"]),
                            }
                            # Copy other dictionary fields
                            for key in point:
                                if key not in ["x", "y", "z", "id"]:
                                    raw_data[key] = point[key]

                        # Method 3: If point is array-like (numpy array, list, tuple)
                        elif hasattr(point, "__getitem__") and len(point) >= 3:
                            position = [
                                float(point[0]),
                                float(point[1]),
                                float(point[2]),
                            ]
                            raw_data = {
                                "id": i + (frame_id * 1000),
                                "x": float(point[0]),
                                "y": float(point[1]),
                                "z": float(point[2]),
                            }

                        # Add position and raw_data if extraction successful
                        if position:
                            point_data["position_3d"] = position
                            point_data["raw_data"] = raw_data
                            rich_points.append(point_data)
                            conversion_success += 1

                    except Exception as e:
                        if i < 3:  # Debug first few errors
                            print(f"🔍 Point {i} conversion error: {e}")

                print(
                    f"🔍 Point conversion: {conversion_success} success, {len(rich_points)} rich points"
                )
                
                # ✅ TRANSFORM POINTS USING SLAM-TO-WORLD ONLY
                if rich_points and self.map_store and hasattr(self.map_store, 'add_geometric_points_rich'):
                    # ✅ PRIMARY: Use SLAM-to-World transform
                    if self.slam_to_world_transform is not None:
                        rich_points = self._transform_points_with_slam_to_world(rich_points)
                        print(f"✅ Transformed {len(rich_points)} points using SLAM→World transform")
                    # ❌ FALLBACK: Keep Umeyama for backward compatibility but don't use it
                    elif hasattr(self, '_umeyama_aligner') and self._umeyama_aligner is not None:
                        print(f"⚠️ Umeyama available but using SLAM-to-World transform only")
                        # ❌ Don't transform with Umeyama - wait for SLAM-to-World
                        print(f"📝 Points stored untransformed - will transform when SLAM-to-World available")
                    
                    # Store points
                    self.map_store.add_geometric_points_rich(rich_points)

            print(f"💾 ORB-SLAM: Added {len(rich_points)} points to memory map")
            print(f"   🎯 Current Frame: {frame_id}")
            print(f"   🤖 Robot Position: {current_pose}")
             
                

        except Exception as e:
            print(f"❌ ORB-SLAM: Failed to update MapStore: {e}")
            import traceback
            traceback.print_exc()
            
    def _habitat_pose_to_matrix(self, position, quaternion):
        """Convert habitat position + quaternion to 4×4 transformation matrix"""
        # Extract quaternion components
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z
        
        # Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # Create 4×4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        return T

    def _translate_tracking_state(self, state) -> str:
        """✅ USE ORB-SLAM3's BUILT-IN TRACKING STATE"""
        # Handle both integer states and dictionary states
        if isinstance(state, dict):
            # Extract state from dictionary
            state_value = state.get("state", state.get("status", 0))
        else:
            state_value = state

        state_mapping = {
            0: "",
            1: "NO_IMAGES_YET",
            2: "NOT_INITIALIZED",
            3: "OK",
            4: "LOST",
        }
        return state_mapping.get(state_value, f"UNKNOWN({state_value})")

    def _format_map_points(self, map_points: List) -> List[Dict[str, Any]]:
        """Format ALL ORB-SLAM3 map points - EXTRACT EVERYTHING"""
        formatted_points = []
        print(f"🔍 DEBUG - Formatting {len(map_points)} raw map points")

        if not map_points:
            return []

        # Check what keys the first dictionary has
        first_point = map_points[0]
        print(f"🔍 DEBUG - First point keys: {list(first_point.keys())}")
        print(f"🔍 DEBUG - First point values: {first_point}")

        extracted_count = 0
        for i, mp in enumerate(map_points):
            try:
                # ✅ EXTRACT ALL POSSIBLE DATA FROM EACH POINT
                point_data = {
                    "point_id": i,
                    "is_tracked": True,
                    "raw_data": mp,  # ✅ SAVE EVERYTHING initially
                }

                # Try to extract position from ANY possible key
                position = None
                if isinstance(mp, dict):
                    # Try ALL possible position key combinations
                    if "x" in mp and "y" in mp and "z" in mp:
                        position = [mp["x"], mp["y"], mp["z"]]
                    elif "world_pos" in mp:
                        position = mp["world_pos"]
                    elif "position" in mp:
                        position = mp["position"]
                    elif "pos" in mp:
                        position = mp["pos"]
                    elif "coordinates" in mp:
                        position = mp["coordinates"]
                    elif hasattr(mp, "__getitem__"):
                        # Try array-like access
                        try:
                            position = [mp[0], mp[1], mp[2]]
                        except:
                            pass

                    # If we found position, add it
                    if position is not None:
                        if hasattr(position, "tolist"):
                            point_data["position_3d"] = position.tolist()
                        elif isinstance(position, (list, tuple, np.ndarray)):
                            point_data["position_3d"] = [
                                float(position[0]),
                                float(position[1]),
                                float(position[2]),
                            ]
                        extracted_count += 1
                    else:
                        # Even if no position, save the raw data for debugging
                        point_data["position_3d"] = None
                        point_data["note"] = "no_position_extracted"

                formatted_points.append(point_data)

            except Exception as e:
                print(f"❌ DEBUG - Failed to format point {i}: {e}")
                # Still save the point with error info
                formatted_points.append(
                    {
                        "point_id": i,
                        "is_tracked": True,
                        "position_3d": None,
                        "error": str(e),
                        "raw_data": (
                            str(mp) if hasattr(mp, "__str__") else "unserializable"
                        ),
                    }
                )
                continue

        print(
            f"🔍 DEBUG - Successfully extracted positions for {extracted_count}/{len(map_points)} points"
        )
        print(f"🔍 DEBUG - Total formatted points: {len(formatted_points)}")
        return formatted_points

    def shutdown(self):
        """Clean shutdown using ORB-SLAM3's built-in methods"""
        print("Shutting down ORB-SLAM3 integration...")

        try:
            # ✅ STOP PROCESSING THREAD FIRST
            self.is_running = False
            
            # Wait a bit for thread to stop
            if hasattr(self, "processing_thread"):
                self.processing_thread.join(timeout=2.0)

            # 🔥 NEW: save trajectories before shutting down SLAM
            # ADD THESE VERIFICATION PRINTS:
            all_frames_path = "experiments/results/trajectory/all_frames.tum"
            keyframes_path = "experiments/results/trajectory/keyframes.tum"
            
            
            


            if os.path.exists(all_frames_path):
                print(f"✅ TRAJECTORY SAVED: {all_frames_path}")
            else:
                print(f"❌ TRAJECTORY FAILED: {all_frames_path}")

            if os.path.exists(keyframes_path):
                print(f"✅ KEYFRAMES SAVED: {keyframes_path}")
            else:
                print(f"❌ KEYFRAMES FAILED: {keyframes_path}")

        except Exception as e:
            print(f"❌ TRAJECTORY SAVE ERROR: {e}")

            # ❌❌❌ DELETE THIS REDUNDANT CODE - YOU ALREADY HAVE THE POINTS!
            # Get current map points directly from ORB-SLAM3 as backup
            if self.slam_system is not None:
                try:
                    current_points = (
                        self.slam_system.get_tracked_map_points()
                    )  # ❌ REDUNDANT
                    if current_points:
                        formatted_points = self._format_map_points(
                            current_points
                        )  # ❌ REDUNDANT
                        print(
                            f"🔍 SHUTDOWN - Total {len(formatted_points)} points from ORB-SLAM3"
                        )
                except Exception as e:
                    print(f"⚠️ SHUTDOWN - Could not get points from ORB-SLAM3: {e}")

                # 🔥 NEW: save trajectories before shutting down SLAM
            try:
                self.slam_system.save_trajectory_tum(
                    "experiments/results/trajectory/all_frames.tum"
                )
                self.slam_system.save_keyframe_trajectory_tum(
                    "experiments/results/trajectory/keyframes.tum"
                )
            except Exception as e:
                print(f"Could not save trajectories: {e}")

            # ADD THESE VERIFICATION PRINTS:
            all_frames_path = "experiments/results/trajectory/all_frames.tum"
            keyframes_path = "experiments/results/trajectory/keyframes.tum"

            if os.path.exists(all_frames_path):
                print(f"✅ TRAJECTORY SAVED: {all_frames_path}")
            else:
                print(f"❌ TRAJECTORY FAILED: {all_frames_path}")

            if os.path.exists(keyframes_path):
                print(f"✅ KEYFRAMES SAVED: {keyframes_path}")
            else:
                print(f"❌ KEYFRAMES FAILED: {keyframes_path}")

        except Exception as e:
            print(f"❌ TRAJECTORY SAVE ERROR: {e}")

            if self.slam_system is not None:
                self.slam_system.shutdown()
                self.slam_system = None

            self.is_initialized = False
            print("ORB-SLAM3 integration shutdown complete")

        except Exception as e:
            print(f"ORB-SLAM3 shutdown failed: {e}")
 
    def get_system_stats(self) -> Dict[str, Any]:
        """Get ORB-SLAM3 RGB-D system statistics"""
        if not self.is_initialized or self.slam_system is None:
            return {}

        try:
            map_points = self.slam_system.get_tracked_map_points()
            tracking_info = self.slam_system.get_tracking_status()

            print(f"📊 ORB-SLAM RGB-D STATS:")
            print(f"   🔵 Total Map Points: {len(map_points) if map_points else 0}")
            print(
                f"   📍 Tracking State: {self._translate_tracking_state(tracking_info)}"
            )
            print(f"   🎯 Sensor: RGB-D")

            return {
                "tracking_state": self._translate_tracking_state(tracking_info),
                "map_points_count": len(map_points) if map_points else 0,
                "sensor_type": "RGB-D",
                "is_initialized": self.is_initialized,
            }

        except Exception as e:
            print(f"💥 ORB-SLAM: Failed to get system stats: {e}")
            print(f"Failed to get ORB-SLAM3 stats: {e}")
            return {}

    def reset_system(self):
        """Reset ORB-SLAM3 system using built-in method"""
        try:
            if self.slam_system is not None:
                # ✅ USE ORB-SLAM3's BUILT-IN RESET
                self.slam_system.Reset()
                print("ORB-SLAM3 system reset")
            else:
                print("ORB-SLAM3 system not available for reset")

        except Exception as e:
            print(f"ORB-SLAM3 reset failed: {e}")


if __name__ == "__main__":
    print(level=print)

    slam = ORBSLAMIntegration()

    if slam.initialize():
        print("✅ ORB-SLAM3 Ready - Testing with Synthetic Frames")

        frame_id = 0

        try:
            for i in range(100):  # Process 100 test frames
                # Create synthetic frames with features for ORB-SLAM3
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

                # Add some features to help ORB-SLAM3 track
                # Horizontal lines
                cv2.line(frame, (0, i * 5), (640, i * 5), (255, 0, 0), 2)
                # Vertical lines
                cv2.line(frame, (i * 7, 0), (i * 7, 480), (0, 255, 0), 2)
                # Moving circle
                cv2.circle(frame, (320 + i * 2, 240), 30, (0, 0, 255), 3)

                # Process frame with ORB-SLAM3
                timestamp = time.time()

                # Get current stats
                stats = slam.get_system_stats()
                print(
                    f"Frame {frame_id}: {stats['tracking_state']} - Points: {stats['map_points_count']}"
                )

                frame_id += 1
                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("⏹️ Stopping...")
        finally:
            slam.shutdown()
            print("✅ ORB-SLAM3 shutdown completed")
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            