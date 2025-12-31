#!/usr/bin/env python3
"""
MAIN ORCHESTRATOR - Pure coordinator that manages all pipelines
ONLY coordinates, NO direct processing or frame ID passing
"""

import sys
import os
import argparse
import numpy as np
from typing import List, Dict, Any, Optional
import time
import torch
import gc
import json
from collections import deque
from enum import Enum
import threading
import cv2
from queue import Queue
import yaml
from concurrent.futures import ThreadPoolExecutor
import os
import time  # Add at top of file if not already imported
import habitat_sim
import os
import numpy as np
import magnum as mn
import math

# ✅ ADD THIS IMPORT (with other imports)
from src.stores.habitat_store import global_habitat_store

# 🔧 FORCE OFFLINE MODE - USE CACHED MODELS IMMEDIATELY
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

print("🔧 OFFLINE MODE: Using cached OWL-ViT (no internet needed)")

# ✅ SET PYTHON PATH ONCE - Use main directory as root
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MAIN_DIR)

print(f"🔧 Running from: {os.getcwd()}")
print(f"🔧 MAIN_DIR: {MAIN_DIR}")
print(f"🔧 Python path: {sys.path}")

# ✅ ABSOLUTE IMPORTS FROM MAIN DIRECTORY
from src.stores.task_store import TaskStore
from src.stores.frame_buffer import global_frame_buffer  # ✅ GLOBAL SINGLETON
from src.stores.central_map_store import CentralMapStore
from src.stores.prediction_store import PredictionStore
from src.stores.user_command_store import UserCommandStore

from experiments.paper_experiments.Main_Experiments.policies.ours_full.run_ours_full_perception_pipeline import (
    PerceptionPipeline,
)
from experiments.paper_experiments.Main_Experiments.policies.ours_full.run_ours_full_prediction_pipeline import (
    PredictionPipeline,
)
from experiments.paper_experiments.Main_Experiments.policies.ours_full.run_ours_full_reasoning_pipeline import (
    ReasoningPipeline,
)
from experiments.paper_experiments.Main_Experiments.policies.ours_full.run_ours_full_action_pipeline import (
    ActionPipeline,
)
from src.perception_pipeline.orb_slam_integration import ORBSLAMIntegration


class OrchestrationState(Enum):
    INITIALIZING = "initializing"
    COLLECTING = "collecting"
    PROCESSING = "processing"
    SHUTDOWN = "shutdown"


class TaskStatus(Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    STUCK = "stuck"


class OursFullPolicy:
    """
    PURE COORDINATOR - Only manages pipelines, no processing
    """

    def __init__(self, config: Dict = None):
        self.state = OrchestrationState.INITIALIZING
        self.config = config

        self.agent_rigid = None

        # 🆕 ADD THIS: Store frame save directory from config
        self.frame_save_dir = config.get("frame_save_dir", None) if config else None
        if self.frame_save_dir:
            print(f"📁 Frame save directory configured: {self.frame_save_dir}")

            # ✅ FIXED (you had typo in _metriccs_ → _metrics_)

        # ✅ CREATE STORES
        self.map_store = CentralMapStore()
        self.task_store = TaskStore()
        self.prediction_store = PredictionStore()
        self.user_command_store = UserCommandStore() 

        # ✅ CREATE HABITAT SIMULATOR FIRST
        self.env = self._create_hm3d_habitat_env()
        print(
            f"checking self.env instance for habitat sim  inside create hm3d habitat env function {self.env}"
        )
        global_habitat_store.set_simulator(self.env)

        # ✅ CREATE ORB-SLAM INSTANCE HERE (before pipelines)
        self.orb_slam = ORBSLAMIntegration(
            map_store=self.map_store,
            task_store=self.task_store,
            prediction_store=self.prediction_store,
            config=self.config,
        )

        # ✅ INITIALIZE ORB-SLAM (starts thread, does coordinate alignment)
        success = self.orb_slam.initialize()  # Capture return value!
        if success:
            print("✅ ORB-SLAM: Successfully initialized!")
        else:
            print("❌ ORB-SLAM: FAILED to initialize!")

        # ✅ PASS ORB-SLAM + STORES to PerceptionPipeline
        perception_stores = {
            "map_store": self.map_store,
            "task_store": self.task_store,
            "prediction_store": self.prediction_store,
            "user_command_store": self.user_command_store,
            "orb_slam": self.orb_slam,  # 🆕 ADD ORB-SLAM INSTANCE
        }

        # ✅ PASS ONLY NON-FRAME STORES TO PIPELINES
        common_stores = {
            "map_store": self.map_store,
            "task_store": self.task_store,
            "prediction_store": self.prediction_store,
            "user_command_store": self.user_command_store,
        }

        # ✅ INITIALIZE PIPELINES with different parameters
        self.perception_pipeline = PerceptionPipeline(
            **perception_stores
        )  # Gets ORB-SLAM
        self.reasoning_pipeline = ReasoningPipeline(**common_stores)  # No ORB-SLAM
        self.prediction_pipeline = PredictionPipeline(**common_stores)  # No ORB-SLAM
        self.action_pipeline = ActionPipeline(**common_stores)  # No ORB-SLAM

        self.state = OrchestrationState.COLLECTING
        print("🎛️ Main Orchestrator - ORB-SLAM created, simulator ready")

    def _create_hm3d_habitat_env(self):
        """Create HM3D environment - DEBUGGED VERSION"""

        print("🔄 Creating HM3D environment with direct GLB loading...")

        # ✅ CREATE SEPARATE VARIABLE FOR CLARITY
        scene_path = "/mnt/d/Coding/Business/Kulfi_Startup_Code/robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/datasets/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"

        print(f"🔄 Loading scene from: {scene_path}")

        # ✅ VERIFY FILE EXISTS FIRST
        if not os.path.exists(scene_path):
            print(f"❌ Scene file not found: {scene_path}")
            return None

        # SIMPLE DIRECT APPROACH
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene_path  # ✅ Use the variable here
        # FORCE CPU MODE - skip GPU issues

        cfg.scene_dataset_config_file = (
            "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
            "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
            "datasets/hm3d_annotated_basis.scene_dataset_config.json"
        )

        cfg.load_semantic_mesh = True

        cfg.gpu_device_id = -1
        cfg.enable_physics = True  # ✅ ADDED: ENABLE PHYSICS
        cfg.force_separate_semantic_scene_graph = True

        # PROPER AGENT CONFIG WITH EXPLICIT SENSOR
        agent_cfg = habitat_sim.AgentConfiguration()

        # AFTER: agent_cfg = habitat_sim.AgentConfiguration()
        agent_cfg.height = 1.00  # ← ADD LINE 1
        agent_cfg.radius = 0.18  # ← ADD LINE 2

        # CURRENT - Only RGB sensor:
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.uuid = "rgba_camera"
        camera_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        agent_cfg.sensor_specifications = [camera_sensor_spec]

        # FIXED - Add depth sensor:
        sensor_specs = []

        # RGB Camera
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgba_camera"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [640, 480]
        # AFTER: rgb_spec = habitat_sim.CameraSensorSpec() block
        sensor_specs.append(rgb_spec)
        rgb_spec.hfov = 90.0  # 🆕 ADD THIS: Horizontal Field of View (degrees)
        rgb_spec.position = [0.0, 1.00, 0.0]  # ← ADD LINE 3 (camera at TOP)

        # ✅ ADD DEPTH SENSOR
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_camera"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [640, 480]
        sensor_specs.append(depth_spec)
        depth_spec.hfov = 90.0  # Must match RGB camera!
        # AFTER: depth_spec = habitat_sim.CameraSensorSpec() block
        depth_spec.position = [0.0, 1.00, 0.0]  # ← ADD LINE 4 (match RGB)

        spectator_cfg = habitat_sim.AgentConfiguration()

        # ONLY a camera sensor
        third_rgb = habitat_sim.CameraSensorSpec()
        third_rgb.uuid = "third_rgb"
        third_rgb.sensor_type = habitat_sim.SensorType.COLOR
        third_rgb.resolution = [640, 480]
        third_rgb.hfov = 90.0

        spectator_cfg.sensor_specifications = [third_rgb] 

        agent_cfg.sensor_specifications = sensor_specs
        print(f"Agent config: {agent_cfg}")

        # Check action space
        action_space = agent_cfg.action_space
        print(f"Action space: {list(action_space.keys())}")

        # Create simulator configuration
        sim_config = habitat_sim.Configuration(cfg, [agent_cfg , spectator_cfg])

        try:
            sim = habitat_sim.Simulator(sim_config)
            print("✅ HM3D Scene Loaded Successfully!")

            # 🚨 SEMANTIC DEBUG — MUST MATCH STRAIGHT-TURN
            print("\n🔍 === SEMANTIC MESH DEBUG (OURS_FULL) ===")
            print(f"[DEBUG] cfg.load_semantic_mesh = {cfg.load_semantic_mesh}")
            print(f"[DEBUG] sim.semantic_scene is None? {sim.semantic_scene is None}")

            if sim.semantic_scene is not None:
                objs = sim.semantic_scene.objects
                print(f"[DEBUG] semantic objects count = {len(objs)}")

                print("[DEBUG] FIRST 10 SEMANTIC OBJECTS:")
                for i, obj in enumerate(objs[:10]):
                    try:
                        cat_name = (
                            obj.category.name() if obj.category else "NO_CATEGORY"
                        )
                        print(f"  {i}: '{cat_name}'")
                    except Exception as e:
                        print(f"  {i}: ERROR {e}")
            else:
                print("❌ semantic_scene is None → semantics NOT loaded")

            # CONFIG FILE CHECK
            config_path = cfg.scene_dataset_config_file
            print(
                f"[DEBUG] scene_dataset_config_file exists: {os.path.exists(config_path)}"
            )
            print("🔍 === END SEMANTIC DEBUG ===\n")

            # 🔥 ADD THESE 12 LINES HERE:
            # 🔥 FIXED: Convert magnum.Deg → float BEFORE numpy

            width, height = rgb_spec.resolution
            hfov_deg_float = float(rgb_spec.hfov)  # 🔥 CONVERT TO FLOAT!
            hfov_rad = np.deg2rad(hfov_deg_float)  # ✅ NOW SAFE!

            fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
            fy = fx
            cx, cy = width / 2.0, height / 2.0

            intrinsics = {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "width": width,
                "height": height,
                "hfov_deg": float(hfov_deg_float),
            }

            global_habitat_store.set_camera_intrinsics(intrinsics)
            print("📷 Camera intrinsics extracted & stored:")
            print(intrinsics)

            # ✅ ADDED: LOAD NAVMESH (AFTER SIMULATOR CREATION)
            navmesh_path = scene_path.replace(".glb", ".navmesh")
            if os.path.exists(navmesh_path):
                print(f"🔄 Loading navmesh: {navmesh_path}")
                sim.pathfinder.load_nav_mesh(navmesh_path)
                print(f"✅ NavMesh loaded: {sim.pathfinder.is_loaded}")

            else:
                print(f"❌ NavMesh not found: {navmesh_path}")

            # 🔥 LOCOBOT EMBODIMENT (Docs + Your Assets)
            print("🔥 === LOCOBOT ROBOT (Your Assets) ===")

            # 1. YOUR LoCoBot path
            locobot_path = "./datasets/versioned_data/locobot_merged_0.2"

            # 2. Load template (Docs exact!)
            obj_template_mgr = sim.get_object_template_manager()
            locobot_template_id = obj_template_mgr.load_configs(locobot_path)[0]

            # 3. Spawn + attach agent (Docs exact!)
            rigid_obj_mgr = sim.get_rigid_object_manager()
            self.robot_id = rigid_obj_mgr.add_object_by_template_id(
                locobot_template_id,
                sim.get_agent(0).scene_node,  # Agent cameras follow LoCoBot!
            )

            # 🔥 PHYSICS STABILITY (ONCE AT CREATION - PERFECT!)
            robot = self.robot_id
            robot.rolling_friction_coefficient = 1.0  # Preventative: Max wheel grip
            robot.spinning_friction_coefficient = 1.0  # Preventative: Max spin damping
            print(
                f"🔧 ROBOT CREATED: friction=roll{robot.rolling_friction_coefficient},spin{robot.spinning_friction_coefficient}"
            )

            print(f"✅ LOCOBOT ID={self.robot_id.object_id} | Agent embodied!")
            print("🔥 === FULL ROBOT PHYSICS READY ===")

            # Test frame capture immediately with more debugging
            test_obs = sim.get_sensor_observations()
            print(f"🔍 Test sensors available: {list(test_obs.keys())}")

            if "rgba_camera" in test_obs:
                test_frame = test_obs["rgba_camera"]
                print(f" {test_frame.shape} | dtype: {test_frame.dtype}")
            else:
                print(f"❌ No rgba_camera in test observations")

            if "depth_camera" in test_obs:
                depth_frame = test_obs["depth_camera"]
                print(f"✅ Depth: {depth_frame.shape} | dtype: {depth_frame.dtype}")
                # Expected: (480, 640) float32

                print(
                    f"📷 RGB: HFOV={rgb_spec.hfov}°, Resolution={rgb_spec.resolution}"
                )

                print(
                    f"📷 Depth: HFOV={depth_spec.hfov}°, Resolution={depth_spec.resolution}"
                )

            return sim
        except Exception as e:
            print(f"❌ Failed to create habitat simulator: {e}")
            return None

    def run_orchestration(self, task_description: str = "explore and map environment"):
        print(f"🚀 Starting main orchestration for task: {task_description}")
        print(
            f"checking self.env instance for habitat sim inside run_orchestration function {self.env}"
        )

        # YOUR CODE (CORRECT!)
        agent = self.env.get_agent(0)
        start_state = agent.get_state()
        print(f"📍 METRICS: START POS = {start_state.position}")  # NEW DEBUG
        global_habitat_store._metrics_trajectory.append(start_state.position.copy())
        print(
            f"📍 METRICS: Trajectory len = {len(global_habitat_store._metrics_trajectory)}"
        )  # NEW
        self._metrics_start_time = time.time()
        print("⏱️  METRICS: Start time recorded")  # NEW

        # 🆕 GET FRAME LIMIT FROM CONFIG
        max_frames = self.config.get("num_frames", 100)
        print(f"🎯 Processing limit: {max_frames} frames")

        if hasattr(self.task_store, "set_initial_task"):
            # ========== SINGLE UNIFIED PLAN CREATION ==========
            current_timestamp = time.time()
            reasoning_id = f"reason_{int(current_timestamp * 1000)}"
            task_id = reasoning_id  # Use same ID

            # Create the complete reasoning plan
            complete_action_plan = {
                # ==================== METADATA ====================
                "reasoning_cycle_id": reasoning_id,
                "timestamp": current_timestamp,
                "component": "final_action_plan",
                "planning_stages": [
                    "strategy_selection",
                    "detailed_planning",
                    "fusion",
                ],
                "llm_used": True,
                # ==================== TASK CONTEXT ====================
                "task_context": {
                    "task_id": task_id,
                    "description": task_description,
                    "parameters": {
                        "object_queries": [
                            "table",
                            "chair",
                            "door",
                            "cabinet",
                            "nightstand",
                        ]
                    },
                },
                # ==================== INDIVIDUAL COMPONENT PLANS ====================
                "component_plans": {
                    "tree_of_thoughts": {
                        "strategy_decision": {
                            "strategy_type": "priority_first",
                            "strategy_reasoning": "Keys typically found on elevated surfaces first",
                            "confidence": 0.85,
                        },
                        "detailed_plan": {
                            "target_objects": ["table", "cabinet", "nightstand"],
                            "search_pattern": "priority_first_pattern",
                        },
                    },
                    "spatial_reasoning": {
                        "navigability_score": 0.8,
                        "optimized_path": [
                            {"x": 1.0, "y": 2.0, "type": "waypoint"},
                            {"x": 2.5, "y": 2.0, "type": "waypoint"},
                        ],
                    },
                },
                # ==================== FUSED FINAL PLAN ====================
                "fused_decision": {
                    "selected_action": "move_to_table",
                    "final_confidence": 0.78,
                    "fusion_logic": "tree_of_thoughts_primary_with_spatial_adjustment",
                },
                # ==================== EXECUTION-READY PLAN ====================
                "execution_ready_plan": {
                    "waypoints": [
                        {"x": 1.0, "y": 2.0, "object": "table", "action": "scan"},
                        {"x": 2.5, "y": 2.0, "object": "cabinet", "action": "scan"},
                    ],
                    "strategy_type": "priority_first",
                    "target_objects": ["table", "cabinet", "nightstand"],
                    "expected_duration": "medium",
                    "navigation_constraints": {"max_speed": 0.3, "avoid_areas": []},
                },
                # ==================== TASK STATUS ====================
                "task_status": {
                    "status": "pending_execution",
                    "execution_attempts": 0,
                    "completion_status": "not_started",
                },
            }

            # ========== SAVE TO TASKSTORE USING ONE METHOD ==========
            # 1. Store the complete plan directly
            self.task_store.current_action_plan = complete_action_plan

            # 2. Also store in reasoning_plans
            if hasattr(self.task_store, "reasoning_plans"):
                if not isinstance(self.task_store.reasoning_plans, dict):
                    self.task_store.reasoning_plans = {}
                self.task_store.reasoning_plans[reasoning_id] = {
                    "tree_of_thoughts": complete_action_plan["component_plans"][
                        "tree_of_thoughts"
                    ],
                    "spatial_reasoning": complete_action_plan["component_plans"][
                        "spatial_reasoning"
                    ],
                    "final_action": complete_action_plan,
                }

            # 3. Call set_initial_task with JUST the description (for mission goals)
            self.task_store.set_initial_task(task_description)

            print(f"✅ Created task: {task_description}")
            print(f"📋 Complete reasoning plan stored with ID: {reasoning_id}")
            print(
                f"📊 Action plan has {len(complete_action_plan['execution_ready_plan']['waypoints'])} waypoints"
            )

            # 4. ALSO store in intermediate_reasoning (for OWL to find)
            if hasattr(self.task_store, "intermediate_reasoning"):
                if not isinstance(self.task_store.intermediate_reasoning, dict):
                    self.task_store.intermediate_reasoning = {}

                # Create the structure OWL is looking for
                self.task_store.intermediate_reasoning[reasoning_id] = {
                    "component_reasoning": {
                        "tree_of_thoughts": complete_action_plan["component_plans"][
                            "tree_of_thoughts"
                        ],
                        "spatial_reasoning": complete_action_plan["component_plans"][
                            "spatial_reasoning"
                        ],
                    },
                    "fused_reasoning": complete_action_plan["fused_decision"],
                    "execution_plan": complete_action_plan["execution_ready_plan"],
                    "timestamp": current_timestamp,
                    "reasoning_cycle_id": reasoning_id,
                }

                print(f"📝 Also stored in intermediate_reasoning for OWL access")

        else:
            # ========== EMERGENCY FALLBACK ONLY ==========
            current_timestamp = time.time()
            reasoning_id = f"reason_{int(current_timestamp * 1000)}"

            # Create minimal structure if set_initial_task doesn't exist
            self.task_store.current_task = type(
                "MockTask",
                (),
                {
                    "task_id": reasoning_id,
                    "description": task_description,
                    "timestamp": current_timestamp,
                },
            )()

            print(f"🆘 Emergency fallback - No set_initial_task method")

        # ✅ START PIPELINE PROCESSING FIRST
        try:
            print("🔄 Starting pipeline processing...")
            self.perception_pipeline.initialize_processing()
            self.reasoning_pipeline.initialize_processing()
            self.prediction_pipeline.initialize_processing()
            self.action_pipeline.initialize_processing()
            print("✅ All pipelines started successfully")

        except Exception as e:
            print(f"❌ Failed to start pipelines: {e}")
            return

        # 🆕 TRACKING VARIABLES
        frame_id = 0
        successful_frames = 0
        failed_frames = 0

        # 🆕 START TIMER
        start_time = time.time()

        # THE REST OF YOUR CODE STAYS EXACTLY THE SAME FROM HERE...
        try:

            # After initializing env and before starting frame capture
            if self.env is not None:
                for _ in range(3):
                    print("🔄 Stepping simulator 3 times to warm up sensors...")
                    print("DEBUG: about to step simulator")
                    self.env.step("move_forward")
                    print("✅ Simulator warm-up complete, first frames ready")

            while (
                frame_id < max_frames
                and not (
                    self.task_store.goal_reached
                    or self.task_store.is_mission_complete()
                )  # 🔥 NEW CONDITION
                and self.state != OrchestrationState.SHUTDOWN
            ):

                if self.state == OrchestrationState.SHUTDOWN:
                    print("🛑 Shutdown signal received")
                    break

                print(f"\n--- FRAME {frame_id + 1}/{max_frames} ---")

                try:

                    # ✅ CAPTURE FRAME FROM HABITAT
                    if self.env is None:
                        print("❌ Habitat simulator not initialized")
                        failed_frames += 1
                        frame_id += 1
                        continue

                    # ✅ CAPTURE FRAME FROM HABITAT
                    # ✅ CAPTURE FRAME FROM HABITAT
                    print("DEBUG: about to get sensor observations")
                    observations = self.env.get_sensor_observations()

                    print(
                        f"🎯 CURRENT SENSORS: {list(observations.keys())}"
                    )  # <-- ADD THIS LINE

                    print("DEBUG: about to extract rgb and depth")
                    rgb_frame = observations["rgba_camera"]
                    depth_frame = observations["depth_camera"]

                    # ========== ADD THIS DEBUG BLOCK HERE ==========
                    print(
                        f"🎯 DEPTH STATS - Min: {depth_frame.min()}, Max: {depth_frame.max()}, Mean: {depth_frame.mean():.3f}"
                    )

                    # Step 1: Verify valid depth exists
                    if np.count_nonzero(depth_frame) == 0:
                        print(
                            "🚨 CRITICAL: Depth frame is ALL ZEROS! Habitat depth sensor not working!"
                        )
                    else:
                        print(
                            f"✅ Depth frame has {np.count_nonzero(depth_frame)} valid pixels"
                        )

                        # Step 2: Save a correctly normalized 16-bit PNG for debugging
                        if depth_frame.dtype == np.uint16:
                            # Normalize the valid data range to 0-65535 for 16-bit PNG
                            if depth_frame.max() > depth_frame.min():
                                depth_normalized = (
                                    (depth_frame - depth_frame.min())
                                    / (depth_frame.max() - depth_frame.min())
                                    * 65535
                                )
                                depth_normalized = depth_normalized.astype(np.uint16)
                            else:
                                depth_normalized = np.zeros_like(
                                    depth_frame, dtype=np.uint16
                                )

                    # ========== END DEBUG BLOCK ==========

                    # Convert RGBA -> RGB if needed
                    if rgb_frame.shape[2] == 4:
                        print(f"DEBUG: RGBA -> RGB conversion for frame {frame_id}")
                        rgb_frame = rgb_frame[:, :, :3]

                    # Ensure correct shape (480, 640, 3)
                    if rgb_frame.shape[:2] != (480, 640):
                        print(
                            f"DEBUG: Transposing rgb_frame from {rgb_frame.shape} to (480, 640, 3)"
                        )
                        rgb_frame = np.transpose(rgb_frame, (1, 0, 2))

                    # 🆕 ADD THIS: Transpose depth frame from (640, 480) to (480, 640)
                    print(
                        f"DEBUG: Depth frame shape before transpose: {depth_frame.shape}"
                    )
                    if depth_frame.shape == (640, 480):
                        depth_frame = depth_frame.T  # Or np.transpose(depth_frame)
                        print(
                            f"DEBUG: Depth frame shape after transpose: {depth_frame.shape}"
                        )

                    # 🆕 ALSO: Convert depth dtype if needed (float32 → uint16)
                    if depth_frame.dtype == np.float32:
                        depth_frame = (depth_frame * 1000).astype(
                            np.uint16
                        )  # meters → millimeters
                        print(f"DEBUG: Converted depth dtype to uint16")

                    print("DEBUG: about to rotate rgb frame")

                    print("DEBUG: about to scale depth")

                    # ✅ WRITE BOTH RGB AND DEPTH TO GLOBAL BUFFER
                    timestamp = frame_id * 0.033  # 30 FPS relative timestamps

                    agent_state = self.env.get_agent(0).get_state()
                    habitat_position = agent_state.position
                    habitat_rotation = agent_state.rotation

                    frame_dict = {
                        "rgb": rgb_frame,
                        "depth": depth_frame,
                        "timestamp": timestamp,
                        "habitat_position": habitat_position,
                        "habitat_rotation": habitat_rotation,
                    }
                    success = global_frame_buffer.write_frame(
                        frame_dict, frame_id, timestamp
                    )

                    if success:

                        print(f"  Wrote frame {frame_id} to buffer successfully ✅")

                    else:
                        print(f"  ❌ Failed to write frame {frame_id} to buffer")

                    print(
                        f"  ✅ Frame {frame_id + 1} captured - RGB: {rgb_frame.shape}, Depth: {depth_frame.shape}"
                    )

                    print(
                        f"🟦 DEBUG: Calling ORB-SLAM process_frame() for frame {frame_id}"
                    )

                    if frame_id == 0:
                        self.orb_slam.process_frame()

                    print(
                        f"🟩 DEBUG: Returned from ORB-SLAM process_frame() for frame {frame_id}"
                    )

                    # 2. ✅ CHECK FOR QUEUED ACTIONS (ADD THIS HERE)
                    print(f"🔍 Checking for queued actions...")
                    while True:
                        # Pull next action from HabitatStore queue
                        action_item = (
                            global_habitat_store.pull_action_from_habitat_store()
                        )
                        if not action_item:
                            break  # No more actions in queue

                        print(f"🎯 MAINORCH: Executing action '{action_item['type']}'")

                        # Execute the action in MAIN THREAD (OpenGL safe!)
                        result = self._execute_habitat_action(action_item)

                        print(f"Action in habitat simulation result : {result}")

                        # Store result so ActionExecutor can retrieve it
                        global_habitat_store.store_action_result(
                            action_item["action_id"], result
                        )

                        print(
                            f"✅ Action '{action_item['type']}' completed, result stored"
                        )
                    # ==================== END OF NEW CODE ====================

                    # 🆕 🎯 CALL PREDICTION PIPELINE EVERY 10 FRAMES
                    # 🆕 🎯 CALL PREDICTION PIPELINE EVERY 10 FRAMES
                    if frame_id % 10 == 0:  # Run predictions every 10 frames
                        try:
                            print(
                                f"🔮 Running prediction cycle for frame {frame_id}..."
                            )
                            predictions = (
                                self.prediction_pipeline.process_prediction_cycle(
                                    frame_id
                                )
                            )

                            print(f"🔍 PREDICTIONS DEBUG - Type: {type(predictions)}")

                            if predictions:
                                print(
                                    f"🔍 PREDICTIONS DEBUG - Keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'Not dict'}"
                                )

                                safest_action = None

                                # 🆕 CASE 1: DIRECT FUSED RESULTS (CURRENT STRUCTURE)
                                if (
                                    isinstance(predictions, dict)
                                    and "recommended_action" in predictions
                                ):
                                    print("🎯 Using direct fused predictions structure")
                                    safest_action = predictions.get(
                                        "recommended_action", "move_forward"
                                    )

                                    # Show action decisions if available
                                    if "action_decisions" in predictions:
                                        action_scores = predictions["action_decisions"]
                                        print(f"  🔮 ACTION DECISIONS:")
                                        for action, data in action_scores.items():
                                            risk = data.get("combined_risk_score", 0.5)
                                            decision = data.get("decision", "unknown")
                                            print(
                                                f"  📊 {action}: risk={risk:.3f}, decision={decision}"
                                            )

                                    print(
                                        f"  🎯 Fused prediction suggests: {safest_action}"
                                    )

                                # 🆕 CASE 2: NESTED PREDICTION STORE STRUCTURE (OLD STRUCTURE)
                                elif isinstance(predictions, dict) and any(
                                    "pred_" in str(key) for key in predictions.keys()
                                ):
                                    print("🎯 Using nested prediction store structure")
                                    pred_key = list(predictions.keys())[0]

                                    if pred_key:
                                        predictors = predictions[pred_key].get(
                                            "predictors", {}
                                        )
                                        collision_data = predictors.get(
                                            "collision_risk", {}
                                        ).get("data", {})
                                        structural_data = predictors.get(
                                            "structural_continuity", {}
                                        ).get("data", {})

                                        collision_risks = collision_data.get(
                                            "action_risks", {}
                                        )
                                        structural_risks = structural_data.get(
                                            "structural_risks", {}
                                        )

                                        # Find action with lowest combined risk
                                        action_scores = {}
                                        for action in [
                                            "move_forward",
                                            "turn_left",
                                            "turn_right",
                                        ]:
                                            collision_score = collision_risks.get(
                                                action, {}
                                            ).get("risk_score", 0.5)
                                            structural_score = structural_risks.get(
                                                action, {}
                                            ).get("risk_score", 0.5)
                                            total_score = (collision_score * 0.7) + (
                                                structural_score * 0.3
                                            )
                                            action_scores[action] = total_score

                                        safest_action = min(
                                            action_scores.items(), key=lambda x: x[1]
                                        )[0]

                                        print(f"  🔮 PREDICTION RESULTS:")
                                        print(
                                            f"  📊 Risk scores - Forward:{action_scores['move_forward']:.3f}, "
                                            f"Left:{action_scores['turn_left']:.3f}, Right:{action_scores['turn_right']:.3f}"
                                        )
                                        print(
                                            f"  🎯 Prediction suggests: {safest_action}"
                                        )

                                        # 🆕 DEBUG: Show what we found in predictions
                                        print(
                                            f"  🔍 PREDICTION DEBUG: Found {len(predictions.keys())} prediction entries"
                                        )
                                        print(
                                            f"  🔍 PREDICTIONS DEBUG - Keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'NO PREDICTIONS'}"
                                        )
                                    else:
                                        print("  🔮 No valid prediction key found")
                                        safest_action = None

                                # 🆕 CASE 3: UNKNOWN STRUCTURE OR ERROR
                                else:
                                    print(
                                        f"❌ Unknown prediction structure or error result"
                                    )
                                    print(f"   Type: {type(predictions)}")
                                    print(f"   Value: {predictions}")
                                    safest_action = None

                            else:
                                print("  🔮 No predictions generated this cycle")
                                safest_action = None

                        except Exception as e:
                            print(f"❌ Prediction cycle error: {e}")
                            import traceback

                            traceback.print_exc()
                            safest_action = None

                    ## 🖼️ SAVE FRAMES FOR DEBUGGING - MULTIPLE CHECKPOINTS
                    save_frames = [2, 4, 5, 15, 23, 35, 45, 65, 85]
                    if frame_id in save_frames:
                        # 🆕 CHECK FRAME ORIENTATION BEFORE SAVING
                        frame_shape = rgb_frame.shape
                        print(f"  🖼 Frame {frame_id} shape: {frame_shape}")

                        if frame_shape[0] == 480 and frame_shape[1] == 640:
                            print("  ✅ Frame orientation: PORTRAIT (480x640)")
                            # ROTATE ONLY FOR DISPLAY/SAVING
                            rgb_frame_display = cv2.rotate(
                                rgb_frame, cv2.ROTATE_90_CLOCKWISE
                            )
                            # USE:
                            if frame_id in save_frames:
                                if (
                                    hasattr(self, "frame_save_dir")
                                    and self.frame_save_dir
                                ):
                                    save_path = os.path.join(
                                        self.frame_save_dir,
                                        f"ours_full_frame_{frame_id}.png",
                                    )
                                else:
                                    save_path = f"ours_full_frame_{frame_id}.png"

                                # Rest of orientation logic...
                                cv2.imwrite(save_path, rgb_frame_display)
                                print(f"  🖼 Saved to: {save_path}")  # Updated message
                        elif frame_shape[0] == 640 and frame_shape[1] == 480:
                            print("  ✅ Frame orientation: LANDSCAPE (640x480)")
                            cv2.imwrite(
                                f"ours_full_frame_{frame_id}.png", rgb_frame
                            )  # No rotation needed
                        else:
                            print(f"  ⚠️  Unexpected frame shape: {frame_shape}")
                            cv2.imwrite(f"ours_full_frame_{frame_id}.png", rgb_frame)

                        print(f"  🖼 Saved ours_full_frame_{frame_id}.png")

                    rgb_frame_display = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                    print("DEBUG: about to write rgb frame")

                    # USE:
                    if hasattr(self, "frame_save_dir") and self.frame_save_dir:
                        save_path = os.path.join(
                            self.frame_save_dir,
                            f"ours_full_frame_before_{frame_id}.png",
                        )
                    else:
                        save_path = f"ours_full_frame_before_{frame_id}.png"  # Fallback
                    cv2.imwrite(save_path, rgb_frame_display)
                    
                    
                    spectator_obs = self.env.get_sensor_observations(1)
 
                    print(f"🎥 Agent-1 sensors: {list(spectator_obs.keys())}")

                    # Capture debug camera frame from sensor directly
                    if "third_rgb" in spectator_obs:
                        third_frame = spectator_obs["third_rgb"]

                        if third_frame.shape[-1] == 4:
                            third_frame = third_frame[..., :3]

                        third_frame_bgr = cv2.cvtColor(third_frame, cv2.COLOR_RGB2BGR)

                        third_save_path = save_path.replace(
                            "ours_full_frame_",
                            "spectator_view_",
                        )

                        cv2.imwrite(third_save_path, third_frame_bgr)
                        print(f"🎥 Spectator frame saved: {third_save_path}")
                    else:
                        print("❌ agent_1 : third_rgb missing")


                    # 💡 BRIGHTNESS CHECK
                    mean_brightness = rgb_frame.mean()
                    print(
                        f"💡 Processing frame {frame_id} brightness: {mean_brightness:.1f}"
                    )

                    # SUCCESS PATH
                    successful_frames += 1
                    print(
                        f"✅ Frame {frame_id} captured - RGB: {rgb_frame.shape}, Depth: {depth_frame.shape}"
                    )

                except Exception as e:
                    print(f"❌ Frame {frame_id} processing failed: {e}")
                    failed_frames += 1
                    # Do NOT increment here — we increment once below

                frame_id += 1

                # 🆕 PROGRESS LOGGING EVERY 2 FRAMES
                if (frame_id - 1) % 2 == 0:  # frame_id was just incremented
                    progress = (frame_id / max_frames) * 100
                    print(
                        f"📊 Progress: {frame_id}/{max_frames} frames ({progress:.1f}%)"
                    )

                # 🆕 VRAM MANAGEMENT
                if frame_id % 5 == 0:
                    self._manage_vram_usage(frame_id)

                time.sleep(0.1)  # Small delay between frames

            # 🆕 FINAL SUMMARY
            # 🆕 FINAL SUMMARY
            self._print_final_summary(
                frames_written=successful_frames,  # ✅ Frames successfully written to buffer
                frames_failed=failed_frames,  # ✅ Frames that failed
                start_time=start_time,
                next_frame_id=frame_id,  # ✅ Next frame ID (optional, for info)
            )

        except KeyboardInterrupt:
            print("🛑 Orchestration interrupted by user")
        except Exception as e:
            print(f"❌ Orchestration error: {e}")
        # finally:
        #     self.shutdown_orchestration()

    def _execute_habitat_action(self, action_item: dict) -> dict:
        """
        Execute Habitat action in main thread (OpenGL safe).
        Supports BOTH:
        - Discrete Habitat actions (move_forward, turn_left, turn_right)
        - Continuous velocity control (type == "velocity_control")

        BACKWARD COMPATIBLE:
        - Same return structure
        - Same HabitatStore contract
        """
        try:
            action_type = action_item["type"]
            parameters = action_item.get("parameters", {})
            metadata = action_item.get("metadata", {})

            print("=" * 80)
            print(f"🎯 Executing Habitat action: {action_type}")
            print(f"📦 Parameters: {parameters}")
            print(f"📦 Metadata: {metadata}")
            print("=" * 80)

            # ==========================================================
            # 🛑 ZERO-VELOCITY STOP + FACE GOAL (PHYSICS-SAFE!)
            # ==========================================================
            if action_type == "velocity_control" and metadata.get(
                "is_stop_command", False
            ):
                print("🛑 EXECUTING ZERO-VELOCITY STOP + FACE GOAL")

                # 🔥 SET POSITION SUCCESS
                self.task_store.goal_reached = True
                print(f"🎯 GOAL POSITION REACHED - Flag set!")

                # 🔥 1. GET LOCOBOT TRUTH
                # rigid_mgr = self.env.get_rigid_object_manager()
                # locobot = rigid_mgr.get_object_by_id(self.locobot.object_id)
                # rb_pos = locobot.translation
                # rb_rot = locobot.rotation

                locobot = self.robot_id  # ✅ Keep this
                rb_pos = locobot.translation
                rb_rot = locobot.rotation

                print(
                    f"🔍 DEBUG: LoCoBot pos=[{rb_pos.x:.3f}, {rb_pos.y:.3f}, {rb_pos.z:.3f}]"
                )

                # 🔥 2. PHYSICS STOP (Zero velocity)
                vc = locobot.velocity_control
                vc.linear_velocity = mn.Vector3(0.0, 0.0, 0.0)
                vc.angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
                vc.controlling_lin_vel = True
                vc.controlling_ang_vel = True
                self.env.step_physics(0.1)  # Settle physics

                # 1. Floor Lock
                if locobot.translation.y < 0.05:
                    locobot.translation = mn.Vector3(
                        locobot.translation.x, 0.05, locobot.translation.z
                    )

                print("✅ LoCoBot STOPPED (zero velocity)")

                # 🔥 3. COMPUTE DESIRED ANGULAR VELOCITY (No teleport!)
                alignment_yaw = None
                if "target_xyz" in metadata:
                    goal_pos = np.array(metadata["target_xyz"])
                    current_pos = np.array([rb_pos.x, rb_pos.y, rb_pos.z])
                    delta = goal_pos - current_pos
                    desired_yaw = math.atan2(delta[0], -delta[2])

                    # 🔥 COMPUTE YAW ERROR
                    current_quat = [
                        rb_rot.vector.x,
                        rb_rot.vector.y,
                        rb_rot.vector.z,
                        rb_rot.scalar,
                    ]
                    current_yaw = self._yaw_from_quat(current_quat)
                    yaw_error = desired_yaw - current_yaw

                    # Normalize to [-pi, pi]
                    yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi

                    print(f"🎯 Goal={goal_pos}, LoCoBot={current_pos}")
                    print(
                        f"📐 Current yaw={math.degrees(current_yaw):.1f}° → Desired={math.degrees(desired_yaw):.1f}° → Error={math.degrees(yaw_error):.1f}°"
                    )

                    # 🔥 ANGULAR VELOCITY TO FACE GOAL (Physics!)
                    turn_speed = 2.0  # rad/s
                    vc.angular_velocity = mn.Vector3(
                        0.0, yaw_error * turn_speed, 0.0
                    )  # Y-axis turn!
                    vc.controlling_ang_vel = True

                    # Quick turn physics step
                    self.env.step_physics(0.2)  # Turn time based on error

                    if locobot.translation.y < 0.05:
                        locobot.translation = mn.Vector3(
                            locobot.translation.x, 0.05, locobot.translation.z
                        )

                    alignment_yaw = desired_yaw
                    print(f"✅ PHYSICS ALIGNMENT → {math.degrees(desired_yaw):.1f}°")
                else:
                    print("ℹ️ No target_xyz - skipping alignment")

                # 🔥 4. FINAL PHYSICS TRUTH (After alignment physics)
                final_rb_pos = locobot.translation
                final_rb_rot = locobot.rotation
                final_quat = [
                    final_rb_rot.vector.x,
                    final_rb_rot.vector.y,
                    final_rb_rot.vector.z,
                    final_rb_rot.scalar,
                ]

                observations = self.env.get_sensor_observations()
                print(
                    f"🔍 FINAL LoCoBot pos=[{final_rb_pos.x:.3f}, {final_rb_pos.y:.3f}, {final_rb_pos.z:.3f}]"
                )
                print(
                    f"🔍 FINAL quat=[{final_quat[0]:.4f}, {final_quat[1]:.4f}, {final_quat[2]:.4f}, {final_quat[3]:.4f}]"
                )

                # 🔥 5. SAME RETURN FORMAT
                result = {
                    "action_id": action_item["action_id"],
                    "status": "completed",
                    "action": "velocity_control",
                    "observations": observations,
                    "collided": False,
                    "new_position": [final_rb_pos.x, final_rb_pos.y, final_rb_pos.z],
                    "new_rotation": final_quat,
                    "timestamp": time.time(),
                    "is_stop_result": True,
                    "stop_success": True,
                    "facing_goal_yaw": float(alignment_yaw) if alignment_yaw else None,
                }

                print(f"📤 Result pos={result['new_position'][:2]}")
                print(f"📤 Result quat={result['new_rotation']}")
                print(f"🛑 PHYSICS STOP + ALIGNMENT COMPLETE ✓")
                return result

            # ==========================================================
            # 🚀 CONTINUOUS VELOCITY CONTROL (MINIMAL FIX)
            # ==========================================================

            # include the commented out setstate logic

            if action_type == "velocity_control" and metadata.get(
                "continuous_nav", False
            ):
                linear_vel = parameters.get("linear_velocity", [0.0, 0.0, 0.0])
                angular_vel = parameters.get("angular_velocity", [0.0, 0.0, 0.0])
                duration = parameters.get("duration", 0.1)

                print(
                    f"🧮 Linear velocity: {linear_vel}, angular velocity: {angular_vel}, dt={duration}"
                )

                robot = self.robot_id  # ✅ Keep this

                # 🔥 2. PHYSICS STOP (Zero velocity)
                vc = robot.velocity_control
                print(f"✅ VC created: {type(vc)}")
                vc.controlling_lin_vel = True
                vc.controlling_ang_vel = True
                vc.lin_vel_is_local = True
                vc.ang_vel_is_local = True
                vc.linear_velocity = mn.Vector3(*linear_vel)
                print(f"✅ VC lin_vel set: {vc.linear_velocity}")
                vc.angular_velocity = mn.Vector3(*angular_vel)
                print(f"✅ VC ang_vel set: {vc.angular_velocity}")

                rb_pos = np.array(
                    [robot.translation.x, robot.translation.y, robot.translation.z]
                )
                print(f"✅ rb_pos: {rb_pos.tolist()}")
                rb_rot = robot.rotation
                print(f"✅ rb_rot: {type(rb_rot)}")

                # 🔥 FIX: Convert Magnum Quaternion → list
                quat_list = [
                    rb_rot.vector.x,
                    rb_rot.vector.y,
                    rb_rot.vector.z,
                    rb_rot.scalar,
                ]
                print(f"✅ quat_list: {quat_list}")

                flat_q = self.flatten_quaternion_to_yaw(quat_list)
                print(
                    f"✅ flat_q: {flat_q}, type={type(flat_q)}, len={len(flat_q) if hasattr(flat_q,'__len__') else 'N/A'}"
                )

                # 🔥 FIXED ERROR LINE (ONLY CHANGE HERE):
                print(
                    f"🔧 flat_q[0:4]={flat_q[:4] if hasattr(flat_q,'__getitem__') else 'NO INDEX'}"
                )
                rigid_state = habitat_sim.RigidState(
                    mn.Quaternion(
                        mn.Vector3(flat_q[0], flat_q[1], flat_q[2]), flat_q[3]
                    ),  # ✅ SAFE!
                    mn.Vector3(rb_pos[0], rb_pos[1], rb_pos[2]),
                )
                print(f"✅ rigid_state created: pos={rigid_state.translation}")

                print(
                    f"📐 RigidState before integrate: pos={rigid_state.translation}, rot={rigid_state.rotation}"
                )

                new_rigid_state = vc.integrate_transform(duration, rigid_state)

                new_pos = np.array(
                    [
                        new_rigid_state.translation.x,
                        new_rigid_state.translation.y,
                        new_rigid_state.translation.z,
                    ]
                )

                print(f"📍 Predicted new_pos from VelocityControl: {new_pos.tolist()}")

                self.env.step_physics(duration)  # ✅ INTEGRATES velocities!
                 
                # THIRD-PERSON CAMERA FOLLOW 
                # Agent 1 = spectator
                agent1 = self.env.get_agent(1)

                robot_pos = self.robot_id.translation
                robot_rot = self.robot_id.rotation
 
                # Camera offset: behind + above
                
                # Convert Magnum Quaternion to list [x, y, z, w]
                robot_rot_list = [
                    robot_rot.vector.x,
                    robot_rot.vector.y, 
                    robot_rot.vector.z,
                    robot_rot.scalar
                ]
                
                offset = mn.Vector3(0.0, 0.0, +2.0)
                cam_pos = robot_pos + robot_rot.transform_vector(offset)
                
                agent1.set_state(
                    habitat_sim.AgentState(
                        position=cam_pos,
                        rotation=robot_rot_list,
                    ),
                    reset_sensors=False,
                )

                print("🎥 [SPECTATOR] synced to robot")
                print(f"🎥 [SPECTATOR] cam_pos = {cam_pos}")
                


                if robot.translation.y < 0.05:
                    robot.translation = mn.Vector3(
                        robot.translation.x, 0.05, robot.translation.z
                    )

                #
                # 🔥 1. POST-PHYSICS TILT KILLER (IMMEDIATELY AFTER STEP)
                #
                post_rot = robot.rotation
                quat_list_post = [
                    post_rot.vector.x,
                    post_rot.vector.y,
                    post_rot.vector.z,
                    post_rot.scalar,
                ]
                flat_q_post = self.flatten_quaternion_to_yaw(quat_list_post)

                # Snap the robot upright BEFORE we log any metrics or take photos
                robot.rotation = mn.Quaternion(
                    mn.Vector3(flat_q_post[0], flat_q_post[1], flat_q_post[2]),
                    flat_q_post[3],
                )
                # Kill tilting momentum
                robot.angular_velocity = mn.Vector3(0.0, robot.angular_velocity.y, 0.0)
 
                # 🔥 PRIMARY COLLISION: Physics contacts (for completeness)
                contacts = self.env.get_physics_contact_points()
                collision_contacts = []

                for contact in contacts:
                    if contact.is_active:  # ✅ YOUR attr!
                        contact_data = {
                            "object_id_a": int(contact.object_id_a),  # ✅
                            "object_id_b": int(contact.object_id_b),  # ✅
                            "position_on_a": [  # ✅
                                float(contact.position_on_a_in_ws[0]),
                                float(contact.position_on_a_in_ws[1]),
                                float(contact.position_on_a_in_ws[2]),
                            ],
                            "position_on_b": [  # ✅
                                float(contact.position_on_b_in_ws[0]),
                                float(contact.position_on_b_in_ws[1]),
                                float(contact.position_on_b_in_ws[2]),
                            ],
                            "normal": [  # ✅
                                float(contact.contact_normal_on_b_in_ws[0]),
                                float(contact.contact_normal_on_b_in_ws[1]),
                                float(contact.contact_normal_on_b_in_ws[2]),
                            ],
                            "distance": float(contact.contact_distance),  # ✅
                            "normal_force": float(contact.normal_force),  # ✅
                            "friction_force1": float(
                                contact.linear_friction_force1
                            ),  # ✅
                            "friction_force2": float(
                                contact.linear_friction_force2
                            ),  # ✅
                            "is_active": bool(contact.is_active),  # ✅
                            "link_id_a": int(contact.link_id_a),  # ✅
                            "link_id_b": int(contact.link_id_b),  # ✅
                        }
                        collision_contacts.append(contact_data)
                    else:
                        continue

                # 🔥 FILTER OUT FLOOR / CEILING CONTACTS
                filtered_contacts, obstacle_dir = self.process_collision_contacts(
                    collision_contacts
                )

                print(
                    f"💥 PHYSICS: {len(collision_contacts)} raw → "
                    f"{len(filtered_contacts)} wall | "
                    f"side={obstacle_dir['side']} depth={obstacle_dir['depth']}"
                )

                print(
                    f"💥 PHYSICS: {len(contacts)} contacts - {self.env.get_physics_step_collision_summary()}"
                )

                physics_collided = len(filtered_contacts) > 0
                collided = physics_collided

                print(f"🎯 FINAL COLLISION = {collided} , physics={physics_collided})")

                # 🔥 HABITATSTORE METRICS (EXACTLY AS BEFORE)

                if collided:
                    global_habitat_store._metrics_collisions += 1
                    print(
                        f"💥 VELOCITY COLLISION: Total collisions = "
                        f"{global_habitat_store._metrics_collisions}"
                    )

                    global_habitat_store._collision_history.append(
                        {
                            "timestamp": time.time(),
                            "action_id": action_item["action_id"],
                            "contacts": filtered_contacts,  # 🔥 ONLY WALL CONTACTS
                            "num_contacts": len(filtered_contacts),
                        }
                    )

                robot_pos = np.array(
                    [robot.translation.x, robot.translation.y, robot.translation.z]
                )
                robot_rot = robot.rotation
                
                # 🔥 PRE-COLLISION PROXIMITY (ZERO-SHOT SAFE)
                obstacle_distance = self.env.pathfinder.distance_to_closest_obstacle(
                    robot_pos
                )
                print(f"📏 Obstacle clearance from the robot before collision= {obstacle_distance:.3f} m")

                print(f"🤖 ROBOT POST-PHYSICS pos = {robot_pos.tolist()}")
                # ✅ CORRECT
                print(
                    f"🤖 ROBOT POST-PHYSICS rot = {[robot_rot.vector.x, robot_rot.vector.y, robot_rot.vector.z, robot_rot.scalar]}"
                )

                # 🔥 METRICS (AUTHORITATIVE)
                global_habitat_store._metrics_trajectory.append(robot_pos.copy())
                print(
                    f"📍 METRICS: Trajectory len = {len(global_habitat_store._metrics_trajectory)}"
                )

                print("✅ Velocity motion applied (projected)")

                observations = self.env.get_sensor_observations()
                # Light‑weight observation debug
                if "rgba_camera" in observations:
                    rgb = observations["rgba_camera"]
                    print(f"👁 rgba_camera shape: {getattr(rgb, 'shape', None)}")
                if "depth_camera" in observations:
                    depth = observations["depth_camera"]
                    print(f"🌊 depth_camera shape: {getattr(depth, 'shape', None)}")

                # 2nd call - SAME SAFETY:
                quat_list2 = [
                    robot_rot.vector.x,
                    robot_rot.vector.y,
                    robot_rot.vector.z,
                    robot_rot.scalar,
                ]
                flat_q = self.flatten_quaternion_to_yaw(quat_list2)  # ✅ List input!

                # 🔥 EXACT SAME RETURN TYPE AS BEFORE - NO CHANGES
                result = {
                    "action_id": action_item["action_id"],
                    "status": "completed",
                    "action": "velocity_control",
                    "obstacle_direction": obstacle_dir,
                    "obstacle_distance": float(obstacle_distance),  # ✅ ADD THIS
                    "observations": observations,
                    "collided": physics_collided,
                    "new_position": robot_pos.tolist(),
                    "new_rotation": flat_q,  # ✅ FLATTENED
                    "timestamp": time.time(),
                    "collision_data": {
                        "num_contacts": len(contacts),
                    },
                    "raw_contacts": collision_contacts[:10],  # ✅ CORRECT ATTRIBUTES!
                    "filtered_contacts": filtered_contacts,
                }

                print(
                    f"📤 Result (velocity_control): status={result['status']}, "
                    f"collided={result['collided']}, new_position={result['new_position']}"
                )
                print("=" * 80)
                return result

            # ==========================================================
            # 🧱 DISCRETE HABITAT ACTIONS (UNCHANGED)
            # ==========================================================
            else:
                with global_habitat_store:
                    observations = self.env.step(action_type)

                    # ✅ ADD THESE 7 LINES HERE:
                    print("🔄 METRICS: Discrete action - tracking position")  # DEBUG 1
                    agent = self.env.get_agent(0)
                    state = agent.get_state()
                    print(
                        f"📍 METRICS: Post-discrete pos = {state.position}"
                    )  # DEBUG 2
                    global_habitat_store._metrics_trajectory.append(
                        state.position.copy()
                    )
                    print(
                        f"📍 METRICS: Trajectory len = {len(global_habitat_store._metrics_trajectory)}"
                    )  # DEBUG 3
                    collided = observations.get("collided", False)
                    print(f"💥 METRICS: Collision detected = {collided}")  # DEBUG 4
                    if collided:
                        global_habitat_store._metrics_collisions += 1
                        print(
                            f"💥 METRICS: Total collisions = {global_habitat_store._metrics_collisions}"
                        )  # DEBUG 5

                new_pos = self.env.get_agent(0).get_state().position

                # Debug discrete step
                print(f"🧱 Discrete action executed: {action_type}")
                print(f"📍 New agent position (discrete): {new_pos}")
                if isinstance(new_pos, np.ndarray):
                    new_pos_list = new_pos.tolist()
                else:
                    try:
                        new_pos_list = list(new_pos)
                    except TypeError:
                        new_pos_list = [float(new_pos)]

                if "rgba_camera" in observations:
                    rgb = observations["rgba_camera"]
                    print(
                        f"👁 rgba_camera shape (discrete): {getattr(rgb, 'shape', None)}"
                    )
                if "depth_camera" in observations:
                    depth = observations["depth_camera"]
                    print(
                        f"🌊 depth_camera shape (discrete): {getattr(depth, 'shape', None)}"
                    )

                result = {
                    "action_id": action_item["action_id"],
                    "status": "completed",
                    "action": action_type,
                    "observations": observations,
                    "collided": observations.get("collided", False),
                    "new_position": new_pos_list,
                    "new_rotation": state.rotation.tolist(),
                    "timestamp": time.time(),
                }

                print(
                    f"📤 Result (discrete): status={result['status']}, "
                    f"collided={result['collided']}, new_position={result['new_position']},"
                )
                print("=" * 80)
                print(
                    f"📤 Result (discrete control): Observations from _execute_habitat_action : discrete action control: {result}"
                )

                return result

        except Exception as e:
            print(f"❌ Habitat action execution failed: {e}")
            import traceback

            traceback.print_exc()

            return {
                "action_id": action_item.get("action_id", "unknown"),
                "status": "error",
                "error": str(e),
                "timestamp": time.time(),
            }

    def _yaw_from_quat(self, q):
        # 🔥 HANDLE Magnum Quaternion EXPLICITLY
        if hasattr(q, "vector") and hasattr(q, "scalar"):
            x = float(q.vector.x)
            y = float(q.vector.y)
            z = float(q.vector.z)
            w = float(q.scalar)
        else:
            q = list(q)
            x, y, z, w = q[:4]

        return math.atan2(
            2.0 * (w * y + x * z),
            1.0 - 2.0 * (y * y + x * x),  # ✅ FIXED LINE
        )

    def flatten_quaternion_to_yaw(self, q):
        yaw = self._yaw_from_quat(q)  # Now safe!
        return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]  # Return list!

    def _print_final_summary(
        self,
        frames_written: int,  # ✅ Actually: frames successfully written to buffer
        frames_failed: int,  # ✅ Frames that failed
        start_time: float,
        next_frame_id: int = None,  # ✅ Optional: next frame ID (for info)
    ):
        """Print final orchestration summary"""
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n{'='*50}")
        print(f"🏁 ORCHESTRATION COMPLETED")
        print(f"{'='*50}")
        print(f"🎯 Target frames: {self.config.get('num_frames', 10)}")
        print(f"✅ Frames written to buffer: {frames_written}")  # ✅ FIXED LABEL
        print(f"❌ Frames failed: {frames_failed}")
        print(f"📊 Total attempted: {frames_written + frames_failed}")

        if next_frame_id is not None:
            print(f"🎯 Next frame ID (if continued): {next_frame_id}")

        print(f"⏱️  Total time: {total_time:.2f} seconds")
        if total_time > 0:
            print(f"📈 Average FPS: {frames_written/total_time:.2f}")

        # Buffer statistics
        buffer_stats = global_frame_buffer.get_buffer_stats()
        print(f"\n🔧 Buffer Statistics:")
        print(f"   Utilization: {buffer_stats['utilization_percent']:.1f}%")
        print(f"   Health Score: {buffer_stats['health_score']}/100")

        # Fix: Check if 'active_readers' exists before accessing
        if "active_readers" in buffer_stats:
            print(f"   Active Readers: {buffer_stats['active_readers']}")
        else:
            print(f"   Active Readers: Not available")

        mission_status = (
            "COMPLETED" if self.task_store.is_mission_complete() else "INCOMPLETE"
        )
        print(f"\n🎯 Mission status: {mission_status}")
        print(f"{'='*50}")

    def _manage_vram_usage(self, frame_id: int):
        """Strategic VRAM management based on pipeline priorities"""

        # Check VRAM every 30 frames
        if frame_id % 30 == 0:
            try:
                if torch.cuda.is_available():
                    vram_used = torch.cuda.memory_allocated() / 1024**3  # GB
                    vram_total = (
                        torch.cuda.get_device_properties(0).total_memory / 1024**3
                    )
                    usage_ratio = vram_used / vram_total

                    print(
                        f"📊 VRAM: {vram_used:.1f}GB/{vram_total:.1f}GB ({usage_ratio:.1%})"
                    )

                    # 🎯 STRATEGIC UNLOADING - ALL 4 PIPELINES
                    if usage_ratio > 0.85:  # Critical - unload in priority order
                        print("⚠️ CRITICAL VRAM - Strategic unloading...")

                        # Unload in REVERSE priority order (least critical first)
                        if self.action_pipeline:
                            self.action_pipeline.shutdown()  # 🦾 4th priority

                        if self.prediction_pipeline:
                            self.prediction_pipeline.shutdown()  # 🔮 3rd priority

                        if self.reasoning_pipeline:
                            self.reasoning_pipeline.shutdown()  # 🧠 2nd priority

                        # 🎥 Perception is LAST RESORT - keep if possible
                        if usage_ratio > 0.9 and self.perception_pipeline:
                            self.perception_pipeline.shutdown()  # 🎥 1st priority

                        torch.cuda.empty_cache()

                    elif usage_ratio > 0.75:  # High - prepare for unloading
                        print("🔶 High VRAM - Monitoring closely")
                        # Could unload perception models if they have shutdown
                        if self.perception_pipeline and hasattr(
                            self.perception_pipeline, "shutdown"
                        ):
                            self.perception_pipeline.shutdown()

            except Exception as e:
                print(f"⚠️ VRAM management error: {e}")

    def shutdown_orchestration(self):
        """Gracefully shutdown all pipelines using threading event"""
        print("🛑 Shutting down main orchestration...")

        self.state = OrchestrationState.SHUTDOWN

        # ✅ Shutdown ALL 4 pipelines in reverse dependency order
        if self.action_pipeline:
            self.action_pipeline.shutdown()  # 🦾 4th
        if self.prediction_pipeline:
            self.prediction_pipeline.shutdown()  # 🔮 3rd
        if self.reasoning_pipeline:
            self.reasoning_pipeline.shutdown()  # 🧠 2nd
        if self.perception_pipeline:
            self.perception_pipeline.shutdown()  # 🎥 1st

        # Close Habitat
        if self.env:
            self.env.close()
 

        # ✅ Cleanup BOTH global stores (ADD THIS LINE)
        global_habitat_store.shutdown()
        global_frame_buffer.shutdown()

        if hasattr(self, "map_store") and self.map_store:
            final_map_path = self.map_store.shutdown_and_save()
            print(f"🗺️ FINAL MAP SAVED: {final_map_path}")

        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

    def get_orchestration_status(self) -> Dict:
        """Get current orchestration status"""
        buffer_stats = global_frame_buffer.get_buffer_stats()

        return {
            "state": self.state.value,
            "mission_complete": self.task_store.is_mission_complete(),
            "current_frame": global_frame_buffer.get_latest_frame_id(),
            "pipelines_ready": {
                "perception": self.perception_pipeline is not None,
                "reasoning": self.reasoning_pipeline is not None,
                "prediction": self.prediction_pipeline is not None,
                "action": self.action_pipeline is not None,
            },
            "buffer_stats": buffer_stats,
        }

    def process_collision_contacts(self, raw_contacts, y_threshold=0.9):
        """
        1. Filters out floor / ceiling contacts
        2. Determines obstacle direction relative to robot using X+Z
        Returns:
            wall_contacts: filtered list
            obstacle_dir: dict with left/right + front/back
        """

        wall_contacts = []

        # --- Robot basis in WORLD frame ---
        # Habitat convention: -Z is forward
        robot_fwd = self.robot_id.rotation.transform_vector(mn.Vector3(0, 0, -1))
        robot_right = self.robot_id.rotation.transform_vector(mn.Vector3(1, 0, 0))

        robot_fwd = np.array([robot_fwd.x, robot_fwd.y, robot_fwd.z])
        robot_right = np.array([robot_right.x, robot_right.y, robot_right.z])

        for contact in raw_contacts:
            normal = np.array(contact.get("normal", [0.0, 0.0, 0.0]))

            # Must involve robot
            if (
                contact.get("object_id_a") != self.robot_id.object_id
                and contact.get("object_id_b") != self.robot_id.object_id
            ):
                continue

            # Remove floor / ceiling
            if abs(normal[1]) > y_threshold:
                continue

            wall_contacts.append(contact)

        obstacle_dir = {"side": "NONE", "depth": "NONE"}  # LEFT / RIGHT  # FRONT / BACK

        if wall_contacts:
            # --- Average world-space normal (wall → robot) ---
            avg_normal = np.mean([c["normal"] for c in wall_contacts], axis=0)
            avg_normal[1] = 0.0  # flatten
            mag = np.linalg.norm(avg_normal)

            if mag > 1e-3:
                n = avg_normal / mag

                # --- Side: LEFT / RIGHT ---
                side_dot = np.dot(n, robot_right)
                obstacle_dir["side"] = "LEFT" if side_dot > 0 else "RIGHT"

                # --- Depth: FRONT / BACK ---
                depth_dot = np.dot(n, robot_fwd)
                obstacle_dir["depth"] = "FRONT" if depth_dot > 0 else "BACK"

        return wall_contacts, obstacle_dir

    def _load_config(self, config: Dict = None) -> Dict:
        """Load configuration from YAML file with override support"""
        # Load base config from YAML file
        config_path = "projects/hybrid_zero_shot_slam_nav/main/configs/simulation/habitat_env_cfg.yaml"
        with open(config_path, "r") as f:
            base_config = yaml.safe_load(f)

        # Add frame schedule defaults if not in YAML
        if "frame_schedule" not in base_config:
            base_config["frame_schedule"] = {
                "slam_every": 1,
                "detection_every": 3,
                "clip_every": 10,
            }

        # Merge with provided config (if any)
        if config:
            base_config.update(config)

        return base_config


def main():
    """Main entry point for the orchestrator"""
    parser = argparse.ArgumentParser(
        description="Main Orchestrator - Coordinates all pipelines"
    )
    parser.add_argument(
        "--scene", type=str, default="gibson_00023", help="Gibson scene to load"
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="Number of frames to process"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="explore and map environment",
        help="Task description",
    )

    args = parser.parse_args()

    # Create orchestrator configuration - NOW MINIMAL
    config = {
        "num_frames": args.frames,
        "target_objects": ["chair", "table", "door", "sofa", "bed", "window"],
        "strategic_loading": True,
    }

    orchestrator = OursFullPolicy(config)

    try:
        print(f"🚀 Starting Main Orchestrator for {args.scene}")
        print(f"   Task: {args.task}")

        orchestrator.run_orchestration(task_description=args.task)

    except Exception as e:
        print(f"❌ Orchestration failed: {e}")

    # ✅ FULL DEBUG VERSION - Copy-Paste Ready!


def run_ours_full(episode: dict):  # ← NO 'self' !
    """WRAPPER for runner - returns straight_turn metrics format"""
    print(f"🚀 OursFull: {episode['episode_id']}")
    print(f"📋 DEBUG: Episode keys = {list(episode.keys())}")

    start_time = time.time()

    # 🆕 CREATE ORGANIZED DIRECTORY (SAME AS STRAIGHT-TURN)
    import os
    from pathlib import Path

    # Get the same base directory as straight_turn_policy
    script_dir = Path(__file__).resolve().parents[1]  # Main_Experiments
    BASE_RESULTS_DIR = script_dir.parent / "results"  # Main_Experiments/results/

    policy_name = "ours_full"
    frame_dir = BASE_RESULTS_DIR / policy_name / f"{episode['episode_id']}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Frame save directory: {frame_dir}")

    # Create config with episode data
    config = {
        "num_frames": episode.get("max_steps", 500),
        "scene_path": episode["scene_path"],  # Pass scene!
        "frame_save_dir": str(frame_dir),  # 🆕 ADD THIS LINE!
    }
    print(f"⚙️  DEBUG: Config created = {config}")

    # Create orchestrator
    print("🏭 DEBUG: Creating OursFullPolicy...")
    orchestrator = OursFullPolicy(config)
    print(f"✅ DEBUG: Orchestrator created = {orchestrator}")

    # Set episode start pose (CRITICAL!)
    print("🎯 DEBUG: Setting episode start pose...")
    agent = orchestrator.env.get_agent(0)
    print(f"👤 DEBUG: Agent 0 acquired")
    state = agent.get_state()
    print(f"📍 DEBUG: Current state = pos:{state.position}, rot:{state.rotation}")

    state.position = np.array(episode["start_pose"]["position"])
    print(f"📍 DEBUG: Set position = {state.position}")

    yaw = episode["start_pose"]["yaw"]
    print(f"🔄 DEBUG: Yaw = {yaw}")

    state.rotation = np.array([0, np.sin(yaw / 2), 0, np.cos(yaw / 2)])
    print(f"🔄 DEBUG: New rotation quat = {state.rotation}")

    agent.set_state(state)
    print("✅ DEBUG: Start pose applied!")

    json_position = np.array(episode["start_pose"]["position"])
    robot = orchestrator.robot_id
    # Set position - MUST be mn.Vector3!
    robot.translation = mn.Vector3(
        float(json_position[0]), float(json_position[1]), float(json_position[2])
    )

    # 🔥 Populate HabitatStore initial_pose for navigation
    global_habitat_store.initial_pose = {
        "position": [
            float(robot.translation.x),
            float(robot.translation.y),
            float(robot.translation.z),
        ],
        "rotation": state.rotation.tolist(),
    }
    print(
        f"💾 DEBUG: initial_pose set in HabitatStore: {global_habitat_store.initial_pose}"
    )

    # Find goal position (copy from straight_turn)
    print("🔍 DEBUG: Searching for goal object...")
    goal_pos = None
    goal_category = episode["goal"]["object_category"]
    print(f"🎯 DEBUG: Target category = '{goal_category}'")

    semantic_objects = list(orchestrator.env.semantic_scene.objects)
    print(f"📊 DEBUG: Found {len(semantic_objects)} semantic objects")

    for i, obj in enumerate(semantic_objects):
        try:
            cat_name = obj.category.name()
            print(f"  🔍 Obj {i}: '{cat_name}'")
            if cat_name == goal_category:
                goal_pos = obj.aabb.center()
                goal_pos = [goal_pos.x, goal_pos.y, goal_pos.z]
                print(f"✅ DEBUG: GOAL FOUND! pos = {goal_pos}")

                # 🆕 ADD THESE 3 LINES HERE:
                global_habitat_store.current_goal_position = goal_pos
                global_habitat_store.current_goal_category = goal_category
                print(
                    f"💾 DEBUG: Goal stored to HabitatStore: {goal_category} at {goal_pos}"
                )

                break
        except Exception as e:
            print(f"  ⚠️  Obj {i}: ERROR {e}")
            continue

    if goal_pos is None:
        goal_pos = [2.0, 0.05, 1.5]
        print(f"❌ DEBUG: Goal NOT found → Using fallback = {goal_pos}")

    print(f"🎯 DEBUG: Final goal_pos = {goal_pos}")

    # Run orchestration
    print("🚀 DEBUG: Starting orchestration...")
    task_desc = f"find {episode['goal']['object_category']}"
    orchestrator.run_orchestration(task_desc)
    print("✅ DEBUG: Orchestration completed!")

    # Extract metrics
    trajectory = global_habitat_store._metrics_trajectory
    print(f"📈 DEBUG: Trajectory extracted, len = {len(trajectory)}")

    collisions = global_habitat_store._metrics_collisions
    print(f"💥 DEBUG: Collisions extracted = {collisions}")

    # Compute metrics (copy straight_turn functions HERE)
    print("📊 DEBUG: Computing final metrics...")

    final_pos = np.array(trajectory[-1])
    print(f"🏁 DEBUG: Final pos = {final_pos}")

    final_dist = np.linalg.norm(final_pos - np.array(goal_pos))
    print(f"📏 DEBUG: Final distance = {final_dist:.3f}m")

    success = int(final_dist < 0.2)
    print(f"✅ DEBUG: Success = {success} (threshold 0.2m)")

    # 🆕 2D XZ PATH LENGTH ONLY (Habitat standard for nav metrics)
    # 🛤️ FIXED 2D XZ PATH LENGTH (Habitat standard)
    if len(trajectory) > 1:
        traj = np.array(trajectory)

        # 🔥 FILTER STATIONARY/JITTER FRAMES (threshold 1mm)
        filtered = [traj[0]]
        for p in traj[1:]:
            if np.linalg.norm(p - filtered[-1]) > 1e-3:  # 1mm move
                filtered.append(p)

        traj_filtered = np.array(filtered)
        print(f"📊 Filtered {len(trajectory)} → {len(filtered)} points")

        # 2D XZ path length
        traj_2d = traj_filtered[:, [0, 2]]  # X,Z only
        path_length = np.sum(np.linalg.norm(np.diff(traj_2d, axis=0), axis=1))
        print(f"🛤️ 2D XZ Path length = {path_length:.3f}m ({len(filtered)} points)")
    else:
        path_length = 0.0

    # 🆕 FIX: Compute SPL correctly
    if success and len(trajectory) > 1:
        straight_dist = np.linalg.norm(np.array(trajectory[0]) - np.array(goal_pos))
        spl = straight_dist / max(path_length, straight_dist)
        print(f"📐 DEBUG: SPL = {spl:.3f} (straight={straight_dist:.3f}m)")
    else:
        spl = 0.0
        print("📐 DEBUG: SPL = 0.0 (no success or short trajectory)")

    failure_mode = (
        "collision" if collisions > 5 else "distance" if not success else "none"
    )
    print(f"🔥 DEBUG: Failure mode = '{failure_mode}'")

    metrics = {
        "success": success,
        "spl": spl,
        "path_length": path_length,
        "collisions": collisions,
        "final_distance": final_dist,
        "failure_mode": failure_mode,
        "trajectory": trajectory,
    }
    print(f"📋 DEBUG: Final metrics = {metrics}")
    
    
    # 🎨 VISUALIZE TRAJECTORY - ADD THIS LINE
    # 🎨 VISUALIZE TRAJECTORY - ADD DEBUGGING FIRST
    print(f"🔍 DEBUG: About to plot trajectory...")
    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Start pos: {trajectory[0]}")
    print(f"   Final pos: {final_pos}")
    print(f"   Goal pos: {goal_pos}")
    print(f"   Frame dir: {frame_dir}")
    print(f"   Final distance: {final_dist:.3f}m")

    # Check if frame_dir exists
    if not os.path.exists(frame_dir):
        print(f"❌ ERROR: frame_dir doesn't exist: {frame_dir}")
    else:
        print(f"✅ frame_dir exists: {frame_dir}")

    try:
        plot_path, topdown_done = plot_trajectory(
            trajectory, trajectory[0], final_pos, goal_pos, frame_dir, final_dist, sim=orchestrator.env 
        )
        print("✅ plot_trajectory() call completed")
    except Exception as e:
        print(f"❌ ERROR in plot_trajectory: {e}")
        import traceback

        traceback.print_exc()

 

    # Only shutdown after both are saved
    if plot_path and topdown_done:
        print("✅ Both files saved, shutting down...")
        orchestrator.shutdown_orchestration()

    print("✅ DEBUG: Shutdown complete!")

    print("🏆 DEBUG: Returning metrics to runner!")
    return metrics


# 📊 VISUALIZATION FUNCTION - ADD RIGHT BEFORE FINAL RETURN

def plot_trajectory(trajectory, start_pos, final_pos, goal_pos, save_dir, final_dist, sim=None):
    """
    Saves two files: 
    1. trajectory_plot.png (The original blue-line XZ plot)
    2. topdown_navmesh.png (The raw Habitat environment map)
    """
    print(f"🎨 Generating visualization results in: {save_dir}")
    
    import matplotlib.pyplot as plt
    import os
    import numpy as np
    import imageio

    # ==========================================================
    # 1. SAVE ORIGINAL TRAJECTORY PLOT (XZ Projection)
    # ==========================================================
    x_vals = [pos[0] for pos in trajectory]
    z_vals = [pos[2] for pos in trajectory]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x_vals, z_vals, "b-", linewidth=2, alpha=0.7, label="Robot Path")
    
    # Markers
    ax.plot(start_pos[0], start_pos[2], "go", markersize=12, markeredgecolor="black", label="Start")
    ax.plot(final_pos[0], final_pos[2], "ro", markersize=12, markeredgecolor="black", label="Final")
    ax.plot(goal_pos[0], goal_pos[2], "b*", markersize=15, markeredgecolor="black", label="Goal")

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.set_title(f"Robot Trajectory - Final Distance: {final_dist:.2f}m")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis("equal")

    plot_path = os.path.join(save_dir, "trajectory_plot.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📈 Trajectory plot saved: {plot_path}")

    # ==========================================================
    # 2. SAVE TOP-DOWN NAVMESH MAP (With Agent & Path)
    # ==========================================================
    if sim is not None and hasattr(sim, 'pathfinder') and sim.pathfinder.is_loaded:
        try:
            from habitat.utils.visualizations import maps
            
            height = start_pos[1]
            meters_per_pixel_map = 0.05

            # 1. Get the base map
            top_down_map = maps.get_topdown_map(sim.pathfinder, height, meters_per_pixel=meters_per_pixel_map)
            grid_dimensions = top_down_map.shape

            # 2. Convert world coordinates to grid coordinates
            start_grid = maps.to_grid(start_pos[2], start_pos[0], grid_dimensions, pathfinder=sim.pathfinder)
            goal_grid = maps.to_grid(goal_pos[2], goal_pos[0], grid_dimensions, pathfinder=sim.pathfinder)

            # 3. Recolor to RGB
            recolor_map = np.array([[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8)
            top_down_map_rgb = recolor_map[top_down_map]

            # 4. DRAW on the map (Optional but helpful)
            # You can draw a point for the goal
            cv2.circle(top_down_map_rgb, (goal_grid[1], goal_grid[0]), 5, (255, 0, 0), -1) # Blue Goal
            cv2.circle(top_down_map_rgb, (start_grid[1], start_grid[0]), 5, (0, 255, 0), -1) # Green Start

            # Save
            topdown_map_file = os.path.join(save_dir, "topdown_navmesh_with_markers.png")
            imageio.imsave(topdown_map_file, top_down_map_rgb)
            topdown_success = True
            
        except Exception as e:
            print(f"⚠️ Top-down map generation failed: {e}")
             

    return plot_path, topdown_success

if __name__ == "__main__":
    main()
