
# habitat - x(right), y(up), z(forward)



# # DIRECT ON Simulator (self.env):
# agent = self.env.get_agent(0)                    # ✅
# self.env.pathfinder.is_navigable(new_pos)        # ✅
# observations = self.env.get_sensor_observations() # ✅
# self.env.step(action_type)                       # ✅
# # TaskStore: "Find chair in kitchen"


#     ↓
# ActionExecutor._read_planned_actions()  # ✅ Reads high-level goal
#     ↓
# ActionExecutor.execute_single_action()  # ✅ Your intelligence layer
#     ↓ (YOUR RESEARCH HAPPENS HERE)
#     1. Semantic goal checking
#     2. Depth-based collision prediction
#     3. I-JEPA analysis for recovery
#     4. Decision to execute or recover
#     ↓
# ActionExecutor pushes to HabitatStore
#     ↓
# MainOrchestrator._execute_habitat_action()  # ✅ Simple Habitat call
#     ↓
# env.step("move_forward")  # ← Habitat's low-level execution
#     ↓
# Returns {"collided": True/False}
#     ↓
# ActionExecutor analyzes result  # ✅ Your intelligence layer
#     ↓
# Updates MapStore, metrics, progress  # ✅ Your tracking


# remove hfhub offline flag - unset
# increase frame limit for owl


# Stores are in-memory Python objects.
# When you run:

# python main.py
# remove all id geenration fallbaks in all files

# every time, all store objects reset.- is there a way to avoid this since we need to build  a complete orbslam map whic wont lose data for seprate navigations


# All pipelines use same stores, just coordinate timing
# implement a global context store
# make every pipeline write to map store


# Real-Time Localization: Determine the robot's precise pose (2$\mathbf{x}, \mathbf{y}, \mathbf{z}$, roll, pitch, yaw) within the map it is building, updating this position multiple times per second.3 This is essential for path planning and collision avoidance.4Map Construction: Generate a consistent representation of the environment, often as an occupancy grid map (for 2D LiDAR) or a 3D point cloud (for cameras/3D LiDAR).5Consistency/Drift Correction: Use Loop Closure to recognize when the robot returns to a previously visited location.6 This triggers a global optimization (like Pose Graph Optimization) that corrects the accumulated error (drift) across the entire map and trajectory, ensuring global consistency.Application Mapping: The raw SLAM output is converted into a navigable map for the specific application. For a warehouse robot, this might mean annotating the map with restricted zones, pick-up points, and charging stations.7


# 2. For OWL, do we need transformed poses, map points, or both?
# ✅ OWL needs BOTH — but for different reasons
# Let’s separate their roles:
# A. Transformed Camera Pose is Mandatory
# OWL cannot place objects in the world without
# A correct world-frame camera pose
# This defines the ray origin for all 2D → 3D projections
# Without transformed pose:
# Every object is placed in a wrong global position
# Even perfect depth becomes useless globally
# ✅ Camera pose is absolutely required.


# B. Transformed Map Points are Conditionally Required
# Map points are needed when:
# You use geometric association
# You fuse object detections with existing structure
# You want semantic labels on SLAM points
# You perform object-level loop closures
# If your program only:
# Shoots a ray using depth
# Drops a semantic object into the worl
# Then map points are not strictly required for OWL itself.
# However:
# ✅ For a semantic SLAMsystem, both pose and map points must be transformed, otherwise:
# Objects and geometry diverge
# Data association becomes unstable
# Long-term consistency is impossible

# ✅ Correct Dependency Rule
# Use Case	Needs Transformed Pose	Needs Transformed Map Points
# Simple 3D object placement	✅ Yes	❌ No
# Semantic SLAM	✅ Yes	✅ Yes
# Planning / navigation	✅ Yes	✅ Yes
# Object-map fusion	✅ Yes	✅

# Enable exactly one consistent projection:
# ORB depth back-projection

# remove fallback intrinsincs

# ➡️ teleports the agent by a fixed delta along its forward axis
# and then runs collision correction if it intersects geometry.
# Internally this is:
# a kinematic SE(2)/SE(3) transform
# not a physics simulation
# So:

# ✅ This is NOT a robot simulation
# ✅ It IS a first-person camera controller in a 3D scene

# Your framing is exactly right.


# ✅ You only need:
# stable egomotion
# smooth SE(3) trajectory
# bounded velocity
# For:
# frontier exploration
# semantic navigation
# object-goal nav
# A kinematic camera agent is perfectly sufficient. tell me are we implemnting this or not ?
# No angular velocity
# No smooth rotation

# ✅ Continuous KINEMATIC MOTION INTEGRATION

# Instead of teleport steps, you integrate pose yourself:

# v = 0.15   # m/s
# w = 0.25   # rad/s
# dt = 0.1


# Then:

# x += v * dt * cos(theta)
# y += v * dt * sin(theta)
# theta += w * dt


# Then directly set the agent pose:

# agent_state = sim.get_agent(0).get_state()
# agent_state.position = new_position
# agent_state.rotation = new_rotation
# sim.get_agent(0).set_state(agent_state)


# This gives you:

# ✅ Smooth velocity
# ✅ Smooth rotation
# ✅ Bounded frame-to-frame motion
# ✅ ORB-SLAM friendly trajectories
# ✅ No force simulation required

# ✅ What You Should Do Next (Recommended Fix)

# Instead of:

# env.step("move_forward")
# env.step("turn_left")
# ✅ Rewrite your current env.step() loop into continuous SE(3) motion

# You should switch to:

# ✅ Velocity-based kinematic integration
# ✅ Direct pose setting
# ✅ High-rate small dt

# This alone will:

# Fix ORB-SLAM tracking loss

# Remove “stuck” behavior

# Keep zero-shot navigation valid

# Keep system fully headless

# Avoid unnecessary physics complexity
# Not always. If the research goal is zero‑shot visual navigation (semantic/goal reasoning, policy learning), accurate low‑level dynamics may be unnecessary. Many navigation pipelines use a kinematic agent (continuous smooth camera motion) rather than a full wheeled robot.

# Example loop:
# Desired velocity v and yaw rate ω from controller.
# Each simulation frame dt: pose += transform(vdt, 0, 0) + rotation(ωdt).
# Query sensors each frame, feed frames to ORB‑SLAM.
# If stuck in collision, reduce step size or let physics resolve penetration by applying small reversing velocity or clearing penetration via a recovery strategy.


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

from experiments.scripts.run_perception_pipeline import PerceptionPipeline
from experiments.scripts.run_prediction_pipeline import PredictionPipeline
from experiments.scripts.run_reasoning_pipeline import ReasoningPipeline
from experiments.scripts.run_action_pipeline import ActionPipeline
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


class MainOrchestrator:
    """
    PURE COORDINATOR - Only manages pipelines, no processing
    """

    def __init__(self, config: Dict = None):
        self.state = OrchestrationState.INITIALIZING
        self.config = config

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
        self._initialize_at_center()

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
        scene_path = "/mnt/d/Coding/Business/Kulfi_Startup_Code/robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/datasets/00809-Qpor2mEya8F/Qpor2mEya8F.basis.glb"

        print(f"🔄 Loading scene from: {scene_path}")

        # ✅ VERIFY FILE EXISTS FIRST
        if not os.path.exists(scene_path):
            print(f"❌ Scene file not found: {scene_path}")
            return None

        # SIMPLE DIRECT APPROACH
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = scene_path  # ✅ Use the variable here
        # FORCE CPU MODE - skip GPU issues
        cfg.gpu_device_id = -1
        cfg.enable_physics = True  # ✅ ADDED: ENABLE PHYSICS
        cfg.force_separate_semantic_scene_graph = True

        # PROPER AGENT CONFIG WITH EXPLICIT SENSOR
        agent_cfg = habitat_sim.AgentConfiguration()

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
        sensor_specs.append(rgb_spec)
        rgb_spec.hfov = 90.0  # 🆕 ADD THIS: Horizontal Field of View (degrees)

        # ✅ ADD DEPTH SENSOR
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_camera"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [640, 480]
        sensor_specs.append(depth_spec)
        depth_spec.hfov = 90.0  # Must match RGB camera!

        agent_cfg.sensor_specifications = sensor_specs
        print(f"Agent config: {agent_cfg}")

        # Check action space
        action_space = agent_cfg.action_space
        print(f"Action space: {list(action_space.keys())}")

        # Create simulator configuration
        sim_config = habitat_sim.Configuration(cfg, [agent_cfg])

        try:
            sim = habitat_sim.Simulator(sim_config)
            print("✅ HM3D Scene Loaded Successfully!")

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

    def _initialize_at_center(self):
        print("🎯 Initializing robot at center with exploration...")

        # 🆕 DEBUG INITIAL POSITION
        initial_state = self.env.get_agent(0).get_state()
        print(f"  🔍 Initial position: {initial_state.position}")
        print(f"  🔍 Initial rotation: {initial_state.rotation}")

        try:
            # Initial exploration movements to position in center
            initial_movements = [
                "move_forward",
                "move_forward",
                "move_forward",
                "turn_left",
                "turn_left",  # 180° turn
                "turn_left",  # 180° turn
                "turn_left",  # <-- ADD THIS ONE: Now 270° turn (faces opposite of wall)
                "move_forward",
                "move_forward",
                "turn_right",
                "turn_right",
                "turn_right",  # 270° turn
                "move_forward",
                "move_forward",
                "turn_left",  # Back to original orientation
                "move_forward",
            ]

            for i, action in enumerate(initial_movements):

                self.env.step(action)
                print(
                    f"  🎯 Initial positioning {i+1}/{len(initial_movements)}: {action}"
                )
                time.sleep(0.1)

            print("✅ Robot positioned at center with exploration pattern")

        except Exception as e:
            print(f"⚠️ Initial positioning failed: {e}")

    def run_orchestration(self, task_description: str = "explore and map environment"):
        print(f"🚀 Starting main orchestration for task: {task_description}")
        print(
            f"checking self.env instance for habitat sim inside run_orchestration function {self.env}"
        )

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

            #  # ✅ CAPTURE INITIAL FRAME FOR ORB-SLAM - ADD THIS AT THE START
            # print("🎯 Capturing initial frame for ORB-SLAM initialization...")
            # initial_observations = self.env.get_sensor_observations()
            # initial_rgb = initial_observations["rgba_camera"]
            # initial_depth = initial_observations["depth_camera"]
            # print("✅ Initial frame captured")

            # # ✅ PROCESS INITIAL FRAME IN ORB-SLAM
            # print("🔄 ORB-SLAM: Processing initial frame... calling orb_slam.process_frame() ")

            #     # Process in ORB-SLAM
            # self.orb_slam.process_frame()

            # # 🎯 SHORTEN WARM-UP TIME
            # print("🔄 ORB-SLAM: Quick warm-up...")
            # time.sleep(0.5)  # REDUCED FROM 2.0 to 0.5 seconds
            # print("✅ ORB-SLAM warm-up complete")

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
            while (
                frame_id < max_frames
                and not self.task_store.is_mission_complete()
                and self.state != OrchestrationState.SHUTDOWN
            ):

                if self.state == OrchestrationState.SHUTDOWN:
                    print("🛑 Shutdown signal received")
                    break

                print(f"\n--- FRAME {frame_id + 1}/{max_frames} ---")

                try:

                    # After initializing env and before starting frame capture
                    if self.env is not None:
                        print("🔄 Stepping simulator 3 times to warm up sensors...")
                        for _ in range(3):
                            print("DEBUG: about to step simulator")
                            self.env.step("move_forward")

                        print("✅ Simulator warm-up complete, first frames ready")

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

                            cv2.imwrite(
                                f"depth_normalized_{frame_id}.png", depth_normalized
                            )
                            print(f"💾 Saved normalized depth frame for visual check")

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

                    rgb_frame_display = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
                    print("DEBUG: about to write rgb frame")

                    cv2.imwrite(f"frame_before_{frame_id}.png", rgb_frame_display)

                    print("DEBUG: about to scale depth")

                    # Scale depth to 0-255 and convert to uint8 for saving
                    # depth_display = (np.clip(depth_frame, 0, 10) / 10 * 255).astype(
                    #     np.uint8
                    # )
                    # cv2.imwrite(f"depth_before_{frame_id}.png", depth_display)

                    # # ✅ FIX DEPTH FRAME SHAPE - Transpose from (640, 480) to (480, 640)
                    # if depth_frame.shape == (640, 480):
                    #     depth_frame = np.transpose(
                    #         depth_frame
                    #     )  # This converts (640, 480) → (480, 640)
                    #     print(f"🔄 Transposed depth frame shape: {depth_frame.shape}")

                    # # ✅ FIX DEPTH FRAME DATA TYPE - Convert float32 to uint16
                    # if depth_frame.dtype == np.float32:
                    #     # Habitat depth is typically in meters (0-10m range)
                    #     # ORB-SLAM expects depth in millimeters (0-10000mm)
                    #     # Scale and convert to uint16
                    #     depth_frame = (depth_frame * 1000).astype(
                    #         np.uint16
                    #     )  # Convert meters to millimeters
                    #     print(
                    #         f"🔄 Converted depth dtype: float32 → uint16, range: {depth_frame.min()}-{depth_frame.max()}"
                    #     )
                    # elif depth_frame.dtype != np.uint16:
                    #     # Fallback conversion
                    #     depth_frame = depth_frame.astype(np.uint16)
                    #     print(f"🔄 Converted depth dtype: {depth_frame.dtype} → uint16")

                    # # ✅ CONVERT RGBA → RGB and fix dimensions for RGB
                    # try:
                    #     # Remove alpha channel (RGBA → RGB)
                    #     if rgb_frame.shape[2] == 4:  # RGBA format
                    #         rgb_frame = rgb_frame[:, :, :3]  # Keep only RGB channels

                    #     # Convert from (640, 480, 3) to (480, 640, 3) if needed
                    #     if rgb_frame.shape == (640, 480, 3):
                    #         rgb_frame = np.transpose(
                    #             rgb_frame, (1, 0, 2)
                    #         )  # Swap width/height

                    #     print(f"🔄 Converted frame shape: {rgb_frame.shape}")

                    # except Exception as e:
                    #     print(f"❌ Frame conversion failed: {e}")
                    #     failed_frames += 1
                    #     frame_id += 1
                    #     continue

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
                        
                        
                        
                                                # 🔥 ADD THESE 4 LINES - EXACTLY HERE (FIRST ACTION ONLY)
                        if not hasattr(global_habitat_store, 'initial_pose'):
                            agent = self.env.get_agent(0)
                            initial_state = agent.get_state()
                            global_habitat_store.initial_pose = {
                                'position': initial_state.position.tolist(),
                                'rotation': initial_state.rotation.tolist()
                            }
                            print(f"✅ INITIAL POSE STORED: {global_habitat_store.initial_pose['position']}")
                            print(f"✅ INITIAL Rotation STORED: {global_habitat_store.initial_pose['rotation']}")


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
                            cv2.imwrite(f"frame_{frame_id}.png", rgb_frame_display)
                        elif frame_shape[0] == 640 and frame_shape[1] == 480:
                            print("  ✅ Frame orientation: LANDSCAPE (640x480)")
                            cv2.imwrite(
                                f"frame_{frame_id}.png", rgb_frame
                            )  # No rotation needed
                        else:
                            print(f"  ⚠️  Unexpected frame shape: {frame_shape}")
                            cv2.imwrite(f"frame_{frame_id}.png", rgb_frame)

                        print(f"  🖼 Saved frame_{frame_id}.png")

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

                # ──────────────────────────────────────────────────────────────
                # ONE AND ONLY PLACE WE INCREMENT frame_id → GUARANTEED ONCE PER LOOP
                # ──────────────────────────────────────────────────────────────
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
        finally:
            self.shutdown_orchestration()

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
            # 🚀 CONTINUOUS VELOCITY CONTROL
            # ==========================================================
            if action_type == "velocity_control" and metadata.get(
                "continuous_nav", False
            ):

                agent = self.env.get_agent(0)
                state = agent.get_state()

                print(f"🧩 Current agent position: {state.position}")
                print(f"🧩 Current agent rotation (quat): {state.rotation}")

                linear_vel = parameters.get("linear_velocity", [0.0, 0.0, 0.0])
                angular_vel = parameters.get("angular_velocity", [0.0, 0.0, 0.0])
                duration = parameters.get("duration", 0.1)

                print(
                    f"🧮 Linear velocity: {linear_vel}, angular velocity: {angular_vel}, dt={duration}"
                )

                vc = habitat_sim.physics.VelocityControl()
                vc.controlling_lin_vel = True
                vc.controlling_ang_vel = True
                vc.lin_vel_is_local = True
                vc.ang_vel_is_local = True
                vc.linear_velocity = mn.Vector3(*linear_vel)
                vc.angular_velocity = mn.Vector3(*angular_vel)

                q = state.rotation
                rigid_state = habitat_sim.RigidState(
                    mn.Quaternion(mn.Vector3(q.x, q.y, q.z), q.w),
                    mn.Vector3(state.position),
                )

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

                # ======================================================
                # 🔒 PHYSICS GATE: PROJECT ONTO NAVMESH
                # ======================================================
                
                
                
                pathfinder = self.env.pathfinder

                xz_pos = np.array([new_pos[0], 0.0, new_pos[2]])

                is_nav = pathfinder.is_navigable(xz_pos)

                print(f"🧭 is_navigable(new_pos) = {is_nav}")

                # if is_nav:
                #     safe_pos = new_pos
                #     print(
                #         f"✅ Using new_pos as safe_pos (on navmesh): {safe_pos.tolist()}"
                #     )
                # else:
                #     snapped  = pathfinder.snap_point(xz_pos)
                #     safe_pos = np.array([snapped[0], new_pos[1], snapped[2]])

                    
                    
                #     print(
                #         f"⚠️ new_pos off-navmesh, snap_point → {None if safe_pos is None else safe_pos}"
                #     )

                #     if safe_pos is None or not pathfinder.is_navigable(safe_pos):
                #         print("⛔ Velocity motion blocked by navmesh after snap_point")
                #         return {
                #             "action_id": action_item["action_id"],
                #             "status": "blocked",
                #             "action": "velocity_control",
                #             "collided": True,
                #             "error": "Projected position not navigable",
                #             "timestamp": time.time(),
                #         }
                
                
                
                # Preserve agent height (navmesh is XZ-only)
                current_y = state.position[1]

                if is_nav:
                    safe_pos = np.array([new_pos[0], current_y, new_pos[2]])
                    print(
                        f"✅ Using XZ from new_pos with preserved Y: {safe_pos.tolist()}"
                    )
                else:
                    snapped = pathfinder.snap_point(xz_pos)

                    if snapped is None:
                        print("⛔ Velocity motion blocked: snap_point returned None")
                        return {
                            "action_id": action_item["action_id"],
                            "status": "blocked",
                            "action": "velocity_control",
                            "collided": True,
                            "error": "snap_point failed",
                            "timestamp": time.time(),
                        }

                    safe_pos = np.array([snapped[0], current_y, snapped[2]])
                    print(
                        f"⚠️ new_pos off-navmesh, snapped XZ with preserved Y: {safe_pos.tolist()}"
                    )

                
                
                
                

                r = new_rigid_state.rotation
                state.position = np.array([safe_pos[0], safe_pos[1], safe_pos[2]])
                state.rotation = np.array(
                    [ r.vector.x, r.vector.y, r.vector.z, r.scalar]
                )
                agent.set_state(state)

                print(f"📌 Final applied position: {state.position.tolist()}")
                print(f"📌 Final applied rotation (quat): {state.rotation.tolist()}")

                #                 2️⃣ Your safety check MAKES IT WORSE

                # This block:

                # if np.isnan(current_pos).any() or det invalid:
                #     current_pos = [0,0,0]
                #     current_quat = identity

                # Resets the robot logically, but:

                # Habitat did not reset

                # Agent is physically elsewhere

                # Yaw now computed from wrong pose

                # Angular velocity dominates → rotation

                observations = self.env.get_sensor_observations()

                # Light‑weight observation debug
                if "rgba_camera" in observations:
                    rgb = observations["rgba_camera"]
                    print(f"👁 rgba_camera shape: {getattr(rgb, 'shape', None)}")
                if "depth_camera" in observations:
                    depth = observations["depth_camera"]
                    print(f"🌊 depth_camera shape: {getattr(depth, 'shape', None)}")

                print("✅ Velocity motion applied (projected)")

                result = {
                    "action_id": action_item["action_id"],
                    "status": "completed",
                    "action": "velocity_control",
                    "observations": observations,
                    "collided": observations.get("collided", False),
                    "new_position": state.position.tolist(),
                    "timestamp": time.time(),
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
                    "timestamp": time.time(),
                }

                print(
                    f"📤 Result (discrete): status={result['status']}, "
                    f"collided={result['collided']}, new_position={result['new_position']}"
                )
                print("=" * 80)
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

    orchestrator = MainOrchestrator(config)

    try:
        print(f"🚀 Starting Main Orchestrator for {args.scene}")
        print(f"   Task: {args.task}")

        orchestrator.run_orchestration(task_description=args.task)

    except Exception as e:
        print(f"❌ Orchestration failed: {e}")


if __name__ == "__main__":
    main()


# # When pulling actions from HabitatStore:
# action_item = self.habitat_store.pull_action_from_habitat_store()

# if action_item:
#     action_type = action_item.get("type")
#     parameters = action_item.get("parameters", {})
#     metadata = action_item.get("metadata", {})  # 🆕 Get metadata

#     if action_type == "velocity_control" and metadata.get("continuous_nav"):
#         print(f"⚡ ORCHESTRATOR: Executing VELOCITY CONTROL command")

#         # Extract velocity data from parameters
#         linear_vel = parameters.get("linear_velocity", [0, 0, 0])
#         angular_vel = parameters.get("angular_velocity", [0, 0, 0])
#         duration = parameters.get("duration", 0.1)

#         # Execute REAL VelocityControl (your test code)
#         agent = simulator.get_agent(0)
#         state = agent.get_state()

#         # Create VelocityControl
#         vc = habitat_sim.physics.VelocityControl()
#         vc.controlling_lin_vel = True
#         vc.controlling_ang_vel = True
#         vc.lin_vel_is_local = True
#         vc.ang_vel_is_local = True
#         vc.linear_velocity = mn.Vector3(*linear_vel) # type: ignore
#         vc.angular_velocity = mn.Vector3(*angular_vel)

#         # Get current state
#         q = state.rotation
#         current_quat = mn.Quaternion(mn.Vector3(q.x, q.y, q.z), q.w)
#         current_pos = mn.Vector3(state.position)

#         # Create RigidState
#         rigid_state = habitat_sim.RigidState(current_quat, current_pos)

#         # Integrate transform
#         new_rigid_state = vc.integrate_transform(duration, rigid_state)

#         # Check navmesh
#         new_pos = np.array([
#             new_rigid_state.translation.x,
#             new_rigid_state.translation.y,
#             new_rigid_state.translation.z
#         ])

#         if simulator.pathfinder.is_navigable(new_pos):
#             # Apply motion
#             r = new_rigid_state.rotation
#             state.position = new_pos
#             state.rotation = np.array([r.scalar, r.vector.x, r.vector.y, r.vector.z])
#             agent.set_state(state)

#             # Get observations
#             observations = simulator.get_sensor_observations()

#             # Store result
#             result = {
#                 "status": "success",
#                 "new_position": new_pos.tolist(),
#                 "observations": observations,
#                 "collided": observations.get("collided", False),
#                 "velocity_command": True  # Flag for ActionExecutor
#             }

#         else:
#             result = {
#                 "status": "blocked",
#                 "error": "Navmesh blocked motion",
#                 "collided": True,
#                 "velocity_command": True
#             }

#         # Store result with original action_id
#         self.habitat_store.store_action_result(action_item["action_id"], result)

#     else:
#         # 🎯 REGULAR DISCRETE ACTIONS (unchanged)
#         print(f"⚡ ORCHESTRATOR: Executing discrete action '{action_type}'")
#         # Your existing discrete action logic...
