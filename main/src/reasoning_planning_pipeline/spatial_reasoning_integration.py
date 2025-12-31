#!/usr/bin/env python3
"""
SPATIAL REASONING INTEGRATION - Geometric intelligence for robotic navigation
Combines map data, predictions, and task context for spatial decision making
Pure geometric algorithms - no LLM dependencies
"""

import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import time
import random  # Add this with other imports

class SpatialReasoningIntegration:
    """
    Geometric spatial reasoning using map data, predictions, and task context
    Provides reachability analysis, risk assessment, and search optimization
    """

    def __init__(
        self,
        map_store=None,
        prediction_store=None,
        task_store=None,
        user_command_store=None,
    ):
        # 🎯 STORE DEPENDENCIES
        self.prediction_store = prediction_store
        self.map_store = map_store
        self.user_command_store = user_command_store
        self.task_store = task_store

        # ✅ DEFERRED INITIALIZATION - Don't calculate until first use
        self.robot_radius = None
        self.min_clearance = None
        self.max_reach_distance = None

        # Search priority rules (geometric heuristics)
        self.object_priority_rules = {
            "table": 10,
            "desk": 10,
            "counter": 9,
            "nightstand": 8,
            "chair": 7,
            "sofa": 6,
            "bed": 5,
            "shelf": 4,
            "floor": 3,
            "wall": 2,
            "ceiling": 1,
        }

        # Risk thresholds
        self.high_risk_threshold = 0.7
        self.medium_risk_threshold = 0.4

        # ✅ ADD INIT FLAG
        self.initialized = True
        self.stores_available = all([map_store, prediction_store, task_store])

        print(
            f"Spatial Reasoning Integration initialized - Stores available: {self.stores_available}"
        )

    def _ensure_calculated(self):
        """Calculate robot parameters using fetched spatial data"""
        if self.robot_radius is None:
            try:
                # Get data using your existing fetch method
                all_data = self._fetch_all_spatial_data()
                map_data = all_data["map_data"]

                # Use the actual calculation methods with real data
                self.robot_radius = self._calculate_robot_radius_from_slam()
                self.min_clearance = self._calculate_min_clearance_from_slam()
                self.max_reach_distance = self._calculate_reach_from_camera()

                print(f"✅ Robot parameters calculated from real data:")
                print(
                    f"   Radius: {self.robot_radius:.2f}m, Clearance: {self.min_clearance:.2f}m, Reach: {self.max_reach_distance:.2f}m"
                )

            except Exception as e:
                print(f"❌ Failed to calculate robot parameters: {e}")
                # Safe fallbacks based on typical robot specs
                self.robot_radius = 0.3
                self.min_clearance = 0.4
                self.max_reach_distance = 2.0

    def analyze_environment(self, reasoning_cycle_id: str) -> bool:
        """Fixed version - uses fetched data and ensures calculations"""
        try:
            print(
                f"🎯 Spatial Reasoning: Starting analysis for cycle {reasoning_cycle_id}"
            )

            # ✅ ADD DEBUG: Check if semantic_objects exist
            if self.map_store and hasattr(self.map_store, "semantic_objects"):
                sem_objs = self.map_store.semantic_objects
                print(f"🔍 DEBUG: Found {len(sem_objs)} semantic objects in MapStore")
                for name, obj in list(sem_objs.items())[:3]:  # Show first 3
                    print(f"  - {name}: position={getattr(obj, 'position', 'NO_POS')}")

            # ✅ Get data using your existing method
            all_data = self._fetch_all_spatial_data()

            # 🎯 MAP TO EXPECTED FORMATS
            map_data = all_data["map_data"]
            prediction_data = all_data["predictions"]
            task_data = all_data["task_goal"]

            # ✅ ENSURE CALCULATIONS ARE DONE BEFORE ANY ANALYSIS
            self._ensure_calculated()

            # 🎯 USE ALL SPATIAL REASONING METHODS (they now have calculated parameters)
            reachability = self._analyze_reachability(map_data, prediction_data)
            search_priority = self._analyze_search_priority(
                map_data, task_data, prediction_data
            )
            risk_zones = self._identify_risk_zones(map_data, prediction_data)
            visibility = self._analyze_visibility(map_data)
            path_feasibility = self._score_path_feasibility(map_data, prediction_data)
            room_connectivity = self._analyze_room_connectivity(map_data, task_data)

            # 🎯 FIXED: Use correct variable names
            spatial_intelligence = {
                # Core spatial analyses
                "reachability_analysis": {
                    "reachable_objects": reachability,
                    "total_reachable": len(reachability),
                    "unreachable_reason": (
                        "distance_or_risk" if not reachability else "all_accessible"
                    ),
                },
                "search_strategy": {
                    "priority_list": search_priority[:5],
                    "top_priority": search_priority[0] if search_priority else None,
                    "total_objects_prioritized": len(search_priority),
                },
                "safety_analysis": {
                    "risk_zones": risk_zones,
                    "high_risk_count": len(
                        [zone for zone in risk_zones if zone["risk_level"] == "high"]
                    ),
                    "medium_risk_count": len(
                        [zone for zone in risk_zones if zone["risk_level"] == "medium"]
                    ),
                },
                "perception_analysis": {
                    "visibility_scores": visibility,
                    "best_visible": (
                        max(visibility.items(), key=lambda x: x[1]["score"])
                        if visibility
                        else None
                    ),
                    "worst_visible": (
                        min(visibility.items(), key=lambda x: x[1]["score"])
                        if visibility
                        else None
                    ),
                },
                "navigation_analysis": {
                    "path_feasibility": path_feasibility,
                    "most_feasible_room": (
                        max(path_feasibility.items(), key=lambda x: x[1])
                        if path_feasibility
                        else None
                    ),
                    "least_feasible_room": (
                        min(path_feasibility.items(), key=lambda x: x[1])
                        if path_feasibility
                        else None
                    ),
                },
                "exploration_analysis": {
                    "room_connectivity": room_connectivity,
                    "unexplored_rooms": [
                        room for room in room_connectivity if not room["explored"]
                    ],
                    "next_exploration_target": next(
                        (room for room in room_connectivity if not room["explored"]),
                        None,
                    ),
                },
                # ✅ FIXED: Use task_data (exists) instead of task_goal (doesn't exist)
                "recommendations": self._generate_recommendations(
                    reachability,
                    search_priority,
                    risk_zones,
                    room_connectivity,
                    task_data,
                ),
                # ✅ FIXED: Use map_data (exists) instead of map_data (doesn't exist)
                "summary_metrics": {
                    "total_objects_analyzed": len(map_data.get("objects", [])),
                    "safe_navigation_paths": len(
                        [score for score in path_feasibility.values() if score > 0.7]
                    ),
                    "exploration_progress": len(
                        [room for room in room_connectivity if room["explored"]]
                    )
                    / max(len(room_connectivity), 1),
                },
                # 🎯 PIPELINE METADATA
                "reasoning_cycle_id": reasoning_cycle_id,
                "component_name": "spatial_reasoning",
                "pipeline_timestamp": time.time(),
                "processing_stages": [
                    "reachability",
                    "priority",
                    "risk",
                    "visibility",
                    "navigation",
                    "exploration",
                ],
            }

            # 💾 SAVE TO TASK STORE
            # 💾 STORE IN INTERMEDIATE_REASONING FOR PIPELINE RETRIEVAL
            if self.task_store:
                # Initialize intermediate_reasoning if it doesn't exist
                if not hasattr(self.task_store, "intermediate_reasoning"):
                    self.task_store.intermediate_reasoning = {}

                # Create or update the cycle data
                if reasoning_cycle_id not in self.task_store.intermediate_reasoning:
                    self.task_store.intermediate_reasoning[reasoning_cycle_id] = {
                        "component_reasoning": {},
                        "timestamp": time.time(),
                    }

            # Store spatial reasoning component data
            self.task_store.intermediate_reasoning[reasoning_cycle_id][
                "component_reasoning"
            ]["spatial_reasoning"] = spatial_intelligence
            print(
                f"✅ Spatial reasoning stored in intermediate_reasoning for {reasoning_cycle_id}"
            )

            print(f"✅ Spatial reasoning completed for cycle: {reasoning_cycle_id}")
            print(
                f"   📊 Results: {len(reachability)} reachable objects, {len(risk_zones)} risk zones, {len(search_priority)} prioritized objects"
            )

            import json

            print(
                f"🔍 SPATIAL REASONING FULL OUTPUT ({reasoning_cycle_id}):\n{json.dumps(spatial_intelligence, indent=2)}"
            )

            return {
                "reasoning_cycle_id": reasoning_cycle_id,
            }

        except Exception as e:
            print(f"❌ Spatial reasoning failed for cycle {reasoning_cycle_id}: {e}")
            import traceback

            traceback.print_exc()
            return False

    def generate_rule_based_plan(self, reasoning_cycle_id: str) -> bool:
        """Generate rule-based plan and save to TaskStore"""
        try:
            # 1. Fetch own data
            map_data = self.map_store.get_map_summary()
            predictions = self.prediction_store.get_latest_prediction()

            # 2. Generate rule-based plan
            plan = self._reason_with_rules(map_data, predictions)

            # 3. Save directly to TaskStore
            plan_data = {
                "reasoning_cycle_id": reasoning_cycle_id,
                "plan_id": f"rule_plan_{reasoning_cycle_id}",
                "strategy_type": plan.get("selected_action", "stop"),
                "confidence": plan.get("confidence", 0.5),
                "reasoning_details": "rule_based_fallback",
                "llm_used": False,
                "timestamp": time.time(),
                "component": "rule_engine",
            }

            self.task_store.write_reasoning_plan(reasoning_cycle_id, plan_data)
            return True

        except Exception as e:
            print(f"Rule-based reasoning failed: {e}")
            return False

    def _fetch_all_spatial_data(self) -> Dict[str, Any]:
        """
        SINGLE METHOD to fetch all data needed for spatial reasoning
        """
        print("=" * 80)
        print("🔍 _fetch_all_spatial_data() - DEBUG START")
        print("=" * 80)

        # Default empty data - NO map_data needed!
        data = {
            "map_data": {},  # ← RENAME from 'map_data' to 'map_data'
            "observations": {},
            "task_goal": {},
            "predictions": {},
        }

        try:
            # 🗺️ BUILD MAP DATA DIRECTLY FROM STORE ATTRIBUTES
            map_data = {
                "objects": [],
                "robot_position": [0, 0, 0],
                "current_room": "unknown",
                "room_boundaries": {},
            }

            print("📊 STORE CHECK:")
            print(f"  • map_store exists: {self.map_store is not None}")
            if self.map_store:
                print(f"  • map_store type: {type(self.map_store)}")
                print(
                    f"  • map_store dir: {[attr for attr in dir(self.map_store) if not attr.startswith('_')][:10]}..."
                )

            # 1. Get objects from semantic_objects
            print("\n🎯 STEP 1: Fetching semantic objects")
            if self.map_store and hasattr(self.map_store, "semantic_objects"):
                sem_objs = self.map_store.semantic_objects
                print(f"  ✅ Found semantic_objects attribute")
                print(f"  • Total semantic objects: {len(sem_objs)}")

                for i, (obj_name, obj) in enumerate(sem_objs.items()):
                    print(f"  • Object {i}: '{obj_name}'")
                    print(f"    - Type: {type(obj)}")

                    # Check position attribute
                    has_position = hasattr(obj, "position")
                    print(f"    - Has 'position' attr: {has_position}")

                    if has_position:
                        pos = obj.position
                        print(f"    - Position: {pos}")
                        print(f"    - Position type: {type(pos)}")
                        print(
                            f"    - Position length: {len(pos) if hasattr(pos, '__len__') else 'N/A'}"
                        )

                        # Add to map_data
                        map_data["objects"].append(
                            {
                                "name": obj_name,
                                "position": list(pos),
                                "confidence": getattr(obj, "confidence", 0.5),
                            }
                        )
                        print(f"    - ✅ Added to map_data")
                    else:
                        print(f"    - ❌ Skipped (no position)")

                    # Show first 3 objects only for readability
                    if i >= 2 and len(sem_objs) > 3:
                        remaining = len(sem_objs) - 3
                        print(f"  • ... and {remaining} more objects")
                        break
            else:
                print(f"  ❌ No semantic_objects found")
                if self.map_store:
                    print(
                        f"  • Available attributes: {[attr for attr in dir(self.map_store) if not attr.startswith('_')]}"
                    )

            # 2. Get robot position from current_pose
            print("\n📍 STEP 2: Fetching robot position")
            if self.map_store and hasattr(self.map_store, "current_pose"):
                print(f"  ✅ Found current_pose attribute")
                if self.map_store.current_pose:
                    pos = self.map_store.current_pose.position
                    print(f"  • Robot position: {pos}")
                    print(f"  • Position type: {type(pos)}")
                    map_data["robot_position"] = pos
                else:
                    print(f"  ⚠️ current_pose exists but is None")
            else:
                print(f"  ❌ No current_pose attribute")
                map_data["robot_position"] = [0, 0, 0]

            # 3. Get current room
            print("\n🚪 STEP 3: Fetching current room")
            if self.map_store and hasattr(self.map_store, "current_room"):
                print(f"  ✅ Found current_room attribute")
                print(f"  • Current room: {self.map_store.current_room}")
                map_data["current_room"] = self.map_store.current_room
            else:
                print(f"  ❌ No current_room attribute")
                map_data["current_room"] = "unknown"

            data["map_data"] = map_data

            print(f"\n📦 MAP_DATA SUMMARY:")
            print(f"  • Objects: {len(map_data['objects'])} items")
            print(f"  • Robot position: {map_data['robot_position']}")
            print(f"  • Current room: {map_data['current_room']}")

            # Show first 2 objects in detail
            for i, obj in enumerate(map_data["objects"][:2]):
                print(f"  • Object {i}: {obj['name']}")
                print(f"    - Position: {obj['position']}")
                print(f"    - Confidence: {obj['confidence']}")

            # 🎯 GET TASK GOAL (FIXED VERSION)
            print("\n🎯 STEP 4: Fetching task goal")
            task_goal = {
                "current_mission": "explore",  # Add default
                "search_progress": {},  # ← ADD THIS DEFAULT
                "explored_rooms": [],  # ← ADD THIS DEFAULT
                "current_task": {"type": "exploration"},  # ← ADD THIS DEFAULT,
                # 🆕 ADD THIS: Umeyama-aligned goals
                "umeyama_aligned_goals": {},
                "mission_goals_list": [],
            }

            if self.task_store:
                print(f"  ✅ task_store exists")
                try:
                    # ✅ FIX 1: Use existing method get_task_summary()
                    if hasattr(self.task_store, "get_task_summary"):
                        print(f"  • Calling get_task_summary()")
                        store_summary = self.task_store.get_task_summary()
                        print(f"  • Store summary keys: {list(store_summary.keys())}")

                        # Extract mission goals from summary
                        mission_data = store_summary.get("mission_goals", {}) 
                        
                        # 🆕 ADD: Get Umeyama-aligned goals
                        if hasattr(self.task_store, "get_umeyama_aligned_goals"):
                            aligned_goals = self.task_store.get_umeyama_aligned_goals()
                            task_goal["umeyama_aligned_goals"] = aligned_goals
                            print(
                                f"  • Found {len(aligned_goals)} Umeyama-aligned goals"
                            )
                            
                        # ✅ CORRECT: Call the actual method on task_store
                        if hasattr(self.task_store, 'get_remaining_goals'):
                            task_goal["mission_goals_list"] = self.task_store.get_remaining_goals()
                        else:
                            task_goal["mission_goals_list"] = []

                        print(f"  • Mission data: {mission_data}")
                    else:
                        print(f"  ❌ task_store has no get_task_summary method")
                except Exception as e:
                    print(f"  ❌ Failed to get task goal: {e}")
            else:
                print(f"  ❌ No task_store available")

            data["task_goal"] = task_goal
            print(f"  • Final task_goal keys: {list(task_goal.keys())}")

            # 📊 GET PREDICTIONS (keep existing logic)
            print("\n🔮 STEP 5: Fetching predictions")
            if self.prediction_store:
                print(f"  ✅ prediction_store exists")
                if hasattr(self.prediction_store, "get_predictions"):
                    print(f"  • Calling get_predictions()")
                    predictions = self.prediction_store.get_predictions()
                    print(f"  • Predictions type: {type(predictions)}")
                    if isinstance(predictions, dict):
                        print(f"  • Predictions keys: {list(predictions.keys())}")
                    data["predictions"] = predictions
                else:
                    print(f"  ❌ prediction_store has no get_predictions method")
            else:
                print(f"  ❌ No prediction_store available")

            print(f"\n✅ FINAL DATA STRUCTURE:")
            print(f"  • map_data objects: {len(data['map_data'].get('objects', []))}")
            print(f"  • task_goal keys: {len(data['task_goal'])}")
            print(f"  • predictions type: {type(data['predictions'])}")

            print("=" * 80)
            print("✅ _fetch_all_spatial_data() - DEBUG COMPLETE")
            print("=" * 80)

            return data

        except Exception as e:
            print(f"❌ _fetch_all_spatial_data() FAILED: {e}")
            import traceback

            traceback.print_exc()
            print("=" * 80)

            # Return empty but valid structure
            return {
                "map_data": {
                    "objects": [],
                    "robot_position": [0, 0, 0],
                    "current_room": "unknown",
                },
                "observations": {},
                "task_goal": {},
                "predictions": {},
            }

    def _calculate_robot_radius_from_slam(self) -> float:
        """Calculate robot radius from SLAM point cloud density"""
        if not self.map_store:
            return 0.3  # Fallback

        try:
            # ✅ USE EXISTING METHOD: get_all_map_points() instead of .get('slam_points')
            slam_points = self.map_store.get_all_map_points()
            if not slam_points:
                return 0.3

            # Find points very close to robot (within typical robot size)
            close_points = []
            for point in slam_points:
                if "position_3d" in point:
                    x, y, z = point["position_3d"]
                    distance = math.sqrt(x**2 + y**2)  # 2D distance from origin
                    if distance < 0.5:  # Within 50cm - likely robot body points
                        close_points.append((x, y))

            if close_points:
                # Calculate bounding circle around close points
                distances = [math.sqrt(x**2 + y**2) for x, y in close_points]
                estimated_radius = max(distances) if distances else 0.3
                return min(estimated_radius, 0.5)  # Cap at reasonable size

            return 0.3  # Fallback

        except Exception as e:
            print(f"Failed to calculate robot radius: {e}")
            return 0.3

    def _calculate_min_clearance_from_slam(self) -> float:
        """Calculate minimum clearance from navigation history in SLAM data"""
        if not self.map_store:
            return 0.4  # Fallback

        try:
            # ✅ USE EXISTING METHOD: get_map_summary() instead of .get('habitat_data')
            map_summary = self.map_store.get_map_summary()
            explored_positions = map_summary.get("explored_positions_count", 1)

            # More exploration = more confidence in tighter spaces
            if explored_positions > 50:
                clearance = 0.3  # Experienced robot
            elif explored_positions > 20:
                clearance = 0.35  # Moderate experience
            else:
                clearance = 0.4  # New environment

            # 2. Adjust based on SLAM point cloud density
            slam_points = self.map_store.get_all_map_points()
            if len(slam_points) > 100:
                # Dense point cloud = better spatial awareness = can handle tighter spaces
                clearance = min(clearance, 0.35)  # Use the tighter clearance

            return clearance

        except Exception as e:
            print(f"Failed to calculate min clearance: {e}")
            return 0.4

    def _calculate_reach_from_camera(self) -> float:
        """Calculate maximum reach distance from camera parameters"""
        if not self.map_store:
            return 2.0  # Fallback

        try:
            # ✅ USE EXISTING METHOD: get_camera_parameters() instead of .get('camera_parameters')
            camera_params = self.map_store.get_camera_parameters() or {}
            if camera_params:
                # Estimate practical interaction range from camera field of view
                fx = camera_params.get("fx", 535.4)  # Focal length x
                width = camera_params.get("width", 640)

                # Calculate horizontal field of view
                fov_horizontal = 2 * math.atan(width / (2 * fx))

                # Practical reach: distance where objects are still clearly visible
                # Assume we want objects to be at least 100 pixels wide for recognition
                object_width_pixels = 100
                practical_distance = (
                    width / object_width_pixels
                ) * 0.01  # More realistic scale

                return min(practical_distance, 3.0)  # Cap at 3 meters

            # Fallback: Use SLAM point maximum distance
            slam_points = self.map_store.get_all_map_points()
            if slam_points:
                max_distance = 0
                for point in slam_points:
                    if "position_3d" in point:
                        x, y, z = point["position_3d"]
                        distance = math.sqrt(x**2 + y**2 + z**2)
                        max_distance = max(max_distance, distance)

                # Use 80% of maximum observed distance for safety
                return max_distance * 0.8 if max_distance > 0 else 2.0

            return 2.0  # Fallback

        except Exception as e:
            print(f"Failed to calculate reach distance: {e}")
            return 2.0

    def _analyze_reachability(
        self, map_data: Dict, prediction_data: Dict
    ) -> List[Dict]:
        """Analyze which objects are physically reachable"""
        reachable_objects = []
        robot_pos = np.array(
            map_data.get("robot_position", [0, 0, 0])
        )  # Fallback to origin if missing

        for obj in map_data["objects"]:
            if "position" not in obj:
                continue

            obj_pos = np.array(obj["position"])
            distance = np.linalg.norm(robot_pos - obj_pos)

            # Basic distance check
            if distance > self.max_reach_distance:
                continue

            # Check if object is in an accessible area (not in corner/tight space)
            is_accessible = self._check_accessible_area(obj_pos, map_data)

            # Get collision risk for path to object - PASS map_data as map_data
            path_risk = self._get_path_risk(
                robot_pos, obj_pos, prediction_data, map_data
            )

            if is_accessible and path_risk < self.high_risk_threshold:
                reachable_objects.append(
                    {
                        "name": obj.get("name", "unknown"),
                        "position": obj["position"],
                        "distance": distance,
                        "risk_score": path_risk,
                        "accessible": True,
                    }
                )

        return reachable_objects

    def _analyze_search_priority(
        self, map_data: Dict, task_data: Dict, prediction_data: Dict
    ) -> List[Dict]:
        """Prioritize objects for search based on multiple factors"""
        prioritized_objects = []

        # ✅ DIRECT FETCH FROM MAP STORE (One line addition)
        semantic_objects = self.map_store.semantic_objects if self.map_store else {}

        # 🆕 GET UMETAMA-ALIGNED GOALS
        aligned_goals = task_data.get("umeyama_aligned_goals", {})
        mission_goals_list = task_data.get("mission_goals_list", [])

        for obj in map_data["objects"]:
            if "name" not in obj:
                continue

                        # ✅ GET COORDINATES - PRIORITIZE UMEYAMA, FALLBACK TO SEMANTIC
            object_name = obj["name"]
            coordinates = [0, 0, 0]  # Default

            # 🎯 PRIORITIZE UMEYAMA-ALIGNED COORDINATES
            if object_name in aligned_goals and aligned_goals[object_name].get("world_coordinates"):
                coordinates = aligned_goals[object_name]["world_coordinates"]
                print(f"🎯 Using Umeyama-aligned coordinates for {object_name}: {coordinates}")
            # 🎯 FALLBACK TO SEMANTIC COORDINATES
            elif object_name in semantic_objects:
                semantic_obj = semantic_objects[object_name]
                if hasattr(semantic_obj, "position"):
                    coordinates = list(semantic_obj.position)
                    print(f"✅ Using semantic coordinates for {object_name}: {coordinates}")

            # 🆕 FIX 3: Check if this object is in Umeyama-aligned goals
            is_aligned_goal = object_name in aligned_goals
            goal_priority_boost = 0.0

            if is_aligned_goal:
                goal_data = aligned_goals[object_name]
                # Boost priority if we have precise world coordinates
                goal_priority_boost = 5.0 if goal_data.get("world_coordinates") else 2.0
                print(
                    f"🎯 Object {object_name} is Umeyama-aligned mission goal! +{goal_priority_boost} boost"
                )

            # Base priority from rules
            base_priority = self.object_priority_rules.get(object_name, 5)

            # Adjust based on mission - PASS task_data to _get_mission_boost
            mission_boost = self._get_mission_boost(
                obj["name"], task_data.get("current_mission", "explore"), task_data
            )

            # Adjust based on risk
            risk_penalty = self._get_risk_penalty(obj, prediction_data)

            search_progress = task_data.get("search_progress", {})  # ← FIXED HERE
            # Adjust based on search progress
            progress_penalty = self._get_progress_penalty(
                obj, search_progress
            )  # ← FIXED HERE
            
            
                        
            # 🆕 CALCULATE DISTANCE TO GOAL IF WE HAVE COORDINATES
            distance_to_goal = float('inf')  # Default: very far
            if is_aligned_goal and goal_data.get("world_coordinates"):
                goal_coords = np.array(goal_data["world_coordinates"])
                robot_pos = np.array(map_data.get("robot_position", [0, 0, 0]))
                distance_to_goal = np.linalg.norm(goal_coords - robot_pos)
                print(f"📏 Distance to goal '{object_name}': {distance_to_goal:.2f}m")
                
                

            # Closer goals get higher priority
            distance_factor = max(0, 1.0 - (distance_to_goal / self.max_reach_distance))

            final_priority = (
                base_priority + mission_boost - risk_penalty - progress_penalty + 
                goal_priority_boost + (distance_factor * 3.0)  # Distance contributes up to +3
            )
            final_priority = max(1, min(15, final_priority))  # Clamp to 1-15 (since goal boost can be +5)

            prioritized_objects.append(
                {
                    "name": obj["name"],
                    "position": obj.get("position", [0, 0, 0]),
                    "coordinates": coordinates,  # 🎯 NEW: ACTUAL COORDINATES!
                    "priority_score": final_priority,
                    "base_priority": base_priority,
                    "mission_boost": mission_boost,
    "goal_priority_boost": goal_priority_boost,  # 🆕 ADD THIS
                "is_umeyama_aligned_goal": is_aligned_goal,  # 🆕 ADD THIS
                "risk_penalty": risk_penalty,
                "distance_to_goal_m": distance_to_goal if is_aligned_goal else None,
                "reason": self._get_priority_reason(base_priority, mission_boost, risk_penalty, is_aligned_goal)  # ✅ FIXED
            })
            

        # Sort by priority score (descending)
        prioritized_objects.sort(key=lambda x: x["priority_score"], reverse=True)
        return prioritized_objects

    def _identify_risk_zones(self, map_data: Dict, prediction_data: Dict) -> List[Dict]:
        """Identify high-risk areas for navigation"""
        risk_zones = []
        robot_pos = np.array(map_data["robot_position"])

        # Check for tight spaces between objects
        for i, obj1 in enumerate(map_data["objects"]):
            for j, obj2 in enumerate(map_data["objects"]):
                if i >= j or "position" not in obj1 or "position" not in obj2:
                    continue

                pos1 = np.array(obj1["position"])
                pos2 = np.array(obj2["position"])
                distance = np.linalg.norm(pos1 - pos2)

                # If objects are close together, might be a tight passage
                if distance < (self.robot_radius * 4):  # 4x robot diameter
                    midpoint = (pos1 + pos2) / 2
                    risk_level = (
                        "high" if distance < (self.robot_radius * 2.5) else "medium"
                    )

                    risk_zones.append(
                        {
                            "type": "narrow_passage",
                            "position": midpoint.tolist(),
                            "risk_level": risk_level,
                            "objects_involved": [
                                obj1.get("name", "unknown"),
                                obj2.get("name", "unknown"),
                            ],
                            "clearance": distance,
                        }
                    )

        # Check areas with high prediction risks
        for action, risk in prediction_data.get("collision_risks", {}).items():
            if risk > self.high_risk_threshold:
                # Estimate position based on action type
                estimated_pos = self._estimate_risk_position(action, robot_pos)
                risk_zones.append(
                    {
                        "type": "predicted_high_risk",
                        "position": estimated_pos,
                        "risk_level": "high",
                        "action": action,
                        "risk_score": risk,
                    }
                )

        return risk_zones

    def _analyze_visibility(self, map_data: Dict) -> Dict[str, Any]:
        """Analyze visibility and occlusion of objects/areas"""
        visibility_analysis = {}
        robot_pos = np.array(map_data["robot_position"])

        for obj in map_data["objects"]:
            if "position" not in obj:
                continue

            obj_pos = np.array(obj["position"])

            # Simple line-of-sight check (simplified)
            has_line_of_sight = self._check_line_of_sight(
                robot_pos, obj_pos, map_data["objects"]
            )

            # Distance-based visibility score (closer = more visible)
            distance = np.linalg.norm(robot_pos - obj_pos)
            distance_score = max(0, 1 - (distance / self.max_reach_distance))

            # Combined visibility score
            visibility_score = (
                distance_score * 0.7 + (1.0 if has_line_of_sight else 0.3) * 0.3
            )

            visibility_analysis[obj.get("name", "unknown")] = {
                "score": visibility_score,
                "has_line_of_sight": has_line_of_sight,
                "distance": distance,
                "occlusions": [] if has_line_of_sight else ["unknown_obstacle"],
            }

        return visibility_analysis

    def _score_path_feasibility(
        self, map_data: Dict, prediction_data: Dict
    ) -> Dict[str, float]:
        """Score feasibility of paths to different areas"""
        feasibility_scores = {}
        robot_pos = np.array(map_data["robot_position"])

        # Score paths to different rooms/areas
        for room_name, boundaries in map_data["room_boundaries"].items():
            if not boundaries:
                continue

            # Estimate room center (simplified)
            room_center = self._estimate_room_center(boundaries)

            # Basic path checking
            has_clear_path = self._check_clear_path(
                robot_pos, room_center, map_data["objects"]
            )

            # Get prediction risks for moving to this room
            move_risk = prediction_data.get("collision_risks", {}).get(
                "move_forward", 0.5
            )

            # Combined feasibility score
            feasibility = (1.0 if has_clear_path else 0.3) * 0.6 + (1 - move_risk) * 0.4

            feasibility_scores[room_name] = feasibility

        return feasibility_scores

    def _analyze_room_connectivity(self, map_data: Dict, task_data: Dict) -> List[Dict]:
        """Analyze room connectivity and accessibility"""
        connectivity_analysis = []
        current_room = self._get_current_room(
            map_data["robot_position"], map_data["room_boundaries"]
        )

        for room_name, boundaries in map_data["room_boundaries"].items():
            if room_name == current_room:
                continue

            # Check if room has been explored
            is_explored = room_name in task_data["explored_rooms"]

            # Simple connectivity check (in real implementation, use door positions)
            is_connected = self._check_room_connectivity(
                current_room, room_name, map_data
            )

            if is_connected:
                connectivity_analysis.append(
                    {
                        "room": room_name,
                        "accessible": True,
                        "explored": is_explored,
                        "priority": (
                            0.8 if not is_explored else 0.3
                        ),  # Unexplored rooms have higher priority
                        "path_required": not is_explored,
                    }
                )

        return connectivity_analysis

    # Helper methods for geometric calculations
    def _check_accessible_area(self, position: np.ndarray, map_data: Dict) -> bool:
        """Check if position is in an accessible area (not corner/tight space)"""
        # Simplified implementation - in real system, use occupancy grid
        for obj in map_data["objects"]:
            if "position" not in obj:
                continue
            obj_pos = np.array(obj["position"])
            if np.linalg.norm(position - obj_pos) < self.min_clearance:
                return False
        return True

    def _get_path_risk(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        prediction_data: Dict,
        map_data: Dict,
    ) -> float:
        """Estimate risk for path between two points using already-fetched data"""
        # Use ACTUAL prediction data from already-fetched data
        actual_risk = prediction_data.get("collision_risks", {}).get(
            "move_forward", 0.5
        )

        # Use habitat_data from already-fetched map_data
        habitat_data = map_data.get("habitat_data", {})
        mission_duration = habitat_data.get("mission_duration", 1)

        # More experience = lower risk confidence
        experience_factor = min(
            1.0, mission_duration / 300
        )  # 5 minutes = full confidence
        return actual_risk * (
            1.0 - experience_factor * 0.3
        )  # Up to 30% risk reduction with experience

    def _get_mission_boost(
        self, object_name: str, mission: str, task_goal: Dict
    ) -> float:
        """Get priority boost based on mission relevance using already-fetched data"""
        # Use ACTUAL mission from already-fetched task_goal
        current_task = task_goal.get("current_task", {})

        # Extract target object from real mission
        mission_text = mission.lower()

        # Simple keyword matching from real mission
        if "key" in mission_text and object_name in ["table", "desk", "counter"]:
            return 3.0
        elif "phone" in mission_text and object_name in ["table", "desk", "bed"]:
            return 3.0
        elif "explore" in mission_text and object_name in ["door", "hallway"]:
            return 2.0

        return 0.0

    def _get_risk_penalty(self, obj: Dict, prediction_data: Dict) -> float:
        # Use ACTUAL structural risks from prediction store
        structural_risks = prediction_data.get("structural_risks", {})
        object_risk = structural_risks.get(obj.get("name", ""), {}).get(
            "risk_score", 0.3
        )

        # Scale based on real robot parameters
        return object_risk * (
            self.robot_radius * 10
        )  # Dynamic scaling based on robot size

    def _get_progress_penalty(self, obj: Dict, search_progress: Dict) -> float:
        object_name = obj.get("name", "")

        # FIX: Ensure search_progress is a dict
        if not isinstance(search_progress, dict):
            return 0.0  # No penalty if data is invalid

        # Use ACTUAL search progress from task store
        if object_name in search_progress:
            # Check how recently it was searched using timestamp if available
            search_time = search_progress[object_name].get("timestamp", 0)
            current_time = time.time()

            # Recent searches get higher penalty
            time_since_search = current_time - search_time
            if time_since_search < 60:  # Searched in last minute
                return 5.0
            elif time_since_search < 300:  # Searched in last 5 minutes
                return 3.0
            else:
                return 1.0  # Old search, lower penalty

        return 0.0

    def _get_priority_reason(
        self,
        base_prio: int,
        mission_boost: float,
        risk_penalty: float,
        is_aligned_goal: bool = False,
    ) -> str:
        """Generate human-readable reason for priority score"""
        reasons = []
        if base_prio >= 8:
            reasons.append("high natural priority")
        if mission_boost > 1:
            reasons.append("mission relevant")
        if is_aligned_goal:
            reasons.append("Umeyama-aligned goal")  # 🆕 NEW REASON

        if risk_penalty > 2:
            reasons.append("high risk area")
        return ", ".join(reasons) if reasons else "standard priority"

    def _check_line_of_sight(
        self, start_pos: np.ndarray, end_pos: np.ndarray, objects: List[Dict]
    ) -> bool:
        # Use ACTUAL robot reach distance instead of hardcoded 1.0m
        check_distance = self.max_reach_distance * 0.5  # 50% of max reach

        for obj in objects:
            if "position" not in obj:
                continue
            obj_pos = np.array(obj["position"])

            # Dynamic distance based on robot capabilities
            if (
                np.linalg.norm(obj_pos - start_pos) < check_distance
                and np.linalg.norm(obj_pos - end_pos) < check_distance
            ):
                return False
        return True

    def _analyze_situation(
        self, map_data: Dict, observations: Dict, task_goal: Dict
    ) -> Dict:
        """Analyze the current situation"""
        current_room = map_data.get("current_room", "unknown")
        clutter_level = observations.get("scene_semantics", {}).get(
            "clutter_level", "unknown"
        )
        visibility = observations.get("scene_semantics", {}).get(
            "visibility_score", 0.5
        )

        return {
            "current_location": current_room,
            "environment_analysis": {
                "clutter_level": clutter_level,
                "visibility": visibility,
                "navigability": self._assess_navigability(map_data),
                "exploration_potential": self._assess_exploration_potential(map_data),
            },
            "task_context": {
                "task_type": task_goal.get("current_task", {}).get(
                    "type", "exploration"
                ),
                "priority": task_goal.get("task_priority", "medium"),
                "complexity": self._assess_task_complexity(task_goal),
            },
            "objects_present": observations.get("perception_signals", {}).get(
                "detections", []
            ),
            "spatial_awareness": self._assess_spatial_awareness(map_data, observations),
        }

    def _assess_risks(self, map_data: Dict, predictions: Dict, task_goal: Dict) -> Dict:
        """Assess risks for different actions"""
        action_risks = {}
        predictions_risks = predictions.get("action_risks", {})

        # Default actions to assess
        actions = ["move_forward", "turn_left", "turn_right", "stop", "explore"]

        for action in actions:
            # Start with prediction risk if available
            base_risk = predictions_risks.get(action, 0.3)

            # Adjust based on world state
            if action == "move_forward":
                # Higher risk if low visibility or high clutter
                visibility = map_data.get("map_summary", {}).get(
                    "visibility_score", 0.5
                )
                base_risk += (1.0 - visibility) * 0.3

            elif action in ["turn_left", "turn_right"]:
                # Turning is generally safer
                base_risk *= 0.8

            elif action == "explore":
                # Exploration risk depends on unknown areas
                exploration_potential = self._assess_exploration_potential(map_data)
                base_risk = 0.2 + (1.0 - exploration_potential) * 0.3

            action_risks[action] = min(max(base_risk, 0.0), 1.0)

        return {
            "action_risks": action_risks,
            "overall_risk_level": self._calculate_overall_risk(
                action_risks, map_data
            ),  # PASS map_data
            "high_risk_actions": [
                act for act, risk in action_risks.items() if risk > 0.6
            ],
            "safe_actions": [act for act, risk in action_risks.items() if risk < 0.3],
        }

    def _calculate_overall_risk(self, action_risks: Dict, map_data: Dict) -> float:
        """Calculate overall risk level using already-fetched data"""
        if not action_risks:
            # Use habitat data from already-fetched map_data
            habitat_data = map_data.get("habitat_data", {})
            mission_duration = habitat_data.get("mission_duration", 1)
            # Longer mission = lower baseline risk (more experience)
            return max(0.1, 0.5 - (mission_duration / 600))  # 10 minutes = 0.4 risk

        # Weight risks by action importance (movement risks matter more)
        weighted_risks = []
        for action, risk in action_risks.items():
            if "move" in action:
                weighted_risks.append(risk * 1.5)  # Movement risks weighted higher
            else:
                weighted_risks.append(risk * 1.0)

        return sum(weighted_risks) / len(weighted_risks)

    def _assess_navigability(self, map_data: Dict) -> float:
        """Assess navigability using already-fetched SLAM data"""
        slam_points = map_data.get("slam_points", [])
        if not slam_points:
            return 0.5

        # Calculate point density around robot (higher density = better mapping)
        robot_pos = np.array(map_data.get("current_pose", [0, 0, 0]))
        nearby_points = 0

        for point in slam_points:
            if "position_3d" in point:
                point_pos = np.array(point["position_3d"])
                distance = np.linalg.norm(robot_pos - point_pos)
                if distance < 2.0:  # Points within 2 meters
                    nearby_points += 1

        # More nearby points = better understanding = higher navigability
        point_density = nearby_points / len(slam_points) if slam_points else 0
        return min(point_density * 2.0, 1.0)  # Scale to 0-1

    def _assess_exploration_potential(self, map_data: Dict) -> float:
        """Assess potential for exploration"""
        known_areas = len(map_data.get("map_summary", {}).get("rooms", []))
        objects_count = map_data.get("map_summary", {}).get("objects_count", 0)

        # More objects and fewer known areas = higher exploration potential
        if known_areas == 0:
            return 1.0

        exploration_potential = (objects_count / max(known_areas, 1)) * 0.3 + 0.7
        return min(exploration_potential, 1.0)

    def _assess_task_complexity(self, task_goal: Dict) -> str:
        """Assess complexity of current task"""
        task_type = task_goal.get("current_task", {}).get("type", "exploration")
        priority = task_goal.get("task_priority", "medium")

        if task_type == "navigation" and priority == "high":
            return "high"
        elif task_type == "manipulation":
            return "high"
        elif task_type == "exploration":
            return "medium"
        else:
            return "low"

    def _assess_spatial_awareness(self, map_data: Dict, observations: Dict) -> Dict:
        """Assess spatial awareness and understanding"""
        objects_detected = len(
            observations.get("perception_signals", {}).get("detections", [])
        )
        rooms_known = len(map_data.get("map_summary", {}).get("rooms", []))

        return {
            "objects_awareness": min(objects_detected / 10.0, 1.0),  # Scale to 0-1
            "environment_awareness": min(rooms_known / 5.0, 1.0),  # Scale to 0-1
            "overall_awareness": (
                min(objects_detected / 10.0, 1.0) + min(rooms_known / 5.0, 1.0)
            )
            / 2,
        }

    def _assess_goal_progress(self, map_data: Dict, task_goal: Dict) -> Dict:
        """Assess progress towards current goal"""
        current_task = task_goal.get("current_task", {})
        task_type = current_task.get("type", "exploration")

        if task_type == "navigation":
            # Simulate navigation progress
            return {
                "progress": random.uniform(0.1, 0.9),  # Placeholder
                "estimated_completion": "unknown",
                "goal_reached": random.random() > 0.8,  # 20% chance
            }
        elif task_type == "exploration":
            exploration_potential = self._assess_exploration_potential(map_data)
            return {
                "progress": 1.0
                - exploration_potential,  # Less potential = more explored
                "estimated_completion": "medium",
                "area_explored": exploration_potential < 0.3,
            }
        else:
            return {
                "progress": 0.5,
                "estimated_completion": "unknown",
                "goal_reached": False,
            }

    def _calculate_confidence(self, situation: Dict, risks: Dict) -> float:
        """Calculate confidence in reasoning results"""
        # Base confidence from spatial awareness
        spatial_confidence = situation.get("spatial_awareness", {}).get(
            "overall_awareness", 0.5
        )

        # Adjust based on risk level
        risk_level = risks.get("overall_risk_level", 0.5)
        risk_penalty = risk_level * 0.3

        # Adjust based on environment clarity
        environment_confidence = situation.get("environment_analysis", {}).get(
            "visibility", 0.5
        )

        confidence = (
            spatial_confidence * 0.4
            + environment_confidence * 0.3
            + (1.0 - risk_penalty) * 0.3
        )

        return min(max(confidence, 0.1), 1.0)

    def _generate_recommendations(self, situation: Dict, risks: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []

        # Environment-based recommendations
        clutter_level = situation.get("environment_analysis", {}).get(
            "clutter_level", "unknown"
        )
        if clutter_level == "high":
            recommendations.append("Move cautiously in cluttered environment")

        visibility = situation.get("environment_analysis", {}).get("visibility", 0.5)
        if visibility < 0.3:
            recommendations.append("Low visibility - consider alternative paths")

        # Risk-based recommendations
        high_risk_actions = risks.get("high_risk_actions", [])
        if high_risk_actions:
            recommendations.append(
                f"Avoid high-risk actions: {', '.join(high_risk_actions)}"
            )

        safe_actions = risks.get("safe_actions", [])
        if safe_actions:
            recommendations.append(f"Prefer safe actions: {', '.join(safe_actions)}")

        # Exploration recommendations
        exploration_potential = situation.get("environment_analysis", {}).get(
            "exploration_potential", 0.5
        )
        if exploration_potential > 0.7:
            recommendations.append("High exploration potential in this area")

        return recommendations

    def _estimate_risk_position(
        self, action: str, robot_pos: np.ndarray
    ) -> List[float]:
        """Estimate position of risk based on action type"""
        action_vectors = {
            "move_forward": [1.0, 0, 0],
            "turn_left": [0.5, 0.5, 0],
            "turn_right": [0.5, -0.5, 0],
            "stop": [0, 0, 0],
        }

        vector = action_vectors.get(action, [0.5, 0, 0])
        estimated_pos = robot_pos + np.array(vector) * 2.0  # 2 meters ahead
        return estimated_pos.tolist()

    def _estimate_room_center(self, boundaries: Any) -> np.ndarray:
        """Estimate center of room from boundaries"""
        # Simplified - in real implementation, use actual room geometry
        return np.array([0, 0, 0])

    def _check_clear_path(
        self, start_pos: np.ndarray, end_pos: np.ndarray, objects: List[Dict]
    ) -> bool:
        """Check if clear path exists between two points"""
        # Simplified implementation
        return True  # Assume clear path for now

    def _get_current_room(self, robot_pos: List[float], room_boundaries: Dict) -> str:
        """Determine current room based on robot position"""
        # Simplified - in real implementation, use room boundary checks
        for room_name in room_boundaries.keys():
            return room_name  # Return first room for now
        return "unknown"

    def _check_room_connectivity(self, room1: str, room2: str, map_data: Dict) -> bool:
        """Check if two rooms are connected"""
        # Simplified - assume all rooms are connected for now
        # In real implementation, use door positions and connectivity graph
        return True

    def _combine_analyses(
        self,
        reachability: List,
        search_priority: List,
        risk_zones: List,
        visibility: Dict,
        path_feasibility: Dict,
        room_connectivity: List,
        task_data: Dict,
    ) -> Dict[str, Any]:
        """Combine all analyses into unified spatial intelligence"""

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            reachability, search_priority, risk_zones, room_connectivity, task_data
        )

        return {
            "reachable_objects": reachability,
            "search_priority": search_priority[:5],  # Top 5 priorities
            "risk_zones": risk_zones,
            "visibility_analysis": visibility,
            "path_feasibility": path_feasibility,
            "room_connectivity": room_connectivity,
            "recommended_actions": recommended_actions,
            "analysis_timestamp": time.time(),
        }

    def _generate_recommendations(
        self,
        reachability: List,
        search_priority: List,
        risk_zones: List,
        room_connectivity: List,
        task_data: Dict,
    ) -> List[str]:
        """Generate human-readable recommendations"""
        recommendations = []

        # Search recommendations
        if search_priority:
            top_priority = search_priority[0]
            recommendations.append(
                f"Search {top_priority['name']} first (priority: {top_priority['priority_score']:.1f})"
            )

        # Risk avoidance recommendations
        high_risk_zones = [zone for zone in risk_zones if zone["risk_level"] == "high"]
        if high_risk_zones:
            recommendations.append(f"Avoid {len(high_risk_zones)} high-risk areas")

        # Room exploration recommendations
        unexplored_rooms = [room for room in room_connectivity if not room["explored"]]
        if unexplored_rooms:
            recommendations.append(f"Explore {len(unexplored_rooms)} unexplored rooms")

        # Mission-specific recommendations
        if "find" in task_data["current_mission"].lower():
            reachable_count = len([obj for obj in reachability if obj["accessible"]])
            recommendations.append(
                f"Focus on {reachable_count} reachable objects for search"
            )

        if not recommendations:
            recommendations.append("Continue current exploration pattern")

        return recommendations

    def _save_to_stores(self, spatial_intelligence: Dict[str, Any]):
        """Save spatial intelligence to relevant stores"""
        try:
            # Save to Map Store
            if self.map_store and hasattr(self.map_store, "update_spatial_analysis"):
                self.map_store.update_spatial_analysis(spatial_intelligence)

            # Save to Task Store
            if self.task_store and hasattr(self.task_store, "update_search_strategy"):
                self.task_store.update_search_strategy(
                    {
                        "spatial_intelligence": spatial_intelligence,
                        "timestamp": time.time(),
                        "recommended_search_targets": [
                            obj["name"]
                            for obj in spatial_intelligence["search_priority"][:3]
                        ],
                    }
                )

            print("Spatial intelligence saved to stores")

        except Exception as e:
            print(f"Failed to save spatial intelligence: {e}")

    def _get_fallback_analysis(self) -> Dict[str, Any]:
        """Provide fallback analysis when main analysis fails"""
        return {
            "reachable_objects": [],
            "search_priority": [],
            "risk_zones": [],
            "visibility_analysis": {},
            "path_feasibility": {},
            "room_connectivity": [],
            "recommended_actions": ["Use basic exploration pattern"],
            "analysis_timestamp": time.time(),
            "processing_time": 0.0,
            "fallback": True,
        }


# Test the spatial reasoning integration
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create test instance
    spatial_reasoner = SpatialReasoningIntegration()

    # Test analysis
    intelligence = spatial_reasoner.analyze_spatial_intelligence()

    print("Spatial Intelligence Analysis:")
    print(f"Reachable objects: {len(intelligence['reachable_objects'])}")
    print(
        f"Search priorities: {[obj['name'] for obj in intelligence['search_priority'][:3]]}"
    )
    print(f"Risk zones: {len(intelligence['risk_zones'])}")
    print(f"Recommended actions: {intelligence['recommended_actions']}")
