# ✅ Correct minimal action (NO new code)
# When run_ours_full starts the episode, it must call the already-existing method:
# TaskStore().set_mission_goas(target_objects)
# set_mission_goals() already:
# Stores them in self.mission_goals
# Clears achieved_goals
# Your reasoning pipeline only writes to:
# self.reasoning_plans
# self.current_action_plan
# self.intermediate_reasoning
# ❗ It never touches mission_goal
# ✅ Where reasoning is allowed to read targets (but not write)
# If reasoning needs to know the target object:
# TaskStore().get_remaining_goals()
# or
# TaskStore().mission_goals
# This is read-only by convention, not enforced by code — but that’s fine given your architecture.


# 1. Target Objects Source:
# Both TOT & Spatial: Uses TOT's target_objects as primary (when both available)

# Only TOT: Uses TOT's target_objects

# Only Spatial: Uses spatial priority list names as target_objects

# Fallback: Empty list []

# 2. Target Coordinates Source:
# Both TOT & Spatial: Gets coordinates from spatial priority list (matches by object name)

# Only TOT: NO coordinates (explicitly sets target_coordinates: None)

# Only Spatial: Uses coordinates from spatial priority list

# Fallback: No coordinates (not applicable since no target objects)

# 3. How They're Used:
# python
# # For BOTH components:
# if spatial_priority_list:
#     # Find this object in spatial priority list
#     for priority_item in spatial_priority_list:
#         if priority_item.get("name") == obj or obj in priority_item.get("name", ""):
#             target_coordinates = priority_item.get("coordinates")  # 🎯 SPATIAL COORDS
#             break

# # Planned action includes BOTH:
# planned_actions.append({
#     "parameters": {
#         "target_object": obj,        # 🎯 FROM TOT
#         "target_coordinates": target_coordinates  # 🎯 FROM SPATIAL (or None)
#     }
# })
# 4. Final Fused Plan Spatial Reasoning Source:
# Spatial component provides coordinates in spatial_priority_list

# TOT component provides object names in target_objects

# Fusion matches them by name (name equality or substring matching)


# {
#   "selected_action": "move_fprward",
#   "planned_actions": [{
#     "type": "move_forward",
#     "target_xyz": [1.23, 0.45, 0.10],  // OWL chair position
#     "max_duration": 10.0,              // Seconds
#     "stop_distance": 0.3,              // Success threshold
#     "priority": "high",                 // From spatial reasoning
#     "reasoning": "closest chair from OWL"
#   }]
# }


# 1. SINGLE DESTINATION COORDINATES (Primary Need)


# distance = task_store.current_action_plan.get('planned_actions', [{}])[0].get('parameters', {}).get('distance_to_goal_m')
# distance = (task_store.current_action_plan.get('planned_actions', [{}])[0]
#            .get('parameters', {})
#            .get('distance_to_goal_m', float('inf')))


#!/usr/bin/env python3
"""
REASONING & PLANNING PIPELINE - Integrated reasoning logic and orchestration
Fetches essential data from stores and makes intelligent decisions directly.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

# ✅ FIXED ABSOLUTE IMPORTS
from src.stores.central_map_store import CentralMapStore
from src.stores.task_store import TaskStore
from src.stores.prediction_store import PredictionStore


class ReasoningPipeline:
    """
    Main reasoning and planning pipeline with integrated reasoning logic
    Fetches ONLY essential data and makes intelligent decisions directly
    """

    def __init__(
        self,
        map_store=None,
        task_store=None,
        prediction_store=None,
        user_command_store=None,
        config: Dict[str, Any] = None,
    ):
        self.config = config or {}
        self.trigger_count = 0

        self.map_store = map_store
        self.prediction_store = prediction_store
        self.task_store = task_store
        self.user_command_store = user_command_store

        # ✅ MODIFY THIS: Updated component initialization
        self.spatial_reasoning = None
        self.tree_of_thoughts = (
            None  # ✅ Tree of Thoughts now handles all LLM reasoning
        )

        # Reasoning parameters (moved from ReasoningEngine)
        self.confidence_threshold = 0.7
        self.risk_tolerance = 0.4
        self.exploration_bias = 0.3

        self.prediction_store.subscribe(self.on_fused_predictions_ready)

        # State memory (moved from ReasoningEngine)
        self.previous_states = []
        self.decision_history = []
        self.max_history = 20

        # Pipeline state
        self.is_running = False
        self.is_initialized = False

        # Decision cache
        self._last_decision = None
        self._last_decision_time = 0
        self._reasoning_interval = 3  # frames between reasoning cycles

        print("Integrated ReasoningPipeline instance created")

    def initialize_processing(self):
        """Initialize reasoning and planning components"""
        print("Initializing integrated reasoning pipeline...")

        try:
            # ✅ FIX: Initialize Spatial Reasoning CORRECTLY
            try:
                from src.reasoning_planning_pipeline.spatial_reasoning_integration import (
                    SpatialReasoningIntegration,
                )

                self.spatial_reasoning = SpatialReasoningIntegration(  # ✅ CORRECT: assign to spatial_reasoning
                    map_store=self.map_store,
                    prediction_store=self.prediction_store,
                    task_store=self.task_store,
                    user_command_store=self.user_command_store,
                )
                print("✅ Spatial Reasoning Integration initialized")
            except ImportError as e:
                print(f"Spatial Reasoning Integration not available: {e}")
                self.spatial_reasoning = None

            # ✅ FIX: Initialize Tree of Thoughts CORRECTLY
            try:
                from src.reasoning_planning_pipeline.tree_of_thoughts_integration import (
                    TreeOfThoughtsIntegration,
                )

                self.tree_of_thoughts = TreeOfThoughtsIntegration(
                    map_store=self.map_store,
                    prediction_store=self.prediction_store,
                    task_store=self.task_store,
                    user_command_store=self.user_command_store,
                    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                )
                print("✅ Tree of Thoughts Integration initialized")
            except ImportError as e:
                print(f"Tree of Thoughts Integration not available: {e}")
                self.tree_of_thoughts = None

            # ✅ ADD DEBUG OUTPUT
            print(f"🔧 INITIALIZATION DEBUG:")
            print(f"   Spatial Reasoning: {self.spatial_reasoning is not None}")
            print(f"   Tree of Thoughts: {self.tree_of_thoughts is not None}")

            self.is_initialized = True
            self.is_running = False

            # Update task store
            self.task_store.update_reasoning_status("initialized")

            print("✅ Integrated reasoning pipeline initialized successfully")
            return True

        except Exception as e:
            print(f"❌ Reasoning pipeline initialization failed: {e}")
            self.is_initialized = False
            return False

    def on_fused_predictions_ready(self, event_type: str, data: Dict):
        """Auto-called when new fused predictions are available"""
        if event_type == "fused_predictions_ready":
            prediction_id = data["prediction_id"]
            #
            # ✅ INCREMENT COUNTER EXACTLY LIKE ACTION PIPELINE
            self.trigger_count += 1

            # ✅ SIMPLE TRIGGER CHECK
            if self.trigger_count == 1:
                print(
                    f"📊 ReasoningPipeline: Trigger count: {self.trigger_count} - running reasoning cycle"
                )
                # Auto-trigger reasoning cycle
                reasoning_cycle_id = data.get("reasoning_cycle_id", 0)
                self.run_reasoning_cycle()
            else:
                print(
                    f"📊 ReasoningPipeline: Trigger count: {self.trigger_count} - already processed, skipping"
                )

    def _get_fallback_reasoning(self) -> Dict:
        """Fallback reasoning when main logic fails"""
        return {
            "timestamp": time.time(),
            "situation_analysis": {"current_location": "unknown"},
            "risk_assessment": {"action_risks": {}, "overall_risk_level": 0.5},
            "goal_progress": {"progress": 0.0, "goal_reached": False},
            "confidence": 0.1,
            "recommendations": ["Use caution - reasoning system degraded"],
            "constraints_violated": [],
            "opportunities_found": [],
            "is_fallback": True,
        }

    def get_reasoning_stats(self) -> Dict:
        """Get reasoning engine statistics"""
        return {
            "total_decisions": len(self.decision_history),
            "average_confidence": (
                np.mean(
                    [r["results"].get("confidence", 0) for r in self.decision_history]
                )
                if self.decision_history
                else 0
            ),
            "recent_risk_level": (
                self.decision_history[-1]["results"]
                .get("risk_assessment", {})
                .get("overall_risk_level", 0.5)
                if self.decision_history
                else 0.5
            ),
        }

    # ==================== PIPELINE ORCHESTRATION ====================

    def run_reasoning_cycle(self) -> Optional[Dict]:
        """
        Single unified reasoning cycle with task status tracking
        Uses reasoning_cycle_id exclusively and retrieves complete Tree of Thoughts data
        """

        # 1. 🎯 Generate single reasoning cycle ID
        reasoning_cycle_id = f"reason_{int(time.time()*1000)}"

        self.task_store.write_reasoning_plan(
            reasoning_cycle_id,
            {
                "reasoning_cycle_id": reasoning_cycle_id,
                "timestamp": time.time(),
                "status": "initialized",
                "source": "reasoning_pipeline",
            },
        )

        print(f"🧠 Starting reasoning cycle: {reasoning_cycle_id}")

        try:
            # 2. 🎯 Run ALL available reasoning components IN PARALLEL
            # ✅ ADD THESE LINES INSTEAD:
            all_component_data = {}

            # Run Tree of Thoughts (if available)
            if self.tree_of_thoughts:
                print("  🧠 Running Tree of Thoughts sequentially...")
                tree_result = self.tree_of_thoughts.generate_comprehensive_plan(
                    reasoning_cycle_id
                )
                if tree_result:
                    all_component_data["tree_of_thoughts"] = tree_result
                    print("  ✅ Tree of Thoughts completed")
                    print(
                        f"  🔍 DEBUG: Tree of Thoughts output: {tree_result}"
                    )  # << ADD THIS

            # Run Spatial Reasoning (if available)
            if self.spatial_reasoning:
                print("  🗺️ Running Spatial Reasoning sequentially...")
                spatial_result = self.spatial_reasoning.analyze_environment(
                    reasoning_cycle_id
                )
                if spatial_result:
                    all_component_data["spatial_reasoning"] = spatial_result
                    print("  ✅ Spatial Reasoning completed")
                    print(
                        f"  🔍 DEBUG: Spatial Reasoning output: {spatial_result}"
                    )  # << ADD THIS

            print(
                f"📊 ReasoningPipeline: Components completed - {list(all_component_data.keys())}"
            )
            # 🎯 NOW check storage - threads have finished
            # all_component_data = {}

            # Check intermediate_reasoning for ALL components
            if hasattr(self.task_store, "intermediate_reasoning"):
                cycle_data = self.task_store.intermediate_reasoning.get(
                    reasoning_cycle_id, {}
                )
                component_reasoning = cycle_data.get("component_reasoning", {})

                # Check for Tree of Thoughts data
                if "tree_of_thoughts" in component_reasoning:
                    all_component_data["tree_of_thoughts"] = component_reasoning[
                        "tree_of_thoughts"
                    ]
                    print(f"✅ Retrieved Tree of Thoughts from intermediate_reasoning")
                else:
                    print(
                        f"❌ No Tree of Thoughts data in intermediate_reasoning for {reasoning_cycle_id}"
                    )

                # Check for Spatial Reasoning data
                if "spatial_reasoning" in component_reasoning:
                    all_component_data["spatial_reasoning"] = component_reasoning[
                        "spatial_reasoning"
                    ]
                    print(f"✅ Retrieved Spatial Reasoning from intermediate_reasoning")
                else:
                    print(
                        f"❌ No Spatial Reasoning data in intermediate_reasoning for {reasoning_cycle_id}"
                    )
            else:
                print(f"❌ No intermediate_reasoning storage available in TaskStore")

            print(
                f"📊 ReasoningPipeline: Retrieved data from {len(all_component_data)} components: {list(all_component_data.keys())}"
            )

            # 5. 🎯 Fuse ALL component reasoning
            fused_reasoning = self._fuse_all_component_reasoning(all_component_data)

            print(
                f"🎯 ReasoningPipeline: Fusion logic = {fused_reasoning.get('fusion_logic')}, Action = {fused_reasoning.get('selected_action')}, Confidence = {fused_reasoning.get('final_confidence', 0.5):.2f}"
            )

            # 6. 🎯 Create final executable action plan WITH STATUS
            final_action_plan = {
                "reasoning_cycle_id": reasoning_cycle_id,
                "timestamp": time.time(),
                "selected_action": fused_reasoning.get("selected_action", "stop"),
                "planned_actions": fused_reasoning.get("planned_actions", []),
                "confidence": fused_reasoning.get("final_confidence", 0.5),
                "reasoning_source": fused_reasoning.get("fusion_logic", "unknown"),
                "components_used": list(all_component_data.keys()),
                "component": "final_action",  # ✅ THIS IS ABSOLUTELY REQUIRED
                # 🆕 TASK STATUS TRACKING
                "task_status": "pending_execution",
                "execution_attempts": 0,
                "last_execution_attempt": None,
                "completion_status": "not_started",
                "execution_metadata": {
                    "requires_confirmation": fused_reasoning.get(
                        "final_confidence", 0.5
                    )
                    < 0.7,
                    "estimated_duration": "pending",
                    "prerequisites_met": True,
                    "safety_clearance": "pending",
                },
            }

            # 🎯 CRITICAL FIX: REMOVE THE PROBLEMATIC LINE
            # ❌ DELETE THIS: self.task_store.append_fused_reasoning(reasoning_cycle_id, complete_reasoning_data)

            # 🎯 ONLY USE write_reasoning_plan - IT HANDLES EVERYTHING
            print(f"💾 Saving action plan via write_reasoning_plan...")
            self.task_store.write_reasoning_plan(reasoning_cycle_id, final_action_plan)

            print(
                f"🧠 Reasoning cycle {reasoning_cycle_id} completed - "
                f"Action: {final_action_plan['selected_action']} "
                f"(Status: {final_action_plan['task_status']}) "
                f"(Confidence: {final_action_plan['confidence']:.2f})"
            )

            # 🎯 RETURN action plan
            return {
                "action_plan": final_action_plan,
                "reasoning_cycle_id": reasoning_cycle_id,
            }

        except Exception as e:
            print(f"❌ Reasoning cycle {reasoning_cycle_id} failed: {e}")
            # Emergency fallback WITH ERROR STATUS
            emergency_plan = {
                "reasoning_cycle_id": reasoning_cycle_id,
                "timestamp": time.time(),
                "selected_action": "explore",
                "planned_actions": [],
                "confidence": 0.1,
                "reasoning_source": "emergency_fallback",
                "components_used": [],
                "task_status": "reasoning_failed",
                "execution_attempts": 0,
                "last_execution_attempt": None,
                "completion_status": "failed",
                "execution_metadata": {
                    "requires_confirmation": True,
                    "estimated_duration": 0,
                    "prerequisites_met": False,
                    "safety_clearance": "rejected",
                    "error_reason": str(e),
                },
            }
            self.task_store.write_reasoning_plan(reasoning_cycle_id, emergency_plan)

            return emergency_plan

    def _fuse_all_component_reasoning(self, all_component_data: Dict) -> Dict:
        """Fuse component reasoning with execution readiness - BACKWARD COMPATIBLE"""

        print("=" * 80)
        print("🔍 FUSION DEBUG: INPUT COMPONENT DATA")
        print(f"All component keys: {list(all_component_data.keys())}")
        print("=" * 80)

        tree_data = all_component_data.get("tree_of_thoughts", {})
        spatial_data = all_component_data.get("spatial_reasoning", {})

        # 1️⃣ CASE: BOTH COMPONENTS AVAILABLE
        if tree_data and spatial_data:
            print("✅ FUSING BOTH: tree_of_thoughts + spatial_reasoning")

            # ✅ BACKWARD COMPATIBLE: Extract existing data structure
            # Tree of Thoughts data extraction (from detailed_plan)
            detailed_plan = tree_data.get("detailed_plan", {})
            target_objects = detailed_plan.get("target_objects", [])
            if isinstance(target_objects, tuple) and len(target_objects) > 0:
                target_objects = list(target_objects[0])  # Extract inner list

            search_pattern = detailed_plan.get(
                "search_pattern", "proximity_first_pattern"
            )

            # ✅ Spatial Reasoning data extraction
            spatial_priority_list = spatial_data.get("search_strategy", {}).get(
                "priority_list", []
            )
            top_priority = spatial_data.get("search_strategy", {}).get(
                "top_priority", {}
            )

            # ✅ CANONICAL SPATIAL OBJECT MAP
            # ✅ FIXED - SAFE VERSION:
            spatial_object_map = {}
            for i, item in enumerate(spatial_priority_list):
                name = item.get("name")
                if name:
                    safe_name = str(name) if isinstance(name, (list, tuple)) else name
                    spatial_object_map[safe_name] = item
            print(f"Spatial Object Map: {spatial_object_map}")

            # 🎯 DEBUG: Check if spatial has coordinates
            print("🔍 SPATIAL DATA CHECK:")
            if spatial_priority_list:
                print(f"  • Priority list items: {len(spatial_priority_list)}")
                for i, item in enumerate(spatial_priority_list[:3]):  # Show first 3
                    has_coords = "coordinates" in item
                    print(f"  • Item {i}: {item.get('name', 'unnamed')}")
                    print(f"    - Has coordinates: {has_coords}")
                    if has_coords:
                        print(f"    - Coordinates: {item.get('coordinates')}")
            else:
                print("  • No spatial priority list")

            # ✅ Build RICH planned_actions while keeping backward compatibility
            planned_actions = []

            # OPTION 1: If spatial has priorities, add exploration actions
            # OPTION 1: If spatial has priorities, add exploration actions
            if spatial_priority_list:
                for priority_item in spatial_priority_list[:2]:
                    obj_name = priority_item.get("name", "")
                    obj_coordinates = priority_item.get(
                        "coordinates"
                    )  # 🎯 GET SEMANTIC COORDINATES

                    # 🎯 OVERRIDE WITH UMEYAMA COORDINATES IF AVAILABLE
                    if obj_name and self.task_store and hasattr(self.task_store, 'get_umeyama_aligned_goal_by_name'):
                        try:
                            umeyama_goals = self.task_store.get_umeyama_aligned_goal_by_name(obj_name)
                            if umeyama_goals and umeyama_goals[0].get("world_coordinates", [0, 0, 0]) != [0, 0, 0]:
                                obj_coordinates = umeyama_goals[0]["world_coordinates"]
                                print(f"🎯 Overriding with Umeyama coordinates for {obj_name}: {obj_coordinates}")
                        except:
                            pass  # Keep semantic coordinates if Umeyama fails

                    if obj_name:
                        planned_actions.append(
                            {
                                "type": "move_forward",
                                "parameters": {
                                    "target_object": obj_name,
                                    "target_coordinates": obj_coordinates,  # ← NOW COULD BE UMEYAMA!
                                    "priority_score": priority_item.get(
                                        "priority_score", 0
                                    ),
                                    "source": "spatial_reasoning",
                                    "distance_to_goal_m": priority_item.get(
                                        "distance_to_goal_m"
                                    ),
                                },
                            }
                        )

            # OPTION 2: If we have target objects, create rich actions
            if target_objects:
                for obj in target_objects:  # Limit to 3 for stability
                    # 🎯 CHECK IF SPATIAL HAS COORDINATES FOR THIS OBJECT
                    target_coordinates = None
                    # 🎯 CANONICAL LOOKUP FIRST
                    spatial_item = spatial_object_map.get(obj)

                    # 🎯 FALLBACK SUBSTRING MATCH (LAST RESORT)
                    if not spatial_item:
                        for name, item in spatial_object_map.items():
                            if obj in name or name in obj:
                                spatial_item = item
                                break

                    target_coordinates = (
                        spatial_item.get("coordinates") if spatial_item else None
                    )
                    
                                        # 🎯 ADD THIS: OVERRIDE WITH UMEYAMA COORDINATES
                    if obj and self.task_store and hasattr(self.task_store, 'get_umeyama_aligned_goal_by_name'):
                        try:
                            umeyama_goals = self.task_store.get_umeyama_aligned_goal_by_name(obj)
                            if umeyama_goals and umeyama_goals[0].get("world_coordinates", [0, 0, 0]) != [0, 0, 0]:
                                target_coordinates = umeyama_goals[0]["world_coordinates"]
                                print(f"🎯 Overriding TOT target with Umeyama coordinates for {obj}: {target_coordinates}")
                        except:
                            pass  # Keep existing coordinates if Umeyama fails


                    if target_coordinates:
                        print(f"🎯 Found coordinates for {obj}: {target_coordinates}")

                    planned_actions.append(
                        {
                            "type": "move_forward",  # Generic type for backward compatibility
                            "action": "move_forward",  # Rich action type in parameters
                            "parameters": {
                                "target_object": obj,
                                "search_pattern": search_pattern,
                                "source": "tree_of_thoughts",
                                "target_coordinates": target_coordinates,  # 🎯 ADD THIS (could be None)
                                "distance_to_goal_m": (
                                    spatial_item.get("distance_to_goal_m")
                                    if spatial_item
                                    else None
                                ),  # ✅ ADD THIS
                            },
                        }
                    )

            # OPTION 3: Fallback to simple move_forward (backward compatibility)
            # 🎯 Prefer actions WITH coordinates

            if not planned_actions:
                planned_actions.append(
                    {
                        "type": "move_forward",
                        "parameters": {"source": "safety_fallback"},
                    }
                )

            action_with_coords = next(
                (
                    a
                    for a in planned_actions
                    if a.get("parameters", {}).get("target_coordinates") is not None
                ),
                None,
            )

            selected_action = (
                action_with_coords if action_with_coords else planned_actions[0]
            )

            # ✅ BACKWARD COMPATIBLE confidence calculation
            tree_confidence = tree_data.get("strategy_decision", {}).get(
                "confidence", 0.5
            )
            spatial_safe_paths = spatial_data.get("summary_metrics", {}).get(
                "safe_navigation_paths", 0
            )
            spatial_confidence = min(0.5, spatial_safe_paths * 0.2)

            final_confidence = (tree_confidence * 0.7) + (spatial_confidence * 0.3)

            fused_result = {
                "selected_action": selected_action,  # ✅ Backward compatible
                "planned_actions": planned_actions,  # ✅ Now has rich data!
                "final_confidence": final_confidence,  # ✅ Backward compatible
                "components_contributing": [
                    "tree_of_thoughts",
                    "spatial_reasoning",
                ],  # ✅ Backward compatible
                "execution_readiness": (
                    "ready" if final_confidence > 0.6 else "needs_confirmation"
                ),  # ✅ Backward compatible
                # ✅ ADD RICH DATA IN EXISTING FIELD (backward compatible)
                "plan_metadata": {
                    "target_objects": target_objects,
                    "search_pattern": search_pattern,
                    "spatial_priorities": [
                        item.get("name") for item in spatial_priority_list[:3]
                    ],
                    "top_priority": (
                        top_priority.get("name")
                        if isinstance(top_priority, dict)
                        else None
                    ),
                },
                "goal_distances": {  # ✅ ADD THIS FOR BACKWARD COMPATIBILITY
                    item.get("name"): item.get("distance_to_goal_m")
                    for item in spatial_priority_list[:5]
                    if item.get("distance_to_goal_m") is not None
                },
                "has_coordinates": any(
                    "coordinates" in item for item in spatial_priority_list
                ),  # 🎯 NEW
                "target_objects": target_objects,
            }

            print(f"✅ FUSED BOTH: Created {len(planned_actions)} rich actions")

        # 2️⃣ CASE: ONLY TOT AVAILABLE (backward compatible)
        elif tree_data:
            print("⚠️ FUSING: Only tree_of_thoughts available")

            detailed_plan = tree_data.get("detailed_plan", {})
            target_objects = detailed_plan.get("target_objects", [])

            # Build actions
            planned_actions = []
            if target_objects:
                for obj in target_objects:
                    # 🎯 TOT doesn't have coordinates - set to None
                    planned_actions.append(
                        {
                            "type": "move_forward",
                            "parameters": {
                                "target_object": obj,
                                "target_coordinates": None,  # 🎯 EXPLICITLY NO COORDINATES
                                "source": "tree_of_thoughts",
                            },
                        }
                    )
            else:
                planned_actions = [{"type": "move_forward", "parameters": {}}]

            action = planned_actions[0].get("type", "move_forward")
            confidence = tree_data.get("strategy_decision", {}).get("confidence", 0.5)

            fused_result = {
                "selected_action": action,
                "planned_actions": planned_actions,
                "final_confidence": confidence,
                "components_contributing": ["tree_of_thoughts"],
                "execution_readiness": (
                    "ready" if confidence > 0.6 else "needs_confirmation"
                ),
                "target_objects": target_objects,
            }

        # 3️⃣ CASE: ONLY SPATIAL REASONING (backward compatible)
        elif spatial_data:
            print("⚠️ FUSING: Only spatial_reasoning available")

            spatial_priority_list = spatial_data.get("search_strategy", {}).get(
                "priority_list", []
            )

            planned_actions = []
            if spatial_priority_list:
                for priority_item in spatial_priority_list:
                    obj_name = priority_item.get("name", "")
                    obj_coordinates = priority_item.get(
                        "coordinates"
                    )  # 🎯 GET COORDINATES!
                    
                            
                            
                                        # 🎯 ADD THIS: Override with Umeyama
                    if obj_name and self.task_store and hasattr(self.task_store, 'get_umeyama_aligned_goal_by_name'):
                        try:
                            umeyama_goals = self.task_store.get_umeyama_aligned_goal_by_name(obj_name)
                            if umeyama_goals and umeyama_goals[0].get("world_coordinates", [0, 0, 0]) != [0, 0, 0]:
                                obj_coordinates = umeyama_goals[0]["world_coordinates"]
                                print(f"🎯 Overriding spatial-only with Umeyama coordinates for {obj_name}")
                        except:
                            pass


                    if obj_name:
                        planned_actions.append(
                            {
                                "type": "move_forward",
                                "parameters": {
                                    "target_object": obj_name,
                                    "target_coordinates": obj_coordinates,  # 🎯 ADD COORDINATES HERE!
                                    "priority_score": priority_item.get(
                                        "priority_score", 0
                                    ),
                                    "source": "spatial_reasoning",
                                    "distance_to_goal_m": priority_item.get(
                                        "distance_to_goal_m"
                                    ),  # ✅ ADD THIS
                                },
                            }
                        )

            if not planned_actions:
                planned_actions = [{"type": "move_forward", "parameters": {}}]

            spatial_safe_paths = spatial_data.get("summary_metrics", {}).get(
                "safe_navigation_paths", 0
            )
            spatial_confidence = min(0.5, spatial_safe_paths * 0.2)

            target_objects = [
                item.get("name") for item in spatial_priority_list if "name" in item
            ]

            fused_result = {
                "selected_action": planned_actions[0].get("type", "move_forward"),
                "planned_actions": planned_actions,
                "final_confidence": spatial_confidence,
                "components_contributing": ["spatial_reasoning"],
                "execution_readiness": (
                    "cautious" if spatial_confidence > 0.4 else "needs_confirmation"
                ),
                "target_objects": target_objects,
            }

            # CHANGE TO:
        # 4️⃣ EMERGENCY FALLBACK with 50+ actions
        else:
            print(
                "❌ FUSING: No component data available - creating 50+ exploration actions"
            )

            # 🎯 MOCK SPATIAL TARGETS (FALLBACK MUST NEVER BE BLIND)
            mock_targets = [
                {"name": "fallback_target_1", "coordinates": [1.0, 0.0, 0.0]},
                {"name": "fallback_target_2", "coordinates": [0.0, 0.0, 1.0]},
                {"name": "fallback_target_3", "coordinates": [-1.0, 0.0, 0.0]},
            ]

            # Create 50+ varied exploration actions
            planned_actions = []

            # Add 20 move_forward actions
            for i in range(20):
                planned_actions.append(
                    {
                        "type": "move_forward",
                        "action": "move_forward",
                        "parameters": {
                            "distance": 0.3,
                            "exploration_step": i + 1,
                            "source": "fallback_exploration",
                        },
                    }
                )

            # Add 15 turn_left actions
            for t in mock_targets:
                planned_actions.append(
                    {
                        "type": "move_forward",
                        "parameters": {
                            "target_object": t["name"],
                            "target_coordinates": t["coordinates"],
                            "source": "fallback_mock_spatial",
                        },
                    }
                )

            # Add 15 turn_right actions
            for i in range(15):
                planned_actions.append(
                    {
                        "type": "turn_right",
                        "action": "turn_right",
                        "parameters": {
                            "angle": 15,
                            "reason": "exploration_scan",
                            "step": i + 1,
                        },
                    }
                )

            target_objects = [t["name"] for t in mock_targets]

            fused_result = {
                "selected_action": "explore_pattern",
                "planned_actions": planned_actions,
                "final_confidence": 0.3,
                "components_contributing": ["fallback_exploration"],
                "execution_readiness": "ready",
                "target_objects": target_objects,
            }

            print(f"🔄 Created {len(planned_actions)} fallback exploration actions")

        print("=" * 80)
        print(f"🔍 FUSION DEBUG: FINAL FUSED RESULT {fused_result}")
        print(f"  Selected action: {fused_result.get('selected_action')}")
        print(f"  Planned actions: {len(fused_result.get('planned_actions', []))}")
        print("=" * 80)

        return fused_result

    def shutdown(self):
        """Shutdown reasoning pipeline"""
        self.is_running = False

        try:
            # Shutdown integration components if they exist
            if self.spatial_reasoning and hasattr(self.spatial_reasoning, "shutdown"):
                self.spatial_reasoning.shutdown()
            if self.tree_of_thoughts and hasattr(self.tree_of_thoughts, "shutdown"):
                self.tree_of_thoughts.shutdown()

            self.task_store.update_reasoning_status("stopped")
            print("Integrated reasoning pipeline shutdown complete")

        except Exception as e:
            print(f"Error during reasoning shutdown: {e}")

        # Simple standalone test


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ✅ INITIALIZE ALL REQUIRED STORES FIRST
    print("🎯 Initializing stores for reasoning pipeline...")

    # 1. Central Map Store
    central_map_store = CentralMapStore.get_global()
    print("✅ CentralMapStore initialized")

    # 2. Prediction Store
    prediction_store = PredictionStore()
    print("✅ PredictionStore initialized")

    # 3. Task Store
    task_store = TaskStore()
    print("✅ TaskStore initialized")

    # 4. User Command Store - either create it or set to None
    try:
        # Try to import and create UserCommandStore if it exists
        from src.stores.user_command_store import UserCommandStore

        user_command_store = UserCommandStore()
        print("✅ UserCommandStore initialized")
    except ImportError:
        # If UserCommandStore doesn't exist, set to None
        user_command_store = None
        print("⚠️ UserCommandStore not available - setting to None")

    # ✅ CORRECT: Pass all required stores (user_command_store is now defined)
    pipeline = ReasoningPipeline(
        map_store=central_map_store,
        prediction_store=prediction_store,
        task_store=task_store,
        user_command_store=user_command_store,  # This is now defined
    )

    if pipeline.initialize_processing():
        try:
            # Run for a few cycles
            for i in range(10):
                result = pipeline.run_reasoning_cycle()
                if result:
                    print(
                        f"Cycle {i}: Selected action: {result['selected_action']} "
                        f"(Confidence: {result['confidence']:.2f})"
                    )
                time.sleep(0.5)
        finally:
            pipeline.shutdown()
    else:
        print("Failed to initialize reasoning pipeline")
