#!/usr/bin/env python3
"""
COLLISION RISK PREDICTOR
Pure geometric safety prediction using depth data and semantic objects
Outputs risk scores for navigation actions
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any
from src.stores.habitat_store import global_habitat_store
from src.stores.prediction_store import PredictionType  # Or


class CollisionRiskPredictor:
    """
    Predicts collision risks for navigation actions using:
    - Depth data for immediate geometric safety
    - Semantic objects for known obstacle risks
    - Robot position for coordinate transformation
    """

    def __init__(
        self,
        map_store=None,
        task_store=None,
        prediction_store=None,
        config: Dict[str, Any] = None,
    ):
        self.config = config or {}
        self.logger = logging.getLogger("CollisionRiskPredictor")

        # Store connections
        self.map_store = map_store
        self.task_store = task_store
        self.prediction_store = prediction_store
        
            # 🎯 SINGLE FETCH: Get simulator ONCE at initialization
        self.habitat_simulator = None
        self.agent = None
      
 

        # Safety parameters
        self.robot_radius = self.config.get("robot_radius", 0.3)
        self.safety_buffer = self.config.get("safety_buffer", 0.5)
        self.min_safe_distance = self.config.get("min_safe_distance", 0.8)
        self.emergency_stop_distance = self.config.get("emergency_stop_distance", 0.4)

        # Risk thresholds
        self.risk_levels = {"low": 0.3, "medium": 0.6, "high": 0.8}

        self.is_initialized = True
        self.logger.info("Collision Risk Predictor initialized")

    def _distance_based_risk(
        self, distance: float, obj_class: str, confidence: float = 1.0
    ) -> float:
        """Calculate risk based on distance and object class"""
        # Base risk by object class
        class_risk = {
            "wall": 0.9,
            "furniture": 0.7,
            "person": 0.8,
            "door": 0.5,
            "window": 0.3,
            "floor": 0.1,
            "ceiling": 0.1,
        }.get(obj_class, 0.5)

        # Distance-based attenuation
        if distance < self.emergency_stop_distance:
            distance_factor = 1.0
        elif distance < self.min_safe_distance:
            distance_factor = 0.8
        elif distance < 2.0:
            distance_factor = 0.5
        elif distance < 5.0:
            distance_factor = 0.3
        else:
            distance_factor = 0.1

        return class_risk * distance_factor * confidence

    def _compute_immediate_geometric_risks(self, point_cloud: np.ndarray, robot_position: list) -> Dict:
        """Calculate immediate collision risks from ORB-SLAM 3D points"""
        if point_cloud is None or len(point_cloud) == 0:
            return {
                "immediate_danger": False,
                "closest_obstacle": float("inf"),
                "risk_score": 0.0,
            }

        try:
            # Calculate distances from robot to all points
            robot_pos = np.array(robot_position)
            distances = np.linalg.norm(point_cloud - robot_pos, axis=1)
            closest_distance = np.min(distances)
            
            # Your existing risk logic (unchanged)
            immediate_danger = closest_distance < self.robot_radius + self.safety_buffer

            if closest_distance < self.emergency_stop_distance:
                risk_score = 0.9
            elif closest_distance < self.min_safe_distance:
                risk_score = 0.6
            else:
                risk_score = 0.3

            return {
                "immediate_danger": immediate_danger,
                "closest_obstacle": float(closest_distance),
                "risk_score": risk_score,
                "safe_navigation_zone": closest_distance > self.min_safe_distance,
                "points_analyzed": len(point_cloud)  # Bonus metric
            }

        except Exception as e:
            self.logger.warning(f"Point cloud processing failed: {e}")
            return {
                "immediate_danger": False,
                "closest_obstacle": float("inf"),
                "risk_score": 0.0,
            }

    def _compute_semantic_object_risks(
        self, semantic_objects: Dict, robot_position: list
    ) -> Dict:
        """Calculate risks from known semantic objects"""
        high_risk_zones = []
        medium_risk_zones = []

        robot_pos_array = np.array(robot_position)

        for obj_id, obj in semantic_objects.items():
            obj_position = obj.get("position_3d")
            if not obj_position:
                continue

            # Calculate distance to object
            obj_pos_array = np.array(obj_position)
            distance = np.linalg.norm(obj_pos_array - robot_pos_array)

            # Skip objects too far away
            if distance > 5.0:
                continue

            # Calculate risk score
            risk_score = self._distance_based_risk(
                distance, obj.get("label", "unknown"), obj.get("confidence", 1.0)
            )

            # Relative position (robot frame: x=forward, y=left, z=up)
            position_relative = (obj_pos_array - robot_pos_array).tolist()

            zone_data = {
                "position_relative": position_relative,
                "distance": float(distance),
                "object_type": obj.get("label", "unknown"),
                "risk_score": risk_score,
            }

            # Categorize by risk level
            if risk_score > self.risk_levels["high"]:
                high_risk_zones.append(zone_data)
            elif risk_score > self.risk_levels["medium"]:
                medium_risk_zones.append(zone_data)

        return {
            "high_risk_zones": high_risk_zones,
            "medium_risk_zones": medium_risk_zones,
        }

    def _get_planned_actions(self) -> List[str]:
        """Get planned navigation actions from TaskStore"""
        default_actions = ["move_forward", "turn_left", "turn_right"]

        if not self.task_store:
            return default_actions

        try:
            current_task = self.task_store.get_current_task()
            if (
                hasattr(current_task, "planned_actions")
                and current_task.planned_actions
            ):
                return current_task.planned_actions
            return default_actions
        except Exception as e:
            self.logger.debug(f"Could not get planned actions: {e}")
            return default_actions

    def _calculate_action_risks(
        self, planned_actions: List[str], immediate_risks: Dict, semantic_risks: Dict
    ) -> Dict:
        """Calculate risk scores for each planned navigation action"""
        action_risks = {}

        for action in planned_actions:
            base_risk = immediate_risks.get("risk_score", 0.0)

            # Adjust risk based on action direction
            if action == "move_forward":
                # Higher risk from objects in front
                front_risks = [
                    zone
                    for zone in semantic_risks.get("high_risk_zones", [])
                    if zone["position_relative"][0] > 0
                ]  # Positive x = forward
                action_risk = base_risk + (0.2 * len(front_risks))

            elif action == "turn_left":
                # Higher risk from objects on right side
                right_risks = [
                    zone
                    for zone in semantic_risks.get("high_risk_zones", [])
                    if zone["position_relative"][1] < 0
                ]  # Negative y = right
                action_risk = base_risk + (0.1 * len(right_risks))

            elif action == "turn_right":
                # Higher risk from objects on left side
                left_risks = [
                    zone
                    for zone in semantic_risks.get("high_risk_zones", [])
                    if zone["position_relative"][1] > 0
                ]  # Positive y = left
                action_risk = base_risk + (0.1 * len(left_risks))

            else:
                action_risk = base_risk

            # Clamp risk between 0 and 1
            action_risk = max(0.0, min(1.0, action_risk))

            # Determine risk level and feasibility
            if action_risk < self.risk_levels["low"]:
                risk_level = "low"
                feasible = True
                speed = "normal"
                factors = ["clear_path"]
            elif action_risk < self.risk_levels["medium"]:
                risk_level = "medium"
                feasible = True
                speed = "caution"
                factors = ["moderate_obstacles"]
            else:
                risk_level = "high"
                feasible = action_risk < 0.9  # Only infeasible if extremely risky
                speed = "slow" if feasible else "stop"
                factors = ["high_risk_obstacles"]

            action_risks[action] = {
                "risk_score": action_risk,
                "risk_level": risk_level,
                "feasible": feasible,
                "primary_risk_factors": factors,
                "recommended_speed": speed,
            }

        return action_risks
    
    
    def _get_robot_position(self) -> list:
        """Get robot position from ORB-SLAM or fallback"""
        try:
            # Try to get from ORB-SLAM map store
            if self.map_store and hasattr(self.map_store, 'current_pose'):
                pose = self.map_store.current_pose
                if pose and hasattr(pose, 'position'):
                    return list(pose.position)  # ORB-SLAM estimated position
        except:
            pass
        
        return [0, 0, 0]  # Fallback
            
    def predict_collision_risk_map(self, prediction_id: str):
        """Main prediction method - follows the 5-step flow"""
        print("🚧 CollisionRiskPredictor: Starting prediction...")
        print(f"  📍 Frame start time: {time.strftime('%H:%M:%S')}")

        start_time = time.time()

        try:
            print("🚧 CollisionRiskPredictor: Step 1 - Getting prediction ID...")
            print(f"🚧 CollisionRiskPredictor: Prediction ID = {prediction_id}")

            if not prediction_id:
                print("❌ CollisionRiskPredictor: No prediction ID - returning fallback")
                return self._create_fallback_result()

            # 2. FETCH DATA FROM STORES
            print("🚧 CollisionRiskPredictor: Step 2 - Fetching data from stores...")
            
            robot_position = self._get_robot_position()  # Use Habitat pose
            print(f"  📍 Robot position: {robot_position}")
            
            semantic_objects = (
                self.map_store.semantic_data.get("semantic_objects", {})
                if self.map_store
                else {}
            )
            print(f"  📊 Semantic objects: {len(semantic_objects)} found")
            
            # Get 3D points around robot from ORB-SLAM map
            print("  🔍 Extracting ORB-SLAM map points...")
            point_cloud = self._get_map_points_around_robot(robot_position, radius=3.0)
            print(f"  📊 ORB-SLAM points: {len(point_cloud) if point_cloud is not None else 0} points")
            
            planned_actions = self._get_planned_actions()
            print(f"  🎯 Planned actions: {planned_actions}")

            # 3. PROCESS RISKS
            print("🚧 CollisionRiskPredictor: Step 3 - Processing risks...")
            
            print("  📏 Computing immediate geometric risks...")
            immediate_risks = self._compute_immediate_geometric_risks(
                point_cloud, robot_position
            )
            print(f"    Immediate danger: {immediate_risks.get('immediate_danger', False)}")
            print(f"    Closest obstacle: {immediate_risks.get('closest_obstacle', 'inf'):.2f}m")
            print(f"    Risk score: {immediate_risks.get('risk_score', 0.0):.2f}")
            
            print("  🏷️ Computing semantic object risks...")
            semantic_risks = self._compute_semantic_object_risks(
                semantic_objects, robot_position
            )
            high_risk_zones = len(semantic_risks.get("high_risk_zones", []))
            medium_risk_zones = len(semantic_risks.get("medium_risk_zones", []))
            print(f"    High risk zones: {high_risk_zones}")
            print(f"    Medium risk zones: {medium_risk_zones}")
            
         
            # Calculate action risks with optional Habitat check
            print("  🎯 Calculating action-specific risks...")
            action_risks = {}
            for action in planned_actions:
                print(f"    Analyzing action: {action}")
                
                # 1. Get base risk (existing logic)
                base_risk = immediate_risks.get("risk_score", 0.0)
                print(f"      Base risk: {base_risk:.2f}")
                
                # Adjust risk based on action direction (existing logic)
                if action == "move_forward":
                    front_risks = [
                        zone
                        for zone in semantic_risks.get("high_risk_zones", [])
                        if zone["position_relative"][0] > 0
                    ]  # Positive x = forward
                    action_risk = base_risk + (0.2 * len(front_risks))
                    print(f"      Front risks: {len(front_risks)} → Action risk: {action_risk:.2f}")
                elif action == "turn_left":
                    right_risks = [
                        zone
                        for zone in semantic_risks.get("high_risk_zones", [])
                        if zone["position_relative"][1] < 0
                    ]  # Negative y = right
                    action_risk = base_risk + (0.1 * len(right_risks))
                    print(f"      Right risks: {len(right_risks)} → Action risk: {action_risk:.2f}")
                elif action == "turn_right":
                    left_risks = [
                        zone
                        for zone in semantic_risks.get("high_risk_zones", [])
                        if zone["position_relative"][1] > 0
                    ]  # Positive y = left
                    action_risk = base_risk + (0.1 * len(left_risks))
                    print(f"      Left risks: {len(left_risks)} → Action risk: {action_risk:.2f}")
                else:
                    action_risk = base_risk
                    print(f"      Default action risk: {action_risk:.2f}")
                
           
                # Clamp risk between 0 and 1
                original_risk = action_risk
                action_risk = max(0.0, min(1.0, action_risk))
                if original_risk != action_risk:
                    print(f"      Risk clamped from {original_risk:.2f} to {action_risk:.2f}")
                
                # Determine risk level and feasibility (existing logic)
                if action_risk < self.risk_levels["low"]:
                    risk_level = "low"
                    feasible = True
                    speed = "normal"
                    factors = ["clear_path"]
                elif action_risk < self.risk_levels["medium"]:
                    risk_level = "medium"
                    feasible = True
                    speed = "caution"
                    factors = ["moderate_obstacles"]
                else:
                    risk_level = "high"
                    feasible = action_risk < 0.9  # Only infeasible if extremely risky
                    speed = "slow" if feasible else "stop"
                    factors = ["high_risk_obstacles"]
                
                print(f"      Final: risk_level={risk_level}, feasible={feasible}, speed={speed}")
                
                action_risks[action] = {
                    "risk_score": action_risk,
                    "risk_level": risk_level,
                    "feasible": feasible,
                    "primary_risk_factors": factors,
                    "recommended_speed": speed,
                }
            # 🎯 MODIFICATION ENDS HERE

            # 4. GENERATE OVERALL ASSESSMENT
            print("🚧 CollisionRiskPredictor: Step 4 - Generating overall assessment...")
            all_risk_scores = [action["risk_score"] for action in action_risks.values()]
            max_risk = max(all_risk_scores) if all_risk_scores else 0.0
            safe_confidence = 1.0 - (max_risk * 0.8)  # Convert to confidence
            
            print(f"  📊 All risk scores: {[f'{r:.2f}' for r in all_risk_scores]}")
            print(f"  📈 Max collision risk: {max_risk:.2f}")
            print(f"  🛡️ Safe navigation confidence: {safe_confidence:.2f}")

            if max_risk < self.risk_levels["low"]:
                overall_level = "low"
            elif max_risk < self.risk_levels["medium"]:
                overall_level = "medium"
            else:
                overall_level = "high"
            
            print(f"  🎯 Overall risk level: {overall_level}")

            # 5. BUILD RESULT
            print("🚧 CollisionRiskPredictor: Step 5 - Building result...")
            result = {
                "prediction_id": prediction_id,
                "predictor_type": "collision_risk",
                "timestamp": time.time(),
                "status": "completed",
                "risk_assessment": {
                    "overall_risk_level": overall_level,
                    "max_collision_risk": max_risk,
                    "safe_navigation_confidence": safe_confidence,
                    "immediate_danger": immediate_risks.get("immediate_danger", False),
                    # 🎯 OPTIONAL: Add habitat info
                    # "habitat_check_used": use_habitat_check,
                },
                "action_risks": action_risks,
                "collision_zones": semantic_risks,
                "safety_margins": {
                    "robot_radius": self.robot_radius,
                    "safety_buffer": self.safety_buffer,
                    "min_safe_distance": self.min_safe_distance,
                    "emergency_stop_distance": self.emergency_stop_distance,
                },
                "data_sources_used": {
                    "robot_position": robot_position != [0, 0, 0],
                    "semantic_objects": len(semantic_objects),
                    "orb_slam_points": len(point_cloud) if point_cloud is not None else 0,
                    "point_cloud_radius": 3.0,
                    # "habitat_collision_check": use_habitat_check,
                },
                "processing_metrics": {
                    "processing_time_ms": (time.time() - start_time) * 1000, 
                    "algorithm_version": "1.0",
                },
            }

            # 6. SAVE TO PREDICTION STORE
            print("🚧 CollisionRiskPredictor: Step 6 - Saving to prediction store...")
            self._save_to_prediction_store(result)

            # 🆕 CRITICAL FIX: Notify PredictionStore that collision prediction is complete
            if self.prediction_store and hasattr(
                self.prediction_store, "update_prediction"
            ):
                # ✅ CORRECT:
                self.prediction_store.update_prediction(
                    prediction_id=result["prediction_id"],
                    prediction_type="collision_risk",  # ✅ CORRECT PARAMETER NAME
                    data=result,
                    source="collision_risk_predictor",
                )
                print("✅ CollisionRiskPredictor: Results stored in PredictionStore")
            else:
                print("⚠️ CollisionRiskPredictor: No prediction store available for saving")

            total_time = (time.time() - start_time) * 1000
            print(f"🚧 CollisionRiskPredictor: Prediction completed in {total_time:.1f}ms")
        
            # Summary
            print(f"\n📋 PREDICTION SUMMARY:")
            print(f"   Prediction ID: {prediction_id}")
            print(f"   Overall risk: {overall_level}")
            print(f"   Actions analyzed: {list(action_risks.keys())}")
            print(f"   Processing time: {total_time:.1f}ms")
            # print(f"   Habitat check: {'✅ Used' if use_habitat_check else '❌ Not used'}")
            
            return result

        except Exception as e:
            error_time = time.strftime('%H:%M:%S')
            print(f"❌❌❌ COLLISION PREDICTION FAILED at {error_time}: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.logger.error(f"Collision risk prediction failed: {e}")
            return self._create_fallback_result()
    
    
    
    
    def _create_fallback_result(self) -> Dict[str, Any]:
        """Create fallback result when prediction fails"""
        return {
            "prediction_id": "fallback",
            "predictor_type": "collision_risk",
            "timestamp": time.time(),
            "status": "completed",
            "risk_assessment": {
                "overall_risk_level": "unknown",
                "max_collision_risk": 0.5,
                "safe_navigation_confidence": 0.5,
                "immediate_danger": False,
            },
            "action_risks": {
                "move_forward": {
                    "risk_score": 0.5,
                    "risk_level": "unknown",
                    "feasible": True,
                    "primary_risk_factors": ["prediction_unavailable"],
                    "recommended_speed": "caution",
                }
            },
            "collision_zones": {"high_risk_zones": [], "medium_risk_zones": []},
            "data_sources_used": {
                "robot_position": False,
                "semantic_objects": 0,
                "depth_data": False,
                "map_points": 0,
            },
        }

    def _save_to_prediction_store(self, result: Dict):
        """Save collision risk prediction to prediction store"""
        if not self.prediction_store:
            print("🚧 CollisionRiskPredictor: ❌ No prediction store available")
            return

        try:
            # 🆕 ADD ATTRIBUTE CHECK:
            if not hasattr(self.prediction_store, "update_prediction"):
                print("🚧 CollisionRiskPredictor: ❌ No update_prediction method")
                return

            # 🆕 USE STRING CONSTANT INSTEAD OF ENUM:
            self.prediction_store.update_prediction(
                prediction_id=result["prediction_id"],
                prediction_type="collision_risk",  # ✅ String constant
                data=result,
                source="collision_risk_predictor",
            )
            print("🚧 CollisionRiskPredictor: ✅ Results stored successfully")

        except Exception as e:
            print(f"🚧 CollisionRiskPredictor: ❌ Store error details: {e}")
            import traceback

            traceback.print_exc()  # 🆕 Show full error details

    def _get_map_points_around_robot(self, robot_position: list, radius: float = 3.0) -> Optional[np.ndarray]:
        """Extract 3D points around robot from existing map store"""
        if not self.map_store or not hasattr(self.map_store, 'map_points'):
            return None
        
        try:
            robot_pos = np.array(robot_position)
            local_points = []
            
            # Use existing map_points from ORB-SLAM
            for point_data in self.map_store.map_points:
                if 'position_3d' in point_data:
                    point_pos = np.array(point_data['position_3d'])
                    distance = np.linalg.norm(point_pos - robot_pos)
                    
                    if distance <= radius:
                        local_points.append(point_data['position_3d'])
            
            return np.array(local_points) if local_points else None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract map points: {e}")
            return None

    def get_immediate_collision_warnings(self) -> List[Dict]:
        """Get immediate collision warnings for real-time safety"""
        warnings = []

        try:
            robot_position = self._get_robot_position()  # Use Habitat pose

            # Get 3D points around robot from ORB-SLAM map
            if self.map_store and hasattr(self.map_store, 'map_points'):
                point_cloud = self._extract_local_point_cloud(robot_position, radius=3.0)
            else:
                point_cloud = None

                # Get point cloud for warnings
                point_cloud = self._get_map_points_around_robot(robot_position, radius=2.0)
                immediate_risks = self._compute_immediate_geometric_risks(
                    point_cloud, robot_position  # ✅ Use point_cloud
                )

            if immediate_risks.get("immediate_danger", False):
                warnings.append(
                    {
                        "type": "CRITICAL",
                        "message": "Immediate collision danger detected",
                        "distance": immediate_risks.get("closest_obstacle", 0.0),
                        "recommendation": "STOP IMMEDIATELY",
                    }
                )
            elif (
                immediate_risks.get("closest_obstacle", float("inf"))
                < self.min_safe_distance
            ):
                warnings.append(
                    {
                        "type": "WARNING",
                        "message": "Close obstacle detected",
                        "distance": immediate_risks.get("closest_obstacle", 0.0),
                        "recommendation": "Proceed with caution",
                    }
                )

        except Exception as e:
            self.logger.warning(f"Failed to get collision warnings: {e}")

        return warnings 
    
    
    
    def shutdown(self):
        """Basic shutdown - release resources"""
        print("🛑 CollisionRiskPredictor shutdown")
        
        # Just set flag and clear caches
        self.is_running = False
        
        # Clear any in-memory data
        if hasattr(self, 'last_prediction'):
            self.last_prediction = None
        
        if hasattr(self, 'collision_map'):
            self.collision_map.clear()
        
        print("✅ CollisionRiskPredictor resources released")


# Test the predictor
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create test predictor
    predictor = CollisionRiskPredictor()

    # Test prediction
    result = predictor.predict_collision_risk_map()
    print("Collision Risk Prediction Result:")
    print(f"Overall risk: {result['risk_assessment']['overall_risk_level']}")
    print(f"Actions: {list(result['action_risks'].keys())}")

    # Test warnings
    warnings = predictor.get_immediate_collision_warnings()
    print(f"Warnings: {len(warnings)}")











