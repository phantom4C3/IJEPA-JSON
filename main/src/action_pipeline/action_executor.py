

#!/usr/bin/env python3
"""
MINIMAL ACTION EXECUTOR - With Continuous Navigation Built-in
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any
import numpy as np
import math
from src.stores.habitat_store import global_habitat_store 

class ActionExecutor:
    """
    Core business logic engine with CONTINUOUS NAVIGATION built-in
    """
    
    def __init__(
        self,
        task_store=None,
        map_store=None,
        prediction_store=None,
        user_command_store=None,
        config=None
    ):        
        self.task_store = task_store 
        self.map_store = map_store
        self.prediction_store = prediction_store
        self.user_command_store = user_command_store
        self.habitat_store = global_habitat_store
        self.config = config or {}
        
        # Execution state
        self._is_executing = False
        self._is_active = False
        self._execution_lock = threading.RLock()
         
        
        # SemExp/ZSE-SLAM style stuck & collision tracking
        self.collision_count = 0
        self.replan_count = 0
        self.max_replan_threshold = 26
        self.max_collision_threshold = 20
        self.goal_found = False
        self.is_stuck = False
        
        # Mission tracking
        self._mission_progress = 0.0
        self._total_actions_executed = 0
        self._successful_actions = 0
        self._current_task_progress = 0.0
        
        # Performance metrics
        self._performance_metrics = {
            "total_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        
        
        
         
        self._consecutive_failures = 0
        self._max_consecutive_failures = 3
        
        print("🚀 Action Executor initialized - Continuous Navigation Built-in")
        
        # FIX: Add current_action_index to track position in plan
        self._current_action_index = 0
        self._current_plan_reference = None
        
         
    def run_execution_cycle(self) -> Dict[str, Any]:
        """
        Main execution cycle - complete action lifecycle management
        """
        print(f"Run Execution cycle from action executor reached")
        
        if not self._is_active:
            print(f"Run Execution cycle from action executor never reached- self._is_active is false ")
            return self._create_execution_result("executor_inactive")
        
        try:
            # 1. ✅ Read planned actions from TaskStore
            if not self.task_store or not self.task_store.current_action_plan:
                planned_actions = []
            else:
                planned_actions = self.task_store.current_action_plan.get("planned_actions", [])
                self._current_plan_reference = self.task_store.current_action_plan
                
                print(f"Current Planned actions : {planned_actions}")
                print(f"Current Planned reference : {self._current_plan_reference}")
            
            # ✅ Pass the plan reference
            execution_results = self._execute_actions_in_habitat(
                planned_actions, 
                self._current_plan_reference
            )
            print(f"execution results :  {execution_results}")
            
            print(f"📊 Checking {len(execution_results)} actions for status...")
            
            result = self._create_execution_result(
                "execution_completed",
                planned_actions_count=len(planned_actions),
                execution_results=execution_results,
                mission_progress=self._mission_progress,
                timestamp=time.time()
            )
            
            print(f"✅ Execution completed in run_execution_cycle: {len(execution_results)} actions executed")
            return result
            
        except Exception as e:
            print(f"Execution cycle failed: {e}")
            return self._create_execution_result("execution_error", error=str(e))
   
    def _execute_actions_in_habitat(self, planned_actions: List[Dict], plan_reference: Dict = None) -> List[Dict]:
        """Execute actions in Habitat with full tracking"""
        print(f"DEBUG: _execute_actions_in_habitat called with {len(planned_actions)} actions")
        execution_results = []
        self._is_executing = True
        
        try:
            i = 0
            # while i < len(planned_actions):
            while i < 1:
                action_data = planned_actions[i]
                action_type = action_data.get('type')
                parameters = action_data.get('parameters', {})
                
                self._current_action_index = i
                
                result = self.execute_single_action(
                    action_type, 
                    parameters, 
                    plan_reference
                )
                execution_results.append(result)
                
                if result.get("halt_execution", False):
                    print(f"🛑 Plan halted by action: {action_type}")
                    continue
                
                
                i += 1
                time.sleep(1)  # Small delay between actions
            
            if execution_results:
                self._update_task_store_with_results(execution_results)
                self._update_mission_progress(execution_results)
                
                final_result = self._create_execution_result(
                    "execution_completed",
                    planned_actions_count=len(planned_actions),
                    execution_results=execution_results,
                    mission_progress=self._mission_progress,
                    mission_complete=self._mission_progress >= 1.0
                )
                
                self._last_execution_result = final_result
                
                print(f"✅ Execution complete in background thread. Mission progress: {self._mission_progress:.1%}")
            
            return execution_results
            
        except Exception as e:
            print(f"Background execution failed: {e}")
            error_result = self._create_execution_result("execution_error", error=str(e))
            self._last_execution_result = error_result
            return []
            
        finally:
            self._is_executing = False

    # ==================== CONTINUOUS NAVIGATION BUILT-IN ====================
    
    def _is_valid_pose(self, pos):
        """Check pose for NaN/Inf - SKIP if invalid"""
        if pos is None or len(pos) < 3:
            return False
        pos_array = np.array(pos)
        return not (np.any(np.isnan(pos_array)) or np.any(np.isinf(pos_array)))

    
    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        """Convert rotation matrix to quaternion [x, y, z, w] - PROPER VERSION"""
        if rotation_matrix is None:
            return [0, 0, 0, 1]
        
        # Proper conversion from rotation matrix to quaternion
        # This matches what Habitat's VelocityControl expects
        import math
        
        # Extract matrix elements
        m00, m01, m02 = rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2]
        m10, m11, m12 = rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2]
        m20, m21, m22 = rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]
        
        # Compute quaternion
        tr = m00 + m11 + m22
        
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        elif (m00 > m11) and (m00 > m22):
            S = math.sqrt(1.0 + m00 - m11 - m22) * 2
            qw = (m21 - m12) / S
            qx = 0.25 * S
            qy = (m01 + m10) / S
            qz = (m02 + m20) / S
        elif m11 > m22:
            S = math.sqrt(1.0 + m11 - m00 - m22) * 2
            qw = (m02 - m20) / S
            qx = (m01 + m10) / S
            qy = 0.25 * S
            qz = (m12 + m21) / S
        else:
            S = math.sqrt(1.0 + m22 - m00 - m11) * 2
            qw = (m10 - m01) / S
            qx = (m02 + m20) / S
            qy = (m12 + m21) / S
            qz = 0.25 * S
        
        # Return as [x, y, z, w] format
        return [qx, qy, qz, qw]
    
    def _yaw_from_quat(self, q):
        """Yaw for Habitat (Y-up), handles BOTH lists AND quaternion objects"""
        # 🔥 SAFETY: Convert quaternion → list
        if hasattr(q, 'components'):
            q = q.components.tolist()
        elif hasattr(q, 'tolist'):
            q = q.tolist()
        
        x, y, z, w = q[:4]  # Safe unpack first 4
        return math.atan2(2.0 * (w*y + x*z), 1.0 - 2.0 * (y*y + z*z))


    def flatten_quaternion_to_yaw(self, q):
        yaw = self._yaw_from_quat(q)  # Now safe!
        return [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)]  # Return list!


    def _execute_continuous_navigation(self, start_position, start_quat, target_coordinates, 
                                     action_type, parameters, plan_reference, start_time):
        """
        Execute continuous navigation loop with ALL ContinuousNavAgent features
        PUSHES actions to HabitatStore (NO direct sim access)
        """
        print(f"🔄 Starting CONTINUOUS NAVIGATION to {target_coordinates}")
        
        # ContinuousNavAgent parameters
        DT = 0.1  # Timestep (seconds)
        MAX_LIN = 0.3  # Maximum linear velocity (m/s)
        MAX_ANG = 4.0  # Maximum angular velocity 1.5 (rad/s)
        STOP_DIST = 1.0
        MAX_STEPS = 100
        
        step_id = 0
        success = False
        collision_occurred = False
        observations = {}
        
        current_pos = np.array(start_position)
        current_quat = start_quat
         
        
        # Store continuous navigation data for result
        continuous_nav_data = {
            "target_xyz": target_coordinates,
            "dist_remaining": 0.0,
            "lin_vel": 0.0,
            "ang_vel": 0.0,
            "yaw_error_deg": 0.0,
            "complete": False
        }
        
        
        
        # 🔥 INSERT AFTER Line 28 (RIGHT BEFORE continuous_nav_data = {...})

        print(f"🔍 DEBUG: START - pos={current_pos}, raw_quat={current_quat}")

        # 1. FLATTEN INITIAL QUAT
        if hasattr(current_quat, 'x') or hasattr(current_quat, 'vector') or len(np.atleast_1d(current_quat)) >= 4:
            current_quat = self.flatten_quaternion_to_yaw(current_quat)
            print(f"🔄 DEBUG: INITIAL QUAT FLATTENED to {current_quat}")
        else:
            print(f"🔄 DEBUG: start_quat already flattened: {current_quat}")

        current_yaw = self._yaw_from_quat(current_quat)
        print(f"🔍 DEBUG: AFTER FLATTEN - yaw={current_yaw:.3f} ({math.degrees(current_yaw):.1f}°)")

        # 2. SINGLE TURN ALIGNMENT (EXECUTE ONCE)
        print(f"🎯 SINGLE TURN ALIGNMENT...")
        target_xyz = np.array(target_coordinates)
        print(f"🔍 DEBUG: target_xyz={target_xyz}")
        
        
        
        dx = target_xyz[0] - current_pos[0]
        dz = target_xyz[2] - current_pos[2]

        # Habitat: yaw=0 faces -Z
        goal_yaw = math.atan2(dx, -dz)

        print(f"🔍 DEBUG: goal_yaw={goal_yaw:.3f} ({math.degrees(goal_yaw):.1f}°)")
        yaw_err = (goal_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        print(f"🔍 DEBUG: yaw_err={yaw_err:.3f} ({math.degrees(yaw_err):.1f}°)")

        if abs(yaw_err) > 0.1:  # >5.7°
            print(f"🔍 DEBUG: ALIGN NEEDED | align_ang={np.clip(yaw_err * 3.0, -MAX_ANG, MAX_ANG):.3f}")
            align_ang = np.clip(yaw_err * 3.0, -MAX_ANG, MAX_ANG)
            
            # 🔥 COMPUTE PHYSICALLY-CORRECT ALIGN DURATION
            # Required rotation: yaw_err (rad)
            # Applied rotation: align_ang * duration
            align_duration = min(
                0.5,  # safety cap (prevents long spins)
                abs(yaw_err) / max(abs(align_ang), 1e-3)
            )

            print(
                f"⏱️ ALIGN DURATION computed = {align_duration:.3f}s "
                f"(yaw_err={math.degrees(yaw_err):.1f}°, ang={align_ang:.2f} rad/s)"
            )

            align_params = {
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, float(align_ang), 0.0],
                "duration": float(align_duration),
                "is_velocity_command": True
            }
            align_metadata = {
            "target_xyz": target_coordinates,
            "dist_remaining": 0.0,  # Pre-align (no dist yet)
            "lin_vel": 0.0,
            "ang_vel": float(align_ang),
            "yaw_error_deg": math.degrees(yaw_err),
            "step_id": 0,
            "current_pos": current_pos.tolist(), 
            "predicted_pos": current_pos.tolist(),  # No motion yet
            "continuous_nav": True,  # ✅ CRITICAL!
            "velocity_data": {
                "lin_vel": 0.0,
                "ang_vel": float(align_ang),
                "duration": float(align_duration)
            }
        }

            print(f"🔍 DEBUG: align_params={align_params}")
            align_id = self.habitat_store.push_action_to_habitat_store(
            action_type="velocity_control",  # ← CHANGE: action_type= NOT raw string!
            parameters=align_params,         # ← parameters= (not positional)
            metadata=align_metadata     # ← metadata= (not positional)
        )

            print(f"🔍 DEBUG: align_id={align_id}")
            
            # FULL WAIT
            print(f"🔍 DEBUG: WAITING for result...")
            result = None 
            while result is None:
                result = self.habitat_store.get_action_result(align_id)
                if result is None:
                    time.sleep(0.05) 
                    print(f"🔍 DEBUG: WAITING - No result yet")
                else:
                    print(f"🔍 DEBUG: RESULT GOT: {result.keys()}")
            
            if result is None:
                print(f"❌ DEBUG: TIMEOUT waiting for align result!")
            else:
                print(f"🔍 DEBUG: Processing result...")
                if result and 'new_position' in result and self._is_valid_pose(result['new_position']):
                    pos_list = np.atleast_1d(result['new_position']).tolist()
                    if len(pos_list) >= 3:
                        current_pos = np.array([pos_list[0], pos_list[1], pos_list[2]])
                        print(f"🔍 DEBUG: new_pos={current_pos}")
                    
                    if 'new_rotation' in result:
                        rot = result['new_rotation']
                        print(f"🔍 DEBUG: raw_rot={rot}")
                        quat_array = None
                        if hasattr(rot, 'x') and hasattr(rot, 'w'):
                            quat_array = np.array([rot.x, rot.y, rot.z, rot.w])
                            print(f"🔍 DEBUG: magnum quat={quat_array}")
                        elif hasattr(rot, 'vector') and hasattr(rot, 'w'):
                            quat_array = np.array([rot.vector.x, rot.vector.y, rot.vector.z, rot.w])
                            print(f"🔍 DEBUG: vector quat={quat_array}")
                        elif hasattr(rot, 'coeffs'):
                            quat_array = np.array(rot.coeffs())
                            print(f"🔍 DEBUG: coeffs quat={quat_array}")
                        else:
                            quat_list = np.atleast_1d(rot).tolist()
                            if len(quat_list) >= 4:
                                quat_array = np.array(quat_list[:4])
                                print(f"🔍 DEBUG: list quat={quat_array}")
                        
                        if quat_array is not None:
                            current_quat = self.flatten_quaternion_to_yaw(quat_array)
                            print(f"🔍 DEBUG: FINAL current_quat={current_quat}")
                        else:
                            print(f"⚠️ DEBUG: NO QUAT ARRAY")
                else:
                    print(f"⚠️ DEBUG: Invalid result or pose")
            
            print(f"✅ SINGLE TURN COMPLETE: was {math.degrees(yaw_err):.1f}°")
            print(f"🚀 DEBUG: FINAL - pos={current_pos}, yaw={self._yaw_from_quat(current_quat):.3f}")
        else:
            print(f"✅ DEBUG: NO ALIGN NEEDED (err < 5.7°)")

  
        while step_id < MAX_STEPS:
            # 🎯 CONTINUOUSNAVAGENT LOGIC START
            
            # 1. Compute FULL 3D vector (ContinuousNavAgent style)
            target_xyz = np.array(target_coordinates)
            delta = target_xyz - current_pos  # [dx, dy, dz]
            
            print(f"Target xyz : {target_xyz}")            
            print(f"Current position : {current_pos}")            
            print(f"Delta vector calculated : {delta}")
            
            # 2. XY distance only (ignore height for ground navigation)
            dist = np.linalg.norm(delta[[0, 2]])
            
            print(f"📊 Step {step_id}: Distance to target = {dist:.2f}m")
            
              
            recovery_bias = 0.0
            if step_id % 3 == 0:  # Check every 3 steps
                recovery_bias = self.detect_stuck_pattern(target_coordinates)  # ← ONE LINE!
                if recovery_bias != 0.0:  # ← CHANGED
                    print(f"🧠 [NAV] SMART ESCAPE: {recovery_bias:+.1f}")

            
            # 3. SUCCESS CHECK (ContinuousNavAgent style)
            if dist < STOP_DIST:
                success = True
                
                print(f"✅ REACHED TARGET! Distance: {dist:.2f}m")
                
                
                # 🔥 PUSH VELOCITY CONTROL WITH ZERO VELOCITY INSTEAD OF "stop"
                stop_params = {
                    "linear_velocity": [0.0, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.0],
                    "duration": 0.1,
                    "is_velocity_command": True
                }
                
                # In _execute_continuous_navigation() - when pushing STOP:
                stop_metadata = {
                    "is_stop_command": True,
                    "target_reached": True,
                    "final_distance": dist,
                    "continuous_nav": True,
                    "align_to_goal": True,  # ← ADD THIS FLAG
                    "target_xyz": target_coordinates  # ← PASS TARGET (backup)
                }

                
                stop_id = self.habitat_store.push_action_to_habitat_store(
                    "velocity_control",  # ← Use velocity_control, not "stop"
                    stop_params,
                    stop_metadata
                )
                print(f"🛑 STOP PUSHED (ID: {stop_id}) - EXITING NAV LOOP")
 
                break

            
            # 4. REACTIVE CONTROL LAW (ContinuousNavAgent EXACT)
            current_yaw = self._yaw_from_quat(current_quat) 


            # Simple goal-directed navigation (NO GRID)
            dx = delta[0]
            dz = delta[2]

            desired_yaw = math.atan2(dx, -dz)

 
 
            yaw_err = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
            
            print(f"YAW : {current_yaw}")
            print(f"Desired YAW : {desired_yaw}")
            print(f"YAW Error : {yaw_err}")
            
            # Velocity control (ContinuousNavAgent EXACT)
            lin_vel = min(MAX_LIN, dist) 
  
            
                        # 🔥 I-JEPA INTELLIGENT NAVIGATION BIAS
            # 🔥 I-JEPA INTELLIGENT NAVIGATION BIAS
            full_prediction = self.get_ijepa_predictions(full_data=True)
            ijepa_bias = 0.0  # Default

            if full_prediction:
                print(f"✅ [NAV] Got I-JEPA prediction with keys: {list(full_prediction.keys())}")
                
                # Get I-JEPA's continuity predictions for turns
                continuity = full_prediction.get("continuity_predictions", {})
                risks = full_prediction.get("structural_risks", {})  # 🔥 ADDED THIS LINE
                
                # 🔥 DEBUG: Check what we actually got
                print(f"📊 [NAV] Continuity keys: {list(continuity.keys())}")
                print(f"📊 [NAV] Risks keys: {list(risks.keys())}")
                
                turn_left_data = continuity.get("turn_left", {})
                turn_right_data = continuity.get("turn_right", {})
                
                # 🔥 ALL FLOATS - NO ARRAYS
                turn_left_risk = float(risks.get("turn_left", {}).get("risk_score", 0.5))
                turn_right_risk = float(risks.get("turn_right", {}).get("risk_score", 0.5))
                left_continuity = float(turn_left_data.get("continuity_score", 0.5))
                right_continuity = float(turn_right_data.get("continuity_score", 0.5))

                
                left_net = float(left_continuity * (1 - turn_left_risk))
                right_net = float(right_continuity * (1 - turn_right_risk))

                
                # DEBUG: Print I-JEPA's analysis
                print(f"🧠 I-JEPA ANALYSIS: L:{left_continuity:.2f} (risk:{turn_left_risk:.2f}) vs R:{right_continuity:.2f} (risk:{turn_right_risk:.2f})")
                
                # Apply bias only if there's a clear winner
                if left_net > right_net + 0.15:
                    ijepa_bias = -0.4  # LEFT BIAS (I-JEPA prefers turning left)
                    print(f"🧠 I-JEPA: LEFT BIAS -0.4 (L:{left_net:.2f} > R:{right_net:.2f})")
                elif right_net > left_net + 0.15:
                    ijepa_bias = +0.4  # RIGHT BIAS (I-JEPA prefers turning right)
                    print(f"🧠 I-JEPA: RIGHT BIAS +0.4 (R:{right_net:.2f} > L:{left_net:.2f})")
                else:
                    ijepa_bias = 0.0
                    print(f"🧠 I-JEPA: NO BIAS (balanced: L:{left_net:.2f} ≈ R:{right_net:.2f})")
            else:
                print(f"⚠️ I-JEPA: No prediction available")
                ijepa_bias = 0.0
 


            if recovery_bias != 0.0:
                # 🚀 CIRCLE ESCAPE MODE: PURE SPIN + NO FORWARD!
                lin_vel = 0.0  # STOP FORWARD MOTION
                ang_vel = np.clip(recovery_bias, -MAX_ANG, MAX_ANG)  # SKIP yaw_err COMPLETELY!
                print(f"🧯 CIRCLE ESCAPE: PURE SPIN ang={ang_vel:.2f} lin={lin_vel:.2f} (NO yaw_err!)")
                print(f"🚀 ESCAPE DEBUG: bias={recovery_bias:.2f} → MAX_LEFT={-MAX_ANG:.2f}")
            else:
                # NORMAL navigation with I-JEPA bias
                # 🔥 PROPORTIONAL MAIN-NAV DURATION (reuse DT — no new vars)

                lin_vel = min(MAX_LIN, dist)
                ang_vel = np.clip(yaw_err * 2.0 + ijepa_bias, -MAX_ANG, MAX_ANG)
                DT = abs(yaw_err) / max(abs(ang_vel), 1e-3)
                print(
                    f"⏱️ MAIN NAV DT = {DT:.3f}s "
                    f"(yaw_err={yaw_err:.2f}, ang_vel={ang_vel:.2f})"
                )
                print(f"✅ NORMAL MODE: lin={lin_vel:.2f} ang={ang_vel:.2f} (yaw_err*2={yaw_err*2:.2f} + ijepa={ijepa_bias:.2f})")
                
                # 🔥 PRE-COLLISION SAFETY GATE (ZERO-SHOT SAFE)
                # Uses navmesh geometry ONLY to limit forward motion
                obs_dist = result.get("obstacle_distance")

                SAFE_DIST = 0.82  # minimum safe distance before slowing
                
                
                obstacle_dir = result.get("obstacle_direction", {})

                side = obstacle_dir.get("side", "NONE")
                depth = obstacle_dir.get("depth", "NONE") 

                if obs_dist < SAFE_DIST: 
                    print(f"🛑 [GATE] Obstacle at {obs_dist:.3f}m ")

                    # 🔹 Single-line angular tweak to turn away from obstacle
                    ang_vel += -0.5 if obstacle_dir.get("side") == "LEFT" else (0.5 if obstacle_dir.get("side") == "RIGHT" else ang_vel)
                    print(f"🌀 Adjusted angular velocity to avoid obstacle: {ang_vel:.2f}")



            print(f"🎯 FINAL VELOCITY: lin_vel={lin_vel:.3f} ang_vel={ang_vel:.3f}")

            
            print(f"Linear velocity : {lin_vel}")
            print(f"Angular veloctiy : -{ang_vel}")
            
            
            # Store continuous nav data
            continuous_nav_data.update({
                "dist_remaining": dist,
                "lin_vel": lin_vel,
                "ang_vel": ang_vel,
                "yaw_error_deg": math.degrees(yaw_err),
                "complete": False
            })
             

            # TO (Z-negative forward) - SAME ACTUAL MOVEMENT:
            predicted_new_pos = current_pos + np.array([
                math.sin(current_yaw) * lin_vel * DT,  # X = -sin(yaw)*velocity
                0.0,
                -math.cos(current_yaw) * lin_vel * DT    # Z = cos(yaw)*velocity
            ]) 
            print(f"🔮 Predicted from ACTUAL pos: {predicted_new_pos}")
             
            
            # 7. PUSH to HabitatStore INLINE (no separate function)
            velocity_metadata = {
                "target_xyz": target_coordinates,
                "dist_remaining": dist,
                "lin_vel": float(lin_vel),
                "ang_vel": float(ang_vel),
                "yaw_error_deg": math.degrees(yaw_err),
                "step_id": step_id,
                "current_pos": current_pos.tolist(), 
                "predicted_pos": predicted_new_pos.tolist(),
                "continuous_nav": True,
                "velocity_data": {
                    "lin_vel": float(lin_vel),
                    "ang_vel": float(ang_vel),
                    "duration": float(DT)
                }
            }
            
            print(f"Velocity metadata : {velocity_metadata}")        

            print(f"  🎯 Velocity: lin={lin_vel:.2f} m/s, ang={ang_vel:.2f} rad/s")

            # 🎯 PUSH VELOCITY COMMAND INLINE
            velocity_parameters = {
                "linear_velocity": [ 0.0, 0.0, -float(lin_vel)],  # m/s in local X
                "angular_velocity": [0.0, float(ang_vel), 0.0,],  # rad/s around local Y 
                "duration": float(DT),  # seconds
                "is_velocity_command": True  # Flag for backward compatibility
            }
            
            print(f"Velocity parameters : {velocity_parameters}")        
            
            action_id = self.habitat_store.push_action_to_habitat_store(
                action_type="velocity_control",  # Special type for orchestrator
                parameters=velocity_parameters,
                metadata=velocity_metadata  # All navigation data
            )
            
            print(f"📤 Pushed velocity command (ID: {action_id})")
            
            # 8. Wait for result from HabitatStore
                        # ✅ WAIT FOR RESULT
            result = None
            while result is None:
                result = self.habitat_store.get_action_result(action_id)
                if result is None:
                    time.sleep(0.05)  # ⏳ allow MainOrch to execute
 
            print(f"✅ Velocity result received for: {action_id}")
 
 
            if not result:
                print(f"⏳ Step {step_id} pending...")
                time.sleep(0.05)
                step_id += 1
                continue  # Keep loop alive!
 
            
            # 9. Update state from result
                        # 9. UPDATE FROM HABITAT FEEDBACK (CRITICAL!)
            observations = result.get("observations", {})
            step_collided = result.get("collided", False)
            filtered_contacts = result.get("filtered_contacts", []) 
            print(f"step collided data from results : {step_collided} and collision contacrts : {len(filtered_contacts)}")
            
            
             # 9. UPDATE FROM HABITAT FEEDBACK (CRITICAL!) 
            
            if (result and 'new_position' in result and 
                self._is_valid_pose(result['new_position'])):

                # 🔥 FORCE 3D POSITION (handles scalar + list)
                pos_list = np.atleast_1d(result['new_position']).tolist()
                if len(pos_list) >= 3:
                    current_pos = np.array([pos_list[0], pos_list[1], pos_list[2]])
                    print(f"🔄 Valid Habitat position update: {current_pos}")
                else:
                    print(f"⚠️ Invalid position format skipped: {result['new_position']}")

                # 🔥 FORCE 4D QUATERNION (Habitat-safe, version-proof)
                if 'new_rotation' in result:
                    rot = result['new_rotation']

                    quat_array = None

                    # Habitat / Magnum quaternion object
                    if hasattr(rot, 'x') and hasattr(rot, 'w'):
                        quat_array = np.array([rot.x, rot.y, rot.z, rot.w], dtype=np.float32)
                    elif hasattr(rot, 'vector') and hasattr(rot, 'w'):
                        quat_array = np.array([rot.vector.x, rot.vector.y, rot.vector.z, rot.w], dtype=np.float32)
                    elif hasattr(rot, 'coeffs'):
                        quat_array = np.array(rot.coeffs(), dtype=np.float32)
                    else:
                        quat_list = np.atleast_1d(rot).tolist()
                        if len(quat_list) >= 4:
                            quat_array = np.array(quat_list[:4], dtype=np.float32)

                    if quat_array is not None:
                        current_quat = self.flatten_quaternion_to_yaw(quat_array)
                        print(f"🔄 Using Habitat rotation (planar): {current_quat}")
                    else:
                        print(f"⚠️ Unrecognized rotation format, enforcing planar fallback")
                        current_quat = self.flatten_quaternion_to_yaw(current_quat)
 
            else:
                print("⚠️ Skipping invalid Habitat pose")


             
            print(f"Step Collided or not  _execute_continuous_navigation: {step_collided}")        
            
            tangent_dir_xz = None  # cache for recovery

           
           
            if step_collided and filtered_contacts:
                    collision_occurred = True
                    self.collision_count += 1
                    print(f"⚠️ Collision at step {step_id} — TANGENT RECOVERY")

                    # --- A. Extract Normal and Project to Ground ---
                    contact = filtered_contacts[0]  # strongest / filtered already
                    raw_normal = np.array(contact["normal"], dtype=np.float32)

                    obj_normal_xz = np.array([raw_normal[0], raw_normal[2]])
                    norm = np.linalg.norm(obj_normal_xz)

                    tangent_dir_xz = None
                    if norm < 1e-4:
                        print("⚠️ Degenerate object normal, skipping tangent")
                    else:
                        obj_normal_xz /= norm
                        print(f"🧱 OBJECT NORMAL (XZ): {obj_normal_xz}")

                        # --- B. Compute both tangents ---
                        tangent_left  = np.array([-obj_normal_xz[1],  obj_normal_xz[0]])
                        tangent_right = np.array([ obj_normal_xz[1], -obj_normal_xz[0]])

                        # --- C. Choose tangent USING OBSTACLE SIDE (STABLE) ---
                        if side == "LEFT":
                            tangent_dir_xz = tangent_right
                            print("🧭 Tangent chosen: RIGHT (obstacle on LEFT)")
                        elif side == "RIGHT":
                            tangent_dir_xz = tangent_left
                            print("🧭 Tangent chosen: LEFT (obstacle on RIGHT)")
                        else:
                            tangent_dir_xz = tangent_right
                            print("🧭 Tangent chosen: DEFAULT RIGHT (side unknown)")

                        # Final normalization
                        tangent_dir_xz /= max(np.linalg.norm(tangent_dir_xz), 1e-4)
                        print(f"➡️ FINAL TANGENT DIR (XZ): {tangent_dir_xz}")

                    # --- D. Determine Recovery Direction ---
                    ang_vel_recovery = 4.0 
                    lin_vel_recovery = 0.3

                    if depth == "FRONT":
                        ang_vel_recovery = -4.0 if side == "LEFT" else 4.0
                        print(f"🧱 [RECOVERY] FRONT + {side} → TURN {'RIGHT' if side=='LEFT' else 'LEFT'}")
                    elif depth == "BACK":
                        ang_vel_recovery = 0.0
                        lin_vel_recovery = 0.3
                        print("🧽 [RECOVERY] BACK contact → FORWARD ONLY")

                    # 🔥 STUCK OVERRIDE (REUSE SAME DECISION)
                    if recovery_bias != 0.0:
                        print("🚨 [COLLISION+STUCK] Using SAME escape bias")
                        ang_vel_recovery = recovery_bias

                    # 🔥 4. FLATTEN ROTATION
                    current_quat = self.flatten_quaternion_to_yaw(current_quat)

                    # 🔥 5. EXECUTE RECOVERY (5 ATTEMPTS)
                    recovery_success = False
                    for recovery_attempt in range(5):
                        print(f"🔄 Recovery attempt {recovery_attempt+1}, ang_vel={ang_vel_recovery}")

                        # 🔥 COMPUTE RECOVERY TURN DURATION (90° escape turn)
                        RECOVERY_YAW = math.pi / 2
                        recovery_duration = min(0.5, RECOVERY_YAW / max(abs(ang_vel_recovery), 1e-3))

                        # Params defined INSIDE the loop as requested
                     
                        linear_speed = 0.2  # forward speed along tangent

                        if tangent_dir_xz is not None:
                            # tangent_dir_xz is a normalized 2D vector [x, z]
                            lin_vel_recovery_x = tangent_dir_xz[0] * linear_speed
                            lin_vel_recovery_z = tangent_dir_xz[1] * linear_speed
                        else:
                            # fallback if tangent not valid
                            lin_vel_recovery_x = 0.0
                            lin_vel_recovery_z = -linear_speed  # move forward in local frame

                        recovery_params = {
                            "linear_velocity": [lin_vel_recovery_x, 0.0, lin_vel_recovery_z],
                            "angular_velocity": [0.0, float(ang_vel_recovery), 0.0],
                            "duration": float(recovery_duration),
                            "is_velocity_command": True
                        }


                        recovery_metadata = {
                            "continuous_nav": True,
                            "recovery": True,
                            "attempt": recovery_attempt,
                            "ang_vel": float(ang_vel_recovery),
                            "tangent_xz": tangent_dir_xz.tolist() if tangent_dir_xz is not None else None
                        }

                        recovery_id = self.habitat_store.push_action_to_habitat_store(
                            "velocity_control", recovery_params, recovery_metadata
                        )

                        recovery_result = None
                        while recovery_result is None:
                            recovery_result = self.habitat_store.get_action_result(recovery_id)
                            if recovery_result is None:
                                time.sleep(0.05)

                        if recovery_result and not recovery_result.get("collided", False):
                            recovery_success = True
                            print("✅ Recovery succeeded (No collision detected)")
                            
                            # Update position/rotation after successful recovery
                            if 'new_position' in recovery_result:
                                current_pos = np.array(recovery_result['new_position'][:3])
                            if 'new_rotation' in recovery_result:
                                current_quat = self.flatten_quaternion_to_yaw(recovery_result['new_rotation'])
                            break
                        else:
                            print(f"❌ Recovery attempt {recovery_attempt+1} still collided")

                    continue  # Next navigation step
           
           
                 
            
            step_id += 1
            time.sleep(DT)  # Match real-time with DT
        
        # FINAL RESULT with ALL ContinuousNavAgent data
        execution_time = time.time() - start_time
        
        print(f"Eecution time: {execution_time}")
        
        # Mark as complete if we exited loop
        if not continuous_nav_data["complete"]:
            continuous_nav_data["complete"] = True
        
        result_dict = self._create_action_result(
            action_type,
            success,
            execution_time,
            "Continuous navigation completed" if success else "Continuous navigation failed",
            parameters,
            collision_occurred=collision_occurred,
            movement_verified=True,
            collision_count=self.collision_count,
            is_stuck=self.is_stuck
        )
        
        print(f"Result Dict : {result_dict}")
        
        # Add ALL ContinuousNavAgent data to result
        result_dict["observations"] = observations
        result_dict["continuous_nav"] = continuous_nav_data
        
        # Add predicted position for collision checking (ContinuousNavAgent feature)
        if "predicted_new_pos" in locals():
            result_dict["predicted_new_pos"] = predicted_new_pos.tolist()
        
        print(f"📋 Continuous navigation result: "
              f"success={success}, steps={step_id}, "
              f"final_dist={continuous_nav_data['dist_remaining']:.2f}m")
        
        return result_dict
    
    def get_ijepa_predictions(self, full_data: bool = True) -> dict:
        """
        SIMPLE: Fetch latest RAW I-JEPA prediction with minimal debug.
        """
        try:
            print("🔍 [I-JEPA] Fetching from RAW predictions...")
            
            if not hasattr(self, "prediction_store"):
                return {}
                
            if not hasattr(self.prediction_store, "predictions"):
                print("❌ No predictions store")
                return {}
            
            # Find all I-JEPA predictions
            ijepa_list = []
            
            for pred_id, entry in self.prediction_store.predictions.items():
                if not isinstance(entry, dict):
                    continue
                
                is_ijepa = False
                ijepa_data = None
                
                # Check for I-JEPA markers
                if (entry.get('predictor_type') == 'structural_continuity' or
                    'continuity_predictions' in entry or
                    'structural_risks' in entry):
                    
                    is_ijepa = True
                    ijepa_data = entry  # Flat structure
                    
                elif ('predictors' in entry and 
                    'structural_continuity' in entry['predictors'] and
                    'data' in entry['predictors']['structural_continuity']):
                    
                    is_ijepa = True
                    ijepa_data = entry['predictors']['structural_continuity']['data']  # Nested
                
                if is_ijepa and ijepa_data:
                    timestamp = entry.get('timestamp', 0)
                    ijepa_list.append({
                        'timestamp': timestamp,
                        'prediction_id': pred_id,
                        'data': ijepa_data
                    })
            
            if not ijepa_list:
                print("⚠️ No I-JEPA predictions found")
                return {}
            
            # Get latest
            ijepa_list.sort(key=lambda x: x['timestamp'], reverse=True)
            latest = ijepa_list[0]
            
            print(f"✅ Using I-JEPA: {latest['prediction_id']}")
            print(f"   Has continuity_predictions: {'continuity_predictions' in latest['data']}")
            print(f"   Has structural_risks: {'structural_risks' in latest['data']}")
            
            if full_data:
                return latest['data']
            else:
                return {
                    "continuity_predictions": latest['data'].get("continuity_predictions", {}),
                    "structural_risks": latest['data'].get("structural_risks", {})
                }
                
        except Exception as e:
            print(f"❌ I-JEPA fetch error: {e}")
            return {}




        
    def execute_single_action(self, action_type: str, parameters: Dict, plan_reference: Dict = None) -> Dict[str, Any]:
        """
        Execute single action - HANDLES BOTH discrete and continuous navigation
        """
        start_time = time.time()
        
        print(f"🎯 ACTION EXECUTOR: Starting execution of execute_single_action: action type : '{action_type}'")
                
        try:
            
            start_position = None 
            if (global_habitat_store.initial_pose and 
                self._is_valid_pose(global_habitat_store.initial_pose['position'])):
                start_position = global_habitat_store.initial_pose['position']
                start_quat = global_habitat_store.initial_pose['rotation']
                print(f"📍 Valid initial pose: {start_position}, quat:{start_quat}")
            else:
                print("⚠️ Invalid initial pose - using safe defaults")
                start_position = [0, 0, 1.0]  # Safe agent height
                start_quat = [0, 0, 0, 1]


            
            # 🎯 STEP 2: Check for target coordinates (continuous navigation trigger)
            target_coordinates = parameters.get("target_coordinates")
            

            print(f"Target coordinates inside action executor: {target_coordinates}")
            
            
            if target_coordinates is None:
                print("⚠️ WARNING: target_coordinates is None! Checking HabitatStore...")
                
                # 🆕 PRIORITY -1: Check HabitatStore FIRST (highest priority)
                if (hasattr(global_habitat_store, 'current_goal_position') and 
                    global_habitat_store.current_goal_position is not None):
                    
                    target_coordinates = global_habitat_store.current_goal_position
                    goal_category = global_habitat_store.current_goal_category or "unknown"
                    print(f"🎯 HABITATSTORE: Using stored goal '{goal_category}' at {target_coordinates}")
                    
                else:
                    # Fallback to microwave goal
                    print("⚠️ HabitatStore has no goal, using microwave fallback...")
                    target_coordinates = [-3.3307390213012695, 0.929732620716095, 0.3535519540309906]
                    print(f"🎯 FALLBACK: Using microwave goal: {target_coordinates}")
            

            
          
            # 🎯 STEP 3: CONTINUOUS NAVIGATION (if we have target coordinates)
            if target_coordinates :
                print(f"🎯 CONTINUOUS NAVIGATION to {target_coordinates}")
                
                if not start_position:
                    print("⚠️ No start position available, falling back to discrete")
                else:
                    current_quat = start_quat  # Use Habitat initial rotation directly
                    print(f"Current quat from Habitat: {current_quat}")

                    
                    # 🎯 Execute continuous navigation (BUILT-IN!)
                    continuous_navigation_result = self._execute_continuous_navigation(
                        start_position=start_position,
                        start_quat=current_quat,
                        target_coordinates=target_coordinates,
                        action_type=action_type,
                        parameters=parameters,
                        plan_reference=plan_reference,
                        start_time=start_time
                    )
                    print(f"Continuous Navigation Result : {continuous_navigation_result}")
                    
                    return continuous_navigation_result

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Action execution failed: {e}")
            
            return self._create_action_result(
                action_type, False, execution_time, f"Execution error: {str(e)}",
                collision_count=self.collision_count,
                is_stuck=self.is_stuck
            )
    
    def detect_stuck_pattern(self, target_coordinates=None) -> float:
        """
        Stuck detection WITH intelligent escape direction.
        Single return type:
        - 0.0  → NOT stuck
        - ±4.0 → stuck, escape yaw bias (direction matters)
        """
        try:
            print(f"🔍 [STUCK_DETECT] Starting detection with escape logic...")

            # 1. Get trajectory
            if not hasattr(global_habitat_store, '_metrics_trajectory'):
                return 0.0

            traj = global_habitat_store._metrics_trajectory
            if len(traj) < 30:
                print(f"⚠️ [STUCK_DETECT] Not enough points: {len(traj)} < 30")
                return 0.0

            # 2. Analyze recent points
            recent_points = min(40, len(traj))
            pts = np.array(traj[-recent_points:])
            print(f"📊 [STUCK_DETECT] Analyzing {len(pts)} points")

            # 3. Path metrics
            path_len = 0.0
            for i in range(1, len(pts)):
                path_len += np.linalg.norm(pts[i] - pts[i - 1])

            displacement = np.linalg.norm(pts[-1] - pts[0])
            progress_ratio = displacement / path_len if path_len > 0 else 0.0

            print(f"📊 Path={path_len:.2f}m | Disp={displacement:.2f}m | Ratio={progress_ratio:.2%}")

            # 4. Heuristics
            h0_stagnant = (path_len < 0.1) and (len(pts) >= 10)
            h1_circle = (path_len > 1.5) and (displacement < 0.4)

            # Motion vectors
            motion_vectors = []
            for i in range(1, len(pts)):
                v = pts[i] - pts[i - 1]
                n = np.linalg.norm(v)
                if n > 0.05:
                    motion_vectors.append(v / n)

            reversals = 0
            for i in range(1, len(motion_vectors)):
                if np.dot(motion_vectors[i], motion_vectors[i - 1]) < -0.3:
                    reversals += 1

            reversal_ratio = reversals / max(1, len(motion_vectors))
            h2_oscillation = (reversal_ratio > 0.25) and (path_len > 3.0)

            h3_inefficient = (progress_ratio < 0.15) and (path_len > 5.0)

            # Stuck-area heuristic
            if len(pts) >= 20:
                s1 = pts[:len(pts)//3]
                s2 = pts[len(pts)//3:2*len(pts)//3]
                s3 = pts[2*len(pts)//3:]
                c1, c2, c3 = np.mean(s1, 0), np.mean(s2, 0), np.mean(s3, 0)
                h5_stuck_area = (
                    np.linalg.norm(c1 - c2) < 2.0 and
                    np.linalg.norm(c2 - c3) < 2.0 and
                    path_len > 4.0
                )
            else:
                h5_stuck_area = False

            stuck = h0_stagnant or h1_circle or h2_oscillation or h3_inefficient or h5_stuck_area

            print(f"""
    📊 HEURISTICS:
    h0_stagnant     = {h0_stagnant}
    h1_circle       = {h1_circle}
    h2_oscillation  = {h2_oscillation} ({reversals}/{len(motion_vectors)})
    h3_inefficient  = {h3_inefficient}
    h5_stuck_area   = {h5_stuck_area}
    FINAL STUCK     = {stuck}
    """)

            if not stuck:
                return 0.0

            # ===========================
            # 🔥 ESCAPE DIRECTION LOGIC
            # ===========================
            escape_bias = -4.0  # default safe turn

            # Circle / stuck-area → escape away from curvature center
            if h1_circle or h5_stuck_area:
                x = pts[:, 0]
                z = pts[:, 2]
                if len(x) > 5:
                    A = np.column_stack([x, np.ones(len(x))])
                    center_x, _ = np.linalg.lstsq(A, z, rcond=None)[0]
                    robot_x = np.mean(pts[-5:, 0])
                    escape_bias = -4.0 if center_x > robot_x else +4.0

            # Oscillation → perpendicular to last motion
            elif h2_oscillation and motion_vectors:
                last_vec = motion_vectors[-1]
                escape_bias = -4.0 if last_vec[0] > 0 else +4.0

            # Inefficient → bias toward goal side if available
            elif h3_inefficient and target_coordinates is not None:
                target = np.array(target_coordinates)
                dx = target[0] - pts[-1, 0]
                escape_bias = -4.0 if dx < 0 else +4.0

            print(f"🚨 [STUCK_DETECT] ESCAPE BIAS = {escape_bias}")
            return escape_bias

        except Exception as e:
            print(f"❌ [STUCK_DETECT] Error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
 

    
    
    def _update_task_store_with_results(self, execution_results: List[Dict]):
        """Update TaskStore with execution results"""
        try:
            if not self.task_store or not execution_results:
                return
            
            successful_actions = [r for r in execution_results if r.get("success")]
            overall_success = len(successful_actions) > 0
            
            if overall_success and hasattr(self.task_store, 'complete_current_task'):
                self.task_store.complete_current_task(success=True)
                print("✅ Task completed successfully")
                
        except Exception as e:
            print(f"Failed to update TaskStore: {e}")

    def _update_mission_progress(self, execution_results: List[Dict]):
        """Update mission progress tracking"""
        try:
            if not execution_results:
                return
            
            successful_actions = [r for r in execution_results if r.get("success")]
            
            if successful_actions:
                progress_increment = len(successful_actions) / max(1, len(execution_results)) * 0.2
                self._mission_progress = min(1.0, self._mission_progress + progress_increment)
                self._current_task_progress = self._mission_progress
                
                print(f"Mission progress: {self._mission_progress:.1%}")
                    
        except Exception as e:
            print(f"Failed to update mission progress: {e}")

    def _update_performance_metrics(self, success: bool, execution_time: float):
        """Update performance metrics"""
        self._performance_metrics["total_actions"] += 1
        self._performance_metrics["total_execution_time"] += execution_time
        
        if success:
            self._performance_metrics["successful_actions"] += 1
        else:
            self._performance_metrics["failed_actions"] += 1
        
        total = self._performance_metrics["total_actions"]
        successful = self._performance_metrics["successful_actions"]
        self._performance_metrics["success_rate"] = successful / total if total > 0 else 0.0

    def _create_action_result(self, action: str, success: bool, execution_time: float, 
                            notes: str, parameters: Dict = None, **kwargs) -> Dict[str, Any]:
        """Create action execution result"""
        result = {
            "action": action,
            "success": success,
            "execution_time": execution_time,
            "parameters": parameters or {},
            "timestamp": time.time(),
            "notes": notes,
        }
        
        result.update(kwargs)
        return result

    def _create_execution_result(self, status: str, **kwargs) -> Dict[str, Any]:
        """Create execution cycle result"""
        result = {
            "status": status,
            "timestamp": time.time(),
            "mission_progress": self._mission_progress,
            "performance_metrics": self._performance_metrics.copy(),
            "collision_count": self.collision_count,
            "goal_found": self.goal_found,
            "is_stuck": self.is_stuck
        }
        
        result.update(kwargs)
        return result

     
 
                            
    # ==================== PUBLIC INTERFACE ====================

    def start_execution(self): 
        self._is_active = True
        print("Action execution started, start_execution called")

    def get_execution_status(self) -> Dict[str, Any]:
        """Get complete execution status"""
        with self._execution_lock:
            return {
                "status": "active" if self._is_active else "inactive",
                "is_executing": self._is_executing,
                "mission_progress": self._mission_progress,
                "performance_metrics": self._performance_metrics.copy(),
                "collision_count": self.collision_count,
                "is_stuck": self.is_stuck
            }
            
            
            
    def shutdown(self):
        """Shutdown executor"""
        with self._execution_lock: 
            self._is_active = False
            self._is_executing = False
            print("Action Executor shutdown complete")

    def reset_stuck_counters(self):
        """Reset stuck and collision counters"""
        self.collision_count = 0
        self.replan_count = 0
        self.is_stuck = False
        print("🔄 Stuck counters reset")

# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test the executor
    executor = ActionExecutor()
    
    # Test execution status
    status = executor.get_execution_status()
    print(f"Executor status: {status}")
    
    executor.shutdown()