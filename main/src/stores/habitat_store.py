 
#!/usr/bin/env python3
"""
HabitatStore - Global singleton for Habitat simulator access
Thread-safe access to simulator instance and camera parameters
"""

import threading
import numpy as np
from typing import Optional, Dict, Any
import habitat_sim
import traceback
import time


class HabitatStore:
    """
    Lightweight wrapper providing:
    - Simulator instance reference
    - Camera intrinsics extraction  
    - Thread-safe access
    - Initialization status
    - Clean shutdown
    """
    
    def __init__(self): 
        self.simulator = None
        self.camera_intrinsics = None
        self._lock = threading.RLock()
        self.action_results = {}  # ADD THIS
        self.initial_pose = None 
        
        
        
                # 🆕 ADD THESE 2 LINES HERE:
        self.current_goal_position = None      # For coordinates [x, y, z]
        self.current_goal_category = None      # For object name ("microwave", "armchair", etc.)

        self._metrics_trajectory = []
        print("✅ METRICS: Initialized trajectory tracker")
        self._metrics_collisions = 0
        print("✅ METRICS: Initialized collisions tracker = 0")
        self._metrics_start_time = None
        print("✅ METRICS: Initialized start_time tracker")

        
        # In your global_habitat_store initialization:
        self._collision_history = []  # List of dicts
        self._collision_stats = {
            "total_contacts": 0,
            "max_penetration": 0.0,
            "deep_jams": 0
        }

        
        
        
        # 🆕 NEW: Motion tracking variables
        self.last_agent_position = None
        self.last_agent_rotation = None
        self.position_history = []  # Store recent positions
        self.motion_stats = {
            'avg_translation': 0.30,  # Will be updated dynamically
            'avg_rotation': 0.10,
            'max_translation': 1.0,
            'min_translation': 0.01
        }
        
        print("🎯 HabitatStore initialized - Thread-safe simulator access")


    def set_simulator(self, simulator) -> None:
        """
        Set simulator reference - called once by orchestrator
        Extracts camera parameters automatically
        """
        with self._lock:
            self.simulator = simulator 
            print("✅ Simulator stored ")

    def set_depth_frame(self, depth_frame, frame_id, timestamp):
        """Store the latest depth frame"""
        with self._lock:
            self.depth_frame = depth_frame
            self.depth_frame_id = frame_id
            self.depth_timestamp = timestamp
            print(f"💾 HabitatStore: Depth frame {frame_id} stored - shape: {depth_frame.shape}")

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get current depth frame from Habitat simulator"""
        with self._lock:
            if self.simulator is None:
                return None
            try:
                observations = self.simulator.get_sensor_observations()
                return observations.get('depth_camera')  # Make sure this matches your sensor UUID
            except Exception as e:
                print(f"❌ Failed to get depth frame: {e}")
                return None
            
    def get_simulator(self) -> Optional[Any]:
        """Get direct simulator access for Habitat's built-in functions"""
        with self._lock:
            return self.simulator

    def get_camera_intrinsics(self) -> Optional[Dict[str, Any]]:
        """Get camera intrinsic parameters"""
        with self._lock:
            return self.camera_intrinsics.copy() if self.camera_intrinsics else None

    def set_camera_intrinsics(self, intrinsics: Dict[str, Any]):
        with self._lock:
            self.camera_intrinsics = intrinsics.copy()


    def is_ready(self) -> bool:
        """Check if simulator is available and ready"""
        with self._lock:
            return self.simulator is not None


    def push_action_to_habitat_store(self, action_type: str, parameters: dict = None, metadata: dict = None) -> str:
        """
        Push action to HabitatStore queue for MainOrchestrator to execute.
        NOW SUPPORTS VELOCITY DATA via action_type="velocity_control"
        """
        print(f"📥 HABITATSTORE: Pushing action '{action_type}' to queue")
        
        # Initialize queue if not exists
        if not hasattr(self, 'action_queue'):
            print("🆕 HABITATSTORE: Creating new action queue")
            self.action_queue = []
            
                # --- get caller info ---
        stack = traceback.extract_stack()
        # -1 = this line, -2 = caller inside HabitatStore, -3 = external caller
        caller = stack[-3]
        caller_file = caller.filename.split("/")[-1].replace(".py", "")
        caller_func = caller.name

        # # keep it short/safe for ID
        # caller_tag = f"{caller_file}_{caller_func}"

        base_id = f"action_{int(time.time()*1000)}"
        action_id = f"{base_id}_{len(self.action_queue)}"  # Remove caller_tag!

        
        # Create action item WITH METADATA SUPPORT
        action_item = {
            "action_id": action_id,
            "type": action_type,
            "parameters": parameters or {},
            "timestamp": time.time(),
            "status": "queued",
            "source_thread": threading.current_thread().name
        }
        
        # 🎯 ADD METADATA IF PROVIDED (for velocity data)
        if metadata:
            action_item["metadata"] = metadata
            print(f"📦 HABITATSTORE: Added metadata to action '{action_type}'")
        
        # Append to queue
        with self._lock:
            self.action_queue.append(action_item)
        
        print(f"✅ HABITATSTORE: Action '{action_type}' queued (ID: {action_item['action_id']})")
        print(f"📊 HABITATSTORE: Queue size: {len(self.action_queue)}")
        
        return action_item["action_id"]
    
             
    # def store_action_result(self, action_id: str, result: dict):
    #     """Store result for ActionExecutor to retrieve"""
    #     with self._lock:
    #         self.action_results[action_id] = result
    
    
    def store_action_result(self, action_id: str, result: dict):
        """Store result for ActionExecutor to retrieve"""
        with self._lock:
            self.action_results[action_id] = result
            print(f"✅ Storing SINGLE result for {action_id}")  # Changed!
            print(f"Storing result using store_action_result() for {action_id} | All stored IDs: {list(self.action_results.keys())}")
            print(f"Stored results for id : {action_id}")

    
    
    
    
    
    def get_action_result(self, action_id: str) -> dict:
        """SIMPLE: Get result if ready"""
        with self._lock:
            if action_id in self.action_results:
                result = self.action_results.pop(action_id)
                print(f"✅ Retrieved result for {action_id} using get_action_result() ")
                return result
        print(f"⏳ No result yet for {action_id}")
        return None  # Let caller handle missing result!


    
    # def pull_action_from_habitat_store(self) -> dict:
    #     """
    #     Pull next action from HabitatStore queue.
    #     NOW PASSES METADATA to MainOrchestrator for velocity control
    #     """
    #     print(f"🎯 HABITATSTORE: Pulling next action from queue")
        
    #     # Check if queue exists and has items
    #     if not hasattr(self, 'action_queue') or not self.action_queue:
    #         print("📭 HABITATSTORE: Action queue is empty")
    #         return None
        
    #     with self._lock: 
    #         # Get and remove first action
    #         action_item = self.action_queue.pop(0)
    #         action_item["status"] = "executing"
    #         action_item["execution_thread"] = threading.current_thread().name
            
    #         # 🎯 DEBUG: Check if this is velocity data
    #         if action_item.get("type") == "velocity_control":
    #             print(f"🚀 HABITATSTORE: Pulled VELOCITY CONTROL command")
    #             print(f"   📦 Metadata: {action_item.get('metadata', {}).keys()}")
    #         else:
    #             print(f"🚀 HABITATSTORE: Pulled action '{action_item['type']}' (ID: {action_item['action_id']})")
            
    #         print(f"📊 HABITATSTORE: Remaining in queue: {len(self.action_queue)}")
        
    #     return action_item
    
    
    
    
    
    
    
    
    
    def pull_action_from_habitat_store(self) -> dict:
        """
        Pull next action from HabitatStore queue.
        NOW PASSES METADATA to MainOrchestrator for velocity control
        🔥 STOP PRIORITY: Preempts stale velocity commands when STOP detected
        """
        print(f"🎯 HABITATSTORE: Pulling next action from queue")
        
        # Check if queue exists and has items
        if not hasattr(self, 'action_queue') or not self.action_queue:
            print("📭 HABITATSTORE: Action queue is empty")
            return None
        
        with self._lock: 
            # 🔥 STEP 1: Scan for STOP priority (NO popping yet)
            stop_index = -1
            for i, action in enumerate(self.action_queue):
                if(action["type"] == "velocity_control" and 
                action.get("metadata", {}).get("is_stop_command", False)):

                    stop_index = i
                    print(f"🛑 STOP PRIORITY DETECTED at index {i}: {action['action_id']}")
                    break
            
            # 🔥 STEP 2-4: If STOP found, cancel ALL actions BEFORE it
            if stop_index > 0:
                cancelled_count = 0
                for i in range(stop_index):
                    cancelled_action = self.action_queue[i]
                    action_id = cancelled_action["action_id"]
                    
                    # Generate skipped result via store_action_result (proper locking)
                    skipped_result = {
                        "status": "skipped_for_stop",
                        "skipped": True,
                        "collided": False,
                        "position": None,
                        "message": f"Preempted by STOP {self.action_queue[stop_index]['action_id']}"
                    }
                    print(f"Skipped action: {skipped_result}")
                    self.store_action_result(action_id, skipped_result)
                    cancelled_count += 1
                
                # Remove cancelled actions [0:stop_index)
                del self.action_queue[:stop_index]
                print(f"🚨 STOP PREEMPTION: Cleared {cancelled_count} stale actions")
            
            # 🔥 STEP 5: Normal FIFO pop (STOP now at index 0, or regular action)
            action_item = self.action_queue.pop(0)
            action_item["status"] = "executing"
            action_item["execution_thread"] = threading.current_thread().name
            
            # 🎯 DEBUG: Check if this is velocity data
            if action_item.get("type") == "velocity_control":
                print(f"🚀 HABITATSTORE: Pulled VELOCITY CONTROL command")
                print(f"   📦 Metadata: {list(action_item.get('metadata', {}).keys())}")
            else:
                print(f"🚀 HABITATSTORE: Pulled action '{action_item['type']}' (ID: {action_item['action_id']})")
            
            print(f"📊 HABITATSTORE: Remaining in queue: {len(self.action_queue)}")
        
        return action_item

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def get_current_agent_state(self) -> Optional[Dict[str, Any]]:
        """Get current agent position and rotation - REAL data from Habitat"""
        with self._lock:
            if self.simulator is None:
                return None
            
            try:
                agent_state = self.simulator.get_agent(0).get_state()
                return {
                    'position': agent_state.position.copy(),  # [x, y, z]
                    'rotation': agent_state.rotation,         # Quaternion object
                    'timestamp': time.time()
                }
            except Exception as e:
                print(f"❌ Failed to get agent state: {e}")
                return None

    def update_motion_tracking(self):
        """Update motion tracking statistics - call this after each movement"""
        with self._lock:
            if self.simulator is None:
                return
            
            current_state = self.get_current_agent_state()
            if not current_state:
                return
            
            current_pos = current_state['position']
            current_rot = current_state['rotation']
            
            # Store in history (keep last 50 positions)
            self.position_history.append({
                'position': current_pos,
                'rotation': current_rot,
                'timestamp': current_state['timestamp']
            })
            if len(self.position_history) > 50:
                self.position_history.pop(0)
            
            # Update last position
            self.last_agent_position = current_pos
            self.last_agent_rotation = current_rot
            
            # Update motion statistics if we have enough history
            if len(self.position_history) >= 2:
                self._update_motion_statistics()

    def _update_motion_statistics(self):
        """Calculate real motion statistics from position history"""
        if len(self.position_history) < 2:
            return
        
        translations = []
        rotations = []
        
        for i in range(1, len(self.position_history)):
            pos1 = self.position_history[i-1]['position']
            pos2 = self.position_history[i]['position']
            
            # Real translation distance
            trans = np.linalg.norm(np.array(pos2) - np.array(pos1))
            translations.append(trans)
            
            # Real rotation difference (quaternion angle)
            rot1 = self.position_history[i-1]['rotation']
            rot2 = self.position_history[i]['rotation']
            rot_diff = self._quaternion_angle(rot1, rot2)
            rotations.append(rot_diff)
        
        if translations:
            self.motion_stats['avg_translation'] = np.mean(translations[-10:])  # Last 10 moves
            self.motion_stats['max_translation'] = np.max(translations)
            self.motion_stats['min_translation'] = np.min(translations)
        
        if rotations:
            self.motion_stats['avg_rotation'] = np.mean(rotations[-10:])

    def _quaternion_angle(self, q1, q2) -> float:
        """Calculate REAL angle between two quaternions in radians"""
        try:
            # Habitat quaternion format: [w, x, y, z]
            if hasattr(q1, 'w'):
                dot = q1.w*q2.w + q1.x*q2.x + q1.y*q2.y + q1.z*q2.z
            elif isinstance(q1, (list, tuple, np.ndarray)) and len(q1) == 4:
                dot = q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]
            else:
                # Default to 0 if format unknown
                return 0.0
            
            # Clamp to valid range
            dot = max(-1.0, min(1.0, dot))
            
            # Angle between quaternions (in radians)
            return 2.0 * np.arccos(abs(dot))
        except Exception as e:
            print(f"⚠️ Quaternion angle calculation failed: {e}")
            return 0.
        
    def __enter__(self):
        """Context manager enter - ensures thread safety"""
        print(f"🔒 CONTEXT MANAGER: Entering (Thread: {threading.current_thread().name})")
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - releases lock"""
        print(f"🔓 CONTEXT MANAGER: Exiting (Thread: {threading.current_thread().name})")
        self._lock.release()




    def shutdown(self):
        """Clean shutdown - release simulator reference"""
        with self._lock:
            self.simulator = None
            self.camera_intrinsics = None
            print("🛑 HabitatStore shutdown complete") 
            
# Global singleton instance
global_habitat_store = HabitatStore()



