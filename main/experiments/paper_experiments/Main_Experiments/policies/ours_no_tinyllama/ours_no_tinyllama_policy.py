#!/usr/bin/env python3
"""
ours_no_tinyllama.py - Ablation: Full system WITHOUT TinyLLaMA language reasoning

Same as ours_full.py but with TinyLLaMA component disabled.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class NoTinyLLaMANavigationCore:
    """
    Navigation core WITHOUT TinyLLaMA language reasoning.
    Identical to ContinuousNavigationCore but with use_tinyllama=False.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize with TinyLLaMA disabled"""
        self.config = config or {}
        
        # Navigation parameters
        self.max_linear_velocity = self.config.get('max_linear_velocity', 0.5)
        self.max_angular_velocity = self.config.get('max_angular_velocity', 1.5)
        self.success_threshold = self.config.get('success_threshold', 0.3)
        self.yaw_gain = self.config.get('yaw_gain', 2.0)
        
        # Keep I-JEPA enabled
        self.use_ijepa = self.config.get('use_ijepa', True)
        self.ijepa_bias_strength = self.config.get('ijepa_bias_strength', -0.3)
        self.ijepa_free_space_threshold = self.config.get('ijepa_free_space_threshold', 0.7)
        
        # 🚫 TinyLLaMA DISABLED (THE ABLATION)
        self.use_tinyllama = False
        
        # Recovery settings
        self.recovery_strategy = self.config.get('recovery_strategy', 'right_only')
        self.recovery_turn_angle = self.config.get('recovery_turn_angle', 1.57)
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        
        # Yaw locking
        self.lock_pitch_roll = self.config.get('lock_pitch_roll', True)
        
        # Object knowledge (without language model)
        self.object_locations = {}  # Simple memory of seen objects
        self.object_common_locations = self._get_default_object_locations()
        
        # Internal state
        self.current_yaw = 0.0
        self.collision_count = 0
        self.is_stuck = False
        self.last_positions = []
        self.recovery_attempts = 0
        self.last_action = None
        self.explored_locations = []  # Track where we've been
        
        print(f"🧭 No-TinyLLaMA Navigation Core initialized:")
        print(f"   Max lin: {self.max_linear_velocity}m/s, Max ang: {self.max_angular_velocity}rad/s")
        print(f"   🎨 I-JEPA: {self.use_ijepa}")
        print(f"   🚫 TinyLLaMA: DISABLED")
        print(f"   Recovery: {self.recovery_strategy}")
    
    def _get_default_object_locations(self) -> Dict[str, List[str]]:
        """
        Default object locations without language reasoning.
        Simple heuristic-based instead of LLM-generated.
        """
        return {
            'chair': ['living_room', 'dining_room', 'office', 'bedroom'],
            'table': ['living_room', 'dining_room', 'kitchen', 'office'],
            'bed': ['bedroom'],
            'couch': ['living_room', 'family_room'],
            'tv': ['living_room', 'bedroom', 'entertainment_room'],
            'refrigerator': ['kitchen'],
            'sink': ['kitchen', 'bathroom'],
            'toilet': ['bathroom'],
            'microwave': ['kitchen'],
            'plant': ['living_room', 'bedroom', 'hallway'],
            'cabinet': ['kitchen', 'bathroom', 'bedroom'],
            'desk': ['office', 'bedroom'],
            'bookshelf': ['office', 'living_room', 'bedroom'],
            'lamp': ['living_room', 'bedroom', 'office'],
            'window': ['living_room', 'bedroom', 'kitchen'],
            'door': ['hallway', 'entrance', 'room_entrances']
        }
    
    def compute_action(self, current_pos: np.ndarray, current_quat: np.ndarray,
                      goal_pos: np.ndarray, collided: bool = False,
                      depth_frame: Optional[np.ndarray] = None,
                      goal_object: Optional[str] = None) -> Dict[str, Any]:
        """
        Compute velocity command WITHOUT TinyLLaMA reasoning.
        
        Args:
            goal_object: Target object category (for object-aware navigation)
        """
        # Extract current yaw
        self.current_yaw = self._yaw_from_quat(current_quat)
        
        # Check if we're in recovery mode
        if collided:
            self.collision_count += 1
            self.recovery_attempts += 1
            
            if self.recovery_attempts <= self.max_recovery_attempts:
                return self._recovery_action()
            else:
                self.is_stuck = True
                self.recovery_attempts = 0
                return self._stuck_recovery()
        
        # Reset recovery attempts if not collided
        if not collided and self.recovery_attempts > 0:
            self.recovery_attempts = 0
        
        # ============================================
        # CONTINUOUS NAVIGATION LOGIC (NO TinyLLaMA)
        # ============================================
        
        # 1. Compute vector to goal
        delta = goal_pos - current_pos
        dist = np.linalg.norm(delta[[0, 2]])  # XY distance only
        
        print(f"  📏 Distance to goal: {dist:.2f}m")
        
        # 2. Check success
        if dist < self.success_threshold:
            print(f"  ✅ Within success threshold ({self.success_threshold}m)")
            return {
                "linear_velocity": [0.0, 0.0, 0.0],
                "angular_velocity": [0.0, 0.0, 0.0],
                "duration": 0.1,
                "is_velocity_command": True,
                "success": True
            }
        
        # 3. Compute yaw error
        desired_yaw = np.arctan2(delta[2], delta[0])
        yaw_err = (desired_yaw - self.current_yaw + np.pi) % (2 * np.pi) - np.pi
        
        print(f"  🧭 Yaw: {math.degrees(self.current_yaw):.1f}°, "
              f"Desired: {math.degrees(desired_yaw):.1f}°, "
              f"Error: {math.degrees(yaw_err):.1f}°")
        
        # 4. Velocity control (P-control)
        lin_vel = min(self.max_linear_velocity, dist)
        ang_vel = np.clip(yaw_err * self.yaw_gain, 
                         -self.max_angular_velocity, 
                         self.max_angular_velocity)
        
        # 5. Add I-JEPA bias if enabled
        if self.use_ijepa and depth_frame is not None:
            ijepa_bias = self._compute_ijepa_bias(depth_frame)
            if ijepa_bias != 0:
                print(f"  🎨 I-JEPA bias: {ijepa_bias:.3f}")
                ang_vel = np.clip(ang_vel + ijepa_bias,
                                -self.max_angular_velocity,
                                self.max_angular_velocity)
        
        # 6. 🚫 NO TinyLLaMA reasoning applied
        # In ours_full.py, this would adjust goal or strategy based on language
        
        # 7. Simple object-aware heuristic (without LLM)
        if goal_object and self._should_adapt_for_object(goal_object, current_pos):
            # Simple heuristic: for certain objects, be more cautious
            if goal_object in ['tv', 'plant', 'lamp']:
                # Fragile objects - slow down
                lin_vel *= 0.7
                print(f"  ⚠️  Caution mode for {goal_object}")
            elif goal_object in ['refrigerator', 'cabinet', 'door']:
                # Large objects - wider approach
                ang_vel *= 1.2
                print(f"  🔄 Wide approach for {goal_object}")
        
        # 8. Apply yaw locking if enabled
        if self.lock_pitch_roll:
            ang_vel_vec = [0.0, float(ang_vel), 0.0]
        else:
            ang_vel_vec = [0.0, float(ang_vel), 0.0]
        
        print(f"  🚀 Velocity: lin={lin_vel:.2f}m/s, ang={ang_vel:.2f}rad/s")
        print(f"  🚫 No TinyLLaMA reasoning (ablation)")
        
        # Return velocity command
        action = {
            "linear_velocity": [float(lin_vel), 0.0, 0.0],
            "angular_velocity": ang_vel_vec,
            "duration": 0.1,
            "is_velocity_command": True,
            "yaw_error": float(yaw_err),
            "distance_to_goal": float(dist),
            "tinyllama_enabled": False  # Mark that TinyLLaMA is disabled
        }
        
        self.last_action = action
        return action
    
    def _should_adapt_for_object(self, goal_object: str, current_pos: np.ndarray) -> bool:
        """
        Simple heuristic for object-aware navigation without LLM.
        """
        # Track explored locations
        self.explored_locations.append(current_pos.copy())
        if len(self.explored_locations) > 100:
            self.explored_locations.pop(0)
        
        # Check if we've been searching too long
        if len(self.explored_locations) > 50:
            print(f"  🔍 Been searching for {goal_object} for a while")
            return True
        
        return False
    
    def _compute_ijepa_bias(self, depth_frame: np.ndarray) -> float:
        """
        I-JEPA free space bias (same as in full system).
        """
        if depth_frame is None or depth_frame.size == 0:
            return 0.0
        
        height, width = depth_frame.shape
        center_col = width // 2
        
        # Sample depth in central column
        front_depths = depth_frame[:, center_col]
        valid_depths = front_depths[front_depths > 0]
        
        if len(valid_depths) == 0:
            return 0.0
        
        avg_depth = np.mean(valid_depths)
        
        # More free space = stronger left bias
        if avg_depth > 2.0:
            return self.ijepa_bias_strength
        elif avg_depth > 1.0:
            return self.ijepa_bias_strength * 0.5
        else:
            return 0.0
    
    def _yaw_from_quat(self, q: np.ndarray) -> float:
        """
        Extract yaw from quaternion (Habitat Y-up convention).
        Same as ours_full.py.
        """
        if len(q) == 4:
            w, x, y, z = q
            x, y, z, w = x, y, z, w  # Convert to [x, y, z, w]
        else:
            x, y, z, w = q
        
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def _recovery_action(self) -> Dict[str, Any]:
        """Recovery action when collision occurs"""
        print(f"  🚨 Collision detected! Attempt {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        recovery_duration = 0.4
        
        if self.recovery_strategy == "right_only":
            ang_vel = self.max_angular_velocity
        elif self.recovery_strategy == "alternating":
            ang_vel = self.max_angular_velocity if (self.recovery_attempts % 2 == 0) else -self.max_angular_velocity
        else:
            ang_vel = self.max_angular_velocity
        
        return {
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, float(ang_vel), 0.0],
            "duration": recovery_duration,
            "is_velocity_command": True,
            "recovery": True,
            "recovery_attempt": self.recovery_attempts,
            "tinyllama_enabled": False
        }
    
    def _stuck_recovery(self) -> Dict[str, Any]:
        """More aggressive recovery when stuck"""
        print(f"  ⚠️ Agent stuck! Performing stuck recovery")
        
        return {
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, float(self.recovery_turn_angle), 0.0],
            "duration": 0.6,
            "is_velocity_command": True,
            "stuck_recovery": True,
            "tinyllama_enabled": False
        }
    
    def reset(self):
        """Reset navigation state"""
        self.current_yaw = 0.0
        self.collision_count = 0
        self.is_stuck = False
        self.last_positions = []
        self.recovery_attempts = 0
        self.last_action = None
        self.explored_locations = []
        print("🧭 No-TinyLLaMA Navigation core reset")


class OursNoTinyLLaMAPolicy:
    """
    Policy interface for system WITHOUT TinyLLaMA.
    """
    
    def __init__(self, env=None, episode_info=None):
        """
        Initialize policy WITHOUT TinyLLaMA.
        
        Args:
            env: Habitat environment instance (optional)
            episode_info: Episode metadata (optional)
        """
        self.env = env
        self.episode_info = episode_info or {}
        
        # Configuration for system WITHOUT TinyLLaMA
        config = {
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 1.5,
            'success_threshold': 0.3,
            'yaw_gain': 2.0,
            'use_ijepa': True,  # Keep I-JEPA enabled
            'ijepa_bias_strength': -0.3,
            'use_tinyllama': False,  # 🚫 KEY DIFFERENCE
            'recovery_strategy': 'right_only',
            'recovery_turn_angle': 1.57,
            'max_recovery_attempts': 3,
            'lock_pitch_roll': True
        }
        
        # Initialize navigation core WITHOUT TinyLLaMA
        self.nav_core = NoTinyLLaMANavigationCore(config)
        
        # Goal tracking
        self.goal_position = None
        self.goal_object = None
        self._initialize_goal()
        
        # Simple exploration strategy (without language reasoning)
        self.exploration_phase = 'initial'
        self.search_patterns = self._get_search_patterns()
        
        print(f"🎯 No-TinyLLaMA Policy initialized")
        print(f"   Target: {self.goal_object}")
        print(f"   🎨 I-JEPA: ENABLED")
        print(f"   🚫 TinyLLaMA: DISABLED (ablation)")
    
    def _get_search_patterns(self):
        """Simple search patterns without LLM reasoning"""
        return {
            'initial': ['spiral', 'wall_follow', 'random'],
            'after_collision': ['right_turn', 'backtrack'],
            'long_search': ['expand_spiral', 'room_scan']
        }
    
    def _initialize_goal(self):
        """Initialize goal from episode info (without language understanding)"""
        if 'goal' in self.episode_info:
            goal_data = self.episode_info['goal']
            self.goal_object = goal_data.get('object_category')
            
            # Simple heuristic: different default positions for different objects
            start_pos = self.episode_info.get('start_pose', {}).get('position', [0, 0, 0])
            
            if self.goal_object in ['chair', 'table', 'couch']:
                # Furniture often in center of rooms
                self.goal_position = np.array([start_pos[0] + 4.0, start_pos[1], start_pos[2]])
            elif self.goal_object in ['refrigerator', 'sink', 'microwave']:
                # Kitchen appliances often along walls
                self.goal_position = np.array([start_pos[0] + 3.0, start_pos[1], start_pos[2] + 2.0])
            elif self.goal_object in ['toilet', 'cabinet']:
                # Bathroom fixtures often in corners
                self.goal_position = np.array([start_pos[0] + 2.0, start_pos[1], start_pos[2] + 3.0])
            else:
                # Default: 3m forward
                self.goal_position = np.array([start_pos[0] + 3.0, start_pos[1], start_pos[2]])
        else:
            self.goal_object = 'explore'
            self.goal_position = np.array([3.0, 0.0, 0.0])
    
    def _adapt_goal_based_on_exploration(self, current_pos: np.ndarray):
        """
        Simple goal adaptation without language reasoning.
        Adjust goal based on exploration history.
        """
        if len(self.nav_core.explored_locations) > 20:
            # If we've explored a lot without finding object, try different area
            avg_explored = np.mean(self.nav_core.explored_locations, axis=0)
            
            # Move goal to less explored area
            direction = np.random.uniform(-1, 1, size=2)
            direction = direction / np.linalg.norm(direction)
            
            self.goal_position = np.array([
                current_pos[0] + direction[0] * 4.0,
                current_pos[1],
                current_pos[2] + direction[1] * 4.0
            ])
            
            print(f"  🗺️ Adjusted goal to less explored area: {self.goal_position}")
    
    def act(self, observation: Dict, goal_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main policy interface WITHOUT TinyLLaMA.
        
        Args:
            observation: Dict with 'position', 'rotation', 'collided', 'depth' (optional)
            goal_info: Optional goal override
        
        Returns:
            Action command
        """
        # Extract current state
        current_pos = observation.get('position', np.zeros(3))
        current_quat = observation.get('rotation', [1.0, 0.0, 0.0, 0.0])
        collided = observation.get('collided', False)
        depth_frame = observation.get('depth')
        
        # Adapt goal if needed (without language reasoning)
        if self.nav_core.step_count % 30 == 0:  # Every 30 steps
            self._adapt_goal_based_on_exploration(current_pos)
        
        # Use provided goal_info or internal goal
        if goal_info and 'position' in goal_info:
            goal_pos = goal_info['position']
        elif self.goal_position is not None:
            goal_pos = self.goal_position
        else:
            goal_pos = np.array([3.0, 0.0, 0.0])
        
        # Compute action WITHOUT TinyLLaMA reasoning
        action = self.nav_core.compute_action(
            current_pos=current_pos,
            current_quat=current_quat,
            goal_pos=goal_pos,
            collided=collided,
            depth_frame=depth_frame,
            goal_object=self.goal_object
        )
        
        # Add policy metadata
        action['policy'] = 'ours_no_tinyllama'
        action['ablation'] = 'no_tinyllama'
        action['goal_object'] = self.goal_object
        action['exploration_phase'] = self.exploration_phase
        
        # Update exploration phase
        self.nav_core.step_count += 1
        if self.nav_core.step_count > 100:
            self.exploration_phase = 'long_search'
        elif self.nav_core.collision_count > 5:
            self.exploration_phase = 'after_collision'
        
        return action
    
    def reset(self, episode_info: Optional[Dict] = None):
        """Reset policy for new episode"""
        if episode_info:
            self.episode_info = episode_info
            self._initialize_goal()
        
        self.nav_core.reset()
        self.exploration_phase = 'initial'
        print(f"🔄 No-TinyLLaMA Policy reset")


def make_policy(env=None, episode_info=None):
    """
    Factory function for No-TinyLLaMA ablation.
    
    Args:
        env: Habitat environment (optional)
        episode_info: Episode configuration
    
    Returns:
        OursNoTinyLLaMAPolicy instance
    """
    policy = OursNoTinyLLaMAPolicy(env=env, episode_info=episode_info)
    return policy


# For backward compatibility
class NoTinyLLaMAPolicy(OursNoTinyLLaMAPolicy):
    """Alias for backward compatibility"""
    pass


# Test if run directly
if __name__ == "__main__":
    print("🧪 Testing No-TinyLLaMA ablation policy")
    
    # Create mock episode info
    mock_episode_info = {
        "episode_id": "test_no_tinyllama",
        "start_pose": {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0
        },
        "goal": {
            "object_category": "refrigerator",
            "position": [3.0, 0.0, 2.0]
        }
    }
    
    # Create policy
    policy = make_policy(episode_info=mock_episode_info)
    
    # Mock observation
    mock_obs = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rotation': np.array([1.0, 0.0, 0.0, 0.0]),
        'collided': False,
        'depth': np.ones((480, 640), dtype=np.float32) * 2.5
    }
    
    # Get action
    action = policy.act(mock_obs)
    
    print(f"\n✅ No-TinyLLaMA Policy created successfully")
    print(f"🎯 Target: {policy.goal_object}")
    print(f"📊 Action:")
    print(f"   Linear velocity: {action['linear_velocity']}")
    print(f"   Angular velocity: {action['angular_velocity']}")
    print(f"   TinyLLaMA enabled: {action.get('tinyllama_enabled', False)}")
    print(f"   Ablation: {action.get('ablation', 'unknown')}")
    print(f"   Exploration phase: {action.get('exploration_phase', 'initial')}")
    
    # Simulate collision and recovery
    print(f"\n🚨 Testing collision recovery...")
    mock_obs['collided'] = True
    recovery_action = policy.act(mock_obs)
    print(f"   Recovery action: {recovery_action.get('recovery', 'normal')}")
    print(f"   Recovery attempt: {recovery_action.get('recovery_attempt', 1)}")