#!/usr/bin/env python3
"""
ours_no_ijepa.py - Ablation: Full system WITHOUT I-JEPA visual priors

Same as ours_full.py but with I-JEPA component disabled.
"""

import numpy as np
import math
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class NoIJEPANavigationCore:
    """
    Navigation core WITHOUT I-JEPA visual priors.
    Identical to ContinuousNavigationCore but with use_ijepa=False.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize with I-JEPA disabled"""
        self.config = config or {}
        
        # Navigation parameters
        self.max_linear_velocity = self.config.get('max_linear_velocity', 0.5)
        self.max_angular_velocity = self.config.get('max_angular_velocity', 1.5)
        self.success_threshold = self.config.get('success_threshold', 0.3)
        self.yaw_gain = self.config.get('yaw_gain', 2.0)
        
        # 🚫 I-JEPA DISABLED (THE ABLATION)
        self.use_ijepa = False  # This is the key difference
        self.ijepa_bias_strength = 0.0  # Zero bias
        
        # Keep TinyLLaMA enabled
        self.use_tinyllama = self.config.get('use_tinyllama', True)
        
        # Recovery settings
        self.recovery_strategy = self.config.get('recovery_strategy', 'right_only')
        self.recovery_turn_angle = self.config.get('recovery_turn_angle', 1.57)
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        
        # Yaw locking
        self.lock_pitch_roll = self.config.get('lock_pitch_roll', True)
        
        # Internal state
        self.current_yaw = 0.0
        self.collision_count = 0
        self.is_stuck = False
        self.last_positions = []
        self.recovery_attempts = 0
        self.last_action = None
        
        print(f"🧭 No-IJEPA Navigation Core initialized:")
        print(f"   Max lin: {self.max_linear_velocity}m/s, Max ang: {self.max_angular_velocity}rad/s")
        print(f"   🚫 I-JEPA: DISABLED")
        print(f"   📝 TinyLLaMA: {self.use_tinyllama}")
        print(f"   Recovery: {self.recovery_strategy}")
    
    def compute_action(self, current_pos: np.ndarray, current_quat: np.ndarray,
                      goal_pos: np.ndarray, collided: bool = False,
                      depth_frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute velocity command WITHOUT I-JEPA bias.
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
        # CONTINUOUS NAVIGATION LOGIC (NO I-JEPA)
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
        
        # 🚫 NO I-JEPA BIAS ADDED HERE (the ablation)
        # In ours_full.py, this would add: ang_vel += ijepa_bias
        
        # Apply TinyLLaMA reasoning if enabled
        if self.use_tinyllama:
            # Placeholder for language reasoning
            # Could adjust ang_vel based on semantic understanding
            pass
        
        # Apply yaw locking if enabled
        if self.lock_pitch_roll:
            ang_vel_vec = [0.0, float(ang_vel), 0.0]
        else:
            ang_vel_vec = [0.0, float(ang_vel), 0.0]
        
        print(f"  🚀 Velocity: lin={lin_vel:.2f}m/s, ang={ang_vel:.2f}rad/s")
        print(f"  🚫 No I-JEPA bias applied (ablation)")
        
        # Return velocity command
        action = {
            "linear_velocity": [float(lin_vel), 0.0, 0.0],
            "angular_velocity": ang_vel_vec,
            "duration": 0.1,
            "is_velocity_command": True,
            "yaw_error": float(yaw_err),
            "distance_to_goal": float(dist),
            "ijepa_enabled": False  # Mark that I-JEPA is disabled
        }
        
        self.last_action = action
        return action
    
    def _yaw_from_quat(self, q: np.ndarray) -> float:
        """
        Extract yaw from quaternion (Habitat Y-up convention).
        Same as ours_full.py.
        """
        if len(q) == 4:
            # Habitat format: [w, x, y, z]
            w, x, y, z = q
            # Convert to [x, y, z, w] for our formula
            x, y, z, w = x, y, z, w
        else:
            # Already in [x, y, z, w] format
            x, y, z, w = q
        
        # ✅ CORRECT Y-up yaw formula
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return yaw
    
    def _recovery_action(self) -> Dict[str, Any]:
        """
        Recovery action when collision occurs.
        Same as ours_full.py.
        """
        print(f"  🚨 Collision detected! Attempt {self.recovery_attempts}/{self.max_recovery_attempts}")
        
        recovery_duration = 0.4
        
        if self.recovery_strategy == "right_only":
            ang_vel = self.max_angular_velocity  # Right turn
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
            "ijepa_enabled": False
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
            "ijepa_enabled": False
        }
    
    def reset(self):
        """Reset navigation state"""
        self.current_yaw = 0.0
        self.collision_count = 0
        self.is_stuck = False
        self.last_positions = []
        self.recovery_attempts = 0
        self.last_action = None
        print("🧭 No-IJEPA Navigation core reset")


class OursNoIJEPAPolicy:
    """
    Policy interface for system WITHOUT I-JEPA.
    """
    
    def __init__(self, env=None, episode_info=None):
        """
        Initialize policy WITHOUT I-JEPA.
        
        Args:
            env: Habitat environment instance (optional)
            episode_info: Episode metadata (optional)
        """
        self.env = env
        self.episode_info = episode_info or {}
        
        # Configuration for system WITHOUT I-JEPA
        config = {
            'max_linear_velocity': 0.5,
            'max_angular_velocity': 1.5,
            'success_threshold': 0.3,
            'yaw_gain': 2.0,
            'use_ijepa': False,  # 🚫 KEY DIFFERENCE
            'ijepa_bias_strength': 0.0,  # Zero bias
            'use_tinyllama': True,  # Keep LLaMA enabled
            'recovery_strategy': 'right_only',
            'recovery_turn_angle': 1.57,
            'max_recovery_attempts': 3,
            'lock_pitch_roll': True
        }
        
        # Initialize navigation core WITHOUT I-JEPA
        self.nav_core = NoIJEPANavigationCore(config)
        
        # Goal tracking
        self.goal_position = None
        self.goal_object = None
        self._initialize_goal()
        
        print(f"🎯 No-IJEPA Policy initialized")
        print(f"   Target: {self.goal_object}")
        print(f"   I-JEPA: DISABLED (ablation)")
    
    def _initialize_goal(self):
        """Initialize goal from episode info"""
        if 'goal' in self.episode_info:
            goal_data = self.episode_info['goal']
            self.goal_object = goal_data.get('object_category')
            
            # If goal position is provided, use it
            if 'position' in goal_data:
                self.goal_position = np.array(goal_data['position'])
            else:
                # Default goal: 3m forward from start
                start_pos = self.episode_info.get('start_pose', {}).get('position', [0, 0, 0])
                self.goal_position = np.array([start_pos[0] + 3.0, start_pos[1], start_pos[2]])
    
    def act(self, observation: Dict, goal_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main policy interface.
        
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
        
        # Use provided goal_info or internal goal
        if goal_info and 'position' in goal_info:
            goal_pos = goal_info['position']
        elif self.goal_position is not None:
            goal_pos = self.goal_position
        else:
            # Default goal
            goal_pos = np.array([3.0, 0.0, 0.0])
        
        # Compute action WITHOUT I-JEPA bias
        action = self.nav_core.compute_action(
            current_pos=current_pos,
            current_quat=current_quat,
            goal_pos=goal_pos,
            collided=collided,
            depth_frame=depth_frame
        )
        
        # Add policy metadata
        action['policy'] = 'ours_no_ijepa'
        action['ablation'] = 'no_ijepa'
        action['goal_object'] = self.goal_object
        
        return action
    
    def reset(self, episode_info: Optional[Dict] = None):
        """Reset policy for new episode"""
        if episode_info:
            self.episode_info = episode_info
            self._initialize_goal()
        
        self.nav_core.reset()
        print(f"🔄 No-IJEPA Policy reset")


def make_policy(env=None, episode_info=None):
    """
    Factory function for No-IJEPA ablation.
    
    Args:
        env: Habitat environment (optional)
        episode_info: Episode configuration
    
    Returns:
        OursNoIJEPAPolicy instance
    """
    policy = OursNoIJEPAPolicy(env=env, episode_info=episode_info)
    return policy


# For backward compatibility
class NoIJEPAPolicy(OursNoIJEPAPolicy):
    """Alias for backward compatibility"""
    pass


# Test if run directly
if __name__ == "__main__":
    print("🧪 Testing No-IJEPA ablation policy")
    
    # Create mock episode info
    mock_episode_info = {
        "episode_id": "test_no_ijepa",
        "start_pose": {
            "position": [0.0, 0.0, 0.0],
            "yaw": 0.0
        },
        "goal": {
            "object_category": "chair",
            "position": [2.0, 0.0, 2.0]
        }
    }
    
    # Create policy
    policy = make_policy(episode_info=mock_episode_info)
    
    # Mock observation
    mock_obs = {
        'position': np.array([0.0, 0.0, 0.0]),
        'rotation': np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
        'collided': False,
        'depth': np.ones((480, 640), dtype=np.float32) * 3.0
    }
    
    # Get action
    action = policy.act(mock_obs)
    
    print(f"\n✅ No-IJEPA Policy created successfully")
    print(f"🎯 Target: {policy.goal_object}")
    print(f"📊 Action:")
    print(f"   Linear velocity: {action['linear_velocity']}")
    print(f"   Angular velocity: {action['angular_velocity']}")
    print(f"   I-JEPA enabled: {action.get('ijepa_enabled', False)}")
    print(f"   Ablation: {action.get('ablation', 'unknown')}")
    
    # Test collision recovery
    mock_obs['collided'] = True
    recovery_action = policy.act(mock_obs)
    print(f"\n🚨 Collision recovery action:")
    print(f"   Type: {recovery_action.get('recovery', 'normal')}")