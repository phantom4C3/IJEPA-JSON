import habitat_sim
import numpy as np

class WaypointController:
    def __init__(self, sim):
        self.sim = sim
        self.velocity = habitat_sim.physics.VelocityControl()
        
    def go_to(self, target_x, target_y, max_speed=0.5):
        agent = self.sim.get_agent(0)
        reached = False
        
        while not reached:
            state = agent.get_state()
            current = state.position[:2]
            target = np.array([target_x, target_y])

            # Calculate direction
            direction = target - current
            distance = np.linalg.norm(direction)

            if distance < 0.05:
                print(f'🎯 Reached target!')
                reached = True
                break

            # Calculate velocity
            forward_speed = min(distance * 2.0, max_speed)

            # Calculate turn (simplified - real code needs quaternion to yaw)
            target_angle = np.arctan2(direction[1], direction[0])
            current_angle = 0.0  # Simplified

            angle_diff = target_angle - current_angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
                
            turn_speed = np.clip(angle_diff * 2.0, -0.5, 0.5)
            
            # Apply - FIXED: Create RigidState from agent state
            self.velocity.linear_velocity = np.array([forward_speed, 0.0, 0.0])
            self.velocity.angular_velocity = np.array([0.0, 0.0, turn_speed])
            
            # Create RigidState from current position and rotation
            rigid_state = habitat_sim.RigidState(state.rotation, state.position)
            
            # Now integrate correctly
            new_rigid_state = self.velocity.integrate_transform(0.1, rigid_state)
            
            # Update agent state
            state.position = new_rigid_state.translation
            state.rotation = new_rigid_state.rotation
            agent.set_state(state)
            
            print(f'📍 Pos: [{state.position[0]:.2f}, {state.position[1]:.2f}], '
                  f'Dist: {distance:.2f}m, Speed: {forward_speed:.2f}m/s')

# Test it
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = 'NONE'
sim = habitat_sim.Simulator(
    habitat_sim.Configuration(sim_cfg, [habitat_sim.AgentConfiguration()])
)

controller = WaypointController(sim)
print('Navigating to waypoint [2.0, 1.0]...')
controller.go_to(2.0, 1.0)

print('\\n✅ Complete waypoint navigation using ONLY Habitat-Sim!')
