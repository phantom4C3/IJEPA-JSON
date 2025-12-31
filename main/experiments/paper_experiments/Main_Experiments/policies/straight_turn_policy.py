
#
# HABITAT ENV CREATION
# #
# def create_habitat_env(scene_path: str):
#     print("[DEBUG] ENTER create_habitat_env()")
#     print(f"[DEBUG] scene_path arg = {scene_path}")

#     print("[DEBUG] Resolving absolute scene path")
#     abs_scene_path = (
#         "/mnt/d/Coding/Business/Kulfi_Startup_Code/robotics_workspace/"
#         "projects/hybrid_zero_shot_slam_nav/main/datasets/"
#         "00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
#     )
#     print(f"[DEBUG] abs_scene_path = {abs_scene_path}")

#     print("[DEBUG] Checking if GLB exists")
#     if not os.path.exists(abs_scene_path):
#         print("[ERROR] GLB missing!")
#         raise FileNotFoundError(f"❌ GLB missing: {abs_scene_path}")
#     print("[DEBUG] GLB exists")


#     print("[DEBUG] Creating SimulatorConfiguration")
#     cfg = habitat_sim.SimulatorConfiguration()
#     print("[DEBUG] SimulatorConfiguration created")

#     sim.seed(42)

#     cfg.scene_id = abs_scene_path
#     print(f"[DEBUG] cfg.scene_id set = {cfg.scene_id}")


#     cfg.scene_dataset_config_file = (  # ← ADD THIS!
#     "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
#     "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
#     "datasets/hm3d_annotated_basis.scene_dataset_config.json"
# )


#     cfg.gpu_device_id = -1
#     print("[DEBUG] cfg.gpu_device_id = -1 (CPU)")

#     cfg.enable_physics = True
#     print("[DEBUG] cfg.enable_physics = True")

#     cfg.load_semantic_mesh = True
#     print("[DEBUG] cfg.load_semantic_mesh = True")

#     cfg.force_separate_semantic_scene_graph = True
#     print("[DEBUG] cfg.force_separate_semantic_scene_graph = True")

#     print("[DEBUG] Creating AgentConfiguration")
#     agent_cfg = habitat_sim.AgentConfiguration()

#     print("[DEBUG] Creating RGB CameraSensorSpec")
#     rgb_spec = habitat_sim.CameraSensorSpec()
#     rgb_spec.uuid = "rgba_camera"
#     print("[DEBUG] rgb_spec.uuid set")

#     rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
#     print("[DEBUG] rgb_spec.sensor_type = COLOR")

#     rgb_spec.resolution = [640, 480]
#     print("[DEBUG] rgb_spec.resolution set")

#     rgb_spec.hfov = 90.0
#     print("[DEBUG] rgb_spec.hfov set")

#     rgb_spec.position = [0.0, 1.7, 0.0]
#     print("[DEBUG] rgb_spec.position set")

#     agent_cfg.sensor_specifications = [rgb_spec]
#     print("[DEBUG] Agent sensor specifications assigned")

#     print("[DEBUG] Creating habitat_sim.Configuration")
#     sim_config = habitat_sim.Configuration(cfg, [agent_cfg])

#     print("[DEBUG] Creating Simulator")
#     sim = habitat_sim.Simulator(sim_config)
#     print("[DEBUG] Simulator created")

#     navmesh_path = abs_scene_path.replace(".glb", ".navmesh")
#     print(f"[DEBUG] navmesh_path = {navmesh_path}")

#     if os.path.exists(navmesh_path):
#         print("[DEBUG] Navmesh exists, loading")
#         sim.pathfinder.load_nav_mesh(navmesh_path)
#         print("[DEBUG] Navmesh loaded")
#     else:
#         print("[WARN] Navmesh NOT found")

#     print("[DEBUG] Checking semantic_scene")
#     print(f"[DEBUG] semantic_scene exists = {sim.semantic_scene is not None}")
#     print(f"[DEBUG] semantic objects count = {len(sim.semantic_scene.objects)}")

#     print("[DEBUG] EXIT create_habitat_env()\n")
#     return sim










# def execute_velocity_step(sim, lin_vel: list, ang_vel: list):
#     print("[DEBUG] ENTER execute_velocity_step()")

#     agent = sim.get_agent(0)
#     print("[DEBUG] Got agent")

#     state = agent.get_state()
#     print(f"[DEBUG] Current position = {state.position}")

#     duration = 1.0 / 60.0
#     print(f"[DEBUG] duration = {duration}")

#     vc = habitat_sim.physics.VelocityControl()
#     print("[DEBUG] VelocityControl created")

#     vc.controlling_lin_vel = True
#     vc.controlling_ang_vel = True
#     vc.lin_vel_is_local = True
#     vc.ang_vel_is_local = True
#     print("[DEBUG] VelocityControl flags set")

#     vc.linear_velocity = mn.Vector3(*lin_vel)
#     vc.angular_velocity = mn.Vector3(*ang_vel)
#     print(f"[DEBUG] lin_vel={lin_vel}, ang_vel={ang_vel}")

#     q = state.rotation
#     print(f"[DEBUG] Current rotation quaternion = {q}")

#     rigid_state = habitat_sim.RigidState(
#         mn.Quaternion(mn.Vector3(q.x, q.y, q.z), q.w),
#         mn.Vector3(state.position),
#     )
#     print("[DEBUG] RigidState created")

#     new_rigid_state = vc.integrate_transform(duration, rigid_state)
#     print("[DEBUG] Integrated transform")

#     new_pos = np.array(
#         [
#             new_rigid_state.translation.x,
#             new_rigid_state.translation.y,
#             new_rigid_state.translation.z,
#         ]
#     )
#     print(f"[DEBUG] new_pos = {new_pos}")

#     pathfinder = sim.pathfinder
#     xz_pos = np.array([new_pos[0], 0.0, new_pos[2]])
#     is_nav = pathfinder.is_navigable(xz_pos)
#     print(f"[DEBUG] is_navigable = {is_nav}")

#     current_y = state.position[1]
#     if is_nav:
#         safe_pos = np.array([new_pos[0], current_y, new_pos[2]])
#         print("[DEBUG] Using navigable position")
#     else:
#         snapped = pathfinder.snap_point(xz_pos)
#         print(f"[DEBUG] snapped = {snapped}")
#         if snapped is None:
#             print("[DEBUG] NAVMESH BLOCKED!")
#             return {"collided": True, "position": state.position.tolist()}  # ← ADD HERE
#         safe_pos = np.array([snapped[0], current_y, snapped[2]])

#     r = new_rigid_state.rotation
#     state.position = safe_pos
#     state.rotation = np.array([r.vector.x, r.vector.y, r.vector.z, r.scalar])
#     agent.set_state(state)

#     print(f"[DEBUG] State updated to {state.position}")

#     # ✅ HABITAT TRUTH COLLISION (like _execute_habitat_action)
#     observations = sim.get_sensor_observations()
#     collided = observations.get("collided", False)  # Habitat's ground truth!
#     print(f"[DEBUG] HABITAT collided={collided} (sensor truth)")

#     return {"collided": collided, "position": state.position.tolist()}




# This debug output shows a critical problem — the category names are not being read correctly from the semantic objects in the scene. The issue is visible in lines like:

# text
# obj.category.name='<bound method PyCapsule.name of <habitat_sim._ext.habitat_sim_bindings.SemanticCategory object at 0x72bcc98c0bb0>>'
# This means that obj.category.name is not returning a string, but is instead returning the method itself as a string representation, not calling it. That's why the goal category 'microwave' is never found — because the system is comparing 'microwave' to strings like '<bound method PyCapsule.name ...>'.

# 🔍 What This Means
# The object categories are NOT wrong — the HM3D semantic file we checked earlier shows the correct category names (microwave, armchair, etc.).

# The problem is in your code where you're trying to access obj.category.name. In Habitat-Sim's Python bindings, obj.category is a C++ object wrapped in Python, and obj.category.name is a method that needs to be called to get the string.

# The fix: You should be calling obj.category.name() — with parentheses — not accessing it as an attribute.


# 6️⃣ What you are ACTUALLY evaluating right now (important)
# Your current system evaluates:
# Semantic object discovery during unguided exploration
# Not:
# ObjectNav
# Goal-conditioned navigation
# SPL-based navigation success
# This is still valid, but metrics must match the task.


# 8️⃣ Why category match can still fail (even with semantics)
# HM3D categories are things like:
# "chair"
# "table"
# "sofa"
# If your episode JSON uses:
# "Chair"
# "DiningTable"
# "couch"
# ➡️ Exact string mismatch = no goal found


# 9️⃣ Final: what to fix (summary)
# ✅ REQUIRED FIX (most important)

# Add this to create_habitat_env:

# cfg.scene_dataset_config_file = (
#     "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
#     "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
#     "datasets/hm3d_annotated_basis.scene_dataset_config.json"
# )

# 🚫 Remove fallback goal (after fixing semantics)

# Fallback hides bugs and breaks metrics validity.

# ⚠️ Collision counting

# Count collisions:

# In both STRAIGHT and TURN

# Or directly accumulate collided == True every step


# SUCCESS:     ❌ 0/1 (expected for blind policy)
# SPL:         0.51 (good! straight paths)
# Path Length: 3.98m (explored decent area)
# Collisions:  0 (navmesh snapping works)
# Final Dist:  3.04m from fallback goal [2.0, 0.05, 1.5]
# Failure:     "distance" (correct)
# Time:        46.6s (500 steps @ 60Hz)


# HM3D ObjectNav:
# FBE:     23.7% SR | 0.123 SPL
# SemExp:  37.9% SR | 0.188 SPL  ← REIMPLEM ENTED (not official)
# OpenFMNav: 52.1% SR | 0.312 SPL

# HM3D ObjectNav Benchmark (Yadav et al., 2022):
# - 80 train scenes, 20 val scenes
# - 2000 episodes total
# - 6 goal classes: chair, couch, potted plant, bed, toilet, tv
# - Standard Habitat metrics (SR, SPL)


# HM3D ObjectNav Benchmark (Yadav et al., 2022):
# - 80 train scenes, 20 val scenes
# - 2000 episodes total
# - 6 goal classes: chair, couch, potted plant, bed, toilet, tv
# - Standard Habitat metrics (SR, SPL)

# Table Format:
# Policy              | SR     | SPL    | Source
# --------------------|--------|--------|------------------------
# Random              | 4.2%   | 0.03   | Habitat 2022 [cite]
# FBE                 | 23.7%  | 0.123  | OpenFMNav [cite]
# SemExp              | 37.9%  | 0.188  | OpenFMNav [cite]
# OpenFMNav           | 52.1%  | 0.312  | OpenFMNav [cite]
# YOUR_straight_turn  | ??%    | ??     | YOUR (run now)
# YOUR_hybrid         | 60%+   | ??     | YOUR


# VELOCITY STEP

# def execute_velocity_step(sim, lin_vel, ang_vel, duration=1.0/60.0):
#     print("[DEBUG] ENTER execute_velocity_step()")

#     print("[DEBUG] Getting agent 0")
#     agent = sim.get_agent(0)

#     print("[DEBUG] Reading current agent state")
#     state = agent.get_state()
#     print(f"[DEBUG] Current position = {state.position}")
#     print(f"[DEBUG] Current rotation = {state.rotation}")

#     print("[DEBUG] Creating VelocityControl")
#     vc = habitat_sim.physics.VelocityControl()

#     print("[DEBUG] Enabling linear velocity control")
#     vc.controlling_lin_vel = True

#     print("[DEBUG] Enabling angular velocity control")
#     vc.controlling_ang_vel = True

#     print("[DEBUG] Setting linear velocity as LOCAL frame")
#     vc.lin_vel_is_local = True

#     print("[DEBUG] Setting angular velocity as LOCAL frame")
#     vc.ang_vel_is_local = True

#     print(f"[DEBUG] Setting linear velocity = {lin_vel}")
#     vc.linear_velocity = mn.Vector3(*lin_vel)

#     print(f"[DEBUG] Setting angular velocity = {ang_vel}")
#     vc.angular_velocity = mn.Vector3(*ang_vel)

#     print("[DEBUG] Attaching VelocityControl to agent.controls")
#     agent.controls = vc  # 🔥 attach controller

#     print(f"[DEBUG] Stepping physics with duration = {duration}")
#     sim.step_physics(duration)

#     print("[DEBUG] Reading new agent state after physics step")
#     new_state = agent.get_state()
#     print(f"[DEBUG] New position = {new_state.position}")
#     print(f"[DEBUG] New rotation = {new_state.rotation}")

#     print("[DEBUG] Computing collision heuristic using np.allclose")
#     collided = np.allclose(
#         state.position,
#         new_state.position,
#         atol=1e-4
#     )

#     print(f"[DEBUG] Collision detected = {collided}")

#     print("[DEBUG] EXIT execute_velocity_step()\n")

#     return {
#         "collided": collided,
#         "position": new_state.position.tolist(),
#     }


import habitat_sim
import habitat_sim.agent
import numpy as np
import time
import os
import math
import magnum as mn
import cv2

print("[DEBUG] straight_turn_policy.py: Imported ALL dependencies")




def create_habitat_env(scene_path: str):
    print("[DEBUG] ENTER create_habitat_env()")
    print(f"[DEBUG] scene_path arg = {scene_path}")

    print("[DEBUG] Resolving absolute scene path")
    abs_scene_path = (
        "/mnt/d/Coding/Business/Kulfi_Startup_Code/robotics_workspace/"
        "projects/hybrid_zero_shot_slam_nav/main/datasets/"
        "00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    )
    print(f"[DEBUG] abs_scene_path = {abs_scene_path}")

    print("[DEBUG] Checking if GLB exists")
    if not os.path.exists(abs_scene_path):
        print("[ERROR] GLB missing!")
        raise FileNotFoundError(f"❌ GLB missing: {abs_scene_path}")
    print("[DEBUG] GLB exists")

    print("[DEBUG] Creating SimulatorConfiguration")
    cfg = habitat_sim.SimulatorConfiguration()
    print("[DEBUG] SimulatorConfiguration created")

    cfg.scene_id = abs_scene_path
    print(f"[DEBUG] cfg.scene_id set = {cfg.scene_id}")

    # 🔧 HM3D SEMANTIC CONFIG
    cfg.scene_dataset_config_file = (
        "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
        "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
        "datasets/hm3d_annotated_basis.scene_dataset_config.json"
    )
    print(f"[DEBUG] cfg.scene_dataset_config_file set")

    cfg.load_semantic_mesh = True
    cfg.force_separate_semantic_scene_graph = True
    cfg.gpu_device_id = -1
    cfg.enable_physics = True
    print("[DEBUG] All cfg settings applied")

    print("[DEBUG] Creating AgentConfiguration")
    agent_cfg = habitat_sim.AgentConfiguration()

    print("[DEBUG] Creating RGB CameraSensorSpec")
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgba_camera"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [640, 480]
    rgb_spec.hfov = 90.0
    rgb_spec.position = [0.0, 1.7, 0.0]
    agent_cfg.sensor_specifications = [rgb_spec]
    print("[DEBUG] Agent sensor specifications assigned")

    print("[DEBUG] Creating habitat_sim.Configuration")
    sim_config = habitat_sim.Configuration(cfg, [agent_cfg])

    print("[DEBUG] Creating Simulator")
    sim = habitat_sim.Simulator(sim_config)
    print("[DEBUG] Simulator created")
    
    
    import random  

    # ✅ SEED AFTER SIM CREATION
    sim.seed(random.randint(0, 1000))  
    print("[DEBUG] sim.seed(42) applied")

    # 🚨 SEMANTIC DEBUG - NOW sim EXISTS!
    print("\n🔍 === SEMANTIC MESH DEBUG ===")
    print(f"[DEBUG] cfg.load_semantic_mesh = {cfg.load_semantic_mesh}")
    print(f"[DEBUG] sim.semantic_scene is None? {sim.semantic_scene is None}")
    print(f"[DEBUG] semantic objects count = {len(sim.semantic_scene.objects)}")

    # LIST FIRST 5 OBJECTS
    print("[DEBUG] FIRST 5 SEMANTIC OBJECTS:")
    for i, obj in enumerate(sim.semantic_scene.objects[:5]):
        cat_name = obj.category.name() if obj.category else "NO_CATEGORY"
        print(f"  {i}: '{cat_name}'")

    # CONFIG FILE CHECK
    config_path = cfg.scene_dataset_config_file
    print(f"[DEBUG] config_file exists: {os.path.exists(config_path)}")
    print("🔍 === END SEMANTIC DEBUG ===\n")

    # Navmesh
    navmesh_path = abs_scene_path.replace(".glb", ".navmesh")
    print(f"[DEBUG] navmesh_path = {navmesh_path}")
    if os.path.exists(navmesh_path):
        print("[DEBUG] Navmesh exists, loading")
        sim.pathfinder.load_nav_mesh(navmesh_path)
        print("[DEBUG] Navmesh loaded")
    else:
        print("[WARN] Navmesh NOT found")

    print("[DEBUG] EXIT create_habitat_env()\n")
    return sim


#
# METRICS
#
def compute_spl(trajectory, goal_pos, success):
    traj = np.array(trajectory)

    if not success or len(traj) < 2:
        return 0.0

    deltas = traj[1:] - traj[:-1]
    path_length = np.sum(np.linalg.norm(deltas, axis=1))

    euclid_dist = np.linalg.norm(traj[0] - np.array(goal_pos))

    spl = euclid_dist / max(path_length, euclid_dist)
    return float(np.clip(spl, 0.0, 1.0))


def compute_metrics(trajectory, goal_pos, collisions, steps):
    print("[DEBUG] ENTER compute_metrics()")

    final_pos = np.array(trajectory[-1])
    final_dist = np.linalg.norm(final_pos - np.array(goal_pos))
    success = final_dist < 0.2

    print(f"[DEBUG] final_dist = {final_dist}, success={success}")

    deltas = np.diff(np.array(trajectory), axis=0)
    path_length = np.sum(np.linalg.norm(deltas, axis=1))
    print(f"[DEBUG] path_length = {path_length}")

    spl = compute_spl(trajectory, goal_pos, success)
    failure_mode = (
        "collision" if collisions > 5 else "distance" if not success else "none"
    )

    metrics = {
        "success": int(success),
        "spl": spl,
        "path_length": path_length,
        "collisions": collisions,
        "final_distance": final_dist,
        "failure_mode": failure_mode,
        "trajectory": trajectory,
    }

    print("[DEBUG] EXIT compute_metrics()\n")
    return metrics


#
# STRAIGHT TURN POLICY
#
def run_straight_turn_policy(episode: dict):
    print("[DEBUG] ENTER run_straight_turn_policy()")
    print(f"[DEBUG] episode_id = {episode['episode_id']}")

    BASE_RESULTS_DIR = os.path.join(
        os.path.dirname(__file__), "..", "results"  # policies/
    )
    BASE_RESULTS_DIR = os.path.abspath(BASE_RESULTS_DIR)

    policy_name = "straight_turn_policy"
    frame_dir = os.path.join(BASE_RESULTS_DIR, policy_name, f"{episode['episode_id']}")

    print(f"[DEBUG] frame_dir = {frame_dir}")

    os.makedirs(frame_dir, exist_ok=True)
    print("[DEBUG] Frame directory ensured")

    start_time = time.time()
    print("[DEBUG] Timer started")

    sim = create_habitat_env(episode["scene_path"])
    agent = sim.get_agent(0)

    start_pos = episode["start_pose"]["position"]
    start_yaw = episode["start_pose"]["yaw"]
    print(f"[DEBUG] start_pos={start_pos}, start_yaw={start_yaw}")

    state = agent.get_state()
    state.position = np.array(start_pos)
    state.rotation = np.array(
        [
            0.0,
            math.sin(start_yaw * 0.5),
            0.0,
            math.cos(start_yaw * 0.5),
        ]
    )
    agent.set_state(state)
    print("[DEBUG] Start state set")
    
    
    
    

    goal_category = episode["goal"]["object_category"]
    print(f"[DEBUG] goal_category='{goal_category}' (type={type(goal_category)})")

    goal_pos = None
    print(f"[DEBUG] TOTAL semantic objects = {len(sim.semantic_scene.objects)}")

    for idx, obj in enumerate(sim.semantic_scene.objects):

        # TRY .name (string) first
        try:
            cat_name = obj.category.name()  # ← METHOD
            print(f"[DEBUG] idx={idx} obj.category.name()='{cat_name}'")
            if cat_name == goal_category:
                print(f"[DEBUG][GOAL HIT] idx={idx} '{cat_name}' == '{goal_category}'")
                c = obj.aabb.center()  # ← METHOD
                goal_pos = [c.x, c.y, c.z]
                break
        except Exception as e:
            print(f"[DEBUG] idx={idx} ERROR: {e}")
            continue

    if goal_pos is None:
        print("[WARN] Goal not found → fallback used")
        goal_pos = [2.0, 0.05, 1.5]
    else:
        print(f"[DEBUG] ✅ GOAL FOUND: {goal_pos}")

    trajectory = [state.position.copy()]
    collisions = 0
    max_steps = episode.get("max_steps", 500)
    mode = "STRAIGHT"
    turn_steps = 0
    turn_direction = 1
    

    print(f"[DEBUG] max_steps={max_steps}")

    for step in range(max_steps):
        print(f"[DEBUG] STEP {step}")

        if mode == "STRAIGHT":
            lin_vel = [0.25, 0.0, 0.0]
            ang_vel = [0.0, 0.0, 0.0]
            print("[DEBUG] Mode STRAIGHT")
        else:
            lin_vel = [0.0, 0.0, 0.0]
            ang_vel = [0.0, 1.57, 0.0]  # ✅ 90° RIGHT
            print("[DEBUG] Mode TURN RIGHT")

                # --- STEP ACTION BASED ON MODE ---
        if mode == "STRAIGHT":
            action = "move_forward"
        else:  # mode == "TURN"
            action = "turn_left"  # or "turn_right"

        obs = sim.step(action)  # Habitat handles movement & collisions
        collided = obs.get("collided", False)
        if collided:
            collisions += 1

        # Append current position to trajectory
        trajectory.append(agent.get_state().position.copy())

        # --- ObjectNav termination ---
        dist_to_goal = np.linalg.norm(np.array(trajectory[-1]) - np.array(goal_pos))
        if dist_to_goal < 0.2:
            print("[DEBUG] GOAL REACHED → STOP")
            break


        obs = sim.get_sensor_observations() 
        rgb = obs["rgba_camera"]
        frame_path = os.path.join(frame_dir, f"{policy_name}_step_{step:04d}.png")

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(frame_path, rgb_bgr)
        print(f"[DEBUG] Frame saved: {frame_path}")

        # ✅ CORRECT LOGIC STRUCTURE
        # ✅ ONE-DIRECTION: Always turn RIGHT 180°
        if mode == "STRAIGHT" and collided:
            mode = "TURN"
            turn_steps = 0
            print("[DEBUG] WALL → TURN")
        elif mode == "TURN":
            turn_steps += 1
            print(f"[DEBUG] turn_steps={turn_steps}")
            if turn_steps >= 2:
                mode = "STRAIGHT"
                turn_steps = 0
                print("[DEBUG] TURN COMPLETE → STRAIGHT")

    print("[DEBUG] Rollout finished")

        # STRAIGHT with no collision → continues automatically


    metrics = compute_metrics(trajectory, goal_pos, collisions, len(trajectory) - 1)
    metrics["time_taken"] = time.time() - start_time
    metrics["policy_type"] = "straight_turn"

    print(f"[DEBUG] FINAL METRICS = {metrics}")

    sim.close()
    print("[DEBUG] Simulator closed")
    print("[DEBUG] EXIT run_straight_turn_policy()")

    return metrics


print("[DEBUG] straight_turn_policy.py: ✅ FULL DEBUG VERSION LOADED")
