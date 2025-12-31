#!/usr/bin/env python3
"""
Minimal HM3D Semantic + Navigation Script (Correct Habitat Usage)

- Loads HM3D geometry (.basis.glb)
- Loads HM3D semantic annotations (.semantic.glb)
- Loads navmesh
- Moves the agent in the scene
- Extracts ALL ground-truth semantic objects
- Saves them to JSON

NO argparse
NO external models
PURE Habitat-Sim
"""

import os
import json
import time
import numpy as np
import habitat_sim


# =========================
# 🔧 HARD-CODED SCENE PATH
# =========================
SCENE_PATH = (
    "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
    "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
    "datasets/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
)

SEMANTIC_PATH = SCENE_PATH.replace(".basis.glb", ".semantic.glb")
NAVMESH_PATH = SCENE_PATH.replace(".basis.glb", ".navmesh")


# =========================
# 🚀 CREATE SIMULATOR
# =========================
def make_simulator():
    assert os.path.exists(SCENE_PATH), "Scene GLB not found"
    assert os.path.exists(SEMANTIC_PATH), "Semantic GLB not found"

    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = SCENE_PATH
    # cfg.semantic_scene_id = SEMANTIC_PATH
    
    
    
    cfg.scene_dataset_config_file = (
    "/mnt/d/Coding/Business/Kulfi_Startup_Code/"
    "robotics_workspace/projects/hybrid_zero_shot_slam_nav/main/"
    "datasets/hm3d_annotated_basis.scene_dataset_config.json"
)



    cfg.load_semantic_mesh = True
    cfg.enable_physics = True
    cfg.gpu_device_id = -1  # CPU only (safe)

    # -------- Agent --------
    agent_cfg = habitat_sim.AgentConfiguration()

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [480, 640]
    rgb.hfov = 90

    depth = habitat_sim.CameraSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [480, 640]
    depth.hfov = 90
    

    agent_cfg.sensor_specifications = [rgb, depth]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent_cfg]))

    # -------- NavMesh --------
    if os.path.exists(NAVMESH_PATH):
        sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
        print("✅ NavMesh loaded")
    else:
        print("⚠️ NavMesh NOT found")

    return sim


# =========================
# 🧭 SIMPLE NAVIGATION
# =========================
def navigate_scene(sim, steps=50):
    print("🧭 Navigating scene...")
    actions = list(sim.config.agents[0].action_space.keys())

    for _ in range(steps):
        action = np.random.choice(actions)
        sim.step(action)


# =========================
# 🧠 EXTRACT SEMANTIC OBJECTS
# =========================
def extract_semantic_objects(sim):
    semantic_scene = sim.semantic_scene
    objects = []

    for obj in semantic_scene.objects:
        if obj is None:
            continue

        data = {
            "object_id": obj.id,
            "category_name": obj.category.name() if obj.category else "unknown",
            "category_id": obj.category.index() if obj.category else -1,
        }
        objects.append(data)

    return objects
# =========================
# 💾 SAVE JSON
# =========================
def save_to_json(objects):
    out = {
        "scene": SCENE_PATH,
        "num_objects": len(objects),
        "objects": objects,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "habitat_sim_version": habitat_sim.__version__,
    }

    with open("hm3d_semantic_objects.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"💾 Saved {len(objects)} objects to hm3d_semantic_objects.json")


# =========================
# 🏁 MAIN
# =========================
if __name__ == "__main__":
    sim = make_simulator()
    navigate_scene(sim, steps=30)
    objects = extract_semantic_objects(sim)
    save_to_json(objects)
    sim.close()
    print("✅ Done.")
