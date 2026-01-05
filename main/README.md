Actively Building !! 
Work in Progress !! 

# 🤖 Modular Robotics Navigation Pipeline

This repository contains a multi-stage pipeline for autonomous robot navigation, featuring SLAM integration, I-JEPA-based world modeling, Tree-of-Thoughts planning, and continuous velocity control.

## 📂 Project Structure

```text
main/src/
├── perception_pipeline/          # 👁️ External World Sensing
│   ├── ORB_SLAM3/                # C++ SLAM core
│   ├── orb_slam_integration.py   # Python wrapper for SLAM tracking
│   ├── orbslam_pybind.cpp        # Pybind11 bindings for C++/Python bridge
│   ├── owl_integration.py        # Open-World Localization/Detection
│   └── umeyama_alignment.py      # Spatial coordinate synchronization
│
├── prediction_pipeline/          # 🧠 Future State Modeling (I-JEPA)
│   ├── ijepa_model/              # Trained weights and architecture
│   ├── ijepa_integration_predictor.py  # I-JEPA inference logic
│   ├── collision_risk_predictor.py     # Structural risk assessment
│   └── ijepa_prototypes.json     # Encoded environmental feature templates
│
├── reasoning_planning_pipeline/  # ⚖️ Decision Making
│   ├── tree_of_thoughts_integration.py # Multi-path heuristic planning
│   └── spatial_reasoning_integration.py # NavMesh & geometric reasoning
│
├── action_pipeline/              # ⚙️ Execution & Control
│   ├── action_executor.py        # High-level action dispatcher
│   └── (ContinuousNavAgent logic) # PID/Velocity control loops
│
└── stores/                       # 🗄️ Central Data Hubs (IPC)
    ├── habitat_store.py          # Bridge to Habitat Sim / Physics feedback
    ├── central_map_store.py      # Global SLAM map & occupancy
    ├── prediction_store.py       # I-JEPA bias & risk scores
    └── task_store.py             # Mission status & goal tracking

```

---

## 🚀 Pipeline Workflow

The system operates in a reactive-semantic loop, allowing for both precise geometric navigation and intelligent "intuition" based on visual features.

### 1. Perception Layer

* **ORB_SLAM3:** Tracks the robot's pose in real-time and builds a point cloud.
* **OWL-ViT:** Identifies semantic objects in the frame.
* **Coordinate Alignment:** Uses the Umeyama algorithm to align the SLAM coordinate system with the simulator's global NavMesh.

### 2. Prediction Layer (I-JEPA Integration)

Unlike traditional pathfinders, this pipeline uses **I-JEPA** (Image Joint-Embedding Predictive Architecture) to predict the "continuity" of the environment.

* **Semantic Bias:** Predicts whether turning left or right leads to better "free space" travel.
* **Structural Risk:** Identifies potential collisions before they appear on the NavMesh.

### 3. Reasoning & Planning

* **Tree of Thoughts (ToT):** Evaluates multiple potential navigation paths based on task requirements.
* **Spatial Reasoning:** Converts high-level LLM commands into specific 3D target coordinates.

### 4. Action & Execution

* **Continuous Navigation:** Executes a `while` loop that pushes velocity commands (`lin_vel`, `ang_vel`) to the Habitat Simulator.
* **Reactive Tweak:** Implements a vector-based "push" that steers the robot away from obstacles using real-time NavMesh distance feedback.

---

## 🛠️ Key Components Detail

### **Action Executor (`action_executor.py`)**

Handles the transition from discrete plans to continuous movement. It manages:

* **Single-Turn Alignment:** Rotating toward the goal before moving.
* **Velocity Control:** Scaling speed based on distance and obstacle proximity.
* **Recovery Loops:** Automatic "Circle Escapes" when the robot detects it is stuck.

### **Habitat Store (`habitat_store.py`)**

The primary interface for asynchronous communication with the simulator. It allows the planning thread to "push" actions and "pull" results (position, rotation, collision flags) without blocking the simulation engine.

### **I-JEPA Predictor (`ijepa_integration_predictor.py`)**

Analyzes incoming frames against a set of prototype images to generate a `continuity_score`. This score is used as an **Angular Velocity Bias**, nudging the robot toward open areas.

---

## 📋 Requirements

* **Python 3.8+**
* **Habitat-Sim** & **Habitat-Lab**
* **Pybind11** (for ORB_SLAM3 bindings)
* **PyTorch** (for I-JEPA inference)
* **NumPy / SciPy** (for Umeyama alignment and vector math)
 
