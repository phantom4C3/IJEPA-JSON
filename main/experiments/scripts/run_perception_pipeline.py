
# What You're Saving:
# SLAM map data from ORB-SLAM3 (keyframes, points, camera poses)
# Navigation graph for path planning
# Exploration progress and mission state
# Session metadata (timestamp, frames processed, etc.)








# check method calls and signatures for consistency # check where we can implement in built tools instead of manually writing everything# PerceptionPipeline.initialize_processing()# → calls# HabitatStore.initialize()# → which starts the simulator

# do not pass any store to any component file via prop drilling ise global instance instead 



# 💠 FULL MODULAR FLOW# ✔ 1. HabitatStore (single responsibility)
# Loads HabitatSim
# Runs simulation
# Manages sensors
# Provides: (rgb, depth, pose, frame_id, timestamp)
# ✔ 2. PerceptionPipeline
# Calls HabitatStore.get_frame()
# Passes frame to ORB-SLAM and OWL
# Receives updated map from SLAM & OWL
# Merges inside CentralMapStore
# ✔ 3. ORBSLAMIntegration
# Uses frame for tracking
# Updates geometry in CentralMapStore
# ✔ 4. OWLIntegration
# Uses frame for embeddings
# Does 2D→3D projection (using pose + depth)
# Updates semantic objects in CentralMapStore
# ✔ 5. CentralMapStore
# Maintains full global map
# Stores fused SLAM + semantic objects




#!/usr/bin/env python3
"""
MODULAR PERCEPTION PIPELINE - Real-Time Semantic-Geometric Mapping
Uses MODULAR COMPONENTS with direct store imports
"""

import sys, os
import numpy as np
import time
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
 
from src.stores.central_map_store import CentralMapStore
from src.stores.task_store import TaskStore
from src.stores.prediction_store import PredictionStore

# ✅ FIXED ABSOLUTE IMPORTS (from main directory)  
from src.perception_pipeline.owl_integration import OWLIntegration

class PerceptionPipeline:
    """
    MODULAR Perception Pipeline - Orchestrates specialized components
    Fuses ORB-SLAM3 geometry with OWL semantic detections via shared stores
    """
     
    def __init__(self, map_store=None, task_store=None, prediction_store=None, user_command_store=None, orb_slam=None):
            # ✅ Use injected stores OR create new ones as fallback 
            self.map_store = map_store or CentralMapStore()
            self.task_store = task_store or TaskStore()
            self.prediction_store = prediction_store or PredictionStore()
            self.user_command_store = user_command_store
            
            self.orb_slam = orb_slam

            self.owl = OWLIntegration( 
                map_store=self.map_store, 
                task_store=self.task_store,
                prediction_store=self.prediction_store
            )
            self.is_running = False

            logging.info("Modular Perception Pipeline instance created (call initialize_processing to start)")
    
    def initialize_processing(self):
        """
        One-time initialization called by main orchestrator
        Initializes all modular components
        """
        logging.info("Initializing modular perception processing...")
        
         
        
        # Initialize state
        self.is_running = True
      
        # ✅ Initialize OWL ONLY ONCE
        if self.owl.initialize():
            # ✅ THEN start continuous processing
            self.owl.start_continuous_processing()
            logging.info("🦉 OWL continuous processing started")
        else:
            logging.error("❌ OWL initialization failed")
              
        # Update task store with perception status
        # Check if method exists, otherwise use fallback
        if hasattr(self.task_store, 'update_perception_status'):
            self.task_store.update_perception_status('initialized')
        else:
            # Fallback: update task data directly or just log
            logging.info("Perception pipeline initialized successfully")
            if hasattr(self.task_store, 'current_task'):
                # Initialize task if None
                if self.task_store.current_task is None:
                    self.task_store.current_task = {'status': 'initialized'}
                else:
                    self.task_store.current_task['status'] = 'initialized'
                            
            # Check if method exists, otherwise use fallback
            if hasattr(self.task_store, 'update_perception_status'):
                self.task_store.update_perception_status('initialized')
            else:
                logging.info("Perception pipeline initialized - TaskStore status updated")
                # Initialize task if needed
                if self.task_store.current_task is None:
                    self.task_store.current_task = {}
                self.task_store.current_task['perception_status'] = 'initialized'  
                      
        logging.info(f"Modular perception processing initialized")
    
    def shutdown(self):
        """Clean shutdown of the perception pipeline"""
        self.is_running = False
        
        # ✅ Shutdown modular components
        self.orb_slam.shutdown()
        self.owl.shutdown()
        
        # Save final map through store if available
        if hasattr(self.map_store, 'save_final_map'):
            self.map_store.save_final_map()
        else:
            # Basic final save 
            filename = "projects/hybrid_zero_shot_slam_nav/main/experiments/results/logs/orbslam_map.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            try:
                with open(filename, 'w') as f:
                    # Get fused map from store
                    fused_map = self.map_store.get_fused_map() if hasattr(self.map_store, 'get_fused_map') else {}
                    json.dump(self._make_serializable(fused_map), f, indent=2)
                logging.info(f"Final map saved: {filename}")
            except Exception as e:
                logging.error(f"Failed to save final map: {e}")
        
        self.task_store.update_perception_status('stopped')
        logging.info("Modular perception pipeline shutdown complete")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Helper to make objects JSON serializable"""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (dict)):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example standalone usage
    config = {
        'save_interval': 30,
        'owl_interval': 3,
        'object_size_priors': {
            'chair': [0.5, 0.5, 0.8],
            'table': [1.0, 1.0, 0.7]
        }
    }
    
    pipeline = PerceptionPipeline(config)
    pipeline.initialize_processing()
    
    try:
        print("Starting pipeline")
    except KeyboardInterrupt:
        pipeline.shutdown()
        print("Ending pipeline")