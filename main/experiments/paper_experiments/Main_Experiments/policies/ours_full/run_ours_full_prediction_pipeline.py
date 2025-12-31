 

# {
#     "pred_000001": {
#         "predictors": {
#             "structural_continuity": {
#                 "data": { /* I-JEPA results */ },
#                 "status": "completed",
#                 "timestamp": 1700000000.1
#             },
#             "collision_risk": {
#                 "data": { /* Collision results */ },
#                 "status": "completed", 
#                 "timestamp": 1700000000.2
#             },
#             "fused_decision": {
#                 "data": { /* Fused results */ },
#                 "status": "completed",
#                 "timestamp": 1700000000.3
#             }
#         }
#     }
# }





# check and tell me how are fused predictions calcaulated 






#!/usr/bin/env python3
"""
PREDICTION PIPELINE - Sequential Coordinator
Runs predictors one after another to avoid race conditions
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Any
from src.prediction_pipeline.collision_risk_predictor import CollisionRiskPredictor
from src.prediction_pipeline.ijepa_integration_predictor import IjepaPredictor
from src.stores.task_store import TaskStore
from src.stores.prediction_store import PredictionStore


class PredictionPipeline:
    """
    Prediction Pipeline Coordinator - SEQUENTIAL MODE
    Runs predictors one after another to avoid race conditions
    """
    
    def __init__(self, map_store=None, task_store=None, prediction_store=None, config: Dict[str, Any] = None, user_command_store=None): 

        self.config = config or {}
        self.logger = logging.getLogger("PredictionPipeline")
        
        # ✅ Stores for business logic coordination
        self.map_store = map_store
        self.task_store = task_store 
        self.prediction_store = prediction_store
        self.user_command_store = user_command_store
        

        # ✅ Initialize predictors
        self.collision_predictor = CollisionRiskPredictor( 
            map_store=self.map_store,
            task_store=self.task_store,
            prediction_store=self.prediction_store,
            config=config
        ) 
        
        self.ijepa_predictor = IjepaPredictor( 
            map_store=self.map_store,
            task_store=self.task_store,
            prediction_store=self.prediction_store,
            config=config
        )
        
        # Pipeline state
        self.is_running = False
        self.is_initialized = False
        
        self.logger.info("🎯 Prediction Pipeline created - SEQUENTIAL Mode")

    def initialize_processing(self):
        """Initialize all prediction components"""
        self.logger.info("Initializing prediction pipeline...")
        
        try:
            # Initialize predictors
            if hasattr(self.ijepa_predictor, 'initialize'):
                self.ijepa_predictor.initialize()
            
            self.is_initialized = True
            self.is_running = True
            
            self.logger.info("✅ Prediction pipeline initialized - SEQUENTIAL mode")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Prediction pipeline initialization failed: {e}")
            return False

    def _initialize_prediction_epoch(self) -> Optional[str]:
        """
        Fetches current task_id from current_action_plan (where main orchestrator stores it)
        """
        try:
            if not self.task_store:
                print("❌ No task store available")
                return None
            
            # DEBUG
            print(f"🔍 DEBUG: Checking TaskStore.current_action_plan...")
            print(f"🔍 DEBUG: TaskStore has current_action_plan: {hasattr(self.task_store, 'current_action_plan')}")
                
            # Look for task_id in current_action_plan (main orchestrator stores here)
            if not hasattr(self.task_store, 'current_action_plan'):
                print("❌ TaskStore has no current_action_plan attribute")
                return None
                
            action_plan = self.task_store.current_action_plan
            if not action_plan:
                print("❌ current_action_plan is None")
                return None
            
            # ========== FIX: MULTIPLE WAYS TO GET TASK_ID ==========
            
            # Method 1: Try task_context first (original method)
            task_id = None
            if 'task_context' in action_plan:
                if 'task_id' in action_plan['task_context']:
                    task_id = action_plan['task_context']['task_id']
                    print(f"✅ Found task_id in task_context: {task_id}")
            
            # Method 2: Fallback to reasoning_cycle_id
            if not task_id and 'reasoning_cycle_id' in action_plan:
                # reasoning_cycle_id is like "reason_123456789"
                # Extract numeric part or use as-is
                reasoning_id = action_plan['reasoning_cycle_id']
                if reasoning_id.startswith("reason_"):
                    task_id = reasoning_id  # Use reasoning_cycle_id as task_id
                    print(f"⚠️  Using reasoning_cycle_id as task_id: {task_id}")
            
            # Method 3: Fallback to timestamp-based ID
            if not task_id:
                timestamp = int(time.time() * 1000)
                task_id = f"reason_{timestamp}"  # Generate a new one
                print(f"⚠️  Generated new task_id from timestamp: {task_id}")
                
                # Also add it back to action_plan for future use
                if 'task_context' not in action_plan:
                    action_plan['task_context'] = {}
                action_plan['task_context']['task_id'] = task_id
                print(f"✅ Added generated task_id back to action_plan")
            
            # Generate prediction_id
            timestamp = int(time.time() * 1000)
            prediction_id = f"pred_{task_id}_{timestamp}"
            
            # Save to prediction store
            if self.prediction_store:
                prediction_metadata = {
                    'prediction_id': prediction_id,
                    'task_id': task_id,
                    'status': 'initialized',
                    'timestamp': time.time(),
                    'predictors_expected': ['collision_risk', 'scene_transition'],
                    'predictors_completed': []
                }
                self.prediction_store.create_prediction_entry(prediction_id, prediction_metadata)
                print(f"✅ Created prediction entry: {prediction_id}")
            
            return prediction_id
            
        except Exception as e:
            print(f"❌ Error in _initialize_prediction_epoch: {e}")
            import traceback
            traceback.print_exc()
            return None      
        
    def process_prediction_cycle(self, frame_id: int) -> Optional[Dict]:
        print(f"🔮 DEBUG: Starting prediction cycle for frame {frame_id}")
        
        try:
            print(f"🔮 Prediction Cycle {frame_id}: Starting predictors...")
            
            # 1. Initialize prediction epoch
            prediction_id = self._initialize_prediction_epoch()
            if not prediction_id:
                print("🔮 DEBUG: No prediction ID generated")
                return None
            
            print(f"🔮 DEBUG: Prediction ID = {prediction_id}")
            
            # 🆕 SEQUENTIAL PROCESSING - ONE AT A TIME
            
            # 2. Run Collision Predictor FIRST and wait for completion
            print("🔮 DEBUG: Calling collision predictor...")
            collision_result = self.collision_predictor.predict_collision_risk_map(prediction_id)
            print(f"🔮 DEBUG: Collision result type = {type(collision_result)}")
             
            
            # 3. THEN run I-JEPA Predictor
            print("🔮 DEBUG: Calling I-JEPA predictor...")
            ijepa_result = self.ijepa_predictor.predict_structural_continuity(prediction_id)
            print(f"🔮 DEBUG: I-JEPA result type = {type(ijepa_result)}")
             
            # 🆕 CHECK FOR SHAPE ATTRIBUTE:
            if hasattr(ijepa_result, 'shape'):
                print(f"🔮 DEBUG: I-JEPA result has shape: {ijepa_result}")
            else:
                print(f"🔮 DEBUG: I-JEPA result is dict with keys: {ijepa_result.keys() if isinstance(ijepa_result, dict) else 'Not dict'}")
            
            # 4. Wait for fused results (should be faster now)
            print("🔮 DEBUG: Waiting for fused results...")
            fused_result = self._fuse_predictions(prediction_id, collision_result, ijepa_result)
            
            
            print(f"✅ Fusion returned: {type(fused_result)}")
            if isinstance(fused_result, dict):
                print(f"✅ Fused keys: {fused_result.keys()}")

                     
            
            return fused_result
                        
        except Exception as e:
            print(f"❌ Prediction cycle failed at step: UNKNOWN - {e}")
            print(f"❌ Full traceback:")
            print(f"🔮 DEBUG: ERROR in prediction cycle: {e}")
            import traceback
            traceback.print_exc()
            
            # ✅ FIX: ALWAYS return a dict, never None or float!
            return {
                'prediction_cycle_error': str(e),
                'recommended_action': 'stop',
                'action_decisions': {},
                'status': 'error',
                'frame_id': frame_id,
                'timestamp': time.time()
            }
        
    def _fuse_predictions(self, prediction_id: str, collision_result, ijepa_result) -> Dict:
        """
        COMPREHENSIVE FUSION: Intelligent risk combination with data validation
        """

        print(f"\n🧠 FUSION START | prediction_id = {prediction_id}")

        try:
            # 1. EXTRACT ALL AVAILABLE DATA WITH VALIDATION
            collision_risk = collision_result.get('risk_assessment', {})
            collision_actions = collision_result.get('action_risks', {})
            ijepa_risks = ijepa_result.get('structural_risks', {})
            ijepa_continuity = ijepa_result.get('continuity_predictions', {})

            print("✅ Data extracted:",
                f"collision_actions={len(collision_actions)},",
                f"structural_risks={len(ijepa_risks)}")

            # 2. ASSESS DATA QUALITY
            has_collision_data = 'robot_position' in collision_result.get('data_sources_used', {})
            max_collision_risk = collision_risk.get('max_collision_risk', 1.0)

            if max_collision_risk == 0.0 and has_collision_data:
                collision_quality = 0.9
            elif max_collision_risk > 0.7:
                collision_quality = 0.8
            elif 0.3 < max_collision_risk <= 0.7:
                collision_quality = 0.6
            else:
                collision_quality = 0.4

            has_structural_data = 'frames_analyzed' in ijepa_result.get('processing_metrics', {})
            max_structural_risk = max(
                [risk.get('risk_score', 0.5) for risk in ijepa_risks.values()]
            ) if ijepa_risks else 0.5

            if max_structural_risk == 0.0 and has_structural_data:
                structural_quality = 0.9
            elif max_structural_risk > 0.7:
                structural_quality = 0.8
            elif 0.3 < max_structural_risk <= 0.7:
                structural_quality = 0.6
            else:
                structural_quality = 0.4

            print(f"📊 Data Quality | collision={collision_quality:.2f} structural={structural_quality:.2f}")

            # 3. SCORE ALL ACTIONS FOR BOTH PREDICTORS
            action_scores = {}
            all_actions = ['move_forward', 'turn_left', 'turn_right']

            for action in all_actions:
                # Collision scoring
                collision_data = collision_actions.get(action, {})
                if collision_data.get('risk_level') == 'high':
                    collision_score = 0.9
                elif collision_data.get('risk_level') == 'medium':
                    collision_score = 0.6
                elif collision_data.get('risk_level') == 'low':
                    collision_score = 0.2
                else:
                    collision_score = 0.5

                collision_score_weighted = collision_score * collision_quality

                # Structural scoring
                structural_data = ijepa_risks.get(action, {})
                continuity_data = ijepa_continuity.get(action, {})

                if structural_data.get('risk_level') == 'high':
                    structural_score = 0.8
                elif structural_data.get('risk_level') == 'medium':
                    structural_score = 0.5
                elif structural_data.get('risk_level') == 'low':
                    structural_score = 0.2
                else:
                    continuity_score = continuity_data.get('continuity_score', 0.5)
                    structural_score = 1.0 - continuity_score

                structural_score_weighted = structural_score * structural_quality

                # 4. COMBINE SCORES
                if collision_quality >= 0.6:
                    combined_score = (collision_score_weighted * 0.8 +
                                    structural_score_weighted * 0.2)
                elif collision_quality >= 0.4:
                    combined_score = (collision_score_weighted * 0.6 +
                                    structural_score_weighted * 0.4)
                else:
                    combined_score = (collision_score_weighted * 0.3 +
                                    structural_score_weighted * 0.7)

                # 5. DECISION
                if combined_score > 0.7:
                    decision = 'stop'
                    confidence = 0.9
                elif combined_score > 0.4:
                    decision = 'proceed_with_caution'
                    confidence = 0.7
                else:
                    decision = 'proceed'
                    confidence = 0.8

                action_scores[action] = {
                    'decision': decision,
                    'combined_risk_score': combined_score,
                    'confidence': confidence,
                    'breakdown': {
                        'collision_risk': collision_score,
                        'collision_quality': collision_quality,
                        'structural_risk': structural_score,
                        'structural_quality': structural_quality
                    }
                }

                print(f"➡️ {action}: combined_risk={combined_score:.3f} decision={decision}")

            # 6. OVERALL SAFETY ASSESSMENT
            all_risk_scores = [score['combined_risk_score'] for score in action_scores.values()]
            max_risk = max(all_risk_scores) if all_risk_scores else 0.5

            if max_risk > 0.7:
                overall_safety = 'unsafe'
                recommended_action = 'stop'
            elif max_risk > 0.4:
                overall_safety = 'caution_required'
                safe_actions = {
                    action: data for action, data in action_scores.items()
                    if data['decision'] != 'stop'
                }
                if safe_actions:
                    recommended_action = min(
                        safe_actions.items(),
                        key=lambda x: x[1]['combined_risk_score']
                    )[0]
                else:
                    recommended_action = 'stop'
            else:
                overall_safety = 'safe'
                recommended_action = 'move_forward'

            print(f"🛡️ Overall Safety={overall_safety} | Recommended={recommended_action} | MaxRisk={max_risk:.3f}")

            fused_data = {
                'timestamp': time.time(),
                'overall_safety_level': overall_safety,
                'recommended_action': recommended_action,
                'max_combined_risk': max_risk,
                'action_decisions': action_scores,
                'data_quality': {
                    'collision_quality': collision_quality,
                    'structural_quality': structural_quality,
                    'overall_confidence': min(collision_quality, structural_quality)
                },
                'risk_summary': {
                    'collision_overall': collision_risk.get('overall_risk_level', 'unknown'),
                    'structural_confidence': ijepa_result.get('structural_analysis', {})
                                                .get('structural_confidence', 0.0)
                }
            }

            # Store fused result
            self.prediction_store.write_fused_predictions(prediction_id, fused_data)
            print("💾 Fused prediction stored successfully")

            return fused_data

            
        except Exception as e:
            print(f"🔮 Fusion error: {e}")
            # Fallback fusion
            fallback_data = {
                'timestamp': time.time(),
                'overall_safety_level': 'unknown',
                'recommended_action': 'stop',
                'max_combined_risk': 0.5,
                'action_decisions': {},
                'data_quality': {'overall_confidence': 0.1},
                'error': str(e)
            }
            
            # 🆕 ADD THIS CALL - Even for fallback cases
            self.prediction_store.write_fused_predictions(prediction_id, fallback_data)
            
            return fallback_data
            

    def shutdown(self):
        """Clean shutdown with disk save"""
        self.is_running = False
        self.is_initialized = False
        
        # 🆕 SAVE TO DISK ON SHUTDOWN
        print("💾 Saving final predictions to disk...")
        try:
            filepath = "experiments/results/predictions/final_predictions.json"
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            success = self.prediction_store.save_to_disk(filepath)
            if success:
                print(f"✅ Final predictions saved to: {filepath}")
            else:
                print(f"❌ Failed to save final predictions")
        except Exception as e:
            print(f"❌ Final save error: {e}")
        
        if hasattr(self.ijepa_predictor, 'shutdown'):
            self.ijepa_predictor.shutdown()
        
        self.logger.info("✅ Prediction pipeline shutdown complete")

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'initialized': self.is_initialized,
            'running': self.is_running,
            'components_ready': {
                'collision_predictor': True,
                'ijepa_predictor': self.ijepa_predictor.is_initialized if hasattr(self.ijepa_predictor, 'is_initialized') else True
            }
        }

# Simple test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🧠 Testing SINGLE Prediction Cycle...")
    
    pipeline = PredictionPipeline()
    
    if pipeline.initialize_processing():
        try:
            # 🎯 CHANGE: Run ONLY ONE cycle instead of 5
            result = pipeline.process_prediction_cycle(0)  # ⬅️ SINGLE CYCLE
            if result:
                print(f"🔮 Single cycle completed successfully")
            
            # Save to disk
            print("\n" + "="*50)
            print("💾 SAVING PREDICTION TO DISK")
            print("="*50)

            filepath = "experiments/outputs/single_prediction.json"
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 🆕 ADD DEBUG PRINT HERE
            print(f"🔍 DEBUG: About to save predictions to: {filepath}")
            print(f"🔍 DEBUG: Prediction store has {len(pipeline.prediction_store.predictions)} predictions")
            print(f"🔍 DEBUG: Prediction store object: {pipeline.prediction_store}")

            # 🎯 EXACT SAVE LINE - ADD DEBUG BEFORE AND AFTER
            print("🔍 DEBUG: Calling save_to_disk()...")
            success = pipeline.prediction_store.save_to_disk(filepath)
            print(f"🔍 DEBUG: save_to_disk() returned: {success}")

            print(f"✅ Prediction saved to: {filepath}")

            # 🆕 ADD VERIFICATION
            if os.path.exists(filepath):
                print(f"✅ VERIFIED: File exists at {filepath}")
                file_size = os.path.getsize(filepath)
                print(f"✅ VERIFIED: File size: {file_size} bytes")
            else:
                print(f"❌ VERIFICATION FAILED: File not found at {filepath}")
            
        finally:
            pipeline.shutdown()