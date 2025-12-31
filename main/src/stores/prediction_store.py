#!/usr/bin/env 

 
"""
PREDICTION STORE - Dedicated store for prediction pipeline outputs
Manages prediction history, risk assessments, and safety recommendations
"""

import logging
import time
import threading
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import json
import pickle
    
class PredictionType(Enum):
    JEPA_FUTURE = "jepa_future"
    GEOMETRIC_SAFETY = "geometric_safety"
    COMBINED_RISK = "combined_risk"
    ACTION_RECOMMENDATION = "action_recommendation"

class RiskLevel(Enum):
    VERY_SAFE = "very_safe"      # 0.0 - 0.2
    SAFE = "safe"               # 0.2 - 0.4  
    CAUTION = "caution"         # 0.4 - 0.6
    RISKY = "risky"             # 0.6 - 0.8
    VERY_RISKY = "very_risky"   # 0.8 - 1.0

@dataclass
class PredictionRecord:
    """Individual prediction record with metadata"""
    prediction_id: str
    timestamp: float
    prediction_type: PredictionType
    data: Dict[str, Any]
    frame_counter: int
    confidence: float = 1.0
    source_component: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to serializable dictionary"""
        result = asdict(self)
        result['prediction_type'] = self.prediction_type.value
        return result

class PredictionStore:
    """
    Dedicated store for prediction pipeline outputs
    Provides fast access to current predictions and historical trends
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # ✅ INITIALIZE BOTH STORAGES - predictions (used by pipeline) and _predictions (legacy)
        self.predictions = {}  # ✅ Used by create_prediction_entry and update_prediction
        self.fused_predictions = {}  # Legacy storage
        self._predictions = {}  # Legacy storage
        
        self._initialized = True
        self.logger = logging.getLogger("PredictionStore")
        
        # 🗃️ Core prediction storage
        self._current_predictions: Dict[str, Any] = {}
        self._prediction_history: deque = deque(maxlen=1000)  # Keep last 1000 predictions
        self._prediction_by_type: Dict[PredictionType, List[PredictionRecord]] = defaultdict(list)
        self._prediction_by_action: Dict[str, List[PredictionRecord]] = defaultdict(list)
        
        self._subscribers = []  # List of callback function
        
        
        # 📊 Statistics and trends
        self._risk_trends: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # action -> [(timestamp, risk)]
        self._safety_scores: Dict[str, List[float]] = defaultdict(list)
        self._prediction_accuracy: List[Tuple[float, bool]] = []  # [(confidence, was_correct)]
        
        # ⚙️ Configuration
        self._max_history_per_type = 100
        self._trend_window_seconds = 300  # 5 minutes for trend analysis
        self._cleanup_interval = 60  # Clean old data every 60 seconds
        
        self._last_cleanup = time.time()
        self._prediction_counter = 0
        
        self.logger.info("Prediction Store initialized")
     
     
    # 🆕 ADD THIS METHOD - Subscriber management
    def subscribe(self, callback):
        """Add a subscriber callback that will be notified on prediction updates"""
        with self._lock:
            self._subscribers.append(callback)
            print(f"📢 PredictionStore: New subscriber added. Total: {len(self._subscribers)}")

    # 🆕 ADD THIS METHOD - Notification system
    def _notify_subscribers(self, event_type: str, data: Dict):
        """Notify all subscribers about prediction updates"""
        with self._lock:
            for callback in self._subscribers:
                try:
                    callback(event_type, data)
                except Exception as e:
                    print(f"❌ PredictionStore: Subscriber notification failed: {e}")

            
    def write_fused_predictions(self, prediction_id: str, fused_data: Dict):
        """Store fused predictions and notify subscribers"""
        with self._lock:
            # Initialize storage if it doesn't exist
            if not hasattr(self, 'fused_predictions'):
                self.fused_predictions = {}
            
            # Store the fused results
            self.fused_predictions[prediction_id] = fused_data
            
            # Notify subscribers
            self._notify_subscribers('fused_predictions_ready', {
                'prediction_id': prediction_id,
                'timestamp': time.time()
            })
    
    def create_prediction_entry(self, prediction_id: str, metadata: Dict):
        with self._lock:
            if not hasattr(self, 'predictions'):
                self.predictions = {}
            
            self.predictions[prediction_id] = {
                'metadata': metadata,
                'predictors': {},
                'created_at': time.time()
            }
            print(f"📊 PredictionStore: Created entry {prediction_id}")

    def get_prediction(self, prediction_id: str) -> Optional[Dict]:
        with self._lock:
            if hasattr(self, 'predictions') and prediction_id in self.predictions:
                return self.predictions[prediction_id]
            return None

    def update_prediction(self, prediction_id: str, prediction_type: str, data: Dict, status: str = "completed", source: str = None):
        """Update prediction with results from a specific predictor"""
        with self._lock:
            if hasattr(self, 'predictions') and prediction_id in self.predictions:
                # Store source information if provided
                predictor_data = {
                    'data': data,
                    'status': status,
                    'completed_at': time.time()
                }
                if source:
                    predictor_data['source'] = source
                
                self.predictions[prediction_id]['predictors'][prediction_type] = predictor_data
                print(f"📊 PredictionStore: Updated {prediction_id} with {prediction_type} results")
                
                                

                # ✅ ADD THIS: Auto-save to disk
                if prediction_type == "structural_continuity":  # Only for I-JEPA to avoid too many saves
                    self.save_to_disk("predictions.json")
                    
                return True
            else:
                print(f"❌ PredictionStore: {prediction_id} not found for {prediction_type} update")
                return False


    def save_to_disk(self, filepath: str = "predictions.json"):
        """Save all predictions to JSON file - FIXED VERSION"""
        try:
            # ✅ FIX: Save from the correct storage (self.predictions) used by the pipeline
            if not hasattr(self, 'predictions') or not self.predictions:
                print("❌ No predictions found in self.predictions to save")
                return False
                
            with open(filepath, 'w') as f:
                # ✅ FIX: Save COMPLETE predictor data, not just filtered fields
                serializable_data = {}
                for pred_id, pred_data in self.predictions.items():
                    serializable_data[pred_id] = {
                        'created_at': pred_data.get('created_at'),
                        'predictors': pred_data.get('predictors', {})  # ✅ SAVE ENTIRE PREDICTORS DICT
                    }
                
                json.dump(serializable_data, f, indent=2, default=str)
            
            print(f"💾 {len(self.predictions)} predictions saved to {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _store_prediction_record(self, record: PredictionRecord):
        """Store prediction record in appropriate collections"""
        # Add to history
        self._prediction_history.append(record)
        
        # Store by type
        type_history = self._prediction_by_type[record.prediction_type]
        type_history.append(record)
        
        # Keep only recent history per type
        if len(type_history) > self._max_history_per_type:
            type_history.pop(0)
        
        # Extract actions and store by action
        if 'action_risks' in record.data:
            for action, risk in record.data['action_risks'].items():
                self._prediction_by_action[action].append(record)
                self._update_risk_trend(action, record.timestamp, risk)
        
        # Update safety scores
        if 'safety_scores' in record.data:
            for action, score in record.data['safety_scores'].items():
                self._safety_scores[action].append(score)
                if len(self._safety_scores[action]) > 100:
                    self._safety_scores[action].pop(0)
    
    def _update_risk_trend(self, action: str, timestamp: float, risk: float):
        """Update risk trend for an action"""
        self._risk_trends[action].append((timestamp, risk))
        
        # Remove old trend data
        cutoff_time = timestamp - self._trend_window_seconds
        self._risk_trends[action] = [
            (ts, r) for ts, r in self._risk_trends[action] 
            if ts >= cutoff_time
        ]
    
    def _update_current_predictions(self, record: PredictionRecord):
        """Update current predictions with latest data"""
        prediction_type = record.prediction_type.value
        
        # Always update the specific prediction type
        self._current_predictions[prediction_type] = record.data
        
        # Update combined predictions if we have both JEPA and geometric
        if (PredictionType.JEPA_FUTURE in self._prediction_by_type and 
            PredictionType.GEOMETRIC_SAFETY in self._prediction_by_type):
            
            jepa_latest = self._prediction_by_type[PredictionType.JEPA_FUTURE][-1].data
            safety_latest = self._prediction_by_type[PredictionType.GEOMETRIC_SAFETY][-1].data
            
            combined = self._combine_predictions(jepa_latest, safety_latest)
            self._current_predictions['combined'] = combined
    
    def _combine_predictions(self, jepa_predictions: Dict, safety_predictions: Dict) -> Dict:
        """Combine JEPA and geometric predictions into unified recommendation"""
        combined = {
            'timestamp': time.time(),
            'jepa_predictions': jepa_predictions,
            'safety_predictions': safety_predictions,
            'action_recommendations': [],
            'risk_assessment': {},
            'confidence_scores': {},
            'safety_levels': {}
        }
        
        # Get all unique actions
        jepa_actions = jepa_predictions.get('action_risks', {}).keys()
        safety_actions = safety_predictions.get('collision_risks', {}).keys()
        all_actions = set(jepa_actions) | set(safety_actions)
        
        for action in all_actions:
            jepa_risk = jepa_predictions.get('action_risks', {}).get(action, 0.5)
            safety_risk = safety_predictions.get('collision_risks', {}).get(action, 0.5)
            
            # Weighted combination (JEPA gets higher weight for intelligence)
            combined_risk = (jepa_risk * 0.7 + safety_risk * 0.3)
            
            combined['risk_assessment'][action] = combined_risk
            combined['confidence_scores'][action] = 1.0 - combined_risk
            combined['safety_levels'][action] = self._get_risk_level(combined_risk)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(action, combined_risk)
            combined['action_recommendations'].append(recommendation)
        
        return combined
    
    def _get_risk_level(self, risk: float) -> RiskLevel:
        """Convert risk score to risk level"""
        if risk <= 0.2:
            return RiskLevel.VERY_SAFE
        elif risk <= 0.4:
            return RiskLevel.SAFE
        elif risk <= 0.6:
            return RiskLevel.CAUTION
        elif risk <= 0.8:
            return RiskLevel.RISKY
        else:
            return RiskLevel.VERY_RISKY
    
    def _generate_recommendation(self, action: str, risk: float) -> str:
        """Generate human-readable recommendation"""
        risk_level = self._get_risk_level(risk)
        
        recommendations = {
            RiskLevel.VERY_SAFE: f"✅ {action}: Very safe (risk: {risk:.3f})",
            RiskLevel.SAFE: f"✓ {action}: Safe (risk: {risk:.3f})", 
            RiskLevel.CAUTION: f"⚠️ {action}: Use caution (risk: {risk:.3f})",
            RiskLevel.RISKY: f"🚨 {action}: Risky (risk: {risk:.3f})",
            RiskLevel.VERY_RISKY: f"❌ {action}: Very risky (risk: {risk:.3f})"
        }
        
        return recommendations.get(risk_level, f"{action}: Unknown risk")
        
    # 🔍 QUERY METHODS
    
    def get_current_predictions(self, prediction_type: str = None) -> Dict:
        """
        Get current predictions
        
        Args:
            prediction_type: Specific type or None for all current predictions
            
        Returns:
            Current prediction data
        """
        with self._lock:
            if prediction_type:
                return self._current_predictions.get(prediction_type, {})
            else:
                return self._current_predictions
    
    def get_safest_action(self) -> Optional[str]:
        """Get the currently safest action based on combined predictions"""
        combined = self.get_current_predictions('combined')
        risk_assessment = combined.get('risk_assessment', {})
        
        if not risk_assessment:
            return None
        
        return min(risk_assessment, key=risk_assessment.get)
    
    def get_risk_trend(self, action: str, window_seconds: float = 60) -> List[float]:
        """
        Get risk trend for an action over time
        
        Args:
            action: Action to get trend for
            window_seconds: Time window in seconds
            
        Returns:
            List of risk values in the specified window
        """
        with self._lock:
            trend_data = self._risk_trends.get(action, [])
            cutoff_time = time.time() - window_seconds
            
            recent_risks = [risk for ts, risk in trend_data if ts >= cutoff_time]
            return recent_risks
    
    def get_prediction_history(self, prediction_type: PredictionType = None, 
                             limit: int = 50) -> List[PredictionRecord]:
        """
        Get prediction history
        
        Args:
            prediction_type: Filter by type or None for all
            limit: Maximum number of records to return
            
        Returns:
            List of prediction records
        """
        with self._lock:
            if prediction_type:
                records = self._prediction_by_type.get(prediction_type, [])
            else:
                records = list(self._prediction_history)
            
            return records[-limit:] if limit else records
     
        
        
        
    def get_action_statistics(self, action: str) -> Dict[str, Any]:
        """
        Get statistics for a specific action
        
        Args:
            action: Action to analyze
            
        Returns:
            Action statistics dictionary
        """
        with self._lock:
            action_predictions = self._prediction_by_action.get(action, [])
            safety_scores = self._safety_scores.get(action, [])
            risk_trend = self._risk_trends.get(action, [])
            
            if not action_predictions:
                return {}
            
            recent_risks = [risk for _, risk in risk_trend[-10:]]  # Last 10 risks
            recent_safety = safety_scores[-10:] if safety_scores else []
            
            return {
                'total_predictions': len(action_predictions),
                'current_risk': recent_risks[-1] if recent_risks else 0.5,
                'average_risk': np.mean(recent_risks) if recent_risks else 0.5,
                'risk_volatility': np.std(recent_risks) if recent_risks else 0.0,
                'average_safety': np.mean(recent_safety) if recent_safety else 0.5,
                'trend_direction': 'improving' if len(recent_risks) > 1 and recent_risks[-1] < recent_risks[0] else 'worsening',
                'last_updated': action_predictions[-1].timestamp if action_predictions else 0
            }
    
    def get_store_statistics(self) -> Dict[str, Any]:
        """Get overall store statistics"""
        with self._lock:
            return {
                'total_predictions': len(self._prediction_history),
                'current_predictions_count': len(self._current_predictions),
                'prediction_types_stored': list(self._prediction_by_type.keys()),
                'actions_tracked': list(self._prediction_by_action.keys()),
                'oldest_prediction': self._prediction_history[0].timestamp if self._prediction_history else 0,
                'newest_prediction': self._prediction_history[-1].timestamp if self._prediction_history else 0
            }
     
    # 💾 PERSISTENCE METHODS
    
    def save_predictions(self, filepath: str) -> bool:
        """Save prediction store to file"""
        try:
            with self._lock:
                save_data = {
                    'current_predictions': self._current_predictions,
                    'prediction_history': [record.to_dict() for record in self._prediction_history],
                    'risk_trends': {action: trends for action, trends in self._risk_trends.items()},
                    'safety_scores': {action: scores for action, scores in self._safety_scores.items()},
                    'metadata': {
                        'prediction_counter': self._prediction_counter,
                        'last_cleanup': self._last_cleanup
                    }
                }
                
                with open(filepath, 'w') as f:
                    json.dump(save_data, f, indent=2, default=self._json_serializer)
                
                self.logger.info(f"Predictions saved to {filepath}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save predictions: {e}")
            return False
    
    def _json_serializer(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

# Global instance for easy access
prediction_store = PredictionStore()

if __name__ == "__main__":
    # Test the prediction store
    logging.basicConfig(level=logging.INFO)
    
    store = PredictionStore()
    
    # Test data
    test_jepa = {
        'action_risks': {'move_forward': 0.1, 'turn_left': 0.05, 'turn_right': 0.2},
        'confidence_scores': {'move_forward': 0.9, 'turn_left': 0.95, 'turn_right': 0.8},
        'future_embeddings': {'move_forward': [0.1, 0.2, 0.3]}
    }
    
    test_safety = {
        'collision_risks': {'move_forward': 0.15, 'turn_left': 0.08, 'turn_right': 0.25},
        'safe_distances': {'move_forward': 1.2, 'turn_left': 1.5, 'turn_right': 0.8},
        'safety_scores': {'move_forward': 0.85, 'turn_left': 0.92, 'turn_right': 0.75}
    }
    
    # Update predictions
    store.update_prediction(PredictionType.JEPA_FUTURE, test_jepa, frame_counter=1, source="jepa_integration")
    store.update_prediction(PredictionType.GEOMETRIC_SAFETY, test_safety, frame_counter=1, source="geometric_predictor")
    
    # Query results
    print("Current combined predictions:", store.get_current_predictions('combined'))
    print("Safest action:", store.get_safest_action())
    print("Store stats:", store.get_store_statistics())
    print("Move forward stats:", store.get_action_statistics('move_forward'))