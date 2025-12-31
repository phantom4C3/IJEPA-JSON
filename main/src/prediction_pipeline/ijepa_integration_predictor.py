
#!/usr/bin/env python3
"""
SCENE TRANSITION PREDICTOR - CORRECTED I-JEPA IMPLEMENTATION
Uses I-JEPA for structural understanding and spatial continuity analysis
Answers: "Does this path maintain structural continuity?"
"""

import torch 
import sys
import os
import logging
import time
import numpy as np 
import torch.nn.functional as F
import cv2
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json
from src.stores.frame_buffer import global_frame_buffer
import threading
from transformers import AutoModel, AutoProcessor
from PIL import Image

# Add the local ijepa_model to Python path
ijepa_path = os.path.join(os.path.dirname(__file__), 'ijepa_model')
sys.path.insert(0, ijepa_path) 
    
class IjepaPredictor:
    """
    Analyzes structural continuity and spatial relationships using I-JEPA embeddings
    Answers: "Will this action maintain structural consistency?"
    """
    
    def __init__(self, map_store=None, task_store=None, prediction_store=None, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("IjepaPredictor")
         
        self.map_store = map_store  
        self.task_store = task_store
        self.prediction_store = prediction_store

        # Threading state
        self._prediction_thread = None
        self._last_prediction_result = None
        self._last_prediction_time = 0
        self._prediction_lock = threading.RLock()

        # Frame tracking
        self.last_processed_frame_id = -1
        self.frame_skip = self.config.get('frame_skip', 1)
        
        # I-JEPA model components
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load prototypes from file
        print("🔮 IjepaPredictor: Loading prototypes directly from JSON...")
        # Get absolute path relative to main project
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
        json_path = os.path.join(PROJECT_ROOT, 'main/src/prediction_pipeline/ijepa_prototypes.json')
        self.prototypes = {}

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # SUPER SIMPLE - skip raw, load everything else
                for key, value in data.items():
                    if key.endswith('_raw') or key == 'metadata':
                        continue
                    
                    # Try to load, if it fails, skip
                    try:
                        tensor = torch.tensor(value, device=self.device, dtype=torch.float32)
                        if tensor.dim() == 1:
                            tensor = tensor.unsqueeze(0)
                        self.prototypes[key] = tensor
                        print(f"✅ {key}: shape {tensor.shape}")
                    except:
                        print(f"⚠️ Skipping {key}: could not convert to tensor")
                
                print(f"✅ Loaded {len(self.prototypes)} centroids")
                
            except Exception as e:
                print(f"❌ JSON loading failed: {e}")
                
                print(f"🔮 Total loaded: {len(self.prototypes)} centroid prototypes")
                
            except Exception as e:
                print(f"🔮 JSON loading failed: {e}")
                self._load_fallback_prototypes()


        self.embedding_history = {}  # frame_id -> embedding
        self.last_processed_frame_id = -1
        self.frame_skip = self.config.get('frame_skip', 8)

        # Structural history
        self.structural_history = deque(maxlen=50)
        self.continuity_history = deque(maxlen=100)
        self.structural_templates = {}  # category -> embedding template
        
        # Analysis parameters
        self.continuity_threshold = self.config.get('continuity_threshold', 0.7)
        self.structural_change_threshold = self.config.get('structural_change_threshold', 0.3)
        
        # ✅ CORRECTED: Initialize I-JEPA properly
        self._initialize_ijepa()
        
        # ✅ CORRECT: Move model.eval() here after initialization
        if self.model:
            self.model.eval()  # Crucial for inference
            
        self.is_initialized = self.model is not None
        self.logger.info(f"Scene Transition Predictor initialized (I-JEPA: {self.is_initialized})")
    
    def _initialize_ijepa(self):
        """Initialize I-JEPA model using transformers library"""
        try:
            self.logger.info("Initializing I-JEPA from transformers...")
            print("🔮 IjepaPredictor: Starting I-JEPA initialization...")
            
            model_id = "facebook/ijepa_vith14_1k"
            
            print(f"🔮 IjepaPredictor: Loading model {model_id}...")
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModel.from_pretrained(model_id).to(self.device)
            
            print("🔮 IjepaPredictor: ✅ Model loaded successfully via transformers")
            self.logger.info("✅ I-JEPA initialized successfully from transformers")
            print("🔮 IjepaPredictor: ✅ I-JEPA fully initialized with transformers!")
            
        except Exception as e:
            self.logger.error(f"I-JEPA transformers initialization failed: {e}")
            print(f"🔮 IjepaPredictor: ❌ I-JEPA transformers failed: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.processor = None
            print("🔮 IjepaPredictor: ✅ Falling back to feature-based analysis")
     
    def _compute_similarities(self, embedding_tensor: torch.Tensor) -> Dict[str, float]:
        """Compute similarities between embedding and all prototypes"""
        print("=" * 60)
        print("🔮 _compute_similarities CALLED")

        if embedding_tensor is None:
            print("🔮 embedding_tensor is None → returning empty similarities")
            return {}

        print(f"🔮 embedding_tensor initial shape = {embedding_tensor.shape}")
        print(f"🔮 embedding_tensor device = {embedding_tensor.device}")

        similarities = {}

        for name, prototype in self.prototypes.items():
            print("-" * 40)
            print(f"🔮 Prototype name = {name}")
            print(f"🔮 Prototype original type = {type(prototype)}")

            try:
                if isinstance(prototype, np.ndarray):
                    prototype = torch.from_numpy(prototype).to(self.device)
                    print("🔮 Converted numpy → torch")
                elif isinstance(prototype, torch.Tensor):
                    prototype = prototype.to(self.device)
                else:
                    print(f"🔮 Warning: Prototype {name} is {type(prototype)}, converting")
                    prototype = torch.tensor(prototype, device=self.device)

                if prototype.dim() == 1:
                    prototype = prototype.unsqueeze(0)
                    print("🔮 Unsqueezed prototype to [1, D]")

                if embedding_tensor.dim() == 1:
                    embedding_tensor = embedding_tensor.unsqueeze(0)
                    print("🔮 Unsqueezed embedding to [1, D]")

                print(f"🔮 Prototype shape = {prototype.shape}")
                print(f"🔮 Embedding shape = {embedding_tensor.shape}")

                if prototype.shape[-1] != embedding_tensor.shape[-1]:
                    min_dim = min(prototype.shape[-1], embedding_tensor.shape[-1])
                    print(f"🔮 Dimension mismatch → truncating to {min_dim}")
                    prototype = prototype[:, :min_dim]
                    embedding_tensor = embedding_tensor[:, :min_dim]

                sim = F.cosine_similarity(embedding_tensor, prototype)
                distance = 1.0 - sim.item()

                similarities[name] = {
                    'similarity': sim.item(),
                    'distance': distance,
                    'best_match': distance < 0.3
                }

                print(f"🔮 similarity = {sim.item():.4f}")
                print(f"🔮 distance = {distance:.4f}")
                print(f"🔮 best_match = {distance < 0.3}")

            except Exception as e:
                print(f"🔮 ❌ Error computing similarity for {name}: {e}")
                similarities[name] = 0.0

        print("🔮 FINAL similarities dict:")
        print(similarities)

        return similarities

     


    def extract_structural_embedding(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Extract structural embedding using I-JEPA
        """
        if not self.model or not self.processor:
            return None
        
        try:
            with torch.no_grad():
                # Convert numpy to PIL Image
                if isinstance(frame, np.ndarray):
                    if frame.shape[2] == 3 and frame.dtype == np.uint8:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image = Image.fromarray(frame_rgb)
                    else:
                        pil_image = Image.fromarray(frame.astype('uint8'))
                else:
                    pil_image = frame
                
                print(f"🔮 I-JEPA: Processing {frame.shape} → PIL {pil_image.size}")
                
                # Use processor and model
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
                
                print(f"🔮 I-JEPA: Input tensor {inputs['pixel_values'].shape}")
                
                outputs = self.model(**inputs)
                
                if not hasattr(outputs, 'last_hidden_state'):
                    print(f"🔮 I-JEPA: ❌ No last_hidden_state in outputs")
                    return None
                
                print(f"🔮 I-JEPA: Hidden state {outputs.last_hidden_state.shape}")
                
                # Get embedding: mean of last_hidden_state
                embedding = outputs.last_hidden_state.mean(dim=1)  # [1, 1280]
                
                print(f"🔮 I-JEPA: ✅ Embedding {embedding.shape}")
                return embedding
                    
        except Exception as e:
            print(f"🔮 I-JEPA: ❌ Extraction failed: {e}")
            return None
        
    def _fallback_structural_embedding_tensor(self, frame) -> torch.Tensor:
        """Fallback that returns tensor for consistency"""
        if not isinstance(frame, np.ndarray):
            print(f"🔮 IjepaPredictor: ❌ Invalid frame type in fallback: {type(frame)}")
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Simple color and texture based features
        if len(frame.shape) == 3:
            # Color histogram features
            hist_features = []
            for channel in range(3):
                hist = cv2.calcHist([frame], [channel], None, [16], [0, 256])
                hist_features.extend(hist.flatten())
            
            # Texture features
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            texture = np.sqrt(sobelx**2 + sobely**2)
            texture_features = [np.mean(texture), np.std(texture)]
            
            # Combine features
            features = np.array(hist_features + texture_features, dtype=np.float32)
            
            # Normalize
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            # Ensure 1280 dimensions
            features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
            current_dim = features_tensor.shape[1]

            if current_dim < 1280:
                padding = torch.zeros(1, 1280 - current_dim).to(self.device)
                return torch.cat([features_tensor, padding], dim=1)
            elif current_dim > 1280:
                return features_tensor[:, :1280]
            else:
                return features_tensor
        else:
            return torch.randn(1, 1280).to(self.device)
        
    def _store_embedding(self, frame_id: int, embedding: torch.Tensor):
        """Store embedding for future similarity search"""
        embedding_np = embedding.cpu().numpy().flatten()
        self.embedding_history[frame_id] = embedding_np
        
        # Keep only recent embeddings to save memory
        if len(self.embedding_history) > 100:
            oldest_key = min(self.embedding_history.keys())
            del self.embedding_history[oldest_key]
            
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Thread-safe way to get latest prediction result"""
        with self._prediction_lock:
            if (self._last_prediction_result and 
                time.time() - self._last_prediction_time < 5.0):
                return self._last_prediction_result
            return None
        
    def predict_structural_continuity(self, prediction_id: str, async_mode: bool = False) -> Dict[str, Any]:
        """Main prediction method - keeps original signature"""
        print(f"🔮 IjepaPredictor: Called with prediction_id={prediction_id}")
        
        try:
            result = self._execute_prediction_logic(prediction_id)
            print(f"🔮 IjepaPredictor: Result type = {type(result)}")
            print(f"Result dict : {result}")
            
            if hasattr(result, 'shape'):
                print(f"🔮 IjepaPredictor: ❌ ERROR: Result has shape attribute!")
                result = {"error": "wrong_return_type", "data": result}
            
            return result
            
        except Exception as e:
            print(f"🔮 IjepaPredictor: ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_result(prediction_id)
        
        
    def _execute_prediction_logic(self, prediction_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute the actual prediction logic"""
        print(f"🔮 IjepaPredictor: _execute_prediction_logic started")
        print(f"🔮 DEBUG: prediction_id = {prediction_id}")
        print(f"🔮 DEBUG: self.model available = {self.model is not None}")
        print(f"🔮 DEBUG: self.processor available = {self.processor is not None}")
        print(f"🔮 DEBUG: prototypes loaded = {len(self.prototypes)}")
        
        start_time = time.time()
        
        try:
            if not prediction_id:
                print(f"🔮 DEBUG: No prediction_id provided, using fallback")
                result = self._create_fallback_result("No active prediction ID")
                self._store_thread_result(result)
                return result
            
            # Get frame from frame buffer
            print(f"🔮 IjepaPredictor: Getting frame from buffer")
            frame_data = global_frame_buffer.get_latest_frame()
            print(f"🔮 DEBUG: frame_data = {frame_data is not None}")
            
            if frame_data is None:
                print(f"🔮 DEBUG: No frame in buffer, using fallback")
                result = self._create_fallback_result("No frame available")
                self._store_thread_result(result)
                return result
            
            current_frame, metadata = frame_data
            frame_id = metadata['frame_id']
            print(f"🔮 IjepaPredictor: Got frame {frame_id}")
            print(f"🔮 DEBUG: metadata keys = {metadata.keys()}")
            
            # Frame skipping for performance
            print(f"🔮 DEBUG: last_processed_frame_id = {self.last_processed_frame_id}, frame_skip = {self.frame_skip}")
            if (self.last_processed_frame_id != -1 and 
                frame_id - self.last_processed_frame_id < self.frame_skip):
                print(f"🔮 DEBUG: Frame skipping active, using cached analysis")
                if hasattr(self, '_cached_analysis'):
                    self._store_thread_result(self._cached_analysis)
                    return self._cached_analysis
            
            self.last_processed_frame_id = frame_id
            print(f"🔮 DEBUG: Processing frame {frame_id}")
                
            # Extract numpy array from frame dictionary
            if isinstance(current_frame, dict):
                print(f"🔮 IjepaPredictor: Frame is dict, extracting numpy array")
                print(f"🔮 DEBUG: dict keys = {current_frame.keys()}")
                if 'rgb' in current_frame:
                    frame_array = current_frame['rgb']
                    print(f"🔮 DEBUG: Using 'rgb' key, shape = {frame_array.shape}")
                elif 'color' in current_frame:
                    frame_array = current_frame['color']
                    print(f"🔮 DEBUG: Using 'color' key, shape = {frame_array.shape}")
                else:
                    print(f"🔮 IjepaPredictor: ❌ Unknown frame dict structure")
                    frame_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    print(f"🔮 DEBUG: Created random frame, shape = {frame_array.shape}")
            else:
                frame_array = current_frame
                print(f"🔮 DEBUG: Frame is numpy array, shape = {frame_array.shape}")
            
            print(f"🔮 IjepaPredictor: Frame shape: {frame_array.shape}")
            print(f"🔮 DEBUG: Frame dtype = {frame_array.dtype}, min = {frame_array.min()}, max = {frame_array.max()}")
            
            # Extract embedding
            print(f"🔮 DEBUG: Calling extract_structural_embedding()")
            current_embedding = self.extract_structural_embedding(frame_array)
            print(f"🔮 DEBUG: Embedding returned = {current_embedding is not None}")
            
            if current_embedding is None:
                print(f"🔮 IjepaPredictor: Using fallback embedding")
                print(f"🔮 DEBUG: I-JEPA extraction failed, using fallback")
                current_embedding = self._fallback_structural_embedding_tensor(frame_array)
            
            print(f"🔮 IjepaPredictor: Embedding shape: {current_embedding.shape}")
            print(f"🔮 DEBUG: Embedding device = {current_embedding.device}")
            print(f"🔮 DEBUG: Embedding dtype = {current_embedding.dtype}")
            
            # ✅ SINGLE PLACE: Save embedding to memory
            raw_embedding_np = None
            if current_embedding is not None:
                # Convert to numpy and store in memory
                raw_embedding_np = current_embedding.cpu().numpy().flatten()
                print(f"🔮 DEBUG: Raw embedding numpy shape = {raw_embedding_np.shape}")
                self._store_embedding(frame_id, current_embedding)
                print(f"🔮 DEBUG: Stored embedding for frame {frame_id}")
            
            # 🚨 CRITICAL: Analyze current structure using prototypes
            print(f"🔮 IjepaPredictor: Analyzing current structure with prototypes")
            current_structure = self._analyze_current_structure(current_embedding)
            print(f"🔮 IjepaPredictor: Structure type = {current_structure['type']}")
            print(f"🔮 IjepaPredictor: Confidence = {current_structure['confidence']:.3f}")
            
            print(f"🔮 DEBUG: Structure properties = {list(current_structure.get('properties', {}).keys())}")
            if 'properties' in current_structure:
                for proto_name, score in current_structure['properties'].items():
                    print(f"🔮 DEBUG: {proto_name}: {score:.3f}")
            
            # Analyze actions (keeping original logic for compatibility)
            planned_actions = ['move_forward', 'turn_left', 'turn_right']
            if self.task_store:
                try:
                    print(f"🔮 DEBUG: Task store available, getting planned actions")
                    current_task = self.task_store.get_current_task()
                    try:
                        if isinstance(current_task, dict):
                            if 'planned_actions' in current_task:
                                planned_actions = current_task['planned_actions']
                                print(f"🔮 DEBUG: Got {len(planned_actions)} actions from task store (dict)")
                        elif hasattr(current_task, 'planned_actions'):
                            planned_actions = current_task.planned_actions
                            print(f"🔮 DEBUG: Got {len(planned_actions)} actions from task store (object)")
                    except Exception as e:
                        print(f"🔮 DEBUG: Could not get planned actions: {e}")
                except Exception as e:
                    print(f"🔮 DEBUG: Could not get planned actions: {e}")
            
            print(f"🔮 DEBUG: Using actions = {planned_actions}")
            
            action_continuity = {}
            structural_risks = {}
            
            print(f"🔮 IjepaPredictor: Analyzing {len(planned_actions)} actions")
            for action in planned_actions:
                print(f"🔮 IjepaPredictor: Analyzing action: {action}")
                continuity_analysis = self._analyze_action_continuity(
                    current_embedding, current_structure, action, None
                )
                if continuity_analysis:
                    action_continuity[action] = continuity_analysis
                    # 🚨 CRITICAL: Assess risk based on REAL prototype similarities
                    structural_risks[action] = self._assess_structural_risk(continuity_analysis, current_structure)
                    print(f"🔮 DEBUG: {action} continuity = {continuity_analysis.get('continuity_score', 0):.3f}, risk = {structural_risks[action].get('risk_level', 'unknown')}")
            
            # Build complete result
            print(f"🔮 IjepaPredictor: Building result dictionary")
            result = {
                "prediction_id": prediction_id,
                "predictor_type": "structural_continuity",
                "timestamp": time.time(),
                "status": "completed",
                
                "structural_analysis": {
                    "current_structure_type": current_structure['type'],
                    "structural_confidence": current_structure['confidence'],
                    "embedding_stability": self._calculate_embedding_stability(current_embedding),
                    "frame_id": frame_id,
                    # 🚨 ADDED: Include all similarity scores for debugging
                    "similarity_scores": current_structure.get('properties', {})
                },
                
                "continuity_predictions": action_continuity,
                "structural_risks": structural_risks,
                 
                
                "processing_metrics": {
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "frame_id": frame_id,
                    "model_used": "ijepa" if self.model else "fallback",
                    "prototypes_loaded": len(self.prototypes)
                },
                
                "data_sources_used": {
                    "frame_buffer": True,
                    "prediction_store": prediction_id != "fallback",
                    "task_store": self.task_store is not None,
                    "prototypes_available": len(self.prototypes) > 0
                }
            }
            
            # Update history
            print(f"🔮 DEBUG: Updating structural history")
            self._update_structural_history(current_structure, current_embedding, result)
            
            # Save to prediction store
            print(f"🔮 IjepaPredictor: Attempting to save results to store...")
            success = self._save_to_prediction_store(result, prediction_id)
            if success:
                print(f"🔮 IjepaPredictor: ✅ Successfully saved to prediction store")
                
                # Try to save to disk
                print(f"🔮 DEBUG: Attempting save_to_disk()")
                try:
                    self.prediction_store.save_to_disk("predictions.json")
                    print(f"🔮 DEBUG: save_to_disk() completed")
                except Exception as e:
                    print(f"🔮 DEBUG: save_to_disk() failed: {e}")
            else:
                print(f"🔮 IjepaPredictor: ❌ Failed to save to prediction store")
            
            # Cache for frame skipping
            self._cached_analysis = result
            print(f"🔮 DEBUG: Cached analysis for frame skipping")
            
            # Store result for async access
            self._store_thread_result(result)
            
            total_time = (time.time() - start_time) * 1000
            print(f"🔮 IjepaPredictor: _execute_prediction_logic completed successfully in {total_time:.1f}ms")
            print(f"🔮 DEBUG: Final result keys = {list(result.keys())}")
            
            return result
            
        except Exception as e:
            print(f"🔮 IjepaPredictor: ❌ ERROR in _execute_prediction_logic: {e}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_result(prediction_id)
            
            
            
            
            
            
            
    def _save_to_prediction_store(self, result: Dict, prediction_id: str) -> bool:
        """Save results to prediction store"""
        if not self.prediction_store or not hasattr(self.prediction_store, 'update_prediction'):
            print(f"🔮 IjepaPredictor: ❌ No prediction store available")
            return False
        
        try:
            success = self.prediction_store.update_prediction(
                prediction_id=prediction_id,
                prediction_type="structural_continuity",
                data=result, 
                source="ijepa_predictor"
            )
            return success
        except Exception as e:
            print(f"🔮 IjepaPredictor: ❌ Failed to save: {e}")
            return False

    def _store_thread_result(self, result: Dict[str, Any]):
        """Thread-safe result storage"""
        with self._prediction_lock:
            self._last_prediction_result = result
            self._last_prediction_time = time.time()
        
    def _analyze_action_continuity(self,
                              current_embedding: Optional[torch.Tensor],
                              current_structure: Dict,
                              action: str,
                              map_context: Optional[Dict]) -> Optional[Dict[str, Any]]:
        """Analyze structural continuity for a single action"""
        print("=" * 60)
        print(f"🔮 _analyze_action_continuity CALLED")
        print(f"🔮 action = {action}")
        print(f"🔮 current_structure = {current_structure}")
        print(f"🔮 map_context = {map_context}")

        if current_embedding is None:
            print("🔮 current_embedding is None → using fallback continuity")
            fallback = self._fallback_continuity_analysis(current_structure, action)
            print("🔮 Fallback result:", fallback)
            return fallback

        print(f"🔮 current_embedding shape = {current_embedding.shape}")
        print(f"🔮 current_embedding device = {current_embedding.device}")
        print(f"🔮 current_embedding dtype = {current_embedding.dtype}")

        similarities = self._compute_similarities(current_embedding)

        print("🔮 Raw similarities returned:")
        print(similarities)

        similarity_scores = {}
        for name, data in similarities.items():
            if isinstance(data, dict):
                similarity_scores[name] = data['similarity']
            else:
                similarity_scores[name] = data

        print("🔮 Extracted similarity_scores:")
        print(similarity_scores)

        if similarity_scores:
            continuity_score = self._calculate_structural_continuity_from_similarities(
                similarity_scores,
                action,
                current_structure['type']
            )
            print(f"🔮 continuity_score (from similarities) = {continuity_score}")
        else:
            continuity_score = self._estimate_basic_continuity(current_structure, action)
            print(f"🔮 continuity_score (basic estimate) = {continuity_score}")

        map_confidence = 0.5
        if map_context:
            map_confidence = self._incorporate_geometric_context(map_context, action)
            print(f"🔮 map_confidence (from map) = {map_confidence}")
        else:
            print("🔮 No map_context → default map_confidence = 0.5")

        embedding_stability = self._calculate_embedding_stability(current_embedding)
        continuity_type = self._classify_continuity_type(continuity_score)

        result = {
            'action': action,
            'continuity_score': continuity_score,
            'map_context_confidence': map_confidence,
            'structural_confidence': current_structure['confidence'],
            'embedding_stability': embedding_stability,
            'continuity_type': continuity_type,
            'similarity_scores': similarity_scores if similarity_scores else {}
        }

        print("🔮 _analyze_action_continuity RESULT ↓↓↓")
        print(result)
        print("🔮 _analyze_action_continuity RESULT ↑↑↑")

        return result

    
    def _calculate_structural_continuity_from_similarities(self, 
                                                     similarities: Dict[str, float],  # Still expects Dict[str, float]
                                                     action: str,
                                                     current_structure_type: str) -> float:
        """Calculate structural continuity based on REAL prototype similarities"""
        if not similarities:
            return 0.5
        
        # 🚨 CRITICAL: This method already expects Dict[str, float]
        # So we need to make sure we pass the RIGHT data
        
        # For moving forward, high similarity to "free_space" or "corridor" is good
        if action == 'move_forward':
            free_space_sim = similarities.get('free_space', 0.0)
            corridor_sim = similarities.get('corridor', 0.0)
            wall_sim = similarities.get('wall', 0.0)
            
            # Good continuity = high free space, low walls
            continuity = (free_space_sim * 0.5 + corridor_sim * 0.5) - (wall_sim * 0.3)
            
        elif action in ['turn_left', 'turn_right']:
            # For turning, we care about door similarity (possible openings)
            door_sim = similarities.get('door', 0.0)
            free_space_sim = similarities.get('free_space', 0.0)
            
            continuity = door_sim * 0.7 + free_space_sim * 0.3
            
        else:
            # For other actions, use the highest similarity
            max_similarity = max(similarities.values())
            continuity = max_similarity
        
        # Normalize to 0-1 range
        continuity = max(0.0, min(1.0, continuity))
        
        return continuity
    
    def _analyze_current_structure(self, embedding_tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
        """Analyze current structural properties using prototypes"""
        print("=" * 60)
        print("🔮 _analyze_current_structure CALLED")

        if embedding_tensor is None:
            print("🔮 embedding_tensor is None → returning unknown structure")
            return {'type': 'unknown', 'confidence': 0.0, 'properties': {}}

        print(f"🔮 embedding_tensor shape = {embedding_tensor.shape}")
        print(f"🔮 embedding_tensor device = {embedding_tensor.device}")
        print(f"🔮 embedding_tensor dtype = {embedding_tensor.dtype}")

        similarities = self._compute_similarities(embedding_tensor)

        print("🔮 Raw similarities:")
        print(similarities)

        if not similarities:
            print("🔮 No similarities computed → unknown structure")
            return {'type': 'unknown', 'confidence': 0.0, 'properties': {}}

        similarity_scores = {}
        for name, data in similarities.items():
            if isinstance(data, dict):
                similarity_scores[name] = data['similarity']
            else:
                similarity_scores[name] = data

        print("🔮 Extracted similarity_scores:")
        print(similarity_scores)

        structure_type = max(similarity_scores.items(), key=lambda x: x[1])[0]
        confidence = max(similarity_scores.values())

        result = {
            'type': structure_type,
            'confidence': confidence,
            'properties': similarity_scores
        }

        print("🔮 _analyze_current_structure RESULT ↓↓↓")
        print(result)
        print("🔮 _analyze_current_structure RESULT ↑↑↑")

        return result

    
    def _calculate_embedding_stability(self, embedding: torch.Tensor) -> float:
        """Calculate embedding stability over recent history"""
        if len(self.structural_history) < 2:
            return 0.5
        
        recent_embeddings = [entry['embedding'] for entry in list(self.structural_history)[-3:] 
                           if entry['embedding'] is not None]
        
        if len(recent_embeddings) < 2:
            return 0.5
        
        similarities = []
        for i in range(1, len(recent_embeddings)):
            emb1 = torch.from_numpy(recent_embeddings[i-1]).to(self.device)
            emb2 = torch.from_numpy(recent_embeddings[i]).to(self.device)
            
            # Ensure proper shape
            if emb1.dim() == 1:
                emb1 = emb1.unsqueeze(0)
            if emb2.dim() == 1:
                emb2 = emb2.unsqueeze(0)
            
            # Truncate to same dimension if needed
            min_dim = min(emb1.shape[-1], emb2.shape[-1])
            emb1 = emb1[:, :min_dim]
            emb2 = emb2[:, :min_dim]
            
            sim = F.cosine_similarity(emb1, emb2)
            similarities.append(sim.item())
        
        return np.mean(similarities) if similarities else 0.5
    
    def _estimate_basic_continuity(self, current_structure: Dict, action: str) -> float:
        """Estimate basic continuity when no historical data available"""
        # Simple heuristics based on structure type and action
        base_continuity = 0.7
        
        structure_factors = {
            'free_space': 0.9,
            'corridor': 0.8,
            'door': 0.6,
            'wall': 0.3,
            'cluttered': 0.5,
            'unknown': 0.5
        }
        
        action_factors = {
            'move_forward': 1.0,
            'turn_left': 0.8,
            'turn_right': 0.8,
            'stop': 1.0
        }
        
        structure_factor = structure_factors.get(current_structure['type'], 0.6)
        action_factor = action_factors.get(action, 0.8)
        
        return base_continuity * structure_factor * action_factor
    
    def _incorporate_geometric_context(self, map_context: Dict, action: str) -> float:
        """Use actual point cloud density in action direction"""
        if not map_context or 'point_cloud' not in map_context:
            return 0.5
        
        points_in_direction = self._count_points_in_direction(
            map_context['point_cloud'], action
        )
        
        if points_in_direction > 30:
            return 0.8
        elif points_in_direction > 10:
            return 0.6
        else:
            return 0.3
    
    def _count_points_in_direction(self, point_cloud, action: str) -> int:
        """Count 3D points roughly in the action direction"""
        if point_cloud is None or len(point_cloud) == 0:
            return 0
        
        if isinstance(point_cloud, np.ndarray):
            forward_points = [p for p in point_cloud if p[2] > 0.1]
            return len(forward_points)
        elif isinstance(point_cloud, list):
            forward_points = [p for p in point_cloud if p[2] > 0.1]
            return len(forward_points)
        else:
            return 0
    
    def _classify_continuity_type(self, continuity_score: float) -> str:
        """Classify the type of structural continuity"""
        if continuity_score > 0.8:
            return 'high_continuity'
        elif continuity_score > 0.6:
            return 'medium_continuity'
        elif continuity_score > 0.4:
            return 'low_continuity'
        else:
            return 'structural_break'
    
    def _assess_structural_risk(self, continuity_analysis: Dict, current_structure: Dict):
        """Assess structural risk based on prototype similarity."""
        
        continuity_score = continuity_analysis['continuity_score']
        stability = continuity_analysis['embedding_stability']
        
        print(f"[DEBUG] Assessing risk: continuity={continuity_score:.3f}, stability={stability:.3f}")
        
        # 🚨 FIX: Extract similarity values from properties
        properties = current_structure.get('properties', {})
        prototype_scores = []
        
        for name, data in properties.items():
            if isinstance(data, dict):
                prototype_scores.append(data['similarity'])  # Extract similarity
            else:
                prototype_scores.append(data)  # Already a float
        
        if prototype_scores:
            avg_prototype_similarity = np.mean(prototype_scores)
            max_prototype_similarity = np.max(prototype_scores)
            print(f"[DEBUG] Prototype stats: avg={avg_prototype_similarity:.3f}, max={max_prototype_similarity:.3f}")
            
            # ✅ FIX #2: Use prototype similarity to adjust risk
            # Lower similarity → higher risk
            similarity_factor = max_prototype_similarity  # Use max similarity
        else:
            similarity_factor = 0.05  # Default if no prototypes
            print(f"[DEBUG] No prototype scores found, using default")

        
        # ✅ FIX #3: NEW LOGIC - Risk depends on BOTH continuity AND prototype similarity
        # High continuity + high prototype similarity = LOW risk
        # Low continuity + low prototype similarity = HIGH risk
        
        # Base risk from continuity (scaled down)
        continuity_risk = (1.0 - continuity_score) * 0.5
        
        # Risk from poor prototype matching
        prototype_risk = (1.0 - similarity_factor) * 0.3
        
        # Combined risk
        base_risk = continuity_risk + prototype_risk
        
        print(f"[DEBUG] Risk components: continuity={continuity_risk:.3f}, prototype={prototype_risk:.3f}")
        
        # ✅ FIX #4: Stability adjustment (only penalize very unstable)
        stability_penalty = 0.0
        if stability < 0.3:
            stability_penalty = (0.3 - stability) * 0.5
            print(f"[DEBUG] Adding stability penalty: {stability_penalty:.3f}")
        
        risk_score = base_risk + stability_penalty
        risk_score = max(0.0, min(1.0, risk_score))
        
        # ✅ FIX #5: Adjusted thresholds
        if risk_score > 0.7:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        print(f"[DEBUG] Final risk: score={risk_score:.3f}, level={risk_level}")
        
        # ✅ FIX #6: Build informative return structure
        factors = {
            'low_continuity': continuity_score < 0.5,
            'low_prototype_similarity': similarity_factor < 0.3,
            'low_stability': stability < 0.6,
            'structural_break': continuity_score < 0.3,
            'no_prototype_match': similarity_factor < 0.1
        }
        
        # Include all prototype similarities for debugging
        similarity_contributions = current_structure.get('properties', {})
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'continuity_score': continuity_score,
            'prototype_similarity': similarity_factor,
            'stability': stability,
            'factors': factors,
            'similarity_contributions': similarity_contributions,
            'risk_components': {
                'continuity_risk': continuity_risk,
                'prototype_risk': prototype_risk,
                'stability_penalty': stability_penalty
            }
        }
    
    def _update_structural_history(self, current_structure: Dict, embedding: Optional[torch.Tensor], analysis: Dict):
        """Update structural history and learn patterns"""
        embedding_np = embedding.cpu().numpy().flatten() if embedding is not None else None
        
        history_entry = {
            'timestamp': time.time(),
            'structure_type': current_structure['type'],
            'confidence': current_structure['confidence'],
            'embedding': embedding_np,
            'analysis': analysis
        }
        
        self.structural_history.append(history_entry)
        
        # Learn continuity patterns
        if len(self.structural_history) > 1:
            prev_structure = self.structural_history[-2]
            continuity = {
                'from_structure': prev_structure['structure_type'],
                'to_structure': current_structure['type'],
                'timestamp': time.time(),
                'confidence': min(prev_structure['confidence'], current_structure['confidence'])
            }
            self.continuity_history.append(continuity)

    def _fallback_continuity_analysis(self, current_structure: Dict, action: str) -> Dict[str, Any]:
        """Fallback continuity analysis when I-JEPA is unavailable"""
        return {
            'action': action,
            'continuity_score': 0.5,
            'map_context_confidence': 0.5,
            'structural_confidence': current_structure['confidence'],
            'embedding_stability': 0.5,
            'continuity_type': 'unknown',
            'similarity_scores': {}
        }
    
    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """Create fallback result when prediction cannot be computed"""
        return {
            "prediction_id": "fallback",
            "predictor_type": "structural_continuity",
            "timestamp": time.time(),
            "status": "failed",
            "error": reason,
            "structural_analysis": {
                "current_structure_type": "unknown",
                "structural_confidence": 0.0,
                "embedding_stability": 0.0,
                "frame_id": -1,
                "similarity_scores": {}
            },
            "continuity_predictions": {},
            "structural_risks": {},
            "processing_metrics": {
                "processing_time_ms": 0,
                "frame_id": -1,
                "model_used": "fallback",
                "prototypes_loaded": len(self.prototypes)
            },
            "data_sources_used": {
                "frame_buffer": False,
                "prediction_store": False,
                "task_store": self.task_store is not None,
                "prototypes_available": len(self.prototypes) > 0
            }
        }

    def get_structural_statistics(self) -> Dict[str, Any]:
        """Get statistics about structural analysis"""
        total_analyses = len(self.structural_history)
        
        structure_counts = {}
        for entry in self.structural_history:
            structure_type = entry['structure_type']
            structure_counts[structure_type] = structure_counts.get(structure_type, 0) + 1
        
        continuity_counts = {}
        for continuity in self.continuity_history:
            continuity_key = f"{continuity['from_structure']}->{continuity['to_structure']}"
            continuity_counts[continuity_key] = continuity_counts.get(continuity_key, 0) + 1
        
        return {
            'total_structural_analyses': total_analyses,
            'structure_type_distribution': structure_counts,
            'common_continuity_patterns': dict(sorted(continuity_counts.items(), 
                                                    key=lambda x: x[1], reverse=True)[:5]),
            'ijepa_available': self.model is not None,
            'prototypes_available': len(self.prototypes),
            'average_confidence': np.mean([e['confidence'] for e in self.structural_history]) if self.structural_history else 0
        }

# Test the corrected scene transition predictor
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test predictor
    predictor = IjepaPredictor()
    
      