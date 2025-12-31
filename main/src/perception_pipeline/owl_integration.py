
#!/usr/bin/env python3
"""
OWL Integration Module for Semantic-Geometric Mapping
Integrates OWL/VLM object detection with ORB-SLAM3 geometric mapping
"""

import numpy as np
import time
import cv2

cv2.setNumThreads(0)  # Disable multithreading to avoid issues
import torch
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# ✅ ADD THIS IMPORT
from src.stores.frame_buffer import global_frame_buffer
import threading
from src.stores.habitat_store import global_habitat_store  # ADD THIS LINE
import json
import time
import os
from datetime import datetime

# ✅ CRITICAL: Force OpenCV to headless mode FIRST
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # Optional but helpful
os.environ["CV_IO_MAX_IMAGE_PIXELS"] = "0"  # Disable decompression bomb protection


@dataclass
class CameraIntrinsics:
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    width: int = 0
    height: int = 0

    # ✅ ADD THESE METHODS:
    def load_from_store(self, intrinsics_dict: dict):
        """Load camera intrinsics from HabitatStore (thread-safe)"""
        self.fx = intrinsics_dict["fx"]
        self.fy = intrinsics_dict["fy"]
        self.cx = intrinsics_dict["cx"]
        self.cy = intrinsics_dict["cy"]
        self.width = intrinsics_dict["width"]
        self.height = intrinsics_dict["height"]

    def get_camera_matrix(self) -> np.ndarray:  # ✅ ADD THIS METHOD
        """Get OpenCV camera matrix"""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

    def get_distortion_coeffs(self) -> np.ndarray:  # ✅ ADD THIS METHOD
        """Get distortion coefficients (assuming no distortion)"""
        return np.zeros(5)


class ObjectStatus(Enum):
    """Status of tracked semantic objects"""

    ACTIVE = "active"
    STALE = "stale"
    REMOVED = "removed"


@dataclass
class Detection2D:
    """2D object detection result"""

    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    timestamp: float


@dataclass
class Object3D:
    """3D semantic object with tracking information"""

    object_id: str
    label: str
    position_3d: List[float]  # [x, y, z] in world coordinates
    confidence: float
    first_seen: int
    last_seen: int
    observation_count: int
    class_id: int
    status: ObjectStatus = ObjectStatus.ACTIVE


class OWLIntegration:
    """
    OWL Integration - Semantic object detection and mapping
    Fuses OWL detections with ORB-SLAM3 geometric map via shared store

    Features:
    - Object detection using OWL/VLM models
    - 2D to 3D projection using camera pose
    - Object tracking across frames
    - Integration with geometric map store
    - Configurable processing parameters
    """

    def __init__(
        self,
        map_store=None,
        task_store=None,
        prediction_store=None,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize OWL integration - CONSUMER ONLY

        Args:
            map_store: Shared CentralMapStore for geometric-semantic fusion
            task_store: Shared TaskStore for mission coordination
            prediction_store: Shared PredictionStore for future state
            config: Configuration dictionary
        """
        # ✅ ONLY business stores via DI
        self.map_store = map_store
        self.task_store = task_store
        self.prediction_store = prediction_store
        self.global_habitat_store = global_habitat_store

        # ❌ NO frame buffer creation - use global_frame_buffer
        # ❌ NO habitat_store parameter - use global_habitat_store

        # Configuration with defaults
        self.config = config or {}
        self.owl_config = self.config.get("owl", {})

        # 🆕 ADD OWL-SPECIFIC FRAME LIMIT (100 frames only for OWL)
        # self.max_owl_frames = 500  # OWL stops after 100 processed frames
        self.owl_processed_count = 0  # Counter for OWL frames

        # OWL model and state
        self.model = None
        self.processor = None
        self.is_initialized = False

        self.camera_intrinsics = (
            global_habitat_store.get_camera_intrinsics()
            if global_habitat_store
            else None
        )
        
        self.current_remaining_goals = []  

        # 🎯 BATCH PROCESSING CONFIGURATION
        self.batch_size = 2
        self.batch_queue = []
        self.max_batch_wait = 0.1

        # Object tracking state
        self.semantic_objects: Dict[str, Object3D] = {}
        self.object_counter = 0

        # Processing parameters
        self.confidence_threshold = self.owl_config.get("confidence_threshold", 0.05)
        self.processing_interval = self.owl_config.get(
            "processing_interval", 3
        )  # Every 3rd frame
        self.max_frames_missing = self.owl_config.get("max_frames_missing", 50)
        self.association_threshold = self.owl_config.get("association_threshold", 1.5)

        # Performance monitoring
        self.processing_times = []
        self.last_processed_frame_id = -1

        # ✅ ADD THESE LINES in __init__ method:
        self.model_name = "google/owlvit-base-patch32"  # ADD THIS
        self.text_queries = self.owl_config.get(
            "text_queries",
            [
                "ceiling",
                "wall",
                "door",
                "handle",
                "chandelier",
                "wardrobe",
                "tv",
                "cabinet",
                "blanket",
                "pad",
                "bed",
                "pillow",
                "nightstand",
                "book",
                "lamp",
                "toy",
                "window",
                "frame",
                "armchair",
                "floor",
                "mat",
                "towel",
                "bucket",
                "tap",
                "soap",
                "toilet",
                "brush",
                "curtain",
                "photo",
                "sheet",
                "ventilation",
                "vent",
                "light",
                "bicycle",
                "box",
                "couch",
                "basket",
                "magazine",
                "papers",
                "picture",
                "unknown",
                "folder",
                "table",
                "chair",
                "handbag",
                "tower",
                "trashcan",
                "desk",
                "printer",
                "telephone",
                "plant",
                "shirt",
                "bag",
                "newspaper",
                "balustrade",
                "stairs",
                "rod",
                "speaker",
                "fireplace",
                "flower",
                "plate",
                "pillar",
                "alarm",
                "control",
                "clock",
                "flag",
                "refrigerator",
                "appliance",
                "machine",
                "mug",
                "worktop",
                "sink",
                "holder",
                "microwave",
                "item",
                "stove",
                "bowl",
                "dishwasher",
                "paper",
                "seat",
                "shelf",
                "doormat",
                "hood",
                "dresser",
                "casket",
                "decoration",
                "controller",
                "dial",
                "bath",
                "accessory",
                "mirror",
                "bottle",
                "shoe",
                "board",
                "iron",
                "clothes",
                "case",
                "briefcase",
                "backpack",
                "boxes",
            ],
        )
        # ADD THI
        self.position_smoothing_alpha = 0.7  # ADD THIS for tracking

        # ✅ ADD CONTINUOUS PROCESSING FLAG
        self.is_running = False
        self.processing_thread = None

        print("🦉 OWLIntegration created - Consumer only (reads from global buffer)")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("🦉 OWLIntegration created - Consumer only (reads from global buffer)")

    def initialize(self) -> bool:
        """
        Initialize OWL model - Uses global frame buffer ONLY
        """
        print("Initializing OWL integration...")

        try:
            # ✅ Initialize OWL model
            self.model, self.processor = self._load_owl_model()

            # Verify model is on correct device
            print(
                f"🔍 OWL Model device after loading: {next(self.model.parameters()).device}"
            )

            print(
                "🦉 OWL DEBUG: Model loaded successfully - ready for frame buffer processing"
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.is_initialized = True
            print(
                f"✅ OWL integration ready - Processing every {self.processing_interval} frames from buffer"
            )
            return True

        except Exception as e:
            print(f"❌ OWL initialization failed: {e}")
            self.is_initialized = False
            return False

    def start_continuous_processing(self):
        """Start continuous frame reading from global buffer"""
        self.is_running = True

        # 🆕 ADD THESE LINES - Initialize timing metrics
        self.processing_start_time = time.time()
        self.total_owl_frames_processed = 0
        self.total_owl_processing_time = 0.0

        self.processing_thread = threading.Thread(
            target=self._continuous_processing_loop
        )
        self.processing_thread.start()
        print("🦉 OWL: Started continuous processing")

    def stop_continuous_processing(self):
        """Stop continuous processing with timeout"""
        self.is_running = False

        if hasattr(self, "processing_thread") and self.processing_thread:
            # Wait for thread to finish with timeout
            if self.processing_thread.is_alive():
                print("🛑 Stopping OWL thread...")
            self.processing_thread.join(timeout=2.0)  # 2 second timeout

            if self.processing_thread.is_alive():
                print("🦉 OWL: Processing thread didn't terminate gracefully")
            else:
                # 🆕 ADD THESE LINES - Calculate and log total timing
                processing_end_time = time.time()
                total_wall_time = processing_end_time - self.processing_start_time

                print(f"⏱️  OWL TOTAL STATS:")
                print(f"   📊 Frames processed: {self.total_owl_frames_processed}")
                print(f"   ⏰ Total wall time: {total_wall_time:.2f}s")
                print(
                    f"   🔄 Total processing time: {self.total_owl_processing_time:.2f}s"
                )
                print(
                    f"   📈 Average FPS: {self.total_owl_frames_processed / total_wall_time:.2f}"
                )
                print(
                    f"   ⚡ Processing efficiency: {(self.total_owl_processing_time / total_wall_time) * 100:.1f}%"
                )

                print("🛑 OWL: Stopped continuous processing")

    def _continuous_processing_loop(self):
        """
        Continuous frame processing loop.

        Relies entirely on FrameBuffer for:
        - frame uniqueness
        - per-model de-duplication
        - safety against reprocessing
        """

        print("🟢 OWL continuous processing loop started")

        self.is_running = True

        while self.is_running:
            try:
                # ✅ 1. Get latest frame ID (atomic, no guesswork)
                latest_frame_id = global_frame_buffer.get_latest_frame_id()

                print(f"🦉 OWL: starting to Process REAL frame {latest_frame_id} ")
                
                                
                # Add this check:
                if latest_frame_id % self.processing_interval != 0:
                    time.sleep(0.01)
                    continue
                
                
                # ✅ 2. Ask buffer for THIS frame ID
                frame_data = global_frame_buffer.read_frame(latest_frame_id)

                frame_dict, metadata = frame_data

                # ✅ 3. Model-level processing (interval check lives INSIDE)
                self.process_frame(frame_dict, metadata)

                #   # ✅ ADD TO BATCH QUEUE WITH REAL DATA
                #         self.batch_queue.append(
                #             {
                #                 "frame": frame_view,
                #                 "frame_id": frame_id,  # REAL ID
                #                 "timestamp": timestamp,
                #             }
                #         )

            except Exception as e:
                print(f"❌ OWL processing error: {e}")
                time.sleep(0.1)

        print("🛑 OWL continuous processing loop stopped")

    def process_frame(self, frame_dict, metadata) -> bool:
        """
        Process a single RGB-D frame (FrameBuffer-safe, non-duplicating)
        """

        frame_id = metadata["frame_id"]
        timestamp = metadata["timestamp"]

        print(f"\n🟢 OWL DEBUG ─── START FRAME {frame_id} ───")

        try:
            start_time = time.time()

            # ===============================
            # FRAME INTEGRITY
            # ===============================
            print(f"📦 Frame metadata:")
            print(f"   • frame_id      = {frame_id}")
            print(f"   • timestamp     = {timestamp}")
            print(f"   • rgb shape     = {frame_dict['rgb'].shape}")
            print(f"   • depth shape   = {frame_dict['depth'].shape}")

            rgb = frame_dict["rgb"]

            # ===============================
            # 2D DETECTION
            # ===============================
            detections_2d = self._detect_objects_2d(rgb)
            print(f"🎯 2D detections: " f"{len(detections_2d) if detections_2d else 0}")

            # ===============================
            # CAMERA POSE
            # ===============================
            camera_pose = self._get_current_camera_pose(frame_id)

            if camera_pose is None:
                print("⚠️  Camera pose NOT available — skipping 3D projection")
            else:
                print("📐 Camera pose available")

            # ===============================
            # 3D PROJECTION + TRACKING
            # ===============================
            if camera_pose is not None and detections_2d is not None:
                print("🔄 Projecting detections to 3D")

                objects_3d = self._project_detections_to_3d(
                    detections_2d, camera_pose, frame_id, frame_dict
                )

                print(f"🌐 3D objects projected: {len(objects_3d)}")

                updated_objects = self._track_and_update_objects(
                    objects_3d, frame_id, timestamp
                )

                print(
                    f"🧭 Objects after tracking: "
                    f"{len(updated_objects) if updated_objects else 0}"
                )

                self._update_shared_map_store(updated_objects, frame_id, timestamp)

                print("🗺️  Shared map store updated")

            else:
                print("⏭️  Skipping 3D stage " "(no pose or no detections)")

            # ===============================
            # METRICS
            # ===============================
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            self.owl_processed_count += 1
            self.total_owl_frames_processed += 1
            self.total_owl_processing_time += processing_time

            print(f"⏱️  Processing time: {processing_time:.3f}s")
            print(
                f"📊 Counters → "
                f"processed={self.total_owl_frames_processed}, "
                f"avg_time={self.total_owl_processing_time / self.total_owl_frames_processed:.3f}s"
            )

            print(f"✅ OWL DEBUG ─── END FRAME {frame_id} ───\n")
            return True

        except Exception as e:
            print(f"❌ OWL ERROR @ frame {frame_id}: {e}")
            return False

    def _load_owl_model(self) -> Tuple[Any, Any]:
        """
        Load OWL model and processor - SIMPLIFIED WORKING VERSION
        """
        print(f"Loading OWL-ViT model: {self.model_name}")

        try:
            # ✅ SIMPLE IMPORTS - NO NEED FOR DISPLAY MANIPULATION
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            # ✅ SIMPLE LOADING - REMOVE PROBLEMATIC OPTIONS
            processor = OwlViTProcessor.from_pretrained(self.model_name)
            model = OwlViTForObjectDetection.from_pretrained(self.model_name)
            model = model.to(self.device)  # Move to GPU immediately after loading
            model = model.eval()

            print("✅ OWL-ViT model loaded successfully")
            return model, processor

        except ImportError as e:
            print(
                "Transformers library not available. Install with: pip install transformers"
            )
            raise
        except Exception as e:
            print(f"Failed to load OWL-ViT model: {e}")
            raise

    # ✅ MODIFY _detect_objects_2d method to use task store queries
    def _detect_objects_2d(self, frame: np.ndarray) -> List[Detection2D]:
        try:
            print(f"🎯 Detection input: shape={frame.shape}, dtype={frame.dtype}")

            # 🆕 ADD FRAME STATS DEBUG
            print(f"🔍 FRAME STATS:")
            print(f"   - Min pixel value: {frame.min()}")
            print(f"   - Max pixel value: {frame.max()}")
            print(f"   - Mean brightness: {np.mean(frame):.1f}")
            print(f"   - Brightness std: {np.std(frame):.1f}")

            # Check if frame is usable
            if np.mean(frame) < 80:
                print("⚠️ WARNING: Frame is too dark (mean < 80)")
            elif np.mean(frame) > 200:
                print("⚠️ WARNING: Frame is too bright (mean > 200)")

            # ✅ GET QUERIES FROM TASK STORE (ADD THIS LINE)
            current_queries = self._get_text_queries_from_task_store()

            print(f"🔍 CURRENT QUERIES DEBUG:")
            print(f"   - Query count: {len(current_queries)}")
            print(f"   - First 3 queries: {current_queries[:3]}")

            # ✅ USE CURRENT QUERIES (CHANGED THIS LINE)
            inputs = self.processor(
                text=current_queries,  # CHANGED FROM self.text_queries
                images=frame,
                return_tensors="pt",
            ).to(self.device)

            # 🆕 ADD INPUTS DEBUG
            print(f"🔍 INPUTS DEBUG:")
            print(f"   - input_ids shape: {inputs['input_ids'].shape}")
            print(f"   - attention_mask shape: {inputs['attention_mask'].shape}")
            print(f"   - pixel_values shape: {inputs['pixel_values'].shape}")
            print(f"   - pixel_values dtype: {inputs['pixel_values'].dtype}")
            print(
                f"   - pixel_values min/max: {inputs['pixel_values'].min():.3f}/{inputs['pixel_values'].max():.3f}"
            )

            inputs = {
                k: v.to(self.device) for k, v in inputs.items()
            }  # Move inputs to GPU

            print(f"🔍 OWL DEBUG: Inputs keys: {list(inputs.keys())}")
            print(f"🔍 OWL DEBUG: Text queries: {current_queries}")

            with torch.no_grad():
                outputs = self.model(**inputs)

            # 🆕 ADD MODEL OUTPUT DEBUG
            print(f"🔍 MODEL OUTPUT DEBUG:")
            print(f"   - Output type: {type(outputs)}")
            if hasattr(outputs, "logits"):
                print(f"   - logits shape: {outputs.logits.shape}")
                print(
                    f"   - logits min/max: {outputs.logits.min():.3f}/{outputs.logits.max():.3f}"
                )
            if hasattr(outputs, "pred_boxes"):
                print(f"   - pred_boxes shape: {outputs.pred_boxes.shape}")

            # ✅ CORRECT: Process OWL-ViT outputs
            target_sizes = torch.tensor([frame.shape[:2]]).to(self.device)

            # 🆕 ADD PRE-PROCESSING DEBUG
            print(f"🔍 POST-PROCESSING DEBUG:")
            print(f"   - target_sizes: {target_sizes}")
            print(f"   - confidence_threshold: {self.confidence_threshold}")

            # Use the correct post-processing method for OWL-ViT
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold,
            )

            # 🆕 ADD RESULTS DEBUG
            print(f"🔍 POST-PROCESS RESULTS DEBUG:")
            print(f"   - Results type: {type(results)}")
            print(f"   - Results length: {len(results) if results else 0}")

            detections = []

            if results and len(results) > 0:
                result = results[0]  # Get first (and only) result

                # 🆕 ADD RAW RESULT DEBUG
                print(f"🔍 RAW RESULT DEBUG:")
                print(f"   - Scores shape: {result['scores'].shape}")
                print(f"   - Labels shape: {result['labels'].shape}")
                print(f"   - Boxes shape: {result['boxes'].shape}")

                # 🆕 PRINT ALL SCORES
                print(f"🎯 ALL SCORES:")
                all_scores = result["scores"].cpu().numpy()
                for i, score in enumerate(all_scores):
                    print(
                        f"   Score {i}: {score:.4f} {'✓' if score > self.confidence_threshold else '✗'}"
                    )

                # 🆕 PRINT CONFIDENCE STATS
                passing_scores = all_scores[all_scores > self.confidence_threshold]
                print(f"🔍 CONFIDENCE STATS in:")
                print(f"   - Total predictions: {len(all_scores)}")
                print(f"   - Passing threshold: {len(passing_scores)}")

                print(
                    f"🎯 DEBUG: Raw results - scores: {len(result['scores'])}, labels: {len(result['labels'])}"
                )

                for score, label_idx, box in zip(
                    result["scores"], result["labels"], result["boxes"]
                ):
                    label = current_queries[label_idx]  # Assume always valid

                    # 🎯 CRITICAL: Process box coordinates FOR ALL OBJECTS (not just in else block!)
                    box_coords = box.tolist()  # Convert tensor to list

                    print(f"🔍 BOX ANALYSIS for {label}:")
                    print(f"   Raw box: {box_coords}")
                    # Line 1:
                    print(
                        f"   x2 > width? {box_coords[2]} > {self.camera_intrinsics['width']} = {box_coords[2] > self.camera_intrinsics['width']}"
                    )

                    # Line 2:
                    print(
                        f"   y2 > height? {box_coords[3]} > {self.camera_intrinsics['height']} = {box_coords[3] > self.camera_intrinsics['height']}"
                    )

                    detection = Detection2D(
                        label=label,
                        confidence=score.item(),
                        class_id=label_idx,
                        timestamp=time.time(),
                        bbox=box_coords,
                    )
                    detections.append(detection)

                    print(f"🎯 Detected: {label} (confidence: {score.item():.3f})")

            else:
                # 🆕 ADD NO RESULTS DEBUG
                print(f"🔍 NO RESULTS DEBUG:")
                print(f"   - Reason 1: No objects in scene matching queries")
                print(
                    f"   - Reason 2: All confidence scores < {self.confidence_threshold}"
                )
                print(f"   - Reason 3: Model didn't detect anything")
                print(f"   - Action: Lower confidence threshold or adjust queries")

            print(f"✅ OWL-ViT found {len(detections)} objects")
            return detections

        except Exception as e:
            print(f"❌ 2D object detection failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_current_camera_pose(self, frame_id: int = None) -> Optional[np.ndarray]:
        if frame_id is not None and hasattr(self.map_store, "get_pose_matrix_for_frame"):
            pose_matrix = self.map_store.get_pose_matrix_for_frame(frame_id)
            if pose_matrix is not None:
                print(f"✅ OWL: Got ALREADY-TRANSFORMED pose for frame {frame_id}")
                return pose_matrix
            else:
                print(f"❌ OWL: get_pose_matrix_for_frame returned None for {frame_id}")
        print(f"❌ OWL: Could not get transformed pose for frame {frame_id}")
        return None

    def _project_detections_to_3d(
        self,
        detections_2d: List[Detection2D],
        camera_pose: np.ndarray,
        frame_id: int,
        frame_dict: Dict[str, Any],  # ✅ ADD THIS PARAMETER
    ) -> List[Dict[str, Any]]:
        """
        Project 2D detections to 3D using only ORB-SLAM depth (already aligned to world frame).
        Other methods (Habitat raycasting / fallback) are disabled.
        """

        print(
            f"🎯 PROJECTION: Starting 3D projection for {len(detections_2d)} detections"
        )
        print(f"🎯 PROJECTION: Camera pose available: {camera_pose is not None}")

        # Use ORB-SLAM map points osnly
        orb_objects = self._project_using_orb_slam_poses_and_habitat_depth(
            detections_2d, camera_pose, frame_id, frame_dict  # ✅ ADD THIS
        )

        print(f"🎯 PROJECTION: ORB-SLAM returned {len(orb_objects)} objects")
        return orb_objects

    def _project_using_orb_slam_poses_and_habitat_depth(
        self, detections_2d: List[Detection2D], camera_pose: np.ndarray, frame_id: int,     frame_dict: Dict[str, Any]  # ✅ ADD THIS PARAMETER
    ) -> List[Dict[str, Any]]:
        """
        CORRECT projection:
        - OWL gives pixel coordinates
        - Habitat depth gives metric depth
        - ORB-SLAM gives ALIGNED camera pose (Habitat world frame)
        - Result saved to map_store (single map)

        BACKWARD COMPATIBLE:
        - Same function name
        - Same arguments
        - Same return format
        """

        print(f"🎯 PROJ DEBUG 1: Starting projection for frame {frame_id}")
        print(f"🎯 PROJ DEBUG 1.5: Detections count: {len(detections_2d)}")
        if detections_2d:
            print(f"🎯 PROJ DEBUG 1.6: First detection: {detections_2d[0].label}")

        objects_3d = []

        if self.map_store is None:
            print("❌ ORB-SLAM: No map store available")
            return []

        # 🔒 Camera pose must already be Umeyama-aligned
        if camera_pose is None or camera_pose.shape != (4, 4):
            print("❌ ORB-SLAM: Invalid camera pose")
            return []

        print(f"🎯 ORB-SLAM VERIFY: Camera pose OK for frame {frame_id}")
        print(f"   Position: {camera_pose}")

        try:
             
                        # ✅ USE PASSED FRAME_DICT INSTEAD OF READING FROM BUFFER
            print(
                f"🎯 PROJ DEBUG 9: Using passed frame_dict, keys: {frame_dict.keys() if frame_dict else 'None'}"
            )


            depth_frame = frame_dict.get("depth")
            print(f"🎯 PROJ DEBUG 10: depth_frame type: {type(depth_frame)}")

            if depth_frame is None:
                print("❌ PROJ DEBUG 11: No depth frame available")
                return []

            height, width = depth_frame.shape
            print(f"🎯 PROJ DEBUG 13: Height: {height}, Width: {width}")

            # ✅ Camera intrinsics (already stored once)
            intrinsics = self.global_habitat_store.get_camera_intrinsics()
            print(
                f"🎯 PROJ DEBUG 14: Intrinsics: fx={intrinsics.get('fx')}, fy={intrinsics.get('fy')}"
            )
            fx, fy = intrinsics["fx"], intrinsics["fy"]
            cx, cy = intrinsics["cx"], intrinsics["cy"]

            for i, detection in enumerate(detections_2d):

                print(
                    f"🎯 PROJ DEBUG 16.{i}: Processing detection {i}: {detection.label}"
                )
                print(f"🎯 PROJ DEBUG 16.{i}.5: Bbox: {detection.bbox}")

                try:
                    x1, y1, x2, y2 = detection.bbox
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    print(f"🎯 PROJ DEBUG 16.{i}.6: Center: ({center_x}, {center_y})")

                    # Bounds check
                    if not (0 <= center_x < width and 0 <= center_y < height):
                        print(f"❌ PROJ DEBUG 16.{i}.7: Center out of bounds")
                        continue

                    depth = float(depth_frame[center_y, center_x])
                    print(f"🎯 PROJ DEBUG 16.{i}.9: Depth value: {depth}")
                    if depth <= 0 or not np.isfinite(depth):
                        continue

                    # 🎯 Pixel → camera ray
                    x_cam = (center_x - cx) * depth / fx
                    y_cam = (center_y - cy) * depth / fy
                    z_cam = depth

                    print(
                        f"🎯 PROJ DEBUG 16.{i}.12: Camera coords: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})"
                    )

                    cam_point = np.array([x_cam, y_cam, z_cam, 1.0])

                    print(f"🎯 PROJ DEBUG 16.{i}.13: Transforming to world coordinates")
                    world_point = (camera_pose @ cam_point)[:3]
                    print(f"🎯 PROJ DEBUG 16.{i}.14: World coords: {world_point}")

                    object_3d = {
                        "label": detection.label,
                        "confidence": detection.confidence,
                        "position_3d": world_point.tolist(),
                        "bbox_2d": detection.bbox,
                        "attributes": {
                            "frame_id": frame_id,
                            "category": detection.label,
                            "class_id": detection.class_id,
                            "timestamp": detection.timestamp,
                            "source": "owl_pixel_projection",
                            "depth": depth,
                            "camera_pose_verified": True,
                            "world_coordinates": True,
                        },
                        "is_new": True,
                        "source": "owl_pixel_projection",
                        "coordinate_frame": "habitat_world",
                    }

                    objects_3d.append(object_3d)
                    
                            
                    print(f"🔄 OWL FULL: cam=[{x_cam:.1f},{y_cam:.1f},{z_cam:.1f}] → world={world_point.tolist()}")
                    
                    # 🎯 NEW: Store mission goal coordinates if this is a mission goal
                    # 🎯 NEW: Store mission goal coordinates if this is a mission goal
                    if self.task_store and hasattr(self.task_store, 'set_umeyama_aligned_goal'):
                        # Check if this detected object is a mission goal (USE STORED LIST)
                        if hasattr(self, 'current_remaining_goals') and detection.label in self.current_remaining_goals:
                            print(f"🎯 OWL: Mission goal '{detection.label}' detected! Storing aligned coordinates")
                            self.task_store.set_umeyama_aligned_goal(
                                goal_name=detection.label,
                                world_coordinates=world_point.tolist(),
                                room="detected_by_owl"
                            )
                            
                     # 🔥 ACCUMULATIVE UPDATE TO GRID + SEMANTICS WITH DEBUG
                    if hasattr(self.map_store, 'occupancy_grid_2d') and self.map_store.occupancy_grid_2d is not None:
                        world_x, _, world_z = world_point
                        print(f"🔄 OWL GRID DEBUG: World point = [{world_x:.2f}, {world_z:.2f}]")
                        
                        # Compute grid coordinates
                        grid_x = int((world_x - self.map_store.grid_origin[0]) / self.map_store.grid_resolution)
                        grid_z = int((world_z - self.map_store.grid_origin[1]) / self.map_store.grid_resolution)
                        print(f"   -> Grid coordinates = ({grid_x}, {grid_z})")
                        
                        # Only mark if within bounds
                        if 0 <= grid_x < self.map_store.grid_size and 0 <= grid_z < self.map_store.grid_size:
                            # ✅ Accumulate: increment instead of overwriting
                            self.map_store.occupancy_grid_2d[grid_z, grid_x] += 1.0
                            print(f"   -> Occupancy grid updated: new value = {self.map_store.occupancy_grid_2d[grid_z, grid_x]:.1f}")
                            
                            # Initialize semantics dictionary if not present
                            if not hasattr(self.map_store, 'grid_semantics') or self.map_store.grid_semantics is None:
                                self.map_store.grid_semantics = {}
                                print(f"   -> Initialized grid_semantics dictionary")
                            
                            # ✅ Only add new semantic label if not already stored
                            if (grid_z, grid_x) not in self.map_store.grid_semantics:
                                self.map_store.grid_semantics[(grid_z, grid_x)] = detection.label
                                print(f"   -> Added semantic label '{detection.label}' at ({grid_z}, {grid_x})")
                            else:
                                existing_label = self.map_store.grid_semantics[(grid_z, grid_x)]
                                print(f"   -> Semantic already exists at ({grid_z}, {grid_x}): '{existing_label}'")
                        else:
                            print(f"⚠️ Grid coordinates ({grid_x}, {grid_z}) out of bounds (0-{self.map_store.grid_size-1})")


                    print(
                        f"✅ PROJ DEBUG 16.{i}.15: Successfully projected {detection.label}"
                    )

                except Exception as e:
                    print(f"❌ Projection failed for {detection.label}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

            print(f"✅ Projected {len(objects_3d)} objects using pixel+depth+pose")
            return objects_3d

        except Exception as e:
            print(f"❌ ORB-SLAM projection failed: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _get_text_queries_from_task_store(self) -> List[str]:
        """Fetch object detection queries from task store using reasoning_id"""
        task_queries = []
        
         

        print(f"🦉 OWL DEBUG: Starting _get_text_queries_from_task_store")
        print(f"🦉 OWL DEBUG: self.task_store exists = {self.task_store is not None}")
        
        
        
        
        
        
        
        # 🆕 PRIORITY -1: Check HabitatStore FIRST (highest priority)
        if (hasattr(global_habitat_store, 'current_goal_category') and 
            global_habitat_store.current_goal_category is not None):
            
            print(f"🎯 OWL HABITATSTORE: Using stored goal '{global_habitat_store.current_goal_category}'")
            return [global_habitat_store.current_goal_category]  # 🆕 Return immediately


        
        
        
        
        
        
        
        
        
        # 🎯 🆕 PRIORITY 0: Check remaining goals FIRST (BEFORE anything else)
                # 🎯 🆕 PRIORITY 0: Check remaining goals FIRST (BEFORE anything else)
        if self.task_store is not None:
            remaining_goals = self.task_store.get_remaining_goals()  # ✅ Use existing method
            print(f"🎯 OWL: Remaining mission goals: {remaining_goals}")
            
            # 🎯 NEW: Also check if we already have aligned coordinates for these goals
            if remaining_goals and hasattr(self.task_store, 'get_umeyama_aligned_goals'):
                aligned_goals = self.task_store.get_umeyama_aligned_goals()
                if aligned_goals:
                    print(f"🎯 OWL: Already have aligned coordinates for {len(aligned_goals)} goals")
                    for goal_key, goal_data in aligned_goals.items():
                        print(f"   - {goal_data.get('goal_name')}: {goal_data.get('world_coordinates')}")
            
            if remaining_goals:
                print(f"🎯 OWL: Using mission goals as queries: {remaining_goals}")
                
                # 🎯 STORE THE GOALS LIST FOR LATER USE (avoid double fetch)
                self.current_remaining_goals = remaining_goals.copy()  # Store for projection phase
                
                return remaining_goals  # 🆕 RETURN EARLY - mission goals take priority




        # 🎯 PRIORITY 1: Try to get queries from latest reasoning cycle
        if self.task_store is not None:
            try:
                print(f"🦉 OWL DEBUG: Checking intermediate_reasoning...")

                # PRIORITY 1: Use final plan from reasoning pipeline
                target_objects = (self.task_store.current_action_plan or {}).get(
                    "target_objects", []
                )
                if target_objects:
                    task_queries = target_objects
                    print(
                        f"🎯 OWL: Using target_objects from current_action_plan: {task_queries}"
                    )
                else:
                    print(f"❌ OWL DEBUG: current_action_plan has no target_objects")

                if target_objects:
                    print(
                        f"📝 Extracted from description for owl queries: {target_objects}"
                    )
                else:
                    print(f"❌ OWL DEBUG: No objects extracted from description")

                # # 🚀 NEW: Check intermediate_reasoning for Tree of Thoughts data
                # if hasattr(self.task_store, "intermediate_reasoning"):
                #     print(
                #         f"🦉 OWL DEBUG: task_store HAS intermediate_reasoning attribute"
                #     )

                #     # Wait 2 seconds to ensure reasoning pipeline wrote data
                #     time.sleep(2.0)

                #     # Get all reasoning cycle IDs
                #     reasoning_ids = list(self.task_store.intermediate_reasoning.keys())
                #     print(
                #         f"🦉 OWL DEBUG: Found {len(reasoning_ids)} reasoning_ids: {reasoning_ids}"
                #     )

                #     if reasoning_ids:
                #         # Get the LATEST reasoning cycle
                #         latest_id = reasoning_ids[-1]
                #         print(f"🔍 OWL: Found reasoning_id: {latest_id}")

                #         # Get data for this reasoning cycle
                #         cycle_data = self.task_store.intermediate_reasoning.get(
                #             latest_id, {}
                #         )
                #         print(
                #             f"🦉 OWL DEBUG: cycle_data keys: {list(cycle_data.keys())}"
                #         )

                #         component_data = cycle_data.get("component_reasoning", {})
                #         print(
                #             f"🦉 OWL DEBUG: component_data keys: {list(component_data.keys())}"
                #         )

                #         # 🎯 Check for Tree of Thoughts target_objects
                #         if "tree_of_thoughts" in component_data:
                #             print(
                #                 f"✅ OWL DEBUG: Found tree_of_thoughts in component_data!"
                #             )
                #             tree_data = component_data["tree_of_thoughts"]
                #             print(f"🦉 OWL DEBUG: tree_data type: {type(tree_data)}")
                #             print(
                #                 f"🦉 OWL DEBUG: tree_data keys: {list(tree_data.keys()) if isinstance(tree_data, dict) else 'NOT_DICT'}"
                #             )

                #             detailed_plan = tree_data.get("detailed_plan", {})
                #             print(
                #                 f"🦉 OWL DEBUG: detailed_plan type: {type(detailed_plan)}"
                #             )
                #             print(
                #                 f"🦉 OWL DEBUG: detailed_plan keys: {list(detailed_plan.keys()) if isinstance(detailed_plan, dict) else 'NOT_DICT'}"
                #             )

                #             target_objects = detailed_plan.get("target_objects", [])
                #             print(
                #                 f"🦉 OWL DEBUG: target_objects found: {target_objects} (type: {type(target_objects)})"
                #             )

                #             if target_objects:
                #                 task_queries = target_objects
                #                 print(
                #                     f"🎯 OWL: Using reasoning_id {latest_id} queries: {target_objects}"
                #                 )
                #                 return task_queries
                #             else:
                #                 print(f"❌ OWL DEBUG: target_objects list is EMPTY")
                #         else:
                #             print(
                #                 f"❌ OWL DEBUG: 'tree_of_thoughts' NOT FOUND in component_data"
                #             )
                #             print(
                #                 f"🦉 OWL DEBUG: Available components: {list(component_data.keys())}"
                #             )
                #     else:
                #         print(
                #             f"❌ OWL DEBUG: No reasoning_ids found in intermediate_reasoning"
                #         )
                # else:
                #     print(
                #         f"❌ OWL DEBUG: task_store does NOT have intermediate_reasoning attribute"
                #     )
                #     print(
                #         f"🦉 OWL DEBUG: task_store attributes: {[attr for attr in dir(self.task_store) if not attr.startswith('_')]}"
                #     )
            except Exception as e:
                print(f"❌ OWL DEBUG: ERROR in task store access: {e}")
                import traceback

                traceback.print_exc()

        else:
            print(f"❌ OWL DEBUG: No task_store available")

        # 🎯 PRIORITY 3: Use fallback queries if no reasoning/task-specific ones
        if not task_queries:
            task_queries = self.text_queries  # Your existing default queries
            print(f"🔄 Using default OWL queries: {task_queries}")
        else:
            print(f"✅ OWL DEBUG: Already have {len(task_queries)} queries")

        print(f"🦉 OWL FINAL Queries: {task_queries}")
        return task_queries

    def _track_and_update_objects(
        self, objects_3d: List[Dict[str, Any]], frame_id: int, timestamp: float
    ) -> Dict[str, Object3D]:
        """
        Track objects across frames and update object states
        ✅ REAL fusion logic

        Args:
            objects_3d: Current frame's 3D objects
            frame_id: Current frame identifier
            timestamp: Current timestamp

        Returns:
            Updated semantic objects dictionary
        """
        updated_objects = self.semantic_objects

        for obj_3d in objects_3d:
            # Object association by position and label
            object_id = self._associate_object(obj_3d, updated_objects)

            if object_id is None:
                # New object - assign ID
                object_id = f"{obj_3d['label']}_{self.object_counter}"
                self.object_counter += 1

                updated_objects[object_id] = Object3D(
                    object_id=object_id,
                    label=obj_3d["label"],
                    position_3d=obj_3d["position_3d"],
                    confidence=obj_3d["confidence"],
                    first_seen=frame_id,
                    last_seen=frame_id,
                    observation_count=1,
                    class_id=obj_3d.get("attributes", {}).get("class_id", -1),  # Get from attributes or use default
                    status=ObjectStatus.ACTIVE,
                )
                print(f"New object detected: {object_id} at {obj_3d['position_3d']}")
            else:
                # Existing object - update position and stats
                existing_obj = updated_objects[object_id]

                # Position smoothing
                alpha = self.position_smoothing_alpha
                existing_obj.position_3d = [
                    alpha * new + (1 - alpha) * old
                    for new, old in zip(obj_3d["position_3d"], existing_obj.position_3d)
                ]

                # Update statistics
                existing_obj.last_seen = frame_id
                existing_obj.observation_count += 1
                existing_obj.confidence = max(
                    existing_obj.confidence, obj_3d["confidence"]
                )
                existing_obj.status = ObjectStatus.ACTIVE

                print(
                    f"Updated object: {object_id}, observations: {existing_obj.observation_count}"
                )

        self.semantic_objects = updated_objects
        print(f"Tracking {len(updated_objects)} objects after update")
        return updated_objects

    def _associate_object(
        self, new_object: Dict[str, Any], existing_objects: Dict[str, Object3D]
    ) -> Optional[str]:
        """
        Associate new detection with existing objects

        Args:
            new_object: New object detection
            existing_objects: Currently tracked objects

        Returns:
            Associated object ID or None if new object
        """
        best_match_id = None
        min_distance = float("inf")

        for obj_id, existing_obj in existing_objects.items():
            # Check label match
            if new_object["label"] != existing_obj.label:
                continue

            # Check spatial proximity
            distance = np.linalg.norm(
                np.array(new_object["position_3d"]) - np.array(existing_obj.position_3d)
            )

            if distance < self.association_threshold and distance < min_distance:
                min_distance = distance
                best_match_id = obj_id

        return best_match_id

    def _update_shared_map_store(
        self, semantic_objects: Dict[str, Object3D], frame_id: int, timestamp: float
    ):
        """
        Update shared map store with SEMANTIC OBJECTS ONLY
        """
        if self.map_store is None:
            print("No map store available for semantic updates")
            return

        try:
            # Convert Object3D instances to dictionaries
            semantic_objects_dict = {
                obj_id: {
                    "object_id": obj.object_id,
                    "label": obj.label,
                    "position_3d": obj.position_3d,
                    "confidence": obj.confidence,
                    "first_seen": obj.first_seen,
                    "last_seen": obj.last_seen,
                    "observation_count": obj.observation_count,
                    "class_id": obj.class_id,
                    "status": obj.status.value,
                    "frame_detected": frame_id,  # 🎯 ADD frame association
                }
                for obj_id, obj in semantic_objects.items()
            }

            semantic_data = {
                "frame_id": frame_id,  # 🎯 PASS the actual frame_id
                "timestamp": timestamp,
                "semantic_objects": semantic_objects_dict,
                "objects_count": len(semantic_objects),
                "last_update": time.time(),
            }

            # ✅ CLEAN: Only update SEMANTIC data
            if hasattr(self.map_store, "update_semantic_data"):
                self.map_store.update_semantic_data(semantic_data)
            elif hasattr(self.map_store, "semantic_data"):
                # Direct semantic attribute (still semantic-only)
                self.map_store.semantic_data = semantic_data
            else:
                print("No semantic update interface found in map store")
                # ❌ DON'T fallback to geometric data!

            print(
                f"Updated semantic store for frame {frame_id} with {len(semantic_objects)} objects"
            )

        except Exception as e:
            print(f"Failed to update semantic map store for frame {frame_id}: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        if not self.processing_times:
            return {}

        stats = {
            "total_frames_processed": self.total_owl_frames_processed,  # 🆕 Use the timing counter
            "avg_processing_time": np.mean(self.processing_times),
            "max_processing_time": np.max(self.processing_times),
            "min_processing_time": np.min(self.processing_times),
            "current_objects_count": len(self.semantic_objects),
            "last_processed_frame": self.last_processed_frame_id,
            "text_queries_count": len(self.text_queries),
            "total_processing_time": self.total_owl_processing_time,  # 🆕 Add this
        }

        if hasattr(self, "batch_size"):
            stats["batch_size"] = self.batch_size
            if self.processing_times:
                stats["batch_efficiency"] = (
                    np.mean(self.processing_times) / self.batch_size
                )

        return stats
 

    def shutdown(self):
        """
        Clean shutdown of OWL integration
        """
        print("Shutting down OWL integration...")

        # ✅ FIRST: Stop continuous processing thread
        self.stop_continuous_processing()

        # THEN clean up resources
        self.model = None
        self.processor = None
        self.is_initialized = False

        # Log final statistics
        stats = self.get_performance_stats()
        if stats:
            print(f"OWL performance stats: {stats}")

        print("OWL integration shutdown complete")


if __name__ == "__main__":
    config = {
        "owl": {
            "confidence_threshold": 0.01,
            "processing_interval": 1,
            "text_queries": ["chair", "table", "person"],
        }
    }

    # ✅ CREATE THE INSTANCE FIRST
    owl = OWLIntegration(config=config)

    # ✅ INITIALIZE IT
    if owl.initialize():
        # ✅ START PROCESSING
        owl.start_continuous_processing()

        # Let it run for a bit
        time.sleep(10)

        # ✅ THEN shutdown
        owl.shutdown()


# OWL pixel detection → Get depth at that pixel → Project using camera pose
# "chair at pixel (225,220)" → "depth = 2.3m" → "chair at (1.23, 0.45, 0.12) in world"


# Your goal: Project OWL detections → ORB-SLAM map (which is Umeyama-aligned to Habitat world).
# Answer: Habitat depth + ORB-SLAM pose = exactly what you need.

# STEP-BY-STEP Logic (Single Method)
# 1. OWL gives: Pixel bbox [150,90,300,350] = "chair at these pixels"
# 2. Habitat depth gives: depth_frame[220,225] = 2.3 meters = exact distance
# 3. ORB-SLAM pose gives: camera_pose = camera location in Habitat world (Umeyama aligned)
# 4. Projection math: pixel + depth + pose → 3D point in Habitat world
# 5. Result: "chair" @ (1.23, 0.45, 0.12) → add to ORB-SLAM map

# ✅ Why This is Perfect (No Alternatives Needed)
# What You Need	Habitat Depth	ORB-SLAM Pose	Result
# Depth/Z	✅ Exact meter at pixel	❌ No	✅ Perfect
# Camera Location	❌ No	✅ Aligned to Habitat	✅ Perfect
# World Frame	✅ Habitat native	✅ Umeyama aligned	✅ SAME frame
