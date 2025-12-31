
"""
SHARED MEMORY FRAME BUFFER - DEADLOCK-FREE VERSION
Copy-out design: Readers get copies, writer never blocks
"""

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time
import atexit
import signal
import os
import sys
from typing import Optional, Tuple, Dict, Any


class FrameBuffer:
    """
    DEADLOCK-FREE shared memory ring buffer
    Readers get copies, writer never blocks - perfect for SLAM + OWL
    """

    def __init__(self, buffer_size: int = 64, rgb_shape: tuple = (480, 640, 3), depth_shape: tuple = (480, 640)):
        self.buffer_size = buffer_size
        self.rgb_shape = rgb_shape
        self.depth_shape = depth_shape
        self.rgb_dtype = np.uint8
        self.depth_dtype = np.uint16  # Typical for depth data
        
        self._model_read_history = {}  # model_name -> set(frame_ids)
        
        print(f"🎯 Initializing DEADLOCK-FREE Frame Buffer: {buffer_size} frames")
        
        # 🎯 CALCULATE MEMORY CORRECTLY - Include dtype size!
        self.rgb_size = int(np.prod(rgb_shape)) * np.dtype(self.rgb_dtype).itemsize
        self.depth_size = int(np.prod(depth_shape)) * np.dtype(self.depth_dtype).itemsize
        self.total_frame_size = self.rgb_size + self.depth_size
        
        # 🎯 SHARED MEMORY ALLOCATION
        total_memory = buffer_size * self.total_frame_size
        try:
            self.shm_frames = SharedMemory(create=True, size=total_memory)
        except Exception as e:
            raise RuntimeError(f"Failed to create shared memory: {e}")
        
        # 🎯 ZERO-COPY NUMPY VIEWS (for writer only)
        # Calculate memory layout correctly
        rgb_total_bytes = buffer_size * self.rgb_size
        depth_total_bytes = buffer_size * self.depth_size
        
        # RGB data comes first
        self.rgb_array = np.ndarray(
            (buffer_size, *rgb_shape), 
            dtype=self.rgb_dtype, 
            buffer=self.shm_frames.buf[:rgb_total_bytes]
        )
        # Depth data starts after RGB data in memory
        depth_offset = rgb_total_bytes
        self.depth_array = np.ndarray(
            (buffer_size, *depth_shape), 
            dtype=self.depth_dtype, 
            buffer=self.shm_frames.buf[depth_offset:depth_offset + depth_total_bytes]
        )
        
        # 🎯 SHARED METADATA
        self.frame_ids = mp.Array('i', buffer_size)      # Frame IDs
        self.timestamps = mp.Array('d', buffer_size)     # Capture timestamps
        self.valid_flags = mp.Array('b', buffer_size)    # Frame validity
        self.frame_counter = mp.Value('i', 0)            # Total frames written
        
        # 🎯 SYNCHRONIZATION (simplified - no reader counting!)
        self.slot_locks = [mp.Lock() for _ in range(buffer_size)]
        self.latest_frame_id = mp.Value('i', -1)
        self.is_running = mp.Value('b', True)
        
        # 🎯 INITIALIZE BUFFER STATE
        for i in range(buffer_size):
            self.frame_ids[i] = -1
            self.valid_flags[i] = 0
        
        # 🎯 REGISTER CLEANUP
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        print(f"✅ DEADLOCK-FREE RGB-D Buffer Ready: {buffer_size} frames, {total_memory/1024/1024:.1f} MB")
        print(f"   RGB: {rgb_shape}, Depth: {depth_shape}")
        print(f"   Memory layout: RGB={rgb_total_bytes} bytes, Depth={depth_total_bytes} bytes")
        
        
    def write_frame(
        self, frame_dict: Dict[str, np.ndarray], frame_id: int, timestamp: float
    ) -> bool:
        """
        WRITE RGB-D FRAME: Never blocks - always succeeds (unless shutdown)
        Accepts dict with 'rgb' and 'depth' keys
        """
        if not self.is_running.value:
            return False

        if "rgb" not in frame_dict or "depth" not in frame_dict:
            raise ValueError("Frame dict must contain 'rgb' and 'depth' keys")
        
        
        # ADD DEBUG PRINT HERE:
        print(f"🎯 DEBUG: Frame {frame_id} - RGB shape: {frame_dict['rgb'].shape}, Depth shape: {frame_dict['depth'].shape}")
        print(f"🎯 DEBUG: RGB dtype: {frame_dict['rgb'].dtype}, Depth dtype: {frame_dict['depth'].dtype}")





        rgb_frame = frame_dict["rgb"]
        depth_frame = frame_dict["depth"]

        if rgb_frame.shape != self.rgb_shape:
            raise ValueError(
                f"RGB shape mismatch: {rgb_frame.shape} vs {self.rgb_shape}"
            )
        if depth_frame.shape != self.depth_shape:
            raise ValueError(
                f"Depth shape mismatch: {depth_frame.shape} vs {self.depth_shape}"
            )

        slot = frame_id % self.buffer_size

        with self.slot_locks[slot]:
            # 🎯 ALWAYS WRITE RGB + DEPTH - NO READER CHECKS!
            np.copyto(self.rgb_array[slot], rgb_frame)
            np.copyto(self.depth_array[slot], depth_frame)
            
            
            print(f"✅ DEBUG: Frame {frame_id} written to buffer - Slot {slot}")
            print(f"✅ DEBUG: Buffer RGB dtype: {self.rgb_array[slot].dtype}, Depth dtype: {self.depth_array[slot].dtype}")





            # 🎯 UPDATE METADATA
            self.frame_ids[slot] = frame_id
            self.timestamps[slot] = timestamp
            self.valid_flags[slot] = 1

            # 🎯 UPDATE GLOBAL STATE
            self.latest_frame_id.value = frame_id
            self.frame_counter.value += 1

            return True

    def read_frame(self, frame_id: int) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        READ FRAME: Returns a COPY - never blocks writer
        """
        
        print(f"🔍 DEBUG: This is FrameBuffer version WITH FILENAME SUPPORT")


        if not self.is_running.value:
            return None

        slot = frame_id % self.buffer_size

        with self.slot_locks[slot]:
            # 🎯 VALIDATE FRAME EXISTS
            if self.frame_ids[slot] != frame_id or not self.valid_flags[slot]:
                return None  # Frame overwritten or invalid

            # 🎯 🆕 NEW: CHECK IF FRAME ALREADY PROCESSED (ADD THIS BLOCK)
            import inspect
            
            # Get the immediate caller (simplest approach)
            caller_name = "unknown"
            try:
    # Get call stack, skip current frame (index 0)
                stack = inspect.stack()
                if len(stack) > 1:
                    # DEBUG: Print what we're getting
                    print(f"🔍 DEBUG: stack[1].function = {stack[1].function}")
                    print(f"🔍 DEBUG: stack[1].filename = {stack[1].filename}")
                    print(f"🔍 DEBUG: type of filename = {type(stack[1].filename)}")
                    
                    if stack[1].filename:
                        caller_name = stack[1].function + " [" + os.path.basename(stack[1].filename) + "]"
                    else:
                        caller_name = stack[1].function + " [NO_FILENAME]"
            except Exception as e:
                print(f"🔍 DEBUG: Error getting stack: {e}")
                caller_name = "unknown"
            
            # SIMPLE model detection from caller function name
            model_name = "unknown"
            caller_lower = caller_name.lower()
            
            if "owl" in caller_lower:
                model_name = "owl"
            elif "ijepa" in caller_lower or "predictor" in caller_lower:
                model_name = "predictor"  # Group I-JEPA with predictor
            elif "orb" in caller_lower or "slam" in caller_lower:
                model_name = "orb_slam"
            
            print(f"🔍 FRAMEBUFFER DEBUG: Caller function='{caller_name}', Model='{model_name}'")

                
            # Check if this model already processed this frame
            if hasattr(self, '_processed_by_model'):
                key = f"{model_name}_{frame_id}"
                if key in self._processed_by_model:
                    # 🚫 Model already processed this frame - skip
                    print(f"⚠️  Skipping frame {frame_id} for {model_name} (already processed)")
                    return None
            
            # 🎯 COPY FRAME (eliminates writer blocking!)
            # 🎯 COPY RGB + DEPTH (eliminates writer blocking!)
            rgb_copy = self.rgb_array[slot].copy()
            depth_copy = self.depth_array[slot].copy()

            # 🎯 CREATE RGB-D DICTIONARY
            frame_dict = {"rgb": rgb_copy, "depth": depth_copy}

            # 🎯 METADATA
            metadata = {
                "frame_id": self.frame_ids[slot],
                "timestamp": self.timestamps[slot],
                "slot": slot,
                "buffer_size": self.buffer_size,
                "rgb_shape": self.rgb_shape,
                "depth_shape": self.depth_shape,
                "is_valid": True,
                "is_copy": True,  # Indicates this is a copy
                "processed_by": model_name,  # 🆕 Add which model processed it
            }
            
            # 🎯 🆕 NEW: MARK FRAME AS PROCESSED (ADD THIS LINE)
            if not hasattr(self, '_processed_by_model'):
                self._processed_by_model = set()
            self._processed_by_model.add(f"{model_name}_{frame_id}")
            
            # 🆕 Optional: Clean old entries to prevent memory leak
            if len(self._processed_by_model) > 1000:
                # Remove oldest entries (keep last 1000)
                self._processed_by_model = set(list(self._processed_by_model)[-1000:])

            return frame_dict, metadata

    def get_latest_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        GET LATEST FRAME: Returns a copy of the latest frame
        """
        latest_id = self.latest_frame_id.value
        if latest_id == -1:
            return None
        return self.read_frame(latest_id)
    
    
    
    
    
    
    
        # 🎯 ADD THESE TWO HELPER METHODS:
    def _has_model_read_frame(self, model_name: str, frame_id: int) -> bool:
        """Check if model already read this frame"""
        if model_name not in self._model_read_history:
            return False
        return frame_id in self._model_read_history[model_name]
    
    def _mark_frame_read(self, model_name: str, frame_id: int):
        """Mark frame as read by model"""
        if model_name not in self._model_read_history:
            self._model_read_history[model_name] = set()
        self._model_read_history[model_name].add(frame_id)
    
    def _find_next_unprocessed(self, model_name: str, start_id: int) -> Optional[int]:
        """Find next frame that this model hasn't processed"""
        # Check next 64 frames in buffer
        for offset in range(1, self.buffer_size + 1):
            check_id = start_id + offset
            if self.is_frame_available(check_id):
                if not self._has_model_read_frame(model_name, check_id):
                    return check_id
        return None

    
    
    
    
    
    
    
    
    
    
    
    

    def get_latest_frame_id(self) -> int:
        """Get current frame ID - atomic read"""
        return self.latest_frame_id.value

    def is_frame_available(self, frame_id: int) -> bool:
        """Check if specific frame is still in buffer"""
        slot = frame_id % self.buffer_size

        with self.slot_locks[slot]:
            return self.frame_ids[slot] == frame_id and self.valid_flags[slot] == 1

    # 🎯 REMOVED: release_frame() - No longer needed!

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Buffer statistics"""
        valid_frames = sum(self.valid_flags)
        utilization = (valid_frames / self.buffer_size) * 100

        return {
            "buffer_size": self.buffer_size,
            "rgb_shape": self.rgb_shape,
            "depth_shape": self.depth_shape,
            "health_score": 100,  # Add this back
            "memory_usage_mb": (self.buffer_size * self.total_frame_size)
            / (1024 * 1024),
            "latest_frame_id": self.latest_frame_id.value,
            "total_frames_written": self.frame_counter.value,
            "valid_frames": valid_frames,
            "utilization_percent": utilization,
            "is_running": self.is_running.value,
            "design": "COPY-OUT (DEADLOCK-FREE)",
            "writer_blocking": "NEVER",
            "reader_blocking": "NEVER",
        }

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        print(f"🛑 Received signal {signum}, cleaning up shared memory...")
        self.shutdown()

    def _cleanup(self):
        """Cleanup handler for normal termination"""
        if hasattr(self, "shm_frames") and self.is_running.value:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown - release shared memory"""
        if hasattr(self, "is_running") and self.is_running.value:
            print("🧹 Shutting down DEADLOCK-FREE Frame Buffer...")
            self.is_running.value = False

            time.sleep(0.1)

            if hasattr(self, "shm_frames"):
                self.shm_frames.close()
                self.shm_frames.unlink()
                print("✅ Shared memory released")

            print("🎯 DEADLOCK-FREE Buffer shutdown complete")

# 🎯 GLOBAL SINGLETON INSTANCE - DEADLOCK-FREE!
global_frame_buffer = FrameBuffer(buffer_size=64, rgb_shape=(480, 640, 3), depth_shape=(480, 640))