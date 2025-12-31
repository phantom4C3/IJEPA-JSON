

#!/usr/bin/env python3
"""
TASK STORE - Pure task management only
Handles mission state, task queue, and progress tracking
NO geometric data, NO robot state - that's in map store
"""

import time
import threading
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from collections import deque
import logging
from typing import Dict, Any  # Add this import at the top


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    EXPLORE = "explore"
    FIND_OBJECT = "find_object"
    NAVIGATE_TO = "navigate_to"
    SEARCH_ROOM = "search_room"


@dataclass
class Task:
    id: str
    type: TaskType
    description: str
    status: TaskStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    parameters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.metadata is None:
            self.metadata = {}


class TaskStore:
    """
    Pure task management - no geometric data, no robot state
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

        self._initialized = True

        # TASK QUEUE MANAGEMENT
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        self.current_task: Optional[Task] = None
        
        self.complete_execution_records = []  # List of all execution cycles
        
        self.intermediate_reasoning = {}  # reasoning_cycle_id -> component data

                # 🆕 ADD THIS: Separate storage for Umeyama-aligned coordinates
        self.umeyama_aligned_mission_goals: Dict[str, Any] = {}  # goal_name -> {"world_coordinates": [], "room": ""}
        
        
            # 🆕 ADD THIS: Mission goals coordinates storage
        self.mission_goals_coordinates: Dict[str, Dict[str, Any]] = {}  # goal_name -> coordinates data

        self._subscribers = []  # 🆕 Add subscriber list

        # TASK PROGRESS TRACKING
        self.task_attempts: Dict[str, int] = {}  # task_id -> attempt_count
        self.task_timeouts: Dict[str, float] = {}  # task_id -> max_duration

        # MISSION STATE
        self.mission_start_time: float = time.time()
        self.mission_goals: Set[str] = set()  # Target objects/locations
        self.achieved_goals: Set[str] = set()  # Completed goals

        # PERFORMANCE METRICS
        self.task_durations: Dict[TaskType, List[float]] = {}
        self.success_rates: Dict[TaskType, List[bool]] = {}

        # TASK GENERATION CONTEXT
        self.last_task_generation: float = 0
        self.task_generation_context: Dict[str, Any] = {}
        
        self.goal_reached = False  # 🔥 NEW FLAG: Position-based success

        print("✅ Task Store initialized - Pure task management")

    # ==================== TASK QUEUE MANAGEMENT ====================
 

    def subscribe(self, callback):
        """Add subscriber for action plan updates"""
        print(f"TaskStore[{id(self)}] new subscriber → {callback}")
        self._subscribers.append(callback)

    def _notify_subscribers(self, event_type, data):
        """Notify all subscribers"""
        print(f"TaskStore[{id(self)}] notifying {len(self._subscribers)} subscribers about {event_type}")
        for callback in self._subscribers:
            try:
                callback(event_type, data)
                print(f"TaskStore subscriber Action pipeline initialised 🚀👩‍🚀")
                
            except Exception as e:
                print(f"TaskStore subscriber error: {e}")
    




    def write_reasoning_plan(self, reasoning_cycle_id: str, plan_data: Dict):
        import time
        from collections import deque

        # ========== FIX NULL CYCLE ID ==========
        if reasoning_cycle_id is None:
            reasoning_cycle_id = f"unknown_{int(time.time()*1000)}"
            print(f"Warning: TASKSTORE: Received None cycle_id, generated: {reasoning_cycle_id}")

        # ========== 1. INITIALIZE STORAGE ==========
        if not hasattr(self, 'reasoning_plans'):
            self.reasoning_plans = {}

        if reasoning_cycle_id not in self.reasoning_plans:
            self.reasoning_plans[reasoning_cycle_id] = {}

        # ========== 2. DETECT DATA TYPE & EXTRACT COMPONENT ==========
        if isinstance(plan_data, str):
            component = 'unknown'
            data_to_store = {'text': plan_data}
        
        else:
            # === THIS IS THE KEY FIX: Check for final action plan FIRST ===
            if 'selected_action' in plan_data:
                component = 'final_action'
                print(f"FINAL ACTION PLAN DETECTED → {plan_data['selected_action']} (cycle {reasoning_cycle_id})")

                # Keep history (last 10 plans)
                if not hasattr(self, 'action_plan_history'):
                    self.action_plan_history = deque(maxlen=10)
                self.action_plan_history.append(plan_data)
                self.current_action_plan = plan_data  # ← important for other components

                # NOTIFY SUBSCRIBERS — THIS IS WHAT TRIGGERS ActionPipeline
                print(f"Triggering action_plan_ready for {plan_data['selected_action']}")
                self._notify_subscribers('action_plan_ready', {
                    'action_plan': plan_data,
                    'timestamp': time.time(),
                    'reasoning_cycle_id': reasoning_cycle_id
                })

            # Intermediate reasoning (Tree of Thoughts, Spatial, etc.)
            elif 'component_reasoning' in plan_data or 'fused_reasoning' in plan_data:
                component = 'intermediate_reasoning'
                print(f"Storing INTERMEDIATE reasoning for cycle: {reasoning_cycle_id}")

                if not hasattr(self, 'intermediate_reasoning'):
                    self.intermediate_reasoning = {}
                self.intermediate_reasoning[reasoning_cycle_id] = plan_data

            else:
                # Generic / init message (your first call)
                component = plan_data.get('component', 'metadata')
                print(f"Storing metadata/init for cycle: {reasoning_cycle_id}")

            plan_data['component'] = component
            plan_data['reasoning_cycle_id'] = reasoning_cycle_id
            plan_data['timestamp'] = time.time()

        # ========== 3. STORE THE PLAN (always) ==========
        self.reasoning_plans[reasoning_cycle_id][component] = plan_data

        # ========== 4. AUTO-SAVE TO DISK ==========
        self.save_to_disk()

        return self
 
 
            
    def debug_storage_state(self):
        """Debug what's stored in both intermediate and final storage"""
        print("🔍 TASKSTORE STORAGE STATE:")
        
        # Check final action plans
        if hasattr(self, 'action_plan_history'):
            print(f"  Final Action Plans: {len(self.action_plan_history)}")
            for i, plan in enumerate(list(self.action_plan_history)[-3:]):
                print(f"    {i}: {plan.get('selected_action', 'unknown')} (ID: {plan.get('reasoning_cycle_id', 'unknown')})")
        
        # Check intermediate reasoning
        if hasattr(self, 'intermediate_reasoning'):
            print(f"  Intermediate Reasoning: {len(self.intermediate_reasoning)} cycles")
            for cycle_id in list(self.intermediate_reasoning.keys())[-3:]:
                data = self.intermediate_reasoning[cycle_id]
                components = list(data.get('component_reasoning', {}).keys())
                print(f"    {cycle_id}: {len(components)} components")
        
        # Check current plan
        if hasattr(self, 'current_action_plan'):
            print(f"  Current Action: {self.current_action_plan.get('selected_action', 'none')}")
            
            
    def _store_reasoning_history(self, reasoning_results: Dict):
        """Store reasoning results in history"""
        self.decision_history.append({
            'timestamp': reasoning_results['timestamp'],
            'results': reasoning_results
        })
        
        # Keep only recent history
        if len(self.decision_history) > self.max_history:
            self.decision_history.pop(0)
    

    def get_current_task_details(self) -> Dict:
        """Get detailed current task information"""
        return {
            'id': getattr(self, 'current_task_id', 'task_001'),
            'description': self.get_current_task(),
            'type': getattr(self, 'current_task_type', 'navigation'),
            'priority': getattr(self, 'current_task_priority', 'medium'),
            'status': getattr(self, 'current_task_status', 'in_progress'),
            'created_at': getattr(self, 'current_task_created', time.time())
        }

    def update_perception_status(self, status: str, details: Dict = None):
        """
        Update perception pipeline status with optional details
        
        Args:
            status: Current status 
            details: Additional status information
        """
        if self.current_task is None:
            self.current_task = {
                'task_id': 'perception_pipeline',
                'created_at': time.time()
            }
        
        self.current_task['perception_status'] = status
        self.current_task['last_update'] = time.time()
        
        if details:
            self.current_task.update(details)
        
        logging.info(f"Perception status: {status}")
        
        
    def save_to_disk(self, filename: str = None):
        """Save current state to disk - NO BACKUPS, SINGLE FILE"""
        import time
        import json
        import os
        
        # ========== SINGLE FILE, NO TIMESTAMP ==========
        if filename is None:
            filename = "task_store_current.json"  # Always same name
        
        # Save to main experiments directory (not backups)
        output_dir = "experiments/results"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # ========== COLLECT ONLY ESSENTIAL DATA ==========
        data = {
            'timestamp': time.time(),
            'last_update': time.strftime("%Y%m%d_%H%M%S"),
            
            # Only keep current/recent plans, not all history
            'current_reasoning_plans': {},
            'current_action_plan': getattr(self, 'current_action_plan', None),
            'current_task': getattr(self, 'current_task', None)
        }
        
        # Add only NON-NULL reasoning plans
        reasoning_plans = getattr(self, 'reasoning_plans', {})
        for cycle_id, plans in reasoning_plans.items():
            if cycle_id is not None:  # Skip null keys
                data['current_reasoning_plans'][cycle_id] = plans
        
        # ========== OVERWRITE (NOT APPEND) ==========
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 TASKSTORE: Saved current state to {filepath}")
        return filepath
            
        # ==================== PIPELINE STATUS METHODS ====================

    def update_navigation_status(self, status: str, details: Dict = None):
        """
        Update navigation pipeline status
        
        Args:
            status: Current navigation status
            details: Additional status information
        """
        with self._lock:
            if not hasattr(self, 'navigation_status'):
                self.navigation_status = {}
            
            self.navigation_status = {
                'status': status,
                'last_update': time.time(),
                'details': details or {}
            }
            
            logging.info(f"Navigation status updated: {status}")
            
    def update_reasoning_status(self, status: str, details: Dict = None):
        """
        Update reasoning pipeline status
        
        Args:
            status: Current reasoning status
            details: Additional status information
        """
        with self._lock:
            if not hasattr(self, 'reasoning_status'):
                self.reasoning_status = {}
            
            self.reasoning_status = {
                'status': status,
                'last_update': time.time(),
                'details': details or {}
            }
            
            logging.info(f"Reasoning status updated: {status}")

    def update_prediction_status(self, status: str, details: Dict = None):
        """
        Update prediction pipeline status
        
        Args:
            status: Current prediction status  
            details: Additional status information
        """
        with self._lock:
            if not hasattr(self, 'prediction_status'):
                self.prediction_status = {}
            
            self.prediction_status = {
                'status': status,
                'last_update': time.time(), 
                'details': details or {}
            }
            
            logging.info(f"Prediction status updated: {status}")

    def get_pipeline_statuses(self) -> Dict[str, Any]:
        """
        Get all pipeline statuses
        """
        with self._lock:
            return {
                'reasoning': getattr(self, 'reasoning_status', {}),
                'prediction': getattr(self, 'prediction_status', {}),
                'perception': getattr(self, 'current_task', {}).get('perception_status', 'unknown')
            }
            
    def update_processing_metrics(self, metrics: dict):
        """Update processing performance metrics"""
        if self.current_task is None:
            self.current_task = {}
        
        if 'processing_metrics' not in self.current_task:
            self.current_task['processing_metrics'] = {}
        
        self.current_task['processing_metrics'].update(metrics)
        self.current_task['processing_metrics']['last_update'] = time.time()


    def get_task_queue(self) -> List[Dict]:
        """Get pending tasks queue"""
        return getattr(self, 'task_queue', [])

    def mark_task_completed(self, task_id: str):
        """Mark a task as completed and move to history"""
        if hasattr(self, 'current_task_id') and self.current_task_id == task_id:
            # Move current task to history
            if not hasattr(self, 'task_history'):
                self.task_history = []
            
            completed_task = {
                'id': self.current_task_id,
                'description': self.get_current_task(),
                'status': 'completed',
                'completed_at': time.time()
            }
            self.task_history.append(completed_task)
            
            # Activate next task from queue
            if self.task_queue:
                next_task = self.task_queue.pop(0)
                self.current_task_id = next_task['id']
                self.current_task = next_task['description']
                self.current_task_status = 'in_progress'
            else:
                # No more tasks
                self.current_task_id = None
                self.current_task = None
                self.current_task_status = 'idle'

    def add_to_task_queue(self, task_description: str, task_type: str = 'navigation', priority: str = 'medium'):
        """Add a new task to the queue"""
        if not hasattr(self, 'task_queue'):
            self.task_queue = []
        
        new_task = {
            'id': f'task_{int(time.time())}_{len(self.task_queue)}',
            'description': task_description,
            'type': task_type,
            'priority': priority,
            'status': 'pending',
            'created_at': time.time()
        }
        self.task_queue.append(new_task)
        
    def add_task(
        self,
        task_type: TaskType,
        description: str,
        parameters: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a new task to the queue"""
        with self._lock:
            task_id = f"task_{int(time.time()*1000)}_{len(self.task_queue)}"

            task = Task(
                id=task_id,
                type=task_type,
                description=description,
                status=TaskStatus.PENDING,
                created_at=time.time(),
                parameters=parameters or {},
                metadata=metadata or {},
            )

            self.task_queue.append(task)
            self.task_attempts[task_id] = 0

            print(f"📋 Added task: {task_type.value} - {description}")
            return task_id

    def get_next_task(self) -> Optional[Task]:
        """Get next pending task and mark as in progress"""
        with self._lock:
            if not self.task_queue:
                return None

            next_task = self.task_queue[0]
            if next_task.status == TaskStatus.PENDING:
                next_task.status = TaskStatus.IN_PROGRESS
                next_task.started_at = time.time()
                self.current_task = next_task
                self.task_attempts[next_task.id] += 1

                print(
                    f"🎯 Starting task: {next_task.type.value} - {next_task.description}"
                )
                return next_task

            return None

    def complete_current_task(self, success: bool = True, completion_data: Dict[str, Any] = None) -> bool:
        """Mark current task as completed or failed"""
        with self._lock:
            if not self.current_task:
                return False

            # Check if current_task is a dict or Task object
            if isinstance(self.current_task, dict):
                # It's a dictionary
                self.current_task['completed_at'] = time.time()
                if success:
                    self.current_task['status'] = 'completed'
                    print(f"✅ Task completed: {self.current_task.get('description', 'Unknown')}")
                    # Move to completed tasks if list exists
                    if hasattr(self, 'completed_tasks'):
                        self.completed_tasks.append(self.current_task)
                else:
                    self.current_task['status'] = 'failed'
                    print(f"❌ Task failed: {self.current_task.get('description', 'Unknown')}")
                    # Move to failed tasks if list exists
                    if hasattr(self, 'failed_tasks'):
                        self.failed_tasks.append(self.current_task)
                
                # Record performance metrics if possible
                try:
                    if hasattr(self, '_record_task_metrics'):
                        duration = self.current_task.get('completed_at', time.time()) - \
                                self.current_task.get('started_at', self.current_task.get('created_at', time.time()))
                        self._record_task_metrics(self.current_task.get('type', 'unknown'), duration, success)
                except:
                    pass  # Silently skip if metrics recording fails
            else:
                # It's a Task object
                task = self.current_task
                task.completed_at = time.time()

                if success:
                    task.status = TaskStatus.COMPLETED
                    if hasattr(self, 'completed_tasks'):
                        self.completed_tasks.append(task)
                    print(f"✅ Task completed: {task.description}")
                else:
                    task.status = TaskStatus.FAILED
                    if hasattr(self, 'failed_tasks'):
                        self.failed_tasks.append(task)
                    print(f"❌ Task failed: {task.description}")

                # Remove from queue
                if hasattr(self, 'task_queue') and self.task_queue:
                    # Find task in queue by ID
                    for i, queued_task in enumerate(self.task_queue):
                        if hasattr(queued_task, 'id') and queued_task.id == task.id:
                            self.task_queue.pop(i)
                            break

                # Record performance metrics
                if hasattr(self, '_record_task_metrics'):
                    duration = task.completed_at - (task.started_at or task.created_at)
                    self._record_task_metrics(task.type, duration, success)

            # Store completion data
            if completion_data:
                if isinstance(self.current_task, dict):
                    if 'metadata' not in self.current_task:
                        self.current_task['metadata'] = {}
                    self.current_task['metadata']['completion_data'] = completion_data
                else:
                    if not hasattr(self.current_task, 'metadata'):
                        self.current_task.metadata = {}
                    self.current_task.metadata["completion_data"] = completion_data

            # Clear current task
            self.current_task = None
            return True

    def cancel_current_task(self, reason: str = "User cancelled") -> bool:
        """Cancel the current task"""
        with self._lock:
            if not self.current_task:
                return False

            task = self.current_task
            task.status = TaskStatus.CANCELLED
            task.metadata["cancellation_reason"] = reason

            # Move to failed tasks
            self.failed_tasks.append(task)

            # Remove from queue
            if self.task_queue and self.task_queue[0].id == task.id:
                self.task_queue.pop(0)

            print(f"🛑 Task cancelled: {task.description} - {reason}")
            self.current_task = None
            return True

    # ==================== MISSION GOAL MANAGEMENT ====================

    def set_mission_goals(self, goals: List[str], coordinates_dict: Dict[str, List[float]] = None):
        """Set mission goals (target objects/locations) with optional coordinates"""
        with self._lock:
            self.mission_goals = set(goals)
            self.achieved_goals.clear()
            
            # Store coordinates if provided
            if coordinates_dict:
                if not hasattr(self, 'mission_goals_coordinates'):
                    self.mission_goals_coordinates = {}
                
                for goal_name, coords in coordinates_dict.items():
                    if goal_name in self.mission_goals:
                        self.mission_goals_coordinates[goal_name] = {
                            "world_coordinates": coords,
                            "timestamp": time.time()
                        }
            
            print(f"🎯 Mission goals set: {list(goals)}")
            if coordinates_dict:
                print(f"   With coordinates for: {list(coordinates_dict.keys())}")

    def record_goal_achievement(self, goal: str) -> bool:
        """Record that a mission goal has been achieved"""
        with self._lock:
            if goal in self.mission_goals and goal not in self.achieved_goals:
                self.achieved_goals.add(goal)
                print(f"🏆 Goal achieved: {goal}")
                return True
            return False

    def get_remaining_goals(self, with_coordinates: bool = False) -> List[Any]:
        """Get list of unachieved mission goals"""
        with self._lock:
            remaining_names = list(self.mission_goals - self.achieved_goals)
            
            if not with_coordinates:
                return remaining_names  # Backward compatibility
            
            # Return goals with coordinates when requested
            result = []
            for goal_name in remaining_names:
                if with_coordinates and goal_name in self.mission_goals_coordinates:
                    coords_data = self.mission_goals_coordinates[goal_name]
                    result.append({
                        "name": goal_name,
                        "coordinates": coords_data["world_coordinates"],
                        "timestamp": coords_data.get("timestamp")
                    })
                else:
                    # Fallback: just the name
                    result.append({"name": goal_name, "coordinates": None})
            
            return result
        
        
    def is_mission_complete(self) -> bool:
        """Check if all mission goals are achieved"""
        with self._lock:
            # Minimal debug print - add these 3 lines
            print(f"🔍 Mission check: {len(self.mission_goals)} total goals, {len(self.get_remaining_goals())} remaining")
            print(f"   Mission goals: {self.mission_goals}")
            print(f"   Remaining goals: {self.get_remaining_goals()}")
            
            return len(self.mission_goals) > 0 and len(self.get_remaining_goals()) == 0
        
        
        
    def set_umeyama_aligned_goal(self, goal_name: str, world_coordinates: List[float], room: str = None):
        """Store mission goal with Umeyama-aligned world coordinates - APPENDS multiple instances"""
        with self._lock:
            # Create a unique key for this goal instance
            timestamp = time.time()
            goal_key = f"{goal_name}_{timestamp}"
            
            self.umeyama_aligned_mission_goals[goal_key] = {
                "goal_name": goal_name,
                "world_coordinates": world_coordinates,
                "room": room or "unknown",
                "timestamp": timestamp,
                "achieved": False  # Track achievement status
            }
            print(f"🎯 Umeyama-aligned goal stored: {goal_name} at {world_coordinates}")
            print(f"   Total stored goals: {len(self.umeyama_aligned_mission_goals)}")
            
            
            
    def get_umeyama_aligned_goal_by_name(self, goal_name: str) -> List[Dict[str, Any]]:
        """Get all Umeyama-aligned mission goals with matching name - BACKWARD COMPATIBLE"""
        with self._lock:
            matching_goals = []
            
            # Try to find exact match (original behavior)
            for goal_key, goal_data in self.umeyama_aligned_mission_goals.items():
                if goal_data.get("goal_name") == goal_name and not goal_data.get("achieved", False):
                    matching_goals.append(goal_data.copy())
            
            # ✅ BACKWARD COMPATIBLE: Return list (empty if no matches)
            if matching_goals:
                return matching_goals
            else:
                print(f"⚠️ No Umeyama goal found for '{goal_name}', returning empty list")
                return []  # ← Maintains backward compatibility


    # ==================== TASK GENERATION & PLANNING ====================

    def generate_exploration_task(self, area: str = None) -> str:
        """Generate an exploration task"""
        description = f"Explore {area}" if area else "Explore unknown area"
        return self.add_task(
            TaskType.EXPLORE,
            description,
            parameters={"area": area} if area else {},
            metadata={"auto_generated": True},
        )

    def generate_find_object_task(self, object_name: str, room: str = None) -> str:
        """Generate a find object task"""
        if room:
            description = f"Find {object_name} in {room}"
            parameters = {"object": object_name, "room": room}
        else:
            description = f"Find {object_name}"
            parameters = {"object": object_name}

        return self.add_task(
            TaskType.FIND_OBJECT,
            description,
            parameters=parameters,
            metadata={"auto_generated": True},
        )

    def generate_navigation_task(self, target_room: str) -> str:
        """Generate a navigation task"""
        return self.add_task(
            TaskType.NAVIGATE_TO,
            f"Navigate to {target_room}",
            parameters={"target_room": target_room},
            metadata={"auto_generated": True},
        )

    # ==================== QUERY METHODS ====================

    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of current task state"""
        with self._lock:
            current_task_info = None
            if self.current_task:
                current_task_info = {
                    "id": self.current_task.id,
                    "type": self.current_task.type.value,
                    "description": self.current_task.description,
                    "duration": time.time()
                    - (self.current_task.started_at or self.current_task.created_at),
                    "attempt": self.task_attempts.get(self.current_task.id, 1),
                }

            return {
                "current_task": current_task_info,
                "queued_tasks": len(self.task_queue),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "mission_goals": {
                    "total": len(self.mission_goals),
                    "achieved": len(self.achieved_goals),
                    "remaining": len(self.get_remaining_goals()),
                    "is_complete": self.is_mission_complete(),
                },
                "performance_metrics": self._get_performance_summary(),
            }

    def get_task_context(self) -> Dict[str, Any]:
        """Get context for task planning/generation"""
        with self._lock:
            return {
                "mission_goals": list(self.mission_goals),
                "achieved_goals": list(self.achieved_goals),
                "recent_tasks": [
                    task.description for task in self.completed_tasks[-5:]
                ],
                "current_task_type": (
                    self.current_task.type.value if self.current_task else None
                ),
                "task_queue_length": len(self.task_queue),
            }

    # ==================== PERFORMANCE TRACKING ====================

    def _record_task_metrics(self, task_type: TaskType, duration: float, success: bool):
        """Record performance metrics for a task"""
        if task_type not in self.task_durations:
            self.task_durations[task_type] = []
            self.success_rates[task_type] = []

        self.task_durations[task_type].append(duration)
        self.success_rates[task_type].append(success)

        # Keep only recent history
        if len(self.task_durations[task_type]) > 100:
            self.task_durations[task_type] = self.task_durations[task_type][-50:]
            self.success_rates[task_type] = self.success_rates[task_type][-50:]

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        with self._lock:
            summary = {}
            for task_type in TaskType:
                durations = self.task_durations.get(task_type, [])
                successes = self.success_rates.get(task_type, [])

                if durations:
                    summary[task_type.value] = {
                        "avg_duration": sum(durations) / len(durations),
                        "success_rate": (
                            sum(successes) / len(successes) if successes else 0
                        ),
                        "total_completed": len(successes),
                    }
            return summary

    # ==================== PERSISTENCE & RESET ====================

    def reset_mission(self):
        """Reset mission state (keep performance history)"""
        with self._lock:
            self.task_queue.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            self.current_task = None
            self.mission_goals.clear()
            self.achieved_goals.clear()

            # Keep performance metrics across missions
            # self.task_durations.clear()  # Uncomment to reset performance too
            # self.success_rates.clear()

            self.mission_start_time = time.time()
            print("🔄 Mission reset - ready for new tasks")

    def clear_all(self):
        """Clear all task data including performance history"""
        with self._lock:
            self.task_queue.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()
            self.current_task = None
            self.mission_goals.clear()
            self.achieved_goals.clear()
            self.task_attempts.clear()
            self.task_timeouts.clear()
            self.task_durations.clear()
            self.success_rates.clear()

            self.mission_start_time = time.time()
            print("🧹 Task store completely cleared")

    # ==================== VALIDATION & SAFETY ====================

    def validate_task_parameters(self, task_type: TaskType, parameters: Dict) -> bool:
        """Validate task parameters before creation"""
        if task_type == TaskType.FIND_OBJECT:
            return "object" in parameters
        elif task_type == TaskType.NAVIGATE_TO:
            return "target_room" in parameters
        elif task_type == TaskType.EXPLORE:
            return True  # No required parameters
        return False

    def check_task_timeout(self) -> Optional[str]:
        """Check if current task has timed out"""
        with self._lock:
            if not self.current_task:
                return None

            task_id = self.current_task.id
            max_duration = self.task_timeouts.get(task_id, 300)  # Default 5min

            start_time = self.current_task.started_at or self.current_task.created_at
            if time.time() - start_time > max_duration:
                return f"Task timeout after {max_duration}s"

            return None

    def set_initial_task(self, task_description: str) -> None:
        """Set the initial mission objective (orchestrator interface)"""
        with self._lock:
            # Parse task description into mission goals
            goals = self._parse_task_to_goals(task_description)
            self.set_mission_goals(goals)

            # Create initial exploration task
            self.generate_exploration_task()

            print(f"🎯 Initial task set: {task_description}")

    def get_simulation_command(self) -> Dict[str, Any]:
        """Get next simulation command (advance/pause/shutdown)"""
        with self._lock:
            # Check if mission is complete
            if self.is_mission_complete():
                return {"action": "shutdown", "reason": "mission_complete"}

            # Check if we have a current task
            if not self.current_task:
                next_task = self.get_next_task()
                if not next_task:
                    # No tasks - explore by default
                    self.generate_exploration_task()
                    return {"action": "advance", "movement": "move_forward"}

            # Default: continue advancing
            return {"action": "advance", "movement": "move_forward"}

    def _parse_task_to_goals(self, task_description: str) -> List[str]:
        """Parse natural language task into mission goals"""
        goals = []

        # Simple parsing - extract objects and locations
        import re

        # Find objects (find chair, locate table, get book)
        objects = re.findall(r"(?:find|locate|get)\s+(\w+)", task_description.lower())
        goals.extend(objects)

        # Find locations (go to kitchen, navigate to bedroom)
        locations = re.findall(
            r"(?:go to|navigate to|reach)\s+(\w+)", task_description.lower()
        )
        goals.extend(locations)

        # If no specific goals found, use exploration
        if not goals and "explore" in task_description.lower():
            goals = ["explore_area"]

        return goals


if __name__ == "__main__":
    # Test the task store
    store = TaskStore()

    # Set mission goals
    store.set_mission_goals(["chair", "table", "kitchen"])

    # Add some tasks
    store.generate_exploration_task("living room")
    store.generate_find_object_task("chair", "living room")
    store.generate_navigation_task("kitchen")

    # Process tasks
    while True:
        task = store.get_next_task()
        if not task:
            break

        print(f"Processing: {task.description}")

        # Simulate task completion
        import random

        success = random.choice([True, False])
        store.complete_current_task(success)

        # Record goal if applicable
        if success and task.type == TaskType.FIND_OBJECT:
            obj = task.parameters.get("object")
            if obj:
                store.record_goal_achievement(obj)

    # Print summary
    summary = store.get_task_summary()
    print("Task Summary:", summary)
