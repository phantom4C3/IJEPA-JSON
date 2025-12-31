#!/usr/bin/env python3
"""
RUN ACTION PIPELINE - Pure Orchestrator
Manages pipeline lifecycle, timing, and component coordination
KNOWS WHEN to run things, not WHAT to run
"""

import logging
import time
import sys
import os
import argparse
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)
import random
from src.stores.task_store import TaskStore
from src.stores.central_map_store import CentralMapStore
from src.stores.prediction_store import PredictionStore
from src.action_pipeline.action_executor import ActionExecutor


class ActionPipeline:
    def __init__(
        self,
        user_command_store=None,
        map_store=None,
        task_store=None,
        prediction_store=None,
        config: Dict[str, Any] = None,
    ):  # ✅ ADD STORES
        self.config = config or {}

        # 🎯 STORE ALL REFERENCES FIRST
        self.map_store = map_store
        self.prediction_store = prediction_store
        self.task_store = task_store
        self.user_command_store = user_command_store

        self.trigger_count = 0

        # ✅ INITIALIZE ACTION EXECUTOR WITH SHARED STORES
        self.action_executor = ActionExecutor(
            task_store=self.task_store,  # For reading/writing tasks
            map_store=self.map_store,  # Future: collision mapping
            prediction_store=self.prediction_store,  # Future: risk prediction
            user_command_store=self.user_command_store,  # Future: emergency override
            config=self.config,  # Future: action params from config
        )

        self.task_store.subscribe(self.on_action_plan_ready)

        # ✅ ADD COUNTER HERE

        self.is_running = False
        self.cycle_counter = 0
        self.is_paused = False

    def initialize_processing(self):
        """REAL initialization - Set up action processing with actual checks and setup"""

        try:
            # 🚀 MINIMAL STEP-BY-STEP EXPLORATION (SAFE VERSION)
            from src.stores.habitat_store import global_habitat_store

            print(f"   global_habitat_store imported: {global_habitat_store is not None}")

            MAX_EXPLORATION_STEPS = 0

            # In initialize_processing():
            for step in range(1, MAX_EXPLORATION_STEPS + 1):
                print(f"🌱 Exploration step {step}")

                choice = random.random()
                
                # 🎯 MINIMAL CHANGE: Use velocity_control instead of discrete actions
                if choice < 0.5: 
                    action_type = "velocity_control"
                    parameters = {
                        "linear_velocity": [0.1, 0.0, 0.0],  # 0.3 m/s forward (x-axis)
                        "angular_velocity": [0.0, 0.0, 0.0],  # No rotation
                        "duration": 0.1,  # Execute for 0.5 seconds
                    }
                    # 🎯 ADD METADATA (SAME AS ACTION EXECUTOR!)
                    velocity_metadata = {
                        "exploration_step": step,
                        "continuous_nav": True,  # 🎯 CRITICAL: This triggers velocity path!
                        "exploration_mode": "random_walk",
                        "velocity_data": {
                            "lin_vel": 0.1,
                            "ang_vel": 0.0,
                            "duration": 0.1
                        }
                    }
                    
                elif choice < 0.75:
                    # Turn left at 0.7 rad/s for 0.5 seconds = ~20 degrees
                    action_type = "velocity_control"
                    parameters = {
                        "linear_velocity": [0.0, 0.0, 0.0],  # No forward movement
                        "angular_velocity": [0.0, 0.0, 0.0],  # 0.7 rad/s left (y-axis)
                        "duration": 0.1,  # Execute for 0.5 seconds
                    }
                    # 🎯 ADD METADATA (SAME AS ACTION EXECUTOR!)
                    velocity_metadata = {
                        "exploration_step": step,
                        "continuous_nav": True,  # 🎯 CRITICAL: This triggers velocity path!
                        "exploration_mode": "random_turn_left",
                        "velocity_data": {
                            "lin_vel": 0.0,
                            "ang_vel": 0.0,
                            "duration": 0.1
                        }
                    }
                    
                else:
                    # Turn right at 0.7 rad/s for 0.5 seconds = ~20 degrees
                    action_type = "velocity_control"
                    parameters = {
                        "linear_velocity": [0.02, 0.0, 0.0],  # No forward movement
                        "angular_velocity": [0.0, 0.0, 0.0],  # -0.7 rad/s right (z-axis)
                        "duration": 0.1,  # Execute for 0.5 seconds
                    }
                    # 🎯 ADD METADATA (SAME AS ACTION EXECUTOR!)
                    velocity_metadata = {
                        "exploration_step": step,
                        "continuous_nav": True,  # 🎯 CRITICAL: This triggers velocity path!
                        "exploration_mode": "random_turn_right",
                        "velocity_data": {
                            "lin_vel": 0.02,
                            "ang_vel": 0.0,
                            "duration": 0.1
                        }
                    }

                # 🎯 PUSH ACTION WITH METADATA (EXACTLY LIKE ACTION EXECUTOR!)
                print(f"📤 Pushing exploration action '{action_type}' to HabitatStore")
                action_id = global_habitat_store.push_action_to_habitat_store(
                    action_type, 
                    parameters,
                    velocity_metadata  # 🎯 ADD THIS LINE (same as Action Executor!)
                )
                time.sleep(0.02)
                
                # 🎯 WAIT FOR RESULT
                print(f"⏳ Waiting for exploration action to complete...")
                result = global_habitat_store.get_action_result(action_id)
                print(f"Result from action pipeline exploration: {result}")
                

                
                if result:
                    # 🎯 MINIMAL RESULT CHECKING
                    if not result or result.get("observations", {}).get("collided", False):
                        print(f"❌ Exploration step from action pipeline {step} failed: ")
                    else:
                        # Optional: Log some observations if available
                        observations = result.get("observations", {})
                        if observations:
                            # Check for collisions
                            if result.get("collided", False):
                                print(f"⚠️ Exploration step {step}: Collision detected!")
                            
                            # Check position if available
                            position = result.get("position")
                            if position:
                                print(f"📍 New position after step {step}: {position}")
                                
                else:
                    print(f"⚠️ No result received for exploration step {step}")
 

            print("✅ Exploration budget exhausted, returning control")
            
            # ✅ REAL: Set up initial task if none exists
            current_task = self.task_store.get_current_task_details()
            if not current_task:
                self.task_store.add_to_task_queue(
                    "Explore and map environment", "exploration", "medium"
                ) 

            # ✅ REAL: Set pipeline state
            self.is_running = False  # Will be started explicitly
            self.is_paused = False
            self.cycle_counter = 0

            return True

        except Exception as e:

            return False

    def on_action_plan_ready(self, event_type: str, data: Dict):
        """Auto-called when new action plan is available from Reasoning"""

        import threading

        print(
            f"🧵 Active threads Before on_actionplan_ready: {threading.active_count()}"
        )
        print(f"Thread names: {[t.name for t in threading.enumerate()]}")

        print("🔔" * 40)
        print("ACTION PLAN RECEIVED → STARTING EXECUTION THREAD")
        print(f"🔔 TASKSTORE TRIGGER RECEIVED!")
        print(f"   📡 Event type: {event_type}")
        print(f"   📦 Data keys: {list(data.keys())}")
        print("🔔" * 40)

        # ✅ UPDATE COUNTER FIRST
        self.trigger_count += 1
        print(f"📊 Trigger count: {self.trigger_count}") 
        
        
        
        
        
            # 🔥 ADD THIS 1 LINE:
        if self.trigger_count > 1:
            print(f"🛑 ActionPipeline: Skipping trigger #{self.trigger_count} (STOPPED)")
            return  # ← EARLY EXIT!

        
        
        
        
        

        if event_type == "action_plan_ready" and self.trigger_count == 1:
            action_plan = data["action_plan"]

            # 🎯 Initialize queue if it doesn't exist
            if not hasattr(self, "pending_action_plans"):
                self.pending_action_plans = []
                print("📋 Action plan queue created")

            # 🎯 Add to queue
            self.pending_action_plans.append(action_plan)

            action_name = action_plan.get("selected_action", "unknown")
            confidence = action_plan.get("confidence", 0.0)
            queue_position = len(self.pending_action_plans)

            print(f"📥 ACTION PLAN QUEUED:")
            print(f"   🎯 Action: {action_name}")
            print(f"   📊 Confidence: {confidence:.2f}")
            print(f"   📍 Queue position: {queue_position}")
            print(f"   🆔 Plan ID: {action_plan.get('reasoning_cycle_id', 'unknown')}")

            # 🎯 Track statistics
            if not hasattr(self, "action_plans_executed"):
                self.action_plans_executed = 0
                print("📈 Action plan counter initialized")

            # 🎯 START EVENT-DRIVEN LOOP ONCE (NON-BLOCKING)
            if not getattr(self, "_event_driven_started", False):
                print(
                    f"📥 ACTION PLAN started: using _event_driven_started trigger and run_event_driven finction call "
                )
                import threading

                threading.Thread(target=self.run_event_driven, daemon=True).start()
                self._event_driven_started = True

        print(
            f"🧵 Active threads After on_actionplan_ready: {threading.active_count()}"
        )
        print(f"Thread names: {[t.name for t in threading.enumerate()]}")

    def run_event_driven(self) -> Dict[str, Any]:
        """
        EVENT-DRIVEN execution - Wait for action plans and execute them once
        No loops, just waits for triggers from TaskStore
        """
        print("🎯" * 50)
        print("🦾 ACTION PIPELINE: STARTING EVENT-DRIVEN EXECUTION")
        print("🎯" * 50)
        print("   📡 Mode: Event-driven (no loops)")
        print("   ⏳ Waiting for action plans from ReasoningPipeline...")
        print("   🔔 Listening for TaskStore triggers...")

        self.is_running = True

        try:

            # 🎯 Start the executor (but don't run cycles)
            if hasattr(self.action_executor, "start_execution"):
                self.action_executor.start_execution()
                print("✅ Action executor started")
            else:
                print("ℹ️  Action executor doesn't have start_execution method")

            # 🎯 WAIT FOR ACTION PLANS (event-driven)
            end_time = time.time()
            action_plans_executed = 0
            check_count = 0

            print("🔄 Entering event-driven wait loop...")

            while self.is_running:

                # 🎯 CHECK FOR PENDING ACTION PLANS (triggered by on_action_plan_ready)
                if self.pending_action_plans:
                    action_plan = self.pending_action_plans.pop(0)
                    action_name = action_plan.get("selected_action", "unknown")
                    confidence = action_plan.get("confidence", 0.0)

                    print("🎯" * 40)
                    print(f"🚀 EXECUTING ACTION PLAN: '{action_name}'")
                    print(f"   📊 Confidence: {confidence:.2f}")
                    print(
                        f"   📋 Plan ID: {action_plan.get('reasoning_cycle_id', 'unknown')}"
                    )
                    print(
                        f"   📝 Source: {action_plan.get('reasoning_source', 'unknown')}"
                    )
                    print("🎯" * 40)

                    # 🎯 EXECUTE SINGLE ACTION PLAN (no loop)
                    execution_start = time.time()
                    print(
                        "🦾 Initiating run_execution_cycle method from action executor..."
                    )
                    result = self.action_executor.run_execution_cycle()

                    execution_time = time.time() - execution_start
                    action_plans_executed += 1

                    print("✅" * 40)
                    print(f"✅ ACTION COMPLETED: '{action_name}'")
                    print(f"   ⏱️  Execution time: {execution_time:.2f}s")
                    print(f"   📊 Result: {result.get('status', 'unknown')}")
                    print(f"   🎯 Total executed: {action_plans_executed}")
                    print("✅" * 40)

                    # 🎯 Check if we should stop after this action
                    if self._should_stop_pipeline(result):
                        print("🎯 Mission completed - stopping execution")
                        break
                else:
                    time.sleep(0.1)  # Small sleep to avoid busy waiting

            return {
                "action_plans_executed": action_plans_executed,
                "final_status": (
                    "completed" if action_plans_executed > 0 else "no_actions_received"
                ),
            }

        except KeyboardInterrupt:
            print("⏹️ Execution interrupted by user")
            return {"final_status": "interrupted"}
        except Exception as e:
            print(f"❌ Event-driven execution failed: {e}")
            return {"final_status": "error", "error": str(e)}
        finally:
            print("🔚 Shutting down event-driven execution...")
            self.shutdown()

    def _should_stop_pipeline(self, cycle_result: Dict[str, Any]) -> bool:
        """
        High-level pipeline stopping conditions
        Pure orchestration logic only
        """
        # Mission completed
        if cycle_result.get("mission_complete", False):
            return True

        # Executor requests stop
        if cycle_result.get("should_stop", False):
            return True

        # Critical system failure
        if cycle_result.get("system_health", {}).get("critical_failure", False):
            return True

        return False

    def shutdown(self):
        """Graceful pipeline shutdown"""
        
        print("🔚 Shutting down run_action_pipeline execution...")    

        self.is_running = False
        self.is_paused = False

        # Shutdown executor
        if self.action_executor:
            self.action_executor.shutdown()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Action Pipeline in Event-Driven Mode"
    )

    # ✅ KEEP THESE TASK ARGUMENTS:
    parser.add_argument(
        "--task",
        type=str,
        help="Mission task description (e.g., 'find keys in kitchen')",
    )

    parser.add_argument(
        "--scenario",
        choices=["exploration", "search", "navigation"],
        help="Predefined scenario type",
    )

    parser.add_argument(
        "--object", type=str, help="Object to search for (use with --scenario search)"
    )

    parser.add_argument("--room", type=str, help="Target room (use with --scenario)")

    parser.add_argument(
        "--target", type=str, help="Navigation target (use with --scenario navigation)"
    )

    parser.add_argument(
        "--complexity",
        choices=["simple", "medium", "complex"],
        default="medium",
        help="Task complexity level (default: medium)",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Interactive task input mode"
    )

    parser.add_argument(
        "--fake-sensor-data", action="store_true", help="Use simulated sensor data"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    # ✅ ADD TIMEOUT ARGUMENT
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="Timeout in seconds (default: 300)"
    )

    return parser.parse_args()


def main():
    """Main entry point - pure orchestration"""
    print("🚀" * 50)
    print("🚀 STARTING ACTION PIPELINE (EVENT-DRIVEN MODE)")
    print("🚀" * 50)

    args = parse_arguments()

    try:
        # ✅ INITIALIZE ALL REQUIRED STORES FIRST
        print("🎯 INITIALIZING STORES...")

        # 1. Central Map Store
        from src.stores.central_map_store import CentralMapStore

        central_map_store = CentralMapStore.get_global()
        print("✅ CentralMapStore initialized")

        # 2. Prediction Store
        from src.stores.prediction_store import PredictionStore

        prediction_store = PredictionStore()
        print("✅ PredictionStore initialized")

        # 3. Task Store
        from src.stores.task_store import TaskStore

        task_store = TaskStore()
        print("✅ TaskStore initialized")

        print("🎯 ALL STORES INITIALIZED SUCCESSFULLY")

        runner = None

        try:
            # ✅ INITIALIZE ACTION PIPELINE WITH ALL STORES
            print("🔧 CREATING ACTION PIPELINE...")
            runner = ActionPipeline(
                map_store=central_map_store,
                task_store=task_store,
                prediction_store=prediction_store,
                config={},
            )

            print("✅ ACTION PIPELINE CREATED SUCCESSFULLY")
            print("🎯" * 50)
            print("🦾 ACTION PIPELINE READY FOR EVENT-DRIVEN EXECUTION")
            print("   📡 Mode: Event-driven (no loops)")

            # ✅ USE ARGS TIMEOUT
            timeout_seconds = args.timeout
            print(f"   ⏰ Timeout: {timeout_seconds} seconds")

            # ✅ CHECK IF TASK PROVIDED AND BUILD TASK STRING
            task_to_execute = None
            if args.task:
                task_to_execute = args.task
                print(f"   🎯 Task from command line: '{task_to_execute}'")
                print("   🔧 Will execute task directly (bypassing ReasoningPipeline)")
            elif args.scenario:
                # Build task from scenario arguments
                if args.scenario == "navigation" and args.target:
                    task_to_execute = f"navigate to {args.target}"
                elif args.scenario == "search" and args.object and args.room:
                    task_to_execute = f"find {args.object} in {args.room}"
                elif args.scenario == "exploration":
                    task_to_execute = f"explore {args.room or 'the environment'}"

                if task_to_execute:
                    print(f"   🎯 Built task from scenario: '{task_to_execute}'")
                    print(
                        "   🔧 Will execute task directly (bypassing ReasoningPipeline)"
                    )
                else:
                    print("   🔔 Listening for TaskStore triggers...")
                    print("   📋 Waiting for ReasoningPipeline to send action plans...")
            else:
                print("   🔔 Listening for TaskStore triggers...")
                print("   📋 Waiting for ReasoningPipeline to send action plans...")

            print("🎯" * 50)

        except Exception as e:
            logging.error(f"❌ PIPELINE FAILED: {e}")
            return 1
        finally:
            if runner:
                print("🔚 SHUTTING DOWN ACTION PIPELINE...")
                runner.shutdown()

        return 0

    finally:
        print("🔒 SYSTEM LOCK CLEANUP COMPLETED")


# ✅ ADD THIS AT THE VERY END (AFTER THE main() FUNCTION):
if __name__ == "__main__":
    print("🔧 SCRIPT STARTED - CALLING MAIN FUNCTION...")
    exit(main())
