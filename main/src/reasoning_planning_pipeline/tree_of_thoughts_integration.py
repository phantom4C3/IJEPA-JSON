
#!/usr/bin/env python3
"""
TREE OF THOUGHTS INTEGRATION - Unified LLM-based planning with internal LLM loading
Single component handling commonsense reasoning + strategic planning
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os
import json
import re
import time
from typing import Dict, Optional, Tuple

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


class StrategyType(Enum):
    PROXIMITY_FIRST = "proximity_first"
    PRIORITY_FIRST = "priority_first"
    ROOM_EXPLORATION = "room_exploration"
    LANDMARK_NAVIGATION = "landmark_navigation"
    SYSTEMATIC_SEARCH = "systematic_search"


import instructor
from pydantic import BaseModel
from typing import List, Literal


# Stage 1: Strategy selection
class StrategyResponse(BaseModel):
    mission_analysis: str
    recommended_strategy: Literal[
        "proximity_first", "priority_first", "systematic_search", "room_exploration"
    ]
    strategy_reasoning: str
    confidence: float  # 0.0 to 1.0
    key_considerations: List[str]


# Stage 2: Object planning
class ObjectPlan(BaseModel):
    target_objects: List[str]
    search_pattern: str
    reasoning: str
    expected_success: float
    fallback_objects: List[str]


class TreeOfThoughtsIntegration:
        """
        Unified LLM-based planner with internal LLM loading
        Handles commonsense reasoning + strategic planning in optimized stages
        """

        def __init__(
            self,
            prediction_store=None,
            map_store=None,
            user_command_store=None,
            task_store=None,
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        ):
            """
            Initialize Tree of Thoughts with all required stores

            Args:
                prediction_store: For risk assessments and safety predictions
                map_store: For world state, objects, and robot position
                user_command_store: For mission intent and user commands
                task_store: For task status and reasoning history
                model_name: LLM model to use for planning
            """
            # 🎯 STORE DEPENDENCIES
            self.prediction_store = prediction_store
            self.map_store = map_store
            self.user_command_store = user_command_store
            self.task_store = task_store
            self.model_name = model_name

            # 🎯 VALIDATE CRITICAL STORES
            if not self.map_store:
                logging.warning(
                    "❌ Tree of Thoughts: No map_store provided - world context will be limited"
                )
            if not self.user_command_store:
                logging.warning(
                    "❌ Tree of Thoughts: No user_command_store provided - mission context will be limited"
                )

            # 🎯 INITIALIZE LLM SERVICE
            self.llm_service = self._initialize_llm_service()

            # 🎯 STRATEGY TEMPLATES FOR FALLBACK
            self.strategy_templates = {
                "find_object": [
                    "Check high-priority surfaces first",
                    "Search closest objects first",
                    "Explore room systematically",
                    "Focus on typical locations",
                ],
                "explore_room": [
                    "Navigate to farthest point first",
                    "Follow perimeter/walls",
                    "Check central landmarks then corners",
                    "Systematic grid pattern",
                ],
                "navigate_to": [
                    "Take shortest safe path",
                    "Avoid obstacles and high-risk areas",
                    "Use landmarks for navigation",
                    "Follow systematic waypoint sequence",
                ],
            }

            # 🎯 CACHE FOR COMMON REASONING PATTERNS
            self.reasoning_cache = {}
            self.cache_max_size = 50

            print(f"🌳 Tree of Thoughts Integration initialized with LLM: {model_name}")
            print(f"   📍 Map Store: {'✅' if map_store else '❌'}")
            print(f"   🎯 User Command Store: {'✅' if user_command_store else '❌'}")
            print(f"   📊 Prediction Store: {'✅' if prediction_store else '❌'}")
            print(f"   📝 Task Store: {'✅' if task_store else '❌'}")

        def _initialize_llm_service(self):
            try:
                from transformers import pipeline
                import torch

                print(f"🎯 LLM DEBUG: Starting to load {self.model_name}")

                pipe = pipeline(
                    "text-generation",
                    model=self.model_name,
                    dtype=torch.bfloat16,
                    device_map="cpu",
                )

                print(f"✅ LLM DEBUG: Pipeline loaded successfully on {pipe.device}")
                return pipe

            except Exception as e:
                print(f"❌ LLM DEBUG: Pipeline FAILED: {e}")
                import traceback

                traceback.print_exc()
                return None


 
        def _stage1_strategy_selection(self) -> Tuple[Optional[Dict], Dict]:
            """
            Returns: (strategy_data, reasoning_step) - NO SAVING!
            SIMPLIFIED: Returns raw response for Stage 2 to parse
            """

            print(f"🔍 DEBUG _stage1_strategy_selection: Starting stage 1")
            context = self._extract_minimal_context()
            print(
                f"🔍 DEBUG _stage1_strategy_selection: Context extracted: mission={context.get('mission', 'N/A')}"
            )

            if not self.llm_service:
                return None, {}

            try:
                # ✅ SIMPLE PROMPT - No JSON template
                prompt = f"""Choose best search strategy for finding keys:

        Options: proximity_first, priority_first, systematic_search, room_exploration

        Mission: {context.get('mission', 'find object')}
        Location: {context.get('room_type', 'room')}
        Objects available: {', '.join(context.get('key_objects', [])[:3])}

        Answer with ONE word from the options above, then a brief reason.

        Example: "proximity_first - check closest objects first"

        Your answer:"""

                print(f"🔍 DEBUG Prompt length: {len(prompt)} chars")

                # Get LLM response
                outputs = self.llm_service(
                    prompt,
                    max_new_tokens=100,  # Shorter response
                    do_sample=True,
                    temperature=0.1,     # Lower temperature for consistency
                    repetition_penalty=1.2,
                )

                full_output = outputs[0]["generated_text"]
                response_text = full_output.replace(prompt, "").strip()
                print(f"🔍 DEBUG Raw response: {response_text}")

                # ✅ SIMPLE TEXT EXTRACTION - No JSON parsing here
                strategy_type = "proximity_first"  # Default
                
                # Look for strategy in response
                response_lower = response_text.lower()
                if "priority" in response_lower:
                    strategy_type = "priority_first"
                elif "systematic" in response_lower:
                    strategy_type = "systematic_search"
                elif "room" in response_lower and "exploration" in response_lower:
                    strategy_type = "room_exploration"
                elif "proximity" in response_lower:
                    strategy_type = "proximity_first"

                print(f"✅ Extracted strategy: {strategy_type}")

                # Create strategy_data with raw response for Stage 2
                strategy_data = {
                    "strategy_type": strategy_type,
                    "strategy_reasoning": response_text[:100],  # First 100 chars of raw response
                    "confidence": 0.7,  # Default confidence
                    "mission_analysis": f"Finding keys in {context.get('room_type', 'room')}",
                    "key_considerations": ["speed", "accuracy", "object locations"],
                    # 🔥 CRITICAL: Add raw data for Stage 2
                    "raw_prompt": prompt,       # Stage 1 prompt
                    "raw_response": response_text,  # Stage 1 raw LLM response
                    "context_objects": context.get('key_objects', [])[:5]  # Objects for Stage 2
                }

                reasoning_step = {
                    "stage": "strategy_selection",
                    "timestamp": time.time(),
                    "input_context": context,
                    "llm_prompt": prompt[:300],
                    "llm_response": response_text[:300],
                    "output_reasoning": {
                        "strategy_type": strategy_type,
                        "strategy_reasoning": response_text[:100],
                        "confidence": 0.7,
                    },
                }

                print(
                    f"🔍 DEBUG _stage1_strategy_selection: Strategy decision: {strategy_type}"
                )

                return strategy_data, reasoning_step

            except Exception as e:
                import logging
                logging.warning(f"Stage 1 LLM failed: {e}")
                print(f"❌ Stage 1 error: {e}")

            return None, {}

            
        def _stage2_detailed_planning(self, strategy_decision: Dict) -> Tuple[Dict, Dict]:
            """
            Returns: (object_based_plan, reasoning_step) - NO SAVING!
            Uses Stage 1's raw data as context
            """

            print(
                f"🔍 STAGE2 DEBUG: strategy_decision keys = {list(strategy_decision.keys())}"
            )
            print(
                f"🔍 STAGE2 DEBUG: strategy_type = {strategy_decision.get('strategy_type', 'MISSING!')}"
            )
            print(
                f"🔍 STAGE2 DEBUG: raw_response = {strategy_decision.get('raw_response', 'NO RAW RESPONSE')[:100]}"
            )

            if not self.llm_service:
                basic_plan = {
                    "target_objects": [],
                    "search_pattern": f"{strategy_decision.get('strategy_type', 'proximity_first')}_fallback",
                    "object_priority": [],
                    "reasoning": "LLM unavailable - using minimal fallback",
                    "expected_success": 0.1,
                }
                reasoning_step = {
                    "stage": "detailed_planning",
                    "timestamp": time.time(),
                    "input_context": self._extract_focused_context(strategy_decision),
                    "output_reasoning": {
                        "search_pattern": "rule_based_fallback",
                        "reasoning": "LLM unavailable - using minimal object planning",
                    },
                }
                return basic_plan, reasoning_step

            context = self._extract_focused_context(strategy_decision)
            
            # ✅ USE Stage 1's raw response as context
            stage1_raw_response = strategy_decision.get('raw_response', 'Check nearby objects first')
            stage1_strategy = strategy_decision.get('strategy_type', 'proximity_first')
            
            # Get objects from Stage 1 context or current context
            available_objects = strategy_decision.get('context_objects', [])
            if not available_objects:
                available_objects = context.get('relevant_objects', ['table', 'cabinet', 'nightstand'])

            # ✅ PROMPT with Stage 1's reasoning
            prompt = f"""Based on this strategy reasoning, create a search plan:

        STRATEGY CHOSEN: {stage1_strategy}
        REASONING: {stage1_raw_response}

        CREATE SEARCH PLAN FOR: {context.get('mission', 'find keys')}
        AVAILABLE OBJECTS: {', '.join(available_objects[:5])}

        Return ONLY this JSON:
        {{
            "target_objects": ["object1", "object2", "object3"],
            "search_pattern": "pattern description",
            "reasoning": "why this order",
            "expected_success": 0.7,
            "fallback_objects": ["object4", "object5"]
        }}

        Choose target_objects from the available objects list.
        search_pattern should match the strategy: {stage1_strategy}
        expected_success should be 0.0 to 1.0.

        ONLY output the JSON, no other text:"""

            try:
                # Get LLM response
                outputs = self.llm_service(
                    prompt,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.2,
                    repetition_penalty=1.2,
                )

                full_response = outputs[0]["generated_text"]
                response_text = full_response.replace(prompt, "").strip()
                print(f"🔍 DEBUG Stage 2 Raw response: {response_text[:200]}...")

                # ✅ USE _parse_llm_json_response for Stage 2
                plan_data = self._parse_llm_text_response(response_text)

                # Handle list output
                if isinstance(plan_data, list) and plan_data:
                    plan_data = plan_data[0]

                print(f"🔍 STAGE2 PARSED plan_data type: {type(plan_data)}")
                print(
                    f"🔍 STAGE2 PARSED plan_data keys: {list(plan_data.keys()) if isinstance(plan_data, dict) else 'NOT DICT'}"
                )

                # Ensure plan_data is dict
                if isinstance(plan_data, str):
                    print(f"⚠️  plan_data is string: {plan_data[:200]}")
                    try:
                        plan_data = json.loads(plan_data)
                    except:
                        plan_data = {}

                if plan_data and isinstance(plan_data, dict):
                    # ✅ SIMPLE: Get t    valid_targets = plan_data.get("target_objects", [])
                    valid_targets = plan_data.get("target_objects",[])

                    
                    # If no valid targets, use first 3 available objects
                    if not valid_targets:
                        valid_targets = ["ceiling", "wall", "door", "handle", "chandelier", "wardrobe", "tv", "cabinet", "blanket", "pad", "bed", "pillow", "nightstand", "book", "lamp", "toy", "window", "frame", "armchair", "floor", "mat", "towel", "bucket", "tap", "soap", "toilet", "brush", "curtain", "photo", "sheet", "ventilation", "vent", "light", "bicycle", "box", "couch", "basket", "magazine", "papers", "picture", "unknown", "folder", "table", "chair", "handbag", "tower", "trashcan", "desk", "printer", "telephone", "plant", "shirt", "bag", "newspaper", "balustrade", "stairs", "rod", "speaker", "fireplace", "flower", "plate", "pillar", "alarm", "control", "clock", "flag", "refrigerator", "appliance", "machine", "mug", "worktop", "sink", "holder", "microwave", "item", "stove", "bowl", "dishwasher", "paper", "seat", "shelf", "doormat", "hood", "dresser", "casket", "decoration", "controller", "dial", "bath", "accessory", "mirror", "bottle", "shoe", "board", "iron", "clothes", "case", "briefcase", "backpack", "boxes"],

                    
                    # Get search pattern or generate from strategy
                    search_pattern = plan_data.get("search_pattern", "")
                    if not search_pattern:
                        search_pattern = f"{stage1_strategy}_pattern"
                    
                    detailed_plan = {
                        "target_objects": valid_targets,
                        "search_pattern": search_pattern,
                        "object_priority": valid_targets,
                        "reasoning": plan_data.get("reasoning", f"Based on {stage1_strategy} strategy"),
                        "expected_success": plan_data.get("expected_success", 0.7),
                        "fallback_objects": plan_data.get("fallback_objects", available_objects[3:5] if len(available_objects) > 3 else ["sofa", "chair"]),
                    }

                    reasoning_step = {
                        "stage": "detailed_planning",
                        "timestamp": time.time(),
                        "input_context": {
                            "mission": context.get('mission', ''),
                            "strategy": stage1_strategy,
                            "stage1_reasoning": stage1_raw_response[:100],
                            "available_objects": available_objects
                        },
                        "llm_prompt": prompt[:300],
                        "llm_response": response_text[:300],
                        "output_reasoning": {
                            "target_objects": detailed_plan.get("target_objects", []),
                            "search_pattern": detailed_plan.get("search_pattern", ""),
                            "object_priority": detailed_plan.get("object_priority", []),
                            "reasoning": detailed_plan.get("reasoning", ""),
                            "expected_success": detailed_plan.get("expected_success", 0.7),
                            "parsed_from_json": True if plan_data else False,
                        },
                    }

                    return detailed_plan, reasoning_step

            except Exception as e:
                import logging
                logging.warning(f"Stage 2 LLM failed: {e}")
                print(f"❌ Stage 2 error: {e}")

            # Fallback using Stage 1 data
            fallback_objects = available_objects[:3] if available_objects else ["table", "cabinet", "nightstand"]
            
            basic_plan = {
                "target_objects": fallback_objects,
                "search_pattern": f"{stage1_strategy}_fallback",
                "object_priority": fallback_objects,
                "reasoning": f"Based on Stage 1 strategy: {stage1_strategy}",
                "expected_success": 0.5,
                "fallback_objects": ["ceiling", "wall", "door", "handle", "chandelier", "wardrobe", "tv", "cabinet", "blanket", "pad", "bed", "pillow", "nightstand", "book", "lamp", "toy", "window", "frame", "armchair", "floor", "mat", "towel", "bucket", "tap", "soap", "toilet", "brush", "curtain", "photo", "sheet", "ventilation", "vent", "light", "bicycle", "box", "couch", "basket", "magazine", "papers", "picture", "unknown", "folder", "table", "chair", "handbag", "tower", "trashcan", "desk", "printer", "telephone", "plant", "shirt", "bag", "newspaper", "balustrade", "stairs", "rod", "speaker", "fireplace", "flower", "plate", "pillar", "alarm", "control", "clock", "flag", "refrigerator", "appliance", "machine", "mug", "worktop", "sink", "holder", "microwave", "item", "stove", "bowl", "dishwasher", "paper", "seat", "shelf", "doormat", "hood", "dresser", "casket", "decoration", "controller", "dial", "bath", "accessory", "mirror", "bottle", "shoe", "board", "iron", "clothes", "case", "briefcase", "backpack", "boxes"][3:15],

            }
            
            reasoning_step = {
                "stage": "detailed_planning_fallback",
                "timestamp": time.time(),
                "input_context": {
                    "strategy": stage1_strategy,
                    "stage1_reasoning": stage1_raw_response[:100]
                },
                "output_reasoning": {
                    "reasoning": f"LLM failed - using fallback for {stage1_strategy} strategy",
                    "is_fallback": True,
                },
            }
            
            return basic_plan, reasoning_step

 


        def _extract_minimal_context(self) -> Dict:
            """Extract minimal context from ALL stores (token efficient)"""

            print(f"🔍 DEBUG _extract_minimal_context: Extracting from stores")
            print(
                f"🔍 DEBUG _extract_minimal_context: self.map_store={self.map_store is not None}, self.user_command_store={self.user_command_store is not None}"
            )

            # 🎯 SOURCE 1: User Command Store (Mission Intent)
            active_command = (
                self.user_command_store.get_active_command()
                if hasattr(self, "user_command_store")
                else None
            )
            mission_data = self._get_mission_from_command_store(active_command)

            # 🎯 SOURCE 2: Central Map Store (World State)
            world_data = self._get_world_from_map_store()

            # 🎯 SOURCE 3: Prediction Store (Risk Assessment)
            risk_data = self._get_risk_from_prediction_store()

            return {
                # From User Command Store
                "mission": mission_data["mission"],
                # 'mission_intent': mission_data['intent'],
                # 'command_priority': mission_data['priority'],
                # From Central Map Store
                "room_type": world_data["room_type"],
                "key_objects": world_data["key_objects"],
                "objects_count": world_data["objects_count"],
                # 'current_position': world_data['current_position'],
                # From Prediction Store
                "risk_level": risk_data["risk_level"],
                # 'safety_confidence': risk_data['safety_confidence'],
                # 'immediate_dangers': risk_data['immediate_dangers']
            }

        def _extract_focused_context(self, strategy_decision: Dict) -> Dict:
            """Stage 2: Build context using Stage 1 output"""

            # ✅ GET MISSION FROM STAGE 1 OUTPUT, NOT FROM STORES!
            mission = strategy_decision.get("mission_analysis", "Find objects")

            # Still need room and objects from MapStore
            world_data = self._get_world_from_map_store()

            # Filter objects based on strategy
            relevant_objects = self._get_strategy_relevant_objects(
                world_data.get("key_objects", []), strategy_decision["strategy_type"]
            )

            return {
                "mission": mission[:100],  # From Stage 1 only!
                "room_type": world_data.get("room_type", "unknown"),
                "relevant_objects": relevant_objects,
                "current_position": [0, 0],
            }

       
        def _build_complete_plan(
            self,
            strategy_decision: Dict,
            detailed_plan: Dict,
            reasoning_steps: List[Dict],
            reasoning_cycle_id: str,
        ) -> Dict:  # 🎯 ADD THIS PARAMETER
            """
            Build complete plan with all reasoning in one structured object
            """

            print(
                f"🔍 DEBUG _build_complete_plan: Building complete plan for cycle_id={reasoning_cycle_id}"
            )
            print(
                f"🔍 DEBUG _build_complete_plan: strategy_type={strategy_decision.get('strategy_type')}"
            )

            return {
                "reasoning_cycle_id": reasoning_cycle_id,
                "timestamp": time.time(),
                "planning_stages": ["strategy_selection", "detailed_planning"],
                "llm_used": True,
                # Core plan components
                "strategy_decision": strategy_decision,
                "detailed_plan": detailed_plan,
                # 🎯 ALL REASONING STEPS TOGETHER
                "reasoning_chain": reasoning_steps,
                # Input context
                "input_context": {
                    "minimal_context": self._extract_minimal_context(),
                    "focused_context": self._extract_focused_context(strategy_decision),
                },
                # Execution-ready simplified plan
                "execution_ready_plan": {
                    "strategy_type": strategy_decision.get("strategy_type"),
                    "search_pattern": detailed_plan.get("search_pattern", ""),
                    "expected_duration": detailed_plan.get(
                        "expected_duration", "medium"
                    ),
                },
            }

        def generate_comprehensive_plan(self, reasoning_cycle_id: str) -> bool:
            # 🎯 ADD LLM AVAILABILITY CHECK
            print(f"🎯 TOT DEBUG: LLM available: {self.llm_service is not None}")

            if not self.llm_service:
                print("❌ TOT DEBUG: LLM service is None - using rule-based fallback")
                return self._generate_rule_based_fallback(reasoning_cycle_id)

            print(
                f"🔍 DEBUG generate_comprehensive_plan: Starting with cycle_id={reasoning_cycle_id}"
            )
            print(
                f"🔍 DEBUG generate_comprehensive_plan: LLM available={self.llm_service is not None}"
            )

            try:
                print(f"🔍 DEBUG generate_comprehensive_plan: Stage 1 completed")
                print(f"🔍 DEBUG generate_comprehensive_plan: Stage 2 completed")

                # Stage 1: Strategy Selection
                strategy_decision, stage1_reasoning = self._stage1_strategy_selection()

                if not strategy_decision:
                    return self._generate_rule_based_fallback(reasoning_cycle_id)

                # Stage 2: Detailed Planning
                detailed_plan, stage2_reasoning = self._stage2_detailed_planning(
                    strategy_decision
                )
                
                
                
                # --- INLINE PATCH START ---
                # Enrich detailed_plan target_objects with coordinates from map_store OR task_store
                if self.map_store and hasattr(self.map_store, 'semantic_objects'):
                    for obj in detailed_plan.get("target_objects", []):
                        if isinstance(obj, str):
                            obj_name = obj
                        elif isinstance(obj, dict) and "name" in obj:
                            obj_name = obj["name"]
                        else:
                            continue

                        coord = None
                        coord_source = None
                        
                        # 1. FIRST try: Get from TaskStore Umeyama-aligned goals (most accurate)
                        if self.task_store and hasattr(self.task_store, 'get_umeyama_aligned_goal_by_name'):
                            aligned_goals = self.task_store.get_umeyama_aligned_goal_by_name(obj_name)
                            if aligned_goals:
                                # Use first (most recent) aligned goal
                                goal_data = aligned_goals[0]
                                coord = goal_data.get("world_coordinates")
                                coord_source = "umeyama_aligned_goal"
                                if coord:
                                    print(f"🎯 TOT: Using Umeyama-aligned coordinates for '{obj_name}'")
                        
                        # 2. SECOND try: Get from MapStore semantic objects (fallback)
                        if not coord and self.map_store:
                            map_obj = self.map_store.semantic_objects.get(obj_name)
                            if map_obj and hasattr(map_obj, "position"):
                                coord = list(map_obj.position)
                                coord_source = "map_store_semantic"
                                if coord:
                                    print(f"📍 TOT: Using map_store coordinates for '{obj_name}'")
                        
                        # Add coordinates if found
                        if coord:
                            if isinstance(obj, str):
                                # Replace string with dict for coordinates
                                idx = detailed_plan["target_objects"].index(obj)
                                detailed_plan["target_objects"][idx] = {
                                    "name": obj,
                                    "coordinates": coord,
                                    "coord_source": coord_source
                                }
                            elif isinstance(obj, dict):
                                obj["coordinates"] = coord
                                obj["coord_source"] = coord_source
                # --- INLINE PATCH END ---

                
                

                # 🎯 BUILD COMPLETE PLAN WITH ALL REASONING
                complete_plan = self._build_complete_plan(
                    strategy_decision,
                    detailed_plan,
                        [stage1_reasoning, stage2_reasoning],  # ✅ CORRECT: Simple list
                    reasoning_cycle_id 
                )
                
                print(f"🔍 TOT FULL PLAN ({reasoning_cycle_id}): {complete_plan}")

                # Save final plan to TaskStore
                if complete_plan:
                    # Directly create the structure main() is looking for
                    self.task_store.write_reasoning_plan(
                        reasoning_cycle_id,
                        {
                            "component": "tree_of_thoughts",
                            "component_reasoning": {
                                "tree_of_thoughts": {
                                    "component": "tree_of_thoughts",
                                    **complete_plan,
                                    "reasoning_cycle_id": reasoning_cycle_id,
                                }
                            },
                            "timestamp": time.time(),
                        },
                    )

                print(f"✅ Generated comprehensive plan from Tree Of Thoughts reasoning {reasoning_cycle_id}")
 

                print(f"🔍 TOT DEBUG: Time after completion: {time.time()}")
                return {
            "reasoning_cycle_id": reasoning_cycle_id,
                }

            except Exception as e:
                print(f"❌ TOT DEBUG ERROR: {e}")
                return self._generate_rule_based_fallback()

        def _classify_mission(self, mission: str) -> str:
            """Classify mission type based on keywords"""
            mission_lower = mission.lower()

            if any(
                word in mission_lower
                for word in ["find", "search", "locate", "get", "fetch"]
            ):
                return "find_object"
            elif any(
                word in mission_lower
                for word in ["explore", "scan", "survey", "inspect"]
            ):
                return "explore_room"
            elif any(
                word in mission_lower
                for word in ["navigate", "go to", "move to", "travel to", "reach"]
            ):
                return "navigate_to"
            else:
                return "find_object"  # default

        def _generate_rule_based_fallback(self, reasoning_cycle_id: str = None) -> str:
            """Generate rule-based plan when LLM fails and return reasoning_cycle_id"""

            # At the beginning of the function:
            if reasoning_cycle_id is None:
                reasoning_cycle_id = f"fallback_{int(time.time())}"

            print(f"🔍 DEBUG _generate_rule_based_fallback: Entering")

            print(
                f"🔍 DEBUG _generate_rule_based_fallback: Cycle ID = {reasoning_cycle_id}"
            )

            context = self._extract_minimal_context()
            print(f"🔍 DEBUG _generate_rule_based_fallback: Context extracted")

            mission_type = self._classify_mission(context["mission"])
            print(
                f"🔍 DEBUG _generate_rule_based_fallback: Mission type = {mission_type}"
            )

            # Simple rule-based strategy selection
            if mission_type == "find_object" and context.get("objects_count", 0) > 5:
                strategy = "priority_first"
                reasoning = "Many objects - focus on high-priority locations"
            elif context["risk_level"] == "high":
                strategy = "systematic_search"
                reasoning = "High risk environment - use cautious systematic approach"
            else:
                strategy = "proximity_first"
                reasoning = "Standard proximity-based search for efficiency"

            print(f"🔍 DEBUG _generate_rule_based_fallback: Strategy = {strategy}")

            # ✅ Create basic plan DIRECTLY (no method call needed)
            basic_plan = {
                "target_objects": ["ceiling", "wall", "door", "handle", "chandelier", "wardrobe", "tv", "cabinet", "blanket", "pad", "bed", "pillow", "nightstand", "book", "lamp", "toy", "window", "frame", "armchair", "floor", "mat", "towel", "bucket", "tap", "soap", "toilet", "brush", "curtain", "photo", "sheet", "ventilation", "vent", "light", "bicycle", "box", "couch", "basket", "magazine", "papers", "picture", "unknown", "folder", "table", "chair", "handbag", "tower", "trashcan", "desk", "printer", "telephone", "plant", "shirt", "bag", "newspaper", "balustrade", "stairs", "rod", "speaker", "fireplace", "flower", "plate", "pillar", "alarm", "control", "clock", "flag", "refrigerator", "appliance", "machine", "mug", "worktop", "sink", "holder", "microwave", "item", "stove", "bowl", "dishwasher", "paper", "seat", "shelf", "doormat", "hood", "dresser", "casket", "decoration", "controller", "dial", "bath", "accessory", "mirror", "bottle", "shoe", "board", "iron", "clothes", "case", "briefcase", "backpack", "boxes"],
                "search_pattern": f"{strategy}_rule_based",
                "object_priority": ["table", "nightstand", "cabinet", "desk"],
                "reasoning": reasoning,
                "expected_success": 0.5,
                "is_fallback": True,
            }

            print(
                f"🔍 DEBUG _generate_rule_based_fallback: Basic plan created with {len(basic_plan['target_objects'])} targets"
            )

            # Create reasoning steps
            reasoning_steps = [
                {
                    "stage": "rule_based_fallback",
                    "timestamp": time.time(),
                    "input_context": context,
                    "output_reasoning": {
                        "strategy_type": strategy,
                        "strategy_reasoning": reasoning,
                        "mission_analysis": "Rule-based fallback plan",
                        "key_considerations": [
                            "LLM unavailable",
                            "Using rule-based logic",
                        ],
                        "confidence": 0.6,
                    },
                }
            ]

            print(f"🔍 DEBUG _generate_rule_based_fallback: Building complete plan...")

            # 🎯 BUILD COMPLETE PLAN
            complete_plan = self._build_complete_plan(
                {
                    "strategy_type": strategy,
                    "strategy_reasoning": reasoning,
                    "confidence": 0.6,
                    "mission_analysis": "Rule-based fallback plan",
                    "key_considerations": ["LLM unavailable", "Using rule-based logic"],
                },
                {**basic_plan, "is_fallback": True},
                reasoning_steps,
                reasoning_cycle_id,  # 🎯 Don't forget this parameter!
            )
            
            
            print(f"🔍 DEBUG _generate_rule_based_fallback: Complete plan built")
            
            # 🔹 Single debug to see the entire plan
            print(f"🔍 TOT FULL PLAN - fallback version ({reasoning_cycle_id}): {complete_plan}")


            # 🎯 STORE IN INTERMEDIATE_REASONING FOR PIPELINE RETRIEVAL
            if self.task_store:
                # Initialize intermediate_reasoning if it doesn't exist
                if not hasattr(self.task_store, "intermediate_reasoning"):
                    self.task_store.intermediate_reasoning = {}

                # Create or update the cycle data
                if reasoning_cycle_id not in self.task_store.intermediate_reasoning:
                    self.task_store.intermediate_reasoning[reasoning_cycle_id] = {
                        "component_reasoning": {},
                        "timestamp": time.time(),
                    }

                # Store tree of thoughts component data
                # In _generate_rule_based_fallback():
                self.task_store.write_reasoning_plan(
                    reasoning_cycle_id,
                    {
                        "component": "tree_of_thoughts_fallback",
                        **complete_plan,
                        "reasoning_cycle_id": reasoning_cycle_id,
                        "is_fallback": True,
                    },
                )

            print(
                f"🔍 DEBUG _generate_rule_based_fallback: Returning cycle ID = {reasoning_cycle_id}"
            )
            return reasoning_cycle_id

        def _get_mission_from_command_store(
            self, active_command: Optional[Dict]
        ) -> Dict:
            """Extract mission data from User Command Store"""

            print(
                f"🔍 DEBUG _get_mission_from_command_store: active_command={active_command is not None}"
            )
            if active_command:
                return {
                    "mission": active_command.get("user_input", "unknown"),
                    "intent": active_command.get("parsed_intent", {}),
                    "priority": active_command.get("priority", "normal"),
                }
            else:
                # ✅ USE self.user_command_store directly
                if self.user_command_store:
                    try:
                        if hasattr(self.user_command_store, "get"):
                            task_goal = self.user_command_store.get("task_goal", {})
                        else:
                            task_goal = getattr(
                                self.user_command_store, "task_goal", {}
                            )

                        current_task = task_goal.get("current_task", {})
                        return {
                            "mission": current_task.get("description", "unknown"),
                            "intent": {"intent_type": "unknown"},
                            "priority": "normal",
                        }
                    except:
                        pass

                # Ultimate fallback
                return {
                    "mission": "unknown",
                    "intent": {"intent_type": "unknown"},
                    "priority": "normal",
                }

        def _get_world_from_map_store(self) -> Dict:
            """Extract world data from Central Map Store - WITHOUT CameraPose"""

            print(f"🔍 DEBUG _get_world_from_map_store: Entering")

            if not self.map_store:
                return self._get_fallback_world()

            try:
                # Get the map summary
                map_summary = self.map_store.get_map_summary()
                print(f"✅ Retrieved map_summary successfully")

                # ✅ DEBUG: Check if map_summary is actually a dict
                if not isinstance(map_summary, dict):
                    print(f"❌ map_summary is not a dict, it's: {type(map_summary)}")
                    return self._get_fallback_world()

                print(f"🔍 map_summary keys: {list(map_summary.keys())}")

                # ✅ SIMPLIFIED: Skip CameraPose extraction entirely
                # Use a fixed position or extract from other fields if available
                current_position = [0, 0]  # Fixed default

                # Alternative: Try to get position from map_summary directly if available
                if "agent_position" in map_summary:
                    position_data = map_summary["agent_position"]
                    if isinstance(position_data, list) and len(position_data) >= 2:
                        current_position = position_data[:2]
                        print(
                            f"✅ Using agent_position from map_summary: {current_position}"
                        )

                # Get room type
                room_type = map_summary.get("current_room", "unknown")
                print(f"🔍 room_type: {room_type}")

                # Get semantic objects (same as before)
                semantic_objects = []

                # Method 1: Direct from map_store.semantic_objects
                if hasattr(self.map_store, "semantic_objects"):
                    semantic_objects = list(self.map_store.semantic_objects.keys())
                    print(
                        f"✅ Got {len(semantic_objects)} objects from semantic_objects"
                    )

                # Method 2: From found_targets
                elif "found_targets" in map_summary and map_summary["found_targets"]:
                    semantic_objects = map_summary["found_targets"]
                    print(f"✅ Using found_targets: {semantic_objects}")

                # Method 3: Object count fallback
                elif "objects_count" in map_summary:
                    count = map_summary["objects_count"]
                    semantic_objects = [f"object_{i}" for i in range(count)]
                    print(f"⚠️ Using object count fallback: {count} objects")

                result = {
                    "room_type": room_type,
                    "key_objects": self._extract_key_objects(semantic_objects),
                    "objects_count": len(semantic_objects),
                    "current_position": current_position,  # Using simplified position
                }

                print(f"✅ _get_world_from_map_store SUCCESS (no CameraPose): {result}")
                return result

            except Exception as e:
                print(f"❌ _get_world_from_map_store FAILED: {e}")
                import traceback

                traceback.print_exc()
                return self._get_fallback_world()

        def _get_risk_from_prediction_store(self) -> Dict:
            """Extract risk data from Prediction Store"""

            print(f"🔍 DEBUG _get_risk_from_prediction_store: Entering")
            print(
                f"🔍 DEBUG _get_risk_from_prediction_store: self.prediction_store type={type(self.prediction_store)}"
            )

            if self.prediction_store:
                try:
                    # Try different interface methods
                    if hasattr(self.prediction_store, "get_latest_prediction"):
                        latest_prediction = (
                            self.prediction_store.get_latest_prediction()
                        )
                    elif hasattr(self.prediction_store, "get"):
                        latest_prediction = self.prediction_store.get(
                            "latest_prediction", {}
                        )
                    else:
                        latest_prediction = getattr(
                            self.prediction_store, "latest_prediction", {}
                        )

                    if latest_prediction:
                        risk_assessment = latest_prediction.get("risk_assessment", {})
                        return {
                            "risk_level": risk_assessment.get(
                                "overall_risk_level", "unknown"
                            ),
                            "safety_confidence": risk_assessment.get(
                                "safe_navigation_confidence", 0.5
                            ),
                            "immediate_dangers": risk_assessment.get(
                                "immediate_danger", False
                            ),
                        }
                except Exception as e:
                    print(f"⚠️  Prediction store access failed: {e}")

            # Fallback
            return {
                "risk_level": self._assess_risk_level(),
                "safety_confidence": 0.7,
                "immediate_dangers": False,
            }

        def _extract_key_objects(
            self, objects: List[Dict], max_count: int = 8
        ) -> List[str]:
            """Extract most relevant objects for token efficiency"""
            if not objects:
                return []

            # Priority objects for common tasks
            priority_objects = [
                "table",
                "desk",
                "counter",
                "nightstand",
                "shelf",
                "cabinet",
            ]

            # Get unique object types, prioritizing important ones
            object_types = {}
            for obj in objects:
                obj_type = str(obj).lower()  # Works for both strings and dicts
                if obj_type in priority_objects:
                    object_types[obj_type] = True

            # Add remaining objects up to max_count
            for obj in objects:
                if len(object_types) >= max_count:
                    break
                obj_type = str(obj).lower()  # Works for both strings and dicts
                object_types[obj_type] = True

            return list(object_types.keys())[:max_count]

        def _get_strategy_relevant_objects(
            self, objects: List[Dict], strategy_type: str
        ) -> List[str]:
            """Get objects relevant to specific strategy"""

            if strategy_type == "proximity_first":
                # All objects for proximity sorting
                return [
                    obj if isinstance(obj, str) else obj.get("name", "")
                    for obj in objects[:10]
                ]
            elif strategy_type == "priority_first":
                # Priority objects only
                priority_types = ["table", "desk", "counter", "nightstand"]
                return [
                    obj if isinstance(obj, str) else obj.get("name", "")
                    for obj in objects
                    if (obj if isinstance(obj, str) else obj.get("name", "")).lower()
                    in priority_types
                ][:8]
            else:
                return [
                    obj if isinstance(obj, str) else obj.get("name", "")
                    for obj in objects[:6]
                ]

        def _assess_risk_level(self) -> str:
            """Quick risk assessment for context"""
            if self.prediction_store:
                try:
                    if hasattr(self.prediction_store, "get"):
                        predictions = self.prediction_store.get("predictions", {})
                        risk_level = predictions.get("overall_risk_level", 0.5)
                    elif hasattr(self.prediction_store, "data"):
                        predictions = self.prediction_store.data.get("predictions", {})
                        risk_level = predictions.get("overall_risk_level", 0.5)
                    else:
                        risk_level = 0.5
                except:
                    risk_level = 0.5
            else:
                risk_level = 0.5

            if risk_level > 0.7:
                return "high"
            elif risk_level > 0.4:
                return "medium"
            else:
                return "low"
            
            
            
            
            
        
        def _parse_llm_text_response(self, response_text: str) -> Dict:
            """
            Parse LLM text response and extract structured data for Stage 2.
            Handles both JSON and natural language responses.
            Returns: Dict with target_objects, search_pattern, reasoning, etc.
            """
            print(f"🔍 TEXT PARSER: Parsing {len(response_text)} chars")
            print(f"🔍 TEXT PARSER INPUT: '{response_text[:200]}...'")
            
            if not response_text or not isinstance(response_text, str):
                print("❌ TEXT PARSER: Empty or invalid response text")
                print(f"🔍 TEXT PARSER DEBUG: response_text type = {type(response_text)}, value = {response_text}")
                return {}
            
            # Clean the response
            original_response = response_text
            response_clean = response_text.strip()
            print(f"🔍 TEXT PARSER: Cleaned from {len(original_response)} to {len(response_clean)} chars")
            print(f"🔍 TEXT PARSER CLEANED: '{response_clean[:150]}...'")
            
            # Default result
            result = {
                "target_objects": [],
                "search_pattern": "",
                "reasoning": response_clean[:100],  # Use first 100 chars as reasoning
                "expected_success": 0.7,
                "fallback_objects": ["sofa", "chair"],
                "parsed_from_text": True,
                "parser_debug": {}
            }
            
            try:
                # FIRST: Try to parse as JSON (if LLM actually returns JSON)
                import json
                import re
                
                print(f"🔍 TEXT PARSER PHASE 1: Looking for JSON pattern...")
                
                # Look for JSON pattern
                json_match = re.search(r'\{.*\}', response_clean, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    print(f"✅ TEXT PARSER: Found JSON pattern, {len(json_str)} chars")
                    print(f"🔍 TEXT PARSER JSON EXTRACT: '{json_str[:150]}...'")
                    
                    try:
                        # Try direct JSON parse
                        parsed_json = json.loads(json_str)
                        print(f"✅ TEXT PARSER: JSON parsed successfully")
                        print(f"🔍 TEXT PARSER JSON TYPE: {type(parsed_json)}")
                        
                        # Extract fields from JSON
                        if isinstance(parsed_json, dict):
                            result["target_objects"] = parsed_json.get("target_objects", [])
                            result["search_pattern"] = parsed_json.get("search_pattern", "")
                            result["reasoning"] = parsed_json.get("reasoning", response_clean[:100])
                            result["expected_success"] = parsed_json.get("expected_success", 0.7)
                            result["fallback_objects"] = parsed_json.get("fallback_objects", ["sofa", "chair"])
                            result["parsed_from_json"] = True
                            
                            print(f"✅ TEXT PARSER SUCCESS: Extracted {len(result['target_objects'])} targets from JSON")
                            print(f"🔍 TEXT PARSER JSON RESULT: targets={result['target_objects']}, pattern={result['search_pattern']}")
                            print(f"🔍 TEXT PARSER JSON FULL: {result}")
                            
                            return result
                        else:
                            print(f"⚠️  TEXT PARSER: Parsed JSON is not dict, type={type(parsed_json)}, value={parsed_json}")
                    except json.JSONDecodeError as e:
                        print(f"⚠️  TEXT PARSER: JSON decode failed: {e}")
                        print(f"🔍 TEXT PARSER DEBUG: JSON string was: '{json_str[:100]}...'")
                        print(f"🔍 TEXT PARSER: Falling back to text parsing...")
                else:
                    print(f"🔍 TEXT PARSER: No JSON pattern found in response")
                
                # SECOND: Text parsing for natural language responses
                print(f"🔍 TEXT PARSER PHASE 2: Starting text parsing...")
                
                response_lower = response_clean.lower()
                print(f"🔍 TEXT PARSER LOWERCASE: '{response_lower[:100]}...'")
                
                # Common objects to look for (from your logs)
                common_objects = [
                    "table", "cabinet", "nightstand", "sofa", "chair", 
                    "desk", "counter", "shelf", "drawer", "bed",
                    "couch", "lamp", "plant", "door", "window"
                ]
                
                print(f"🔍 TEXT PARSER: Looking for {len(common_objects)} common objects")
                
                target_objects = []
                found_objects = []
                for obj in common_objects:
                    if obj in response_lower:
                        found_objects.append(obj)
                        target_objects.append(obj)
                        if len(target_objects) >= 3:  # Limit to 3 objects
                            print(f"🔍 TEXT PARSER: Found object '{obj}', reached limit of 3")
                            break
                        else:
                            print(f"🔍 TEXT PARSER: Found object '{obj}'")
                
                result["target_objects"] = target_objects
                print(f"✅ TEXT PARSER: Found {len(target_objects)} target objects: {target_objects}")
                print(f"🔍 TEXT PARSER: All found objects: {found_objects}")
                
                # Extract search pattern from keywords
                print(f"🔍 TEXT PARSER: Analyzing search pattern keywords...")
                
                pattern_keywords = {
                    "spiral_search": ["spiral", "circle", "circular"],
                    "grid_search": ["grid", "systematic", "pattern", "organized"],
                    "proximity_based": ["proximity", "close", "near", "nearby", "closest"],
                    "room_exploration": ["room exploration", "full room", "entire room"],
                    "priority_based": ["priority", "important", "critical", "first"]
                }
                
                detected_pattern = "systematic_search"  # Default
                for pattern_name, keywords in pattern_keywords.items():
                    for keyword in keywords:
                        if keyword in response_lower:
                            detected_pattern = pattern_name
                            print(f"🔍 TEXT PARSER: Found pattern keyword '{keyword}' -> {pattern_name}")
                            break
                    if detected_pattern != "systematic_search":
                        break
                
                result["search_pattern"] = detected_pattern
                print(f"✅ TEXT PARSER: Detected search pattern: {detected_pattern}")
                
                # Extract expected success (look for numbers/percentages)
                print(f"🔍 TEXT PARSER: Looking for success rate...")
                
                success_match = re.search(r'(\d+(\.\d+)?)%', response_clean)
                if success_match:
                    print(f"🔍 TEXT PARSER: Found percentage match: {success_match.group()}")
                    try:
                        percent = float(success_match.group(1))
                        if 0 <= percent <= 100:
                            result["expected_success"] = percent / 100.0
                            print(f"✅ TEXT PARSER: Converted {percent}% to {result['expected_success']}")
                        else:
                            print(f"⚠️  TEXT PARSER: Percentage {percent}% out of range")
                    except Exception as e:
                        print(f"❌ TEXT PARSER: Error parsing percentage: {e}")
                else:
                    print(f"🔍 TEXT PARSER: No percentage found, looking for decimal...")
                    # Look for decimal success
                    decimal_match = re.search(r'(\d+\.\d+) out of 1', response_lower)
                    if decimal_match:
                        print(f"🔍 TEXT PARSER: Found decimal match: {decimal_match.group()}")
                        try:
                            result["expected_success"] = float(decimal_match.group(1))
                            print(f"✅ TEXT PARSER: Set expected_success = {result['expected_success']}")
                        except Exception as e:
                            print(f"❌ TEXT PARSER: Error parsing decimal: {e}")
                    else:
                        print(f"🔍 TEXT PARSER: No success rate found, using default 0.7")
                
                # Extract fallback objects (objects mentioned after "or", "alternatively", etc.)
                print(f"🔍 TEXT PARSER: Looking for fallback objects...")
                
                fallback_keywords = ["alternatively", "otherwise", "or", "fallback", "backup", "second choice"]
                fallback_section = ""
                
                for keyword in fallback_keywords:
                    if keyword in response_lower:
                        print(f"🔍 TEXT PARSER: Found fallback keyword '{keyword}'")
                        # Try to extract text after keyword
                        pattern = rf'{keyword}[:\s]+([^.]+)'
                        match = re.search(pattern, response_lower)
                        if match:
                            fallback_section = match.group(1)
                            print(f"🔍 TEXT PARSER: Fallback section: '{fallback_section}'")
                            break
                
                if fallback_section:
                    fallback_objects = []
                    for obj in common_objects:
                        if obj in fallback_section:
                            fallback_objects.append(obj)
                            print(f"🔍 TEXT PARSER: Found fallback object '{obj}'")
                    
                    if fallback_objects:
                        result["fallback_objects"] = fallback_objects
                        print(f"✅ TEXT PARSER: Found {len(fallback_objects)} fallback objects: {fallback_objects}")
                    else:
                        print(f"🔍 TEXT PARSER: No objects found in fallback section")
                else:
                    print(f"🔍 TEXT PARSER: No fallback keywords found, using defaults")
                
                # Store debug info
                result["parser_debug"] = {
                    "input_length": len(original_response),
                    "cleaned_length": len(response_clean),
                    "found_objects": found_objects,
                    "detected_pattern": detected_pattern,
                    "had_json_pattern": bool(json_match),
                    "success_rate_source": "percentage" if success_match else "decimal" if decimal_match else "default"
                }
                
                print(f"✅ TEXT PARSER FINAL RESULT:")
                print(f"   Targets: {result['target_objects']}")
                print(f"   Pattern: {result['search_pattern']}")
                print(f"   Success: {result['expected_success']}")
                print(f"   Fallbacks: {result['fallback_objects']}")
                print(f"   Parser debug: {result['parser_debug']}")
                
            except Exception as e:
                print(f"❌ TEXT PARSER ERROR: {e}")
                import traceback
                print(f"❌ TEXT PARSER TRACEBACK: {traceback.format_exc()}")
                # Return default result if parsing fails
            
            return result       
            
            
        def shutdown(self):
            """Cleanup resources"""
            self.is_initialized = False
            print("Tree of Thoughts shutdown")




def main():
    """
    Single main function for standalone Tree of Thoughts testing WITH REAL STORES
    """
    import time
    import json
    import argparse
    import os

    print("=" * 60)
    print("🌳 TREE OF THOUGHTS - REAL STORE TEST")
    print("=" * 60)

    # 🎯 ADD IMPORTS FOR REAL STORES
    from src.stores.central_map_store import CentralMapStore
    from src.stores.task_store import TaskStore
    from src.stores.prediction_store import PredictionStore
    from src.stores.user_command_store import UserCommandStore

    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mission", default="find my keys in living room", help="Mission command"
    )
    parser.add_argument(
        "--output", default="tree_of_thoughts_output.json", help="Output file"
    )
    parser.add_argument("--save", action="store_true", help="Save to file")
    args = parser.parse_args()

    # ========== 1. CREATE REAL STORES ==========
    print("\n🔧 1. Creating REAL data stores...")

    # 🎯 REAL STORES
    real_map_store = CentralMapStore()
    real_task_store = TaskStore()
    real_prediction_store = PredictionStore()
    real_user_command_store = UserCommandStore()

    # 🎯 POPULATE REAL STORES WITH TEST DATA
    objects_list = ["table", "sofa", "chair", "cabinet", "nightstand"]

    # Add test data to map store
    if hasattr(real_map_store, "semantic_objects"):
        for obj in objects_list:
            real_map_store.semantic_objects[obj] = {
                "name": obj,
                "position": [1.0, 1.0, 0],
            }

    # Set current room in map store
    if hasattr(real_map_store, "current_room"):
        real_map_store.current_room = "living_room"

    # Set current pose in map store
    if hasattr(real_map_store, "current_pose"):
        # Create a mock CameraPose
        real_map_store.current_pose = type(
            "MockPose",
            (),
            {
                "position": [1.0, 2.0, 0],
                "rotation_quat": [0, 0, 0, 1],
                "transform_matrix": [
                    [1, 0, 0, 1],
                    [0, 1, 0, 2],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
                "timestamp": time.time(),
                "frame_id": 0,
                "tracking_quality": 1.0,
            },
        )()

    # Set mission in user command store
    if hasattr(real_user_command_store, "set_active_command"):
        real_user_command_store.set_active_command(
            {
                "user_input": args.mission,
                "parsed_intent": {"intent_type": "find_object"},
                "priority": "normal",
            }
        )

    print(f"   ✅ Created REAL stores with mission='{args.mission}'")

    # ========== 2. INITIALIZE TREE OF THOUGHTS ==========
    print("\n🔧 2. Initializing Tree of Thoughts with REAL stores...")

    # Initialize with REAL stores
    tot = TreeOfThoughtsIntegration(
        prediction_store=real_prediction_store,
        map_store=real_map_store,
        user_command_store=real_user_command_store,
        task_store=real_task_store,
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    print(f"   ✅ LLM available: {tot.llm_service is not None}")

    # ========== 3. GENERATE COMPREHENSIVE PLAN ==========
    print(f"\n🚀 3. Generating plan for: '{args.mission}'")

    reasoning_cycle_id = f"test_{int(time.time())}"
    start_time = time.time()

    try:
        # 🎯 ADD TIMING DEBUG
        print(f"⏱️  Starting plan generation at: {start_time}")

        plan_generated = tot.generate_comprehensive_plan(reasoning_cycle_id)

        elapsed_time = time.time() - start_time
        print(f"⏱️  Plan generation took: {elapsed_time:.2f} seconds")

    except Exception as e:
        print(f"❌ Plan generation failed with error: {e}")
        import traceback

        traceback.print_exc()
        print("\nTrying fallback generation...")
        plan_generated = False

    # To this:
    if not plan_generated:
        print("⚠️  Plan generation had errors, but saving available data...")

    print(f"   ✅ Plan generated with ID: {reasoning_cycle_id}")

    # ========== 4. CHECK IF FILE WAS SAVED ==========
    print(f"\n📋 4. Checking if data was saved to disk...")

    # Check task store intermediate reasoning
    if hasattr(real_task_store, "intermediate_reasoning"):
        if reasoning_cycle_id in real_task_store.intermediate_reasoning:
            print(f"✅ Plan found in task store intermediate_reasoning")
            tree_data = (
                real_task_store.intermediate_reasoning[reasoning_cycle_id]
                .get("component_reasoning", {})
                .get("tree_of_thoughts")
            )
            if tree_data:
                print(
                    f"✅ Tree of Thoughts data: {tree_data.get('reasoning_cycle_id', 'N/A')}"
                )
            else:
                print(f"❌ No tree_of_thoughts data in intermediate_reasoning")
        else:
            print(f"❌ Plan NOT found in task store intermediate_reasoning")
            print(
                f"🔍 Available keys: {list(real_task_store.intermediate_reasoning.keys())}"
            )

    # ========== 5. MANUALLY SAVE FINAL FUSED DECISIONS ==========
    print(f"\n💾 5. Manually saving fused decisions to disk...")

    # Create output directory
    output_dir = "experiments/results/tree_of_thoughts"
    os.makedirs(output_dir, exist_ok=True)

    # Save timestamped file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{output_dir}/tree_of_thoughts_{timestamp}.json"

    # Collect all data
    final_data = {
        "reasoning_cycle_id": reasoning_cycle_id,
        "timestamp": time.time(),
        "generation_time_seconds": elapsed_time if "elapsed_time" in locals() else 0,
        "mission": args.mission,
        "stores_used": {
            "map_store": real_map_store is not None,
            "prediction_store": real_prediction_store is not None,
            "task_store": real_task_store is not None,
            "user_command_store": real_user_command_store is not None,
        },
    }

    # Add plan data if available
    if "stored_plan" in locals() and stored_plan:
        final_data["plan"] = stored_plan

    # Add task store data
    if hasattr(real_task_store, "intermediate_reasoning"):
        final_data["task_store_data"] = real_task_store.intermediate_reasoning.get(
            reasoning_cycle_id, {}
        )

    # Save to file
    try:
        with open(output_file, "w") as f:
            json.dump(final_data, f, indent=2, default=str)

        print(f"✅ FINAL FUSED DECISIONS saved to: {output_file}")

        # Verify file was saved
        if os.path.exists(output_file):
            file_size = os.path.getsize(output_file)
            print(f"✅ VERIFIED: File exists, size: {file_size} bytes")

            # Show preview
            with open(output_file, "r") as f:
                preview = f.read()[:300]
                print(f"📄 PREVIEW (first 300 chars):\n{preview}...")
        else:
            print(f"❌ VERIFICATION FAILED: File not created")

    except Exception as e:
        print(f"❌ Manual save failed: {e}")
        import traceback

        traceback.print_exc()

    # ========== 6. CLEANUP WITH SAVE ==========
    print(f"\n🧹 6. Cleaning up with final save...")

    tot.shutdown()

    print("\n" + "=" * 60)
    print("🏁 TEST COMPLETE - DATA SAVED TO DISK")
    print("=" * 60)

    # 🎯 FINAL CHECK - LIST ALL SAVED FILES
    print(f"\n📁 ALL SAVED FILES in {output_dir}:")
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            filepath = os.path.join(output_dir, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  📄 {file} ({size} bytes)")


if __name__ == "__main__":
    main()




# owl text queries and action pipeline logic adaptation
# TOT uses LLMs for strategy, but not for open-vocabulary goal interpretation.

