

"""
User Command Store - Stores user intent, mission objectives, and command history
"""
import json
import time
import logging
from typing import Dict, List, Optional, Any
from enum import Enum

class CommandStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class CommandPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"

class UserCommandStore:
    """
    Stores and manages user commands for robotic missions
    """
    
    def __init__(self, storage_path: str = "user_commands.json"):
        self.storage_path = storage_path
        self.commands = {}
        self._load_commands()
        
    def _load_commands(self):
        """Load commands from disk"""
        try:
            with open(self.storage_path, 'r') as f:
                self.commands = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.commands = {}
            logging.info("No existing command store found, starting fresh")
    
    def _save_commands(self):
        """Save commands to disk"""
        with open(self.storage_path, 'w') as f:
            json.dump(self.commands, f, indent=2)
    
    def create_command(self, 
                      user_input: str,
                      command_type: str = "navigation",
                      priority: CommandPriority = CommandPriority.NORMAL,
                      metadata: Dict = None) -> str:
        """
        Create a new user command with automatic parsing
        """
        command_id = f"cmd_{int(time.time()*1000)}"
        
        # Parse command intent
        parsed_intent = self._parse_user_intent(user_input)
        
        command_data = {
            'command_id': command_id,
            'user_input': user_input,
            'parsed_intent': parsed_intent,
            'command_type': command_type,
            'priority': priority.value,
            'status': CommandStatus.PENDING.value,
            'created_at': time.time(),
            'updated_at': time.time(),
            'metadata': metadata or {},
            'execution_history': [],
            'reasoning_cycles_used': []
        }
        
        self.commands[command_id] = command_data
        self._save_commands()
        
        logging.info(f"Created command {command_id}: {user_input}")
        return command_id
    
    def _parse_user_intent(self, user_input: str) -> Dict:
        """
        Parse user input into structured intent
        """
        input_lower = user_input.lower()
        
        # Basic intent parsing (could be enhanced with NLP)
        if any(word in input_lower for word in ['find', 'locate', 'search for']):
            intent_type = "find_object"
            target_object = self._extract_object_name(user_input)
        elif any(word in input_lower for word in ['go to', 'navigate to', 'move to']):
            intent_type = "navigate_location" 
            target_location = self._extract_location_name(user_input)
        elif any(word in input_lower for word in ['explore', 'scan', 'map']):
            intent_type = "explore_room"
            target_room = self._extract_room_name(user_input)
        else:
            intent_type = "general_task"
            target_object = "unknown"
        
        return {
            'intent_type': intent_type,
            'target_object': target_object if 'target_object' in locals() else None,
            'target_location': target_location if 'target_location' in locals() else None,
            'target_room': target_room if 'target_room' in locals() else None,
            'urgency': self._assess_urgency(user_input),
            'complexity': self._assess_complexity(user_input)
        }
    
    def _extract_object_name(self, user_input: str) -> str:
        """Extract object name from find commands"""
        # Simple extraction - enhance with proper NLP
        words = user_input.lower().split()
        find_index = next((i for i, word in enumerate(words) 
                          if word in ['find', 'locate', 'search']), -1)
        if find_index != -1 and find_index + 1 < len(words):
            return words[find_index + 1]
        return "unknown_object"
    
    def _extract_location_name(self, user_input: str) -> str:
        """Extract location name from navigation commands"""
        words = user_input.lower().split()
        go_index = next((i for i, word in enumerate(words) 
                        if word in ['go', 'navigate', 'move']), -1)
        if go_index != -1 and go_index + 2 < len(words) and words[go_index + 1] == 'to':
            return words[go_index + 2]
        return "unknown_location"
    
    def _extract_room_name(self, user_input: str) -> str:
        """Extract room name from exploration commands"""
        room_keywords = ['room', 'area', 'space', 'kitchen', 'living', 'bedroom', 'office']
        words = user_input.lower().split()
        for word in words:
            if word in room_keywords:
                return word
        return "unknown_room"
    
    def _assess_urgency(self, user_input: str) -> str:
        """Assess command urgency"""
        urgent_words = ['now', 'urgent', 'quickly', 'immediately', 'asap']
        if any(word in user_input.lower() for word in urgent_words):
            return "high"
        return "normal"
    
    def _assess_complexity(self, user_input: str) -> str:
        """Assess command complexity"""
        complex_indicators = ['and', 'then', 'after', 'while', 'multiple']
        if any(indicator in user_input.lower() for indicator in complex_indicators):
            return "complex"
        return "simple"
    
    def get_active_command(self) -> Optional[Dict]:
        """Get currently active command"""
        for cmd_id, cmd_data in self.commands.items():
            if cmd_data['status'] == CommandStatus.ACTIVE.value:
                return cmd_data
        return None
    
    def update_command_status(self, command_id: str, status: CommandStatus, 
                            reasoning_cycle_id: str = None) -> bool:
        """Update command status and link to reasoning cycle"""
        if command_id not in self.commands:
            return False
        
        self.commands[command_id]['status'] = status.value
        self.commands[command_id]['updated_at'] = time.time()
        
        if reasoning_cycle_id:
            self.commands[command_id]['reasoning_cycles_used'].append(reasoning_cycle_id)
        
        self._save_commands()
        return True
    
    def add_execution_attempt(self, command_id: str, 
                            action_plan: Dict, 
                            success: bool) -> bool:
        """Record execution attempt for a command"""
        if command_id not in self.commands:
            return False
        
        attempt = {
            'timestamp': time.time(),
            'action_plan': action_plan,
            'success': success,
            'reasoning_cycle_id': action_plan.get('reasoning_cycle_id')
        }
        
        self.commands[command_id]['execution_history'].append(attempt)
        self._save_commands()
        return True
    
    def get_command_history(self, limit: int = 10) -> List[Dict]:
        """Get recent command history"""
        sorted_commands = sorted(self.commands.values(), 
                               key=lambda x: x['created_at'], 
                               reverse=True)
        return sorted_commands[:limit]
    
    def get_commands_by_status(self, status: CommandStatus) -> List[Dict]:
        """Get commands by status"""
        return [cmd for cmd in self.commands.values() 
                if cmd['status'] == status.value]
        
        
        