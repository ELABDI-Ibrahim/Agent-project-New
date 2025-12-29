"""
Shared context for inter-agent communication and data storage.
"""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional
from collections import defaultdict

from .models import AgentMessage
from .enums import AnalysisState


class SharedContext:
    """
    Thread-safe shared context for storing data and managing communication between agents.
    
    This class provides:
    - Thread-safe data storage and retrieval
    - Message logging functionality
    - Analysis state management
    """
    
    def __init__(self):
        """Initialize the shared context with thread-safe storage."""
        self._data_lock = threading.RLock()
        self._message_lock = threading.RLock()
        self._state_lock = threading.RLock()
        
        # Data storage
        self._data_store: Dict[str, Any] = {}
        self._data_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Message logging
        self._message_history: List[AgentMessage] = []
        self._messages_by_agent: Dict[str, List[AgentMessage]] = defaultdict(list)
        
        # Analysis state management
        self._analysis_state = AnalysisState.INITIALIZED
        self._state_history: List[tuple[AnalysisState, datetime]] = [
            (AnalysisState.INITIALIZED, datetime.now())
        ]
    
    def store_data(self, key: str, data: Any, agent_id: str) -> None:
        """
        Store data in the shared context with metadata.
        
        Args:
            key: Unique identifier for the data
            data: The data to store
            agent_id: ID of the agent storing the data
        """
        with self._data_lock:
            self._data_store[key] = data
            self._data_metadata[key] = {
                "agent_id": agent_id,
                "timestamp": datetime.now(),
                "data_type": type(data).__name__
            }
    
    def retrieve_data(self, key: str) -> Any:
        """
        Retrieve data from the shared context.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            The stored data, or None if key doesn't exist
        """
        with self._data_lock:
            return self._data_store.get(key)
    
    def get_data_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about stored data.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            Metadata dictionary or None if key doesn't exist
        """
        with self._data_lock:
            return self._data_metadata.get(key)
    
    def list_data_keys(self) -> List[str]:
        """
        Get a list of all stored data keys.
        
        Returns:
            List of data keys
        """
        with self._data_lock:
            return list(self._data_store.keys())
    
    def remove_data(self, key: str) -> bool:
        """
        Remove data from the shared context.
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            True if data was removed, False if key didn't exist
        """
        with self._data_lock:
            if key in self._data_store:
                del self._data_store[key]
                del self._data_metadata[key]
                return True
            return False
    
    def log_message(self, message: AgentMessage) -> None:
        """
        Log a message in the shared context.
        
        Args:
            message: The message to log
        """
        with self._message_lock:
            self._message_history.append(message)
            self._messages_by_agent[message.sender].append(message)
            self._messages_by_agent[message.recipient].append(message)
    
    def get_message_history(self, agent_filter: Optional[str] = None) -> List[AgentMessage]:
        """
        Get message history, optionally filtered by agent.
        
        Args:
            agent_filter: Optional agent ID to filter messages
            
        Returns:
            List of messages
        """
        with self._message_lock:
            if agent_filter is None:
                return self._message_history.copy()
            else:
                return self._messages_by_agent[agent_filter].copy()
    
    def get_recent_messages(self, count: int = 10, agent_filter: Optional[str] = None) -> List[AgentMessage]:
        """
        Get the most recent messages.
        
        Args:
            count: Number of recent messages to retrieve
            agent_filter: Optional agent ID to filter messages
            
        Returns:
            List of recent messages
        """
        messages = self.get_message_history(agent_filter)
        return messages[-count:] if len(messages) > count else messages
    
    def clear_message_history(self) -> None:
        """Clear all message history."""
        with self._message_lock:
            self._message_history.clear()
            self._messages_by_agent.clear()
    
    def update_analysis_state(self, state: AnalysisState) -> None:
        """
        Update the current analysis state.
        
        Args:
            state: New analysis state
        """
        with self._state_lock:
            if state != self._analysis_state:
                self._analysis_state = state
                self._state_history.append((state, datetime.now()))
    
    def get_analysis_state(self) -> AnalysisState:
        """
        Get the current analysis state.
        
        Returns:
            Current analysis state
        """
        with self._state_lock:
            return self._analysis_state
    
    def get_state_history(self) -> List[tuple[AnalysisState, datetime]]:
        """
        Get the history of analysis state changes.
        
        Returns:
            List of (state, timestamp) tuples
        """
        with self._state_lock:
            return self._state_history.copy()
    
    def get_state_duration(self, state: AnalysisState) -> Optional[float]:
        """
        Get the total duration spent in a particular state.
        
        Args:
            state: The analysis state to measure
            
        Returns:
            Duration in seconds, or None if state was never entered
        """
        with self._state_lock:
            total_duration = 0.0
            current_time = datetime.now()
            
            for i, (hist_state, timestamp) in enumerate(self._state_history):
                if hist_state == state:
                    # Find the end time (next state change or current time)
                    if i + 1 < len(self._state_history):
                        end_time = self._state_history[i + 1][1]
                    else:
                        end_time = current_time
                    
                    duration = (end_time - timestamp).total_seconds()
                    total_duration += duration
            
            return total_duration if total_duration > 0 else None
    
    def clear_all_data(self) -> None:
        """Clear all stored data, messages, and reset state."""
        with self._data_lock, self._message_lock, self._state_lock:
            self._data_store.clear()
            self._data_metadata.clear()
            self._message_history.clear()
            self._messages_by_agent.clear()
            self._analysis_state = AnalysisState.INITIALIZED
            self._state_history = [(AnalysisState.INITIALIZED, datetime.now())]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current shared context state.
        
        Returns:
            Dictionary containing summary information
        """
        with self._data_lock, self._message_lock, self._state_lock:
            return {
                "data_items": len(self._data_store),
                "data_keys": list(self._data_store.keys()),
                "total_messages": len(self._message_history),
                "agents_with_messages": list(self._messages_by_agent.keys()),
                "current_state": self._analysis_state.name,
                "state_changes": len(self._state_history)
            }