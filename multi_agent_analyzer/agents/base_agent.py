"""
Base agent class for the multi-agent data analyzer system.

This module provides the abstract BaseAgent class that defines the interface
and common functionality for all specialized agents in the system.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List

from ..core.models import AgentMessage, ValidationResult
from ..core.enums import MessageType, Priority
from ..core.shared_context import SharedContext
from ..core.llm_coordinator import LLMCoordinator
from ..core.message_protocol import MessageFactory


class AgentResult:
    """Result returned by agent processing."""
    
    def __init__(self, success: bool, data: Optional[Dict[str, Any]] = None, 
                 errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        """
        Initialize agent result.
        
        Args:
            success: Whether the processing was successful
            data: Result data from processing
            errors: List of errors encountered
            warnings: List of warnings generated
        """
        self.success = success
        self.data = data or {}
        self.errors = errors or []
        self.warnings = warnings or []
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat()
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent data analyzer system.
    
    This class provides:
    - Common agent interface and lifecycle management
    - Message sending and receiving capabilities
    - LLM assistance request functionality
    - Error handling and logging infrastructure
    - Integration with shared context and coordinator
    """
    
    def __init__(self, name: str, llm_coordinator: LLMCoordinator, shared_context: SharedContext):
        """
        Initialize the base agent.
        
        Args:
            name: Unique name/identifier for this agent
            llm_coordinator: LLM coordinator for cognitive assistance
            shared_context: Shared context for inter-agent communication
        """
        self.name = name
        self.llm_coordinator = llm_coordinator
        self.shared_context = shared_context
        
        # Set up logging
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Agent state
        self._is_initialized = False
        self._message_handlers: Dict[MessageType, callable] = {}
        self._processing_errors: List[str] = []
        
        # Initialize message handlers
        self._setup_message_handlers()
        
        self.logger.info(f"Agent {name} initialized")
    
    @property
    def is_initialized(self) -> bool:
        """Check if agent is properly initialized."""
        return self._is_initialized
    
    def initialize(self) -> bool:
        """
        Initialize the agent and perform any setup required.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self._initialize_agent()
            self._is_initialized = True
            self.logger.info(f"Agent {self.name} initialization completed")
            return True
        except Exception as e:
            self.logger.error(f"Agent {self.name} initialization failed: {str(e)}")
            self._processing_errors.append(f"Initialization failed: {str(e)}")
            return False
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process input data and return results.
        
        This is the main processing method that each agent must implement
        according to its specific responsibilities.
        
        Args:
            input_data: Input data to process
            
        Returns:
            AgentResult: Processing results
        """
        pass
    
    @abstractmethod
    def _initialize_agent(self) -> None:
        """
        Agent-specific initialization logic.
        
        Subclasses should implement this method to perform any
        agent-specific setup required.
        """
        pass
    
    def send_message(self, recipient: str, message: AgentMessage) -> bool:
        """
        Send a message to another agent.
        
        Args:
            recipient: ID of the recipient agent
            message: Message to send
            
        Returns:
            bool: True if message was sent successfully, False otherwise
        """
        try:
            # Ensure sender is set correctly
            message.sender = self.name
            message.recipient = recipient
            
            # Log the message in shared context
            self.shared_context.log_message(message)
            
            # Store message for recipient to retrieve
            message_key = f"message_{recipient}_{message.correlation_id}"
            self.shared_context.store_data(message_key, message, self.name)
            
            self.logger.debug(f"Sent {message.message_type.name} message to {recipient}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send message to {recipient}: {str(e)}")
            self._processing_errors.append(f"Message send failed: {str(e)}")
            return False
    
    def receive_message(self, message: AgentMessage) -> bool:
        """
        Receive and process a message from another agent.
        
        Args:
            message: Message received
            
        Returns:
            bool: True if message was processed successfully, False otherwise
        """
        try:
            # Log the received message
            self.shared_context.log_message(message)
            
            # Get appropriate handler for message type
            handler = self._message_handlers.get(message.message_type)
            
            if handler:
                handler(message)
                self.logger.debug(f"Processed {message.message_type.name} message from {message.sender}")
                return True
            else:
                self.logger.warning(f"No handler for message type: {message.message_type.name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to process message from {message.sender}: {str(e)}")
            self._processing_errors.append(f"Message processing failed: {str(e)}")
            return False
    
    def request_llm_assistance(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Request assistance from the LLM coordinator.
        
        Args:
            query: Query or question for the LLM
            context: Additional context information
            
        Returns:
            str: Response from the LLM coordinator
        """
        try:
            # Prepare context with agent information
            full_context = {
                "agent_name": self.name,
                "agent_type": self.__class__.__name__,
                "timestamp": datetime.now().isoformat()
            }
            
            if context:
                full_context.update(context)
            
            # Request assistance from LLM coordinator
            response = self.llm_coordinator.resolve_ambiguity(query, full_context)
            
            self.logger.debug(f"Received LLM assistance for query: {query[:50]}...")
            return response
            
        except Exception as e:
            self.logger.error(f"LLM assistance request failed: {str(e)}")
            self._processing_errors.append(f"LLM assistance failed: {str(e)}")
            return f"LLM assistance unavailable: {str(e)}"
    
    def validate_input(self, input_data: Dict[str, Any], required_keys: List[str]) -> ValidationResult:
        """
        Validate input data against required keys and basic structure.
        
        Args:
            input_data: Input data to validate
            required_keys: List of required keys in input data
            
        Returns:
            ValidationResult: Validation result
        """
        errors = []
        warnings = []
        
        # Check if input_data is a dictionary
        if not isinstance(input_data, dict):
            errors.append("Input data must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Check for required keys
        missing_keys = set(required_keys) - set(input_data.keys())
        if missing_keys:
            errors.append(f"Missing required keys: {missing_keys}")
        
        # Check for empty values in required keys
        for key in required_keys:
            if key in input_data and input_data[key] is None:
                warnings.append(f"Required key '{key}' has None value")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def get_processing_errors(self) -> List[str]:
        """
        Get list of processing errors encountered by this agent.
        
        Returns:
            List[str]: List of error messages
        """
        return self._processing_errors.copy()
    
    def clear_processing_errors(self) -> None:
        """Clear the list of processing errors."""
        self._processing_errors.clear()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "initialized": self._is_initialized,
            "error_count": len(self._processing_errors),
            "recent_errors": self._processing_errors[-5:] if self._processing_errors else []
        }
    
    def _setup_message_handlers(self) -> None:
        """Set up default message handlers."""
        self._message_handlers = {
            MessageType.REQUEST: self._handle_request_message,
            MessageType.RESPONSE: self._handle_response_message,
            MessageType.NOTIFICATION: self._handle_notification_message,
            MessageType.ERROR: self._handle_error_message
        }
    
    def _handle_request_message(self, message: AgentMessage) -> None:
        """
        Handle incoming request messages.
        
        Args:
            message: Request message to handle
        """
        try:
            # Default implementation - subclasses can override
            self.logger.info(f"Received request from {message.sender}: {message.content}")
            
            # Send acknowledgment response
            response = MessageFactory.create_response(
                original_message=message,
                content={"status": "received", "agent": self.name}
            )
            self.send_message(message.sender, response)
            
        except Exception as e:
            self.logger.error(f"Failed to handle request message: {str(e)}")
            # Send error response
            error_msg = MessageFactory.create_error(
                sender=self.name,
                recipient=message.sender,
                error_message=f"Failed to process request: {str(e)}",
                original_correlation_id=message.correlation_id
            )
            self.send_message(message.sender, error_msg)
    
    def _handle_response_message(self, message: AgentMessage) -> None:
        """
        Handle incoming response messages.
        
        Args:
            message: Response message to handle
        """
        self.logger.info(f"Received response from {message.sender}")
        # Default implementation - subclasses can override for specific handling
    
    def _handle_notification_message(self, message: AgentMessage) -> None:
        """
        Handle incoming notification messages.
        
        Args:
            message: Notification message to handle
        """
        self.logger.info(f"Received notification from {message.sender}: {message.content}")
        # Default implementation - subclasses can override for specific handling
    
    def _handle_error_message(self, message: AgentMessage) -> None:
        """
        Handle incoming error messages.
        
        Args:
            message: Error message to handle
        """
        error_info = message.content.get("error_message", "Unknown error")
        self.logger.error(f"Received error from {message.sender}: {error_info}")
        self._processing_errors.append(f"Error from {message.sender}: {error_info}")
    
    def _log_processing_start(self, operation: str, input_data: Dict[str, Any]) -> None:
        """
        Log the start of a processing operation.
        
        Args:
            operation: Name of the operation
            input_data: Input data being processed
        """
        data_summary = {k: type(v).__name__ for k, v in input_data.items()}
        self.logger.info(f"Starting {operation} with input: {data_summary}")
    
    def _log_processing_end(self, operation: str, result: AgentResult) -> None:
        """
        Log the end of a processing operation.
        
        Args:
            operation: Name of the operation
            result: Result of the operation
        """
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(f"Completed {operation} - Status: {status}, "
                        f"Errors: {len(result.errors)}, Warnings: {len(result.warnings)}")
    
    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(name='{self.name}', initialized={self._is_initialized})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the agent."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"initialized={self._is_initialized}, "
                f"errors={len(self._processing_errors)})")