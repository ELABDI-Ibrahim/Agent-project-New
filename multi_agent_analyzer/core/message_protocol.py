"""
Message protocol and validation for inter-agent communication.
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from dataclasses import asdict

from .models import AgentMessage
from .enums import MessageType, Priority


class MessageValidator:
    """Validates agent messages against the protocol schema."""
    
    REQUIRED_FIELDS = {
        'sender', 'recipient', 'message_type', 'content', 
        'timestamp', 'correlation_id', 'priority'
    }
    
    VALID_MESSAGE_TYPES = {mt.name for mt in MessageType}
    VALID_PRIORITIES = {p.name for p in Priority}
    
    @classmethod
    def validate_message(cls, message: AgentMessage) -> List[str]:
        """
        Validate an AgentMessage against the protocol schema.
        
        Args:
            message: The message to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        message_dict = message.to_dict()
        missing_fields = cls.REQUIRED_FIELDS - set(message_dict.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Validate sender and recipient
        if not message.sender or not isinstance(message.sender, str):
            errors.append("Sender must be a non-empty string")
        
        if not message.recipient or not isinstance(message.recipient, str):
            errors.append("Recipient must be a non-empty string")
        
        # Validate message type
        if message.message_type.name not in cls.VALID_MESSAGE_TYPES:
            errors.append(f"Invalid message type: {message.message_type.name}")
        
        # Validate priority
        if message.priority.name not in cls.VALID_PRIORITIES:
            errors.append(f"Invalid priority: {message.priority.name}")
        
        # Validate content
        if not isinstance(message.content, dict):
            errors.append("Content must be a dictionary")
        
        # Validate timestamp
        if not isinstance(message.timestamp, datetime):
            errors.append("Timestamp must be a datetime object")
        
        # Validate correlation_id
        if not isinstance(message.correlation_id, str):
            errors.append("Correlation ID must be a string")
        
        return errors
    
    @classmethod
    def validate_json_message(cls, json_data: Dict[str, Any]) -> List[str]:
        """
        Validate a JSON message dictionary against the protocol schema.
        
        Args:
            json_data: Dictionary containing message data
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        missing_fields = cls.REQUIRED_FIELDS - set(json_data.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            return errors  # Can't continue validation without required fields
        
        # Validate field types and values
        if not isinstance(json_data.get('sender'), str) or not json_data.get('sender'):
            errors.append("Sender must be a non-empty string")
        
        if not isinstance(json_data.get('recipient'), str) or not json_data.get('recipient'):
            errors.append("Recipient must be a non-empty string")
        
        if json_data.get('message_type') not in cls.VALID_MESSAGE_TYPES:
            errors.append(f"Invalid message type: {json_data.get('message_type')}")
        
        if json_data.get('priority') not in cls.VALID_PRIORITIES:
            errors.append(f"Invalid priority: {json_data.get('priority')}")
        
        if not isinstance(json_data.get('content'), dict):
            errors.append("Content must be a dictionary")
        
        if not isinstance(json_data.get('correlation_id'), str):
            errors.append("Correlation ID must be a string")
        
        # Validate timestamp format
        try:
            datetime.fromisoformat(json_data.get('timestamp', ''))
        except (ValueError, TypeError):
            errors.append("Timestamp must be a valid ISO format datetime string")
        
        return errors


class MessageRouter:
    """Routes messages between agents and manages message flow."""
    
    def __init__(self):
        """Initialize the message router."""
        self._registered_agents: Set[str] = set()
        self._message_handlers: Dict[str, callable] = {}
        self._routing_rules: Dict[str, List[str]] = {}
        
    def register_agent(self, agent_id: str, message_handler: callable) -> None:
        """
        Register an agent with the router.
        
        Args:
            agent_id: Unique identifier for the agent
            message_handler: Function to handle incoming messages
        """
        self._registered_agents.add(agent_id)
        self._message_handlers[agent_id] = message_handler
    
    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from the router.
        
        Args:
            agent_id: Unique identifier for the agent
        """
        self._registered_agents.discard(agent_id)
        self._message_handlers.pop(agent_id, None)
        self._routing_rules.pop(agent_id, None)
    
    def add_routing_rule(self, sender: str, allowed_recipients: List[str]) -> None:
        """
        Add a routing rule for an agent.
        
        Args:
            sender: Agent ID that is sending messages
            allowed_recipients: List of agent IDs that can receive messages from sender
        """
        self._routing_rules[sender] = allowed_recipients
    
    def is_route_allowed(self, sender: str, recipient: str) -> bool:
        """
        Check if a message route is allowed.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            
        Returns:
            True if route is allowed, False otherwise
        """
        # If no routing rules defined, allow all routes between registered agents
        if not self._routing_rules:
            return sender in self._registered_agents and recipient in self._registered_agents
        
        # Check specific routing rules
        allowed_recipients = self._routing_rules.get(sender, [])
        return recipient in allowed_recipients
    
    def route_message(self, message: AgentMessage) -> bool:
        """
        Route a message to its recipient.
        
        Args:
            message: The message to route
            
        Returns:
            True if message was successfully routed, False otherwise
        """
        # Validate message
        validation_errors = MessageValidator.validate_message(message)
        if validation_errors:
            raise ValueError(f"Message validation failed: {validation_errors}")
        
        # Check if route is allowed
        if not self.is_route_allowed(message.sender, message.recipient):
            raise ValueError(f"Route not allowed: {message.sender} -> {message.recipient}")
        
        # Check if recipient is registered
        if message.recipient not in self._registered_agents:
            raise ValueError(f"Recipient not registered: {message.recipient}")
        
        # Route the message
        handler = self._message_handlers.get(message.recipient)
        if handler:
            try:
                handler(message)
                return True
            except Exception as e:
                raise RuntimeError(f"Message handler failed for {message.recipient}: {e}")
        
        return False
    
    def broadcast_message(self, sender: str, message_content: Dict[str, Any], 
                         message_type: MessageType = MessageType.NOTIFICATION,
                         priority: Priority = Priority.MEDIUM) -> List[str]:
        """
        Broadcast a message to all registered agents except the sender.
        
        Args:
            sender: ID of the sending agent
            message_content: Content of the message
            message_type: Type of message
            priority: Priority of the message
            
        Returns:
            List of agent IDs that successfully received the message
        """
        successful_deliveries = []
        correlation_id = str(uuid.uuid4())
        
        for recipient in self._registered_agents:
            if recipient != sender and self.is_route_allowed(sender, recipient):
                message = AgentMessage(
                    sender=sender,
                    recipient=recipient,
                    message_type=message_type,
                    content=message_content,
                    correlation_id=correlation_id,
                    priority=priority
                )
                
                try:
                    if self.route_message(message):
                        successful_deliveries.append(recipient)
                except Exception:
                    # Continue with other recipients even if one fails
                    continue
        
        return successful_deliveries
    
    def get_registered_agents(self) -> Set[str]:
        """
        Get the set of registered agent IDs.
        
        Returns:
            Set of registered agent IDs
        """
        return self._registered_agents.copy()


class MessageFactory:
    """Factory for creating standardized messages."""
    
    @staticmethod
    def create_request(sender: str, recipient: str, content: Dict[str, Any],
                      priority: Priority = Priority.MEDIUM) -> AgentMessage:
        """
        Create a request message.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content
            priority: Message priority
            
        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.REQUEST,
            content=content,
            correlation_id=str(uuid.uuid4()),
            priority=priority
        )
    
    @staticmethod
    def create_response(original_message: AgentMessage, content: Dict[str, Any],
                       priority: Priority = Priority.MEDIUM) -> AgentMessage:
        """
        Create a response message to an original request.
        
        Args:
            original_message: The original message being responded to
            content: Response content
            priority: Message priority
            
        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=original_message.recipient,
            recipient=original_message.sender,
            message_type=MessageType.RESPONSE,
            content=content,
            correlation_id=original_message.correlation_id,
            priority=priority
        )
    
    @staticmethod
    def create_notification(sender: str, recipient: str, content: Dict[str, Any],
                           priority: Priority = Priority.LOW) -> AgentMessage:
        """
        Create a notification message.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            content: Message content
            priority: Message priority
            
        Returns:
            AgentMessage instance
        """
        return AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.NOTIFICATION,
            content=content,
            correlation_id=str(uuid.uuid4()),
            priority=priority
        )
    
    @staticmethod
    def create_error(sender: str, recipient: str, error_message: str,
                    error_code: Optional[str] = None,
                    original_correlation_id: Optional[str] = None) -> AgentMessage:
        """
        Create an error message.
        
        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            error_message: Error description
            error_code: Optional error code
            original_correlation_id: Correlation ID of the message that caused the error
            
        Returns:
            AgentMessage instance
        """
        content = {"error_message": error_message}
        if error_code:
            content["error_code"] = error_code
        
        return AgentMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.ERROR,
            content=content,
            correlation_id=original_correlation_id or str(uuid.uuid4()),
            priority=Priority.HIGH
        )


def serialize_message(message: AgentMessage) -> str:
    """
    Serialize an AgentMessage to JSON string.
    
    Args:
        message: The message to serialize
        
    Returns:
        JSON string representation
    """
    return json.dumps(message.to_dict(), indent=2)


def deserialize_message(json_str: str) -> AgentMessage:
    """
    Deserialize a JSON string to AgentMessage.
    
    Args:
        json_str: JSON string representation
        
    Returns:
        AgentMessage instance
        
    Raises:
        ValueError: If JSON is invalid or message validation fails
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    
    # Validate the JSON data
    validation_errors = MessageValidator.validate_json_message(data)
    if validation_errors:
        raise ValueError(f"Message validation failed: {validation_errors}")
    
    return AgentMessage.from_dict(data)