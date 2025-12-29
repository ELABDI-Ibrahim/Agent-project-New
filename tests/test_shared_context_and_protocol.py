"""
Test the shared context and message protocol functionality.
"""

import pytest
import threading
import time
from datetime import datetime

from multi_agent_analyzer.core import (
    SharedContext, MessageValidator, MessageRouter, MessageFactory,
    AgentMessage, MessageType, Priority, AnalysisState,
    serialize_message, deserialize_message
)


class TestSharedContext:
    """Test the SharedContext class."""
    
    def test_data_storage_and_retrieval(self):
        """Test basic data storage and retrieval."""
        context = SharedContext()
        
        # Store data
        context.store_data("test_key", {"value": 42}, "test_agent")
        
        # Retrieve data
        data = context.retrieve_data("test_key")
        assert data == {"value": 42}
        
        # Check metadata
        metadata = context.get_data_metadata("test_key")
        assert metadata["agent_id"] == "test_agent"
        assert metadata["data_type"] == "dict"
        assert isinstance(metadata["timestamp"], datetime)
    
    def test_thread_safety(self):
        """Test that SharedContext is thread-safe."""
        context = SharedContext()
        results = []
        
        def store_data(agent_id, value):
            context.store_data(f"key_{agent_id}", value, agent_id)
            results.append(f"stored_{agent_id}")
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_data, args=(f"agent_{i}", i))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all data was stored
        assert len(results) == 10
        for i in range(10):
            assert context.retrieve_data(f"key_agent_{i}") == i
    
    def test_message_logging(self):
        """Test message logging functionality."""
        context = SharedContext()
        
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content={"test": "data"}
        )
        
        context.log_message(message)
        
        # Check message history
        history = context.get_message_history()
        assert len(history) == 1
        assert history[0].sender == "agent1"
        
        # Check agent-filtered history
        agent1_messages = context.get_message_history("agent1")
        assert len(agent1_messages) == 1
        
        agent2_messages = context.get_message_history("agent2")
        assert len(agent2_messages) == 1
    
    def test_analysis_state_management(self):
        """Test analysis state management."""
        context = SharedContext()
        
        # Initial state
        assert context.get_analysis_state() == AnalysisState.INITIALIZED
        
        # Update state
        context.update_analysis_state(AnalysisState.COLLECTING)
        assert context.get_analysis_state() == AnalysisState.COLLECTING
        
        # Check state history
        history = context.get_state_history()
        assert len(history) == 2
        assert history[0][0] == AnalysisState.INITIALIZED
        assert history[1][0] == AnalysisState.COLLECTING


class TestMessageValidator:
    """Test the MessageValidator class."""
    
    def test_valid_message(self):
        """Test validation of a valid message."""
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content={"test": "data"},
            correlation_id="test-123"
        )
        
        errors = MessageValidator.validate_message(message)
        assert len(errors) == 0
    
    def test_invalid_message(self):
        """Test validation of an invalid message."""
        message = AgentMessage(
            sender="",  # Invalid empty sender
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content="not a dict",  # Invalid content type
            correlation_id="test-123"
        )
        
        errors = MessageValidator.validate_message(message)
        assert len(errors) > 0
        assert any("Sender must be a non-empty string" in error for error in errors)
        assert any("Content must be a dictionary" in error for error in errors)
    
    def test_json_validation(self):
        """Test JSON message validation."""
        valid_json = {
            "sender": "agent1",
            "recipient": "agent2",
            "message_type": "REQUEST",
            "content": {"test": "data"},
            "timestamp": datetime.now().isoformat(),
            "correlation_id": "test-123",
            "priority": "MEDIUM"
        }
        
        errors = MessageValidator.validate_json_message(valid_json)
        assert len(errors) == 0
        
        # Test invalid JSON
        invalid_json = valid_json.copy()
        invalid_json["message_type"] = "INVALID_TYPE"
        
        errors = MessageValidator.validate_json_message(invalid_json)
        assert len(errors) > 0


class TestMessageRouter:
    """Test the MessageRouter class."""
    
    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        router = MessageRouter()
        
        def dummy_handler(message):
            pass
        
        # Register agent
        router.register_agent("agent1", dummy_handler)
        assert "agent1" in router.get_registered_agents()
        
        # Unregister agent
        router.unregister_agent("agent1")
        assert "agent1" not in router.get_registered_agents()
    
    def test_message_routing(self):
        """Test message routing between agents."""
        router = MessageRouter()
        received_messages = []
        
        def handler1(message):
            received_messages.append(("agent1", message))
        
        def handler2(message):
            received_messages.append(("agent2", message))
        
        # Register agents
        router.register_agent("agent1", handler1)
        router.register_agent("agent2", handler2)
        
        # Create and route message
        message = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content={"test": "data"}
        )
        
        success = router.route_message(message)
        assert success is True
        assert len(received_messages) == 1
        assert received_messages[0][0] == "agent2"
        assert received_messages[0][1].sender == "agent1"


class TestMessageFactory:
    """Test the MessageFactory class."""
    
    def test_create_request(self):
        """Test creating a request message."""
        message = MessageFactory.create_request(
            sender="agent1",
            recipient="agent2",
            content={"action": "process_data"}
        )
        
        assert message.sender == "agent1"
        assert message.recipient == "agent2"
        assert message.message_type == MessageType.REQUEST
        assert message.content == {"action": "process_data"}
        assert message.priority == Priority.MEDIUM
    
    def test_create_response(self):
        """Test creating a response message."""
        original = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content={"action": "process_data"},
            correlation_id="test-123"
        )
        
        response = MessageFactory.create_response(
            original_message=original,
            content={"result": "success"}
        )
        
        assert response.sender == "agent2"
        assert response.recipient == "agent1"
        assert response.message_type == MessageType.RESPONSE
        assert response.correlation_id == "test-123"
    
    def test_create_error(self):
        """Test creating an error message."""
        error_msg = MessageFactory.create_error(
            sender="agent1",
            recipient="agent2",
            error_message="Processing failed",
            error_code="ERR_001"
        )
        
        assert error_msg.message_type == MessageType.ERROR
        assert error_msg.priority == Priority.HIGH
        assert error_msg.content["error_message"] == "Processing failed"
        assert error_msg.content["error_code"] == "ERR_001"


class TestMessageSerialization:
    """Test message serialization and deserialization."""
    
    def test_serialize_deserialize(self):
        """Test message serialization and deserialization."""
        original = AgentMessage(
            sender="agent1",
            recipient="agent2",
            message_type=MessageType.REQUEST,
            content={"test": "data"},
            correlation_id="test-123",
            priority=Priority.HIGH
        )
        
        # Serialize
        json_str = serialize_message(original)
        assert isinstance(json_str, str)
        
        # Deserialize
        restored = deserialize_message(json_str)
        assert restored.sender == original.sender
        assert restored.recipient == original.recipient
        assert restored.message_type == original.message_type
        assert restored.content == original.content
        assert restored.correlation_id == original.correlation_id
        assert restored.priority == original.priority
    
    def test_deserialize_invalid_json(self):
        """Test deserialization of invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            deserialize_message("invalid json")
        
        with pytest.raises(ValueError, match="Message validation failed"):
            deserialize_message('{"sender": ""}')  # Invalid message