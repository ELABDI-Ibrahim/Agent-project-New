"""
Unit tests for the BaseAgent class.
"""

import pytest
from unittest.mock import Mock, MagicMock
from datetime import datetime

from multi_agent_analyzer.agents import BaseAgent, AgentResult
from multi_agent_analyzer.core.models import AgentMessage
from multi_agent_analyzer.core.enums import MessageType, Priority
from multi_agent_analyzer.core.shared_context import SharedContext
from multi_agent_analyzer.core.llm_coordinator import LLMCoordinator
from multi_agent_analyzer.config import Config


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing purposes."""
    
    def _initialize_agent(self) -> None:
        """Test agent initialization."""
        pass
    
    def process(self, input_data: dict) -> AgentResult:
        """Test processing method."""
        if input_data.get("should_fail"):
            return AgentResult(
                success=False,
                errors=["Test error"],
                warnings=["Test warning"]
            )
        
        return AgentResult(
            success=True,
            data={"processed": True, "input_keys": list(input_data.keys())}
        )


class TestBaseAgent:
    """Test cases for BaseAgent functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=Config)
        config.gemini_api_key = "test_key"
        config.gemini_model = "gemini-1.5-pro"
        config.gemini_temperature = 0.7
        config.gemini_max_tokens = 1000
        config.max_retries = 3
        config.retry_delay = 1
        return config
    
    @pytest.fixture
    def mock_llm_coordinator(self, mock_config):
        """Create a mock LLM coordinator."""
        coordinator = Mock(spec=LLMCoordinator)
        coordinator.resolve_ambiguity.return_value = "Test LLM response"
        return coordinator
    
    @pytest.fixture
    def shared_context(self):
        """Create a real shared context for testing."""
        return SharedContext()
    
    @pytest.fixture
    def test_agent(self, mock_llm_coordinator, shared_context):
        """Create a test agent instance."""
        agent = TestAgent("test_agent", mock_llm_coordinator, shared_context)
        return agent
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization."""
        assert test_agent.name == "test_agent"
        assert not test_agent.is_initialized
        
        # Initialize the agent
        result = test_agent.initialize()
        assert result is True
        assert test_agent.is_initialized
    
    def test_agent_processing_success(self, test_agent):
        """Test successful agent processing."""
        test_agent.initialize()
        
        input_data = {"key1": "value1", "key2": "value2"}
        result = test_agent.process(input_data)
        
        assert result.success is True
        assert result.data["processed"] is True
        assert "key1" in result.data["input_keys"]
        assert "key2" in result.data["input_keys"]
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_agent_processing_failure(self, test_agent):
        """Test agent processing failure."""
        test_agent.initialize()
        
        input_data = {"should_fail": True}
        result = test_agent.process(input_data)
        
        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
    
    def test_input_validation_success(self, test_agent):
        """Test successful input validation."""
        input_data = {"required_key": "value", "optional_key": "value"}
        required_keys = ["required_key"]
        
        result = test_agent.validate_input(input_data, required_keys)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_input_validation_missing_keys(self, test_agent):
        """Test input validation with missing required keys."""
        input_data = {"optional_key": "value"}
        required_keys = ["required_key", "another_required_key"]
        
        result = test_agent.validate_input(input_data, required_keys)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Missing required keys" in result.errors[0]
    
    def test_input_validation_invalid_type(self, test_agent):
        """Test input validation with invalid input type."""
        input_data = "not a dictionary"
        required_keys = ["required_key"]
        
        result = test_agent.validate_input(input_data, required_keys)
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Input data must be a dictionary" in result.errors[0]
    
    def test_message_sending(self, test_agent, shared_context):
        """Test message sending functionality."""
        test_agent.initialize()
        
        message = AgentMessage(
            sender="test_agent",
            recipient="target_agent",
            message_type=MessageType.REQUEST,
            content={"test": "data"}
        )
        
        result = test_agent.send_message("target_agent", message)
        
        assert result is True
        # Check that message was logged in shared context
        messages = shared_context.get_message_history()
        assert len(messages) == 1
        assert messages[0].sender == "test_agent"
        assert messages[0].recipient == "target_agent"
    
    def test_llm_assistance_request(self, test_agent, mock_llm_coordinator):
        """Test LLM assistance request."""
        test_agent.initialize()
        
        query = "Test query"
        context = {"test_context": "value"}
        
        response = test_agent.request_llm_assistance(query, context)
        
        assert response == "Test LLM response"
        mock_llm_coordinator.resolve_ambiguity.assert_called_once()
        
        # Check that the call included agent information in context
        call_args = mock_llm_coordinator.resolve_ambiguity.call_args
        assert call_args[0][0] == query  # First argument is the query
        assert call_args[0][1]["agent_name"] == "test_agent"  # Context includes agent name
        assert call_args[0][1]["test_context"] == "value"  # Original context preserved
    
    def test_agent_status(self, test_agent):
        """Test agent status reporting."""
        status = test_agent.get_agent_status()
        
        assert status["name"] == "test_agent"
        assert status["type"] == "TestAgent"
        assert status["initialized"] is False
        assert status["error_count"] == 0
        
        # Initialize and check status again
        test_agent.initialize()
        status = test_agent.get_agent_status()
        assert status["initialized"] is True
    
    def test_processing_error_tracking(self, test_agent):
        """Test processing error tracking."""
        # Initially no errors
        assert len(test_agent.get_processing_errors()) == 0
        
        # Add some errors by triggering error conditions
        test_agent._processing_errors.append("Test error 1")
        test_agent._processing_errors.append("Test error 2")
        
        errors = test_agent.get_processing_errors()
        assert len(errors) == 2
        assert "Test error 1" in errors
        assert "Test error 2" in errors
        
        # Clear errors
        test_agent.clear_processing_errors()
        assert len(test_agent.get_processing_errors()) == 0
    
    def test_agent_result_serialization(self):
        """Test AgentResult serialization."""
        result = AgentResult(
            success=True,
            data={"key": "value"},
            errors=["error1"],
            warnings=["warning1"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["success"] is True
        assert result_dict["data"]["key"] == "value"
        assert result_dict["errors"] == ["error1"]
        assert result_dict["warnings"] == ["warning1"]
        assert "timestamp" in result_dict
    
    def test_string_representations(self, test_agent):
        """Test string representations of agent."""
        str_repr = str(test_agent)
        assert "TestAgent" in str_repr
        assert "test_agent" in str_repr
        assert "initialized=False" in str_repr
        
        repr_str = repr(test_agent)
        assert "TestAgent" in repr_str
        assert "test_agent" in repr_str
        assert "errors=0" in repr_str