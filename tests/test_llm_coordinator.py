"""
Tests for the LLM Coordinator functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from multi_agent_analyzer.config import Config
from multi_agent_analyzer.core.llm_coordinator import LLMCoordinator, AgentInstruction
from multi_agent_analyzer.core.models import DataSchema, ValidationResult
from multi_agent_analyzer.core.enums import Priority, DataType


class TestLLMCoordinator:
    """Test cases for LLMCoordinator class."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return Config(
            gemini_api_key="test-api-key",
            gemini_model="gemini-1.5-pro",
            gemini_temperature=0.1,
            max_retries=2,
            retry_delay=0.1
        )

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.generativeai module."""
        with patch('multi_agent_analyzer.core.llm_coordinator.genai') as mock:
            mock_model = Mock()
            mock_response = Mock()
            mock_response.text = '{"test": "response"}'
            mock_model.generate_content.return_value = mock_response
            mock.GenerativeModel.return_value = mock_model
            yield mock

    def test_initialization_success(self, config, mock_genai):
        """Test successful initialization of LLMCoordinator."""
        coordinator = LLMCoordinator(config)
        
        assert coordinator.config == config
        mock_genai.configure.assert_called_once_with(api_key="test-api-key")
        mock_genai.GenerativeModel.assert_called_once()

    def test_initialization_no_api_key(self):
        """Test initialization fails without API key."""
        config = Config(gemini_api_key="")
        
        with pytest.raises(ValueError, match="Gemini API key is required"):
            LLMCoordinator(config)

    def test_interpret_data_dictionary_success(self, config, mock_genai):
        """Test successful data dictionary interpretation."""
        # Mock response with valid schema JSON
        mock_response = Mock()
        mock_response.text = json.dumps({
            "columns": {
                "test_col": {
                    "data_type": "NUMERICAL",
                    "description": "Test column",
                    "is_nullable": True,
                    "relationships": []
                }
            },
            "primary_keys": [],
            "foreign_keys": {},
            "business_rules": [],
            "data_quality_rules": []
        })
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        test_dict = {"test_col": {"type": "number", "description": "Test column"}}
        
        result = coordinator.interpret_data_dictionary(test_dict)
        
        assert isinstance(result, DataSchema)
        assert "test_col" in result.columns
        assert result.columns["test_col"].data_type == DataType.NUMERICAL

    def test_resolve_ambiguity_success(self, config, mock_genai):
        """Test successful ambiguity resolution."""
        mock_response = Mock()
        mock_response.text = "This is a clear resolution of the ambiguity."
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator.resolve_ambiguity("What does this column mean?", {"context": "test"})
        
        assert result == "This is a clear resolution of the ambiguity."

    def test_validate_decision_success(self, config, mock_genai):
        """Test successful decision validation."""
        mock_response = Mock()
        mock_response.text = json.dumps({
            "is_valid": True,
            "errors": [],
            "warnings": ["Minor warning"],
            "metadata": {"confidence_score": 0.9}
        })
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        decision = {"action": "test", "confidence": 0.8}
        context = {"data": "test"}
        
        result = coordinator.validate_decision(decision, context)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.warnings) == 1

    def test_generate_explanation_success(self, config, mock_genai):
        """Test successful explanation generation."""
        mock_response = Mock()
        mock_response.text = "This is a business-friendly explanation."
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator.generate_explanation("Technical content", "business")
        
        assert result == "This is a business-friendly explanation."

    def test_coordinate_agents_success(self, config, mock_genai):
        """Test successful agent coordination."""
        mock_response = Mock()
        mock_response.text = json.dumps([
            {
                "agent_name": "CollectorAgent",
                "action": "collect_data",
                "parameters": {"file": "test.csv"},
                "priority": "HIGH",
                "context": {}
            }
        ])
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator.coordinate_agents("data_collection", {"step": "initial"})
        
        assert len(result) == 1
        assert isinstance(result[0], AgentInstruction)
        assert result[0].agent_name == "CollectorAgent"
        assert result[0].priority == Priority.HIGH

    def test_api_call_retry_logic(self, config, mock_genai):
        """Test API call retry logic on failure."""
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.side_effect = [
            Exception("API Error"),
            Mock(text="Success response")
        ]
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator._make_api_call_with_retry("test prompt")
        
        assert result == "Success response"
        assert mock_model.generate_content.call_count == 2

    def test_api_call_max_retries_exceeded(self, config, mock_genai):
        """Test API call fails after max retries."""
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.side_effect = Exception("Persistent API Error")
        
        coordinator = LLMCoordinator(config)
        
        with pytest.raises(Exception, match="API call failed after 2 attempts"):
            coordinator._make_api_call_with_retry("test prompt")

    def test_health_check_success(self, config, mock_genai):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.text = "OK - System is healthy"
        mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator.health_check()
        
        assert result is True

    def test_health_check_failure(self, config, mock_genai):
        """Test health check failure."""
        mock_model = mock_genai.GenerativeModel.return_value
        mock_model.generate_content.side_effect = Exception("Health check failed")
        
        coordinator = LLMCoordinator(config)
        
        result = coordinator.health_check()
        
        assert result is False

    def test_parse_schema_response_invalid_json(self, config, mock_genai):
        """Test handling of invalid JSON in schema response."""
        coordinator = LLMCoordinator(config)
        
        result = coordinator._parse_schema_response("Invalid JSON response")
        
        # Should return minimal schema structure
        assert "columns" in result
        assert "primary_keys" in result
        assert result["columns"] == {}

    def test_parse_validation_response_invalid_json(self, config, mock_genai):
        """Test handling of invalid JSON in validation response."""
        coordinator = LLMCoordinator(config)
        
        result = coordinator._parse_validation_response("Invalid JSON response")
        
        # Should return failed validation
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0

    def test_parse_coordination_response_invalid_json(self, config, mock_genai):
        """Test handling of invalid JSON in coordination response."""
        coordinator = LLMCoordinator(config)
        
        result = coordinator._parse_coordination_response("Invalid JSON response")
        
        # Should return empty list
        assert result == []