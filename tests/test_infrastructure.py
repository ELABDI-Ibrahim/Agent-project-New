"""
Test the basic infrastructure setup.
"""

import pytest
from datetime import datetime
from multi_agent_analyzer import (
    Config, AgentMessage, DataProfile, StatisticalSummary,
    MessageType, Priority, DataType
)


def test_config_creation():
    """Test that Config can be created with default values."""
    config = Config()
    assert config.gemini_model == "gemini-1.5-pro"
    assert config.gemini_temperature == 0.1
    assert config.max_retries == 3
    assert config.property_test_iterations == 100


def test_config_validation_fails_without_api_key():
    """Test that config validation fails without API key."""
    config = Config()
    with pytest.raises(ValueError, match="Gemini API key is required"):
        config.validate()


def test_config_validation_passes_with_api_key():
    """Test that config validation passes with API key."""
    config = Config(gemini_api_key="test-key")
    assert config.validate() is True


def test_agent_message_creation():
    """Test that AgentMessage can be created and serialized."""
    message = AgentMessage(
        sender="test_sender",
        recipient="test_recipient",
        message_type=MessageType.REQUEST,
        content={"test": "data"}
    )
    
    assert message.sender == "test_sender"
    assert message.recipient == "test_recipient"
    assert message.message_type == MessageType.REQUEST
    assert message.content == {"test": "data"}
    assert isinstance(message.timestamp, datetime)


def test_agent_message_serialization():
    """Test that AgentMessage can be serialized and deserialized."""
    original = AgentMessage(
        sender="test_sender",
        recipient="test_recipient", 
        message_type=MessageType.RESPONSE,
        content={"result": "success"},
        correlation_id="test-123",
        priority=Priority.HIGH
    )
    
    # Serialize to dict
    message_dict = original.to_dict()
    assert message_dict["sender"] == "test_sender"
    assert message_dict["message_type"] == "RESPONSE"
    assert message_dict["priority"] == "HIGH"
    
    # Deserialize from dict
    restored = AgentMessage.from_dict(message_dict)
    assert restored.sender == original.sender
    assert restored.recipient == original.recipient
    assert restored.message_type == original.message_type
    assert restored.content == original.content
    assert restored.correlation_id == original.correlation_id
    assert restored.priority == original.priority


def test_data_profile_creation():
    """Test that DataProfile can be created."""
    profile = DataProfile(
        row_count=1000,
        column_count=5,
        data_types={"col1": "int64", "col2": "float64"},
        missing_values={"col1": 0, "col2": 5},
        unique_values={"col1": 1000, "col2": 995},
        memory_usage=1024.0
    )
    
    assert profile.row_count == 1000
    assert profile.column_count == 5
    assert profile.data_types["col1"] == "int64"
    assert profile.missing_values["col2"] == 5


def test_statistical_summary_creation():
    """Test that StatisticalSummary can be created."""
    summary = StatisticalSummary(
        descriptive_stats={
            "col1": {"mean": 50.0, "std": 10.0, "min": 0.0, "max": 100.0}
        },
        outliers={"col1": [150, -10]},
        distribution_tests={"col1": {"shapiro_p": 0.05}}
    )
    
    assert summary.descriptive_stats["col1"]["mean"] == 50.0
    assert summary.outliers["col1"] == [150, -10]
    assert summary.distribution_tests["col1"]["shapiro_p"] == 0.05


def test_imports_work():
    """Test that all expected imports work correctly."""
    from multi_agent_analyzer.core.enums import MessageType, Priority, DataType, RiskLevel
    from multi_agent_analyzer.core.models import AgentMessage, DataProfile
    from multi_agent_analyzer.config import Config
    
    # Test enum values
    assert MessageType.REQUEST is not None
    assert Priority.HIGH is not None
    assert DataType.NUMERICAL is not None
    assert RiskLevel.CRITICAL is not None