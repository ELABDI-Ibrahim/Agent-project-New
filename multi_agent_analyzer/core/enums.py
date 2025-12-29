"""
Enumerations used throughout the multi-agent data analyzer system.
"""

from enum import Enum, auto


class MessageType(Enum):
    """Types of messages that can be exchanged between agents."""
    REQUEST = auto()
    RESPONSE = auto()
    NOTIFICATION = auto()
    ERROR = auto()


class Priority(Enum):
    """Priority levels for messages and recommendations."""
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


class DataType(Enum):
    """Data types for columns in datasets."""
    NUMERICAL = auto()
    CATEGORICAL = auto()
    TEMPORAL = auto()
    TEXT = auto()
    BOOLEAN = auto()


class RiskLevel(Enum):
    """Risk levels for recommendations and decisions."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()
    MINIMAL = auto()


class ExportFormat(Enum):
    """Export formats for reports."""
    JSON = auto()
    CSV = auto()
    TXT = auto()
    MARKDOWN = auto()


class AnalysisState(Enum):
    """States of the analysis process."""
    INITIALIZED = auto()
    COLLECTING = auto()
    ANALYZING = auto()
    DECIDING = auto()
    REPORTING = auto()
    COMPLETED = auto()
    ERROR = auto()