"""
Multi-Agent Data Analyzer

A sophisticated system that leverages specialized AI agents to process CSV data 
and generate comprehensive analytical reports.
"""

__version__ = "0.1.0"
__author__ = "Multi-Agent Data Analyzer Team"

from .core.models import (
    AgentMessage,
    DataProfile,
    StatisticalSummary,
    Recommendation,
    Report,
    ColumnDefinition,
    DataSchema
)

from .core.enums import (
    MessageType,
    Priority,
    DataType,
    RiskLevel,
    ExportFormat,
    AnalysisState
)

from .config import Config

__all__ = [
    "AgentMessage",
    "DataProfile", 
    "StatisticalSummary",
    "Recommendation",
    "Report",
    "ColumnDefinition",
    "DataSchema",
    "MessageType",
    "Priority",
    "DataType",
    "RiskLevel",
    "ExportFormat",
    "AnalysisState",
    "Config"
]