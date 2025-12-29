"""
Configuration management for the multi-agent data analyzer system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for the multi-agent data analyzer."""
    
    # Gemini API Configuration
    gemini_api_key: str = ""
    gemini_model: str ="gemini-2.5-flash-lite"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 8192
    
    # System Configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout_seconds: int = 30
    
    # Data Processing Configuration
    max_rows_sample: int = 1000
    chunk_size: int = 10000
    memory_limit_mb: int = 1024
    
    # Testing Configuration
    property_test_iterations: int = 100
    test_data_seed: int = 42
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Additional settings
    additional_settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> 'Config':
        """Create configuration from environment variables."""
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_model=os.getenv("GEMINI_MODEL", ""),  # Changed
            gemini_temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
            gemini_max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "8192")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0")),
            timeout_seconds=int(os.getenv("TIMEOUT_SECONDS", "30")),
            max_rows_sample=int(os.getenv("MAX_ROWS_SAMPLE", "1000")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "10000")),
            memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "1024")),
            property_test_iterations=int(os.getenv("PROPERTY_TEST_ITERATIONS", "100")),
            test_data_seed=int(os.getenv("TEST_DATA_SEED", "42")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE")
        )

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """Load configuration from a JSON file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)

    def to_file(self, config_path: str) -> None:
        """Save configuration to a JSON file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            "gemini_api_key": self.gemini_api_key,
            "gemini_model": self.gemini_model,
            "gemini_temperature": self.gemini_temperature,
            "gemini_max_tokens": self.gemini_max_tokens,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout_seconds": self.timeout_seconds,
            "max_rows_sample": self.max_rows_sample,
            "chunk_size": self.chunk_size,
            "memory_limit_mb": self.memory_limit_mb,
            "property_test_iterations": self.property_test_iterations,
            "test_data_seed": self.test_data_seed,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "additional_settings": self.additional_settings
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def validate(self) -> bool:
        """Validate the configuration settings."""
        errors = []
        
        # Validate Gemini API key
        if not self.gemini_api_key:
            errors.append("Gemini API key is required")
        
        # Validate numeric ranges
        if self.gemini_temperature < 0 or self.gemini_temperature > 2:
            errors.append("Gemini temperature must be between 0 and 2")
        
        if self.gemini_max_tokens <= 0:
            errors.append("Gemini max tokens must be positive")
        
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        if self.retry_delay < 0:
            errors.append("Retry delay must be non-negative")
        
        if self.timeout_seconds <= 0:
            errors.append("Timeout seconds must be positive")
        
        if self.max_rows_sample <= 0:
            errors.append("Max rows sample must be positive")
        
        if self.chunk_size <= 0:
            errors.append("Chunk size must be positive")
        
        if self.memory_limit_mb <= 0:
            errors.append("Memory limit must be positive")
        
        if self.property_test_iterations <= 0:
            errors.append("Property test iterations must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {', '.join(valid_log_levels)}")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
        
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "gemini_api_key": self.gemini_api_key,
            "gemini_model": self.gemini_model,
            "gemini_temperature": self.gemini_temperature,
            "gemini_max_tokens": self.gemini_max_tokens,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "timeout_seconds": self.timeout_seconds,
            "max_rows_sample": self.max_rows_sample,
            "chunk_size": self.chunk_size,
            "memory_limit_mb": self.memory_limit_mb,
            "property_test_iterations": self.property_test_iterations,
            "test_data_seed": self.test_data_seed,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "additional_settings": self.additional_settings
        }