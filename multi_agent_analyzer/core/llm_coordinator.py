"""
LLM Coordinator for the multi-agent data analyzer system.

This module provides the LLMCoordinator class that integrates with Google's Gemini API
to provide cognitive capabilities for agent coordination, data interpretation, and
decision validation.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .models import (
    DataSchema, ColumnDefinition, ValidationResult, AgentMessage
)
from .enums import DataType, MessageType, Priority
from ..config import Config


@dataclass
class AgentInstruction:
    """Instruction for an agent from the LLM coordinator."""
    agent_name: str
    action: str
    parameters: Dict[str, Any]
    priority: Priority
    context: Dict[str, Any]


class LLMCoordinator:
    """
    LLM Coordinator that uses Gemini API to provide cognitive capabilities
    for the multi-agent data analysis system.
    """

    def __init__(self, config: Config):
        """
        Initialize the LLM Coordinator with Gemini API configuration.
        
        Args:
            config: Configuration object containing API settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configure Gemini API
        if not config.gemini_api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=config.gemini_api_key)
        
        # Initialize the model with safety settings
        self.model = genai.GenerativeModel(
            model_name=config.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=config.gemini_temperature,
                max_output_tokens=config.gemini_max_tokens,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        self.logger.info(f"LLM Coordinator initialized with model: {config.gemini_model}")

    def interpret_data_dictionary(self, dictionary: Dict[str, Any]) -> DataSchema:
        """
        Interpret a data dictionary and convert it to a structured DataSchema.
        
        Args:
            dictionary: Raw data dictionary as a dictionary
            
        Returns:
            DataSchema: Structured schema with column definitions
            
        Raises:
            ValueError: If the dictionary cannot be interpreted
        """
        try:
            prompt = self._build_data_dictionary_prompt(dictionary)
            response = self._make_api_call_with_retry(prompt)
            
            # Parse the response to extract schema information
            schema_data = self._parse_schema_response(response)
            
            # Convert to DataSchema object
            columns = {}
            for col_name, col_info in schema_data.get("columns", {}).items():
                columns[col_name] = ColumnDefinition(
                    name=col_name,
                    data_type=DataType[col_info.get("data_type", "TEXT").upper()],
                    description=col_info.get("description", ""),
                    unit=col_info.get("unit"),
                    valid_range=col_info.get("valid_range"),
                    categorical_values=col_info.get("categorical_values"),
                    is_nullable=col_info.get("is_nullable", True),
                    relationships=col_info.get("relationships", [])
                )
            
            return DataSchema(
                columns=columns,
                primary_keys=schema_data.get("primary_keys", []),
                foreign_keys=schema_data.get("foreign_keys", {}),
                business_rules=schema_data.get("business_rules", []),
                data_quality_rules=schema_data.get("data_quality_rules", [])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to interpret data dictionary: {str(e)}")
            raise ValueError(f"Data dictionary interpretation failed: {str(e)}")

    def resolve_ambiguity(self, query: str, context: Dict[str, Any]) -> str:
        """
        Resolve ambiguities in data interpretation or analysis.
        
        Args:
            query: The ambiguous query or question
            context: Additional context information
            
        Returns:
            str: Clarification or resolution of the ambiguity
        """
        try:
            prompt = self._build_ambiguity_resolution_prompt(query, context)
            response = self._make_api_call_with_retry(prompt)
            
            self.logger.info(f"Resolved ambiguity for query: {query[:50]}...")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to resolve ambiguity: {str(e)}")
            return f"Unable to resolve ambiguity: {str(e)}"

    def validate_decision(self, decision: Dict[str, Any], context: Dict[str, Any]) -> ValidationResult:
        """
        Validate an agent's decision for logical consistency.
        
        Args:
            decision: The decision to validate
            context: Context information for validation
            
        Returns:
            ValidationResult: Validation result with errors and warnings
        """
        try:
            prompt = self._build_decision_validation_prompt(decision, context)
            response = self._make_api_call_with_retry(prompt)
            
            # Parse validation response
            validation_data = self._parse_validation_response(response)
            
            return ValidationResult(
                is_valid=validation_data.get("is_valid", False),
                errors=validation_data.get("errors", []),
                warnings=validation_data.get("warnings", []),
                metadata=validation_data.get("metadata", {})
            )
            
        except Exception as e:
            self.logger.error(f"Decision validation failed: {str(e)}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed: {str(e)}"],
                warnings=[],
                metadata={}
            )

    def generate_explanation(self, technical_content: str, audience: str = "business") -> str:
        """
        Generate human-readable explanations from technical content.
        
        Args:
            technical_content: Technical content to explain
            audience: Target audience (business, technical, general)
            
        Returns:
            str: Human-readable explanation
        """
        try:
            prompt = self._build_explanation_prompt(technical_content, audience)
            response = self._make_api_call_with_retry(prompt)
            
            self.logger.info(f"Generated explanation for {audience} audience")
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"Failed to generate explanation: {str(e)}")
            return f"Unable to generate explanation: {str(e)}"

    def coordinate_agents(self, workflow_step: str, context: Dict[str, Any]) -> List[AgentInstruction]:
        """
        Coordinate agent activities for a specific workflow step.
        
        Args:
            workflow_step: Current step in the analysis workflow
            context: Current system context and state
            
        Returns:
            List[AgentInstruction]: Instructions for agents
        """
        try:
            prompt = self._build_coordination_prompt(workflow_step, context)
            response = self._make_api_call_with_retry(prompt)
            
            # Parse coordination response
            instructions_data = self._parse_coordination_response(response)
            
            instructions = []
            for instr_data in instructions_data:
                instructions.append(AgentInstruction(
                    agent_name=instr_data.get("agent_name", ""),
                    action=instr_data.get("action", ""),
                    parameters=instr_data.get("parameters", {}),
                    priority=Priority[instr_data.get("priority", "MEDIUM").upper()],
                    context=instr_data.get("context", {})
                ))
            
            self.logger.info(f"Generated {len(instructions)} agent instructions for step: {workflow_step}")
            return instructions
            
        except Exception as e:
            self.logger.error(f"Agent coordination failed: {str(e)}")
            return []

    def _make_api_call_with_retry(self, prompt: str) -> str:
        """
        Make an API call to Gemini with retry logic and error handling.
        
        Args:
            prompt: The prompt to send to the API
            
        Returns:
            str: The response from the API
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                self.logger.debug(f"Making API call, attempt {attempt + 1}")
                
                response = self.model.generate_content(prompt)
                
                if response.text:
                    return response.text
                else:
                    raise ValueError("Empty response from Gemini API")
                    
            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {str(e)}")
                
                if attempt < self.config.max_retries - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay * (2 ** attempt)
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_retries} API call attempts failed")
        
        raise Exception(f"API call failed after {self.config.max_retries} attempts: {str(last_exception)}")

    def _build_data_dictionary_prompt(self, dictionary: Dict[str, Any]) -> str:
        """Build prompt for data dictionary interpretation."""
        return f"""
Analyze the following data dictionary and convert it to a structured schema format.

Data Dictionary:
{json.dumps(dictionary, indent=2)}

Please provide a JSON response with the following structure:
{{
    "columns": {{
        "column_name": {{
            "data_type": "NUMERICAL|CATEGORICAL|TEMPORAL|TEXT|BOOLEAN",
            "description": "description of the column",
            "unit": "unit of measurement (if applicable)",
            "valid_range": [min, max] (for numerical columns),
            "categorical_values": ["value1", "value2"] (for categorical columns),
            "is_nullable": true/false,
            "relationships": ["related_column1", "related_column2"]
        }}
    }},
    "primary_keys": ["key1", "key2"],
    "foreign_keys": {{"fk_column": "referenced_table.column"}},
    "business_rules": ["rule1", "rule2"],
    "data_quality_rules": ["rule1", "rule2"]
}}

Focus on identifying:
1. Correct data types based on column descriptions
2. Relationships between columns
3. Business rules that might affect analysis
4. Data quality constraints
"""

    def _build_ambiguity_resolution_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Build prompt for ambiguity resolution."""
        return f"""
Resolve the following ambiguity in data analysis context:

Query: {query}

Context:
{json.dumps(context, indent=2)}

Please provide a clear, concise resolution that:
1. Addresses the specific ambiguity
2. Considers the provided context
3. Suggests the most appropriate interpretation
4. Explains the reasoning behind the resolution

Response should be in plain text, not JSON.
"""

    def _build_decision_validation_prompt(self, decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Build prompt for decision validation."""
        return f"""
Validate the following decision for logical consistency and appropriateness:

Decision:
{json.dumps(decision, indent=2)}

Context:
{json.dumps(context, indent=2)}

Please provide a JSON response with the following structure:
{{
    "is_valid": true/false,
    "errors": ["error1", "error2"],
    "warnings": ["warning1", "warning2"],
    "metadata": {{
        "confidence_score": 0.0-1.0,
        "reasoning": "explanation of validation",
        "suggestions": ["suggestion1", "suggestion2"]
    }}
}}

Check for:
1. Logical consistency
2. Appropriateness given the context
3. Potential risks or issues
4. Missing information or considerations
"""

    def _build_explanation_prompt(self, technical_content: str, audience: str) -> str:
        """Build prompt for generating explanations."""
        audience_guidance = {
            "business": "Use business terminology, focus on impact and value, avoid technical jargon",
            "technical": "Use precise technical language, include implementation details",
            "general": "Use simple language, provide analogies, explain technical terms"
        }
        
        guidance = audience_guidance.get(audience, audience_guidance["general"])
        
        return f"""
Convert the following technical content into a clear explanation for a {audience} audience:

Technical Content:
{technical_content}

Guidelines for {audience} audience: {guidance}

Please provide a clear, well-structured explanation that:
1. Maintains accuracy while improving readability
2. Uses appropriate terminology for the target audience
3. Includes relevant context and implications
4. Highlights key insights and actionable information

Response should be in plain text, not JSON.
"""

    def _build_coordination_prompt(self, workflow_step: str, context: Dict[str, Any]) -> str:
        """Build prompt for agent coordination."""
        return f"""
Coordinate agent activities for the following workflow step:

Workflow Step: {workflow_step}

Current Context:
{json.dumps(context, indent=2)}

Please provide a JSON response with agent instructions:
[
    {{
        "agent_name": "CollectorAgent|AnalyzerAgent|DecisionAgent|ReporterAgent",
        "action": "specific action to perform",
        "parameters": {{"param1": "value1", "param2": "value2"}},
        "priority": "HIGH|MEDIUM|LOW",
        "context": {{"additional": "context information"}}
    }}
]

Consider:
1. Current workflow step requirements
2. Available data and previous results
3. Agent capabilities and responsibilities
4. Optimal sequence and dependencies
5. Error handling and fallback options
"""

    def _parse_schema_response(self, response: str) -> Dict[str, Any]:
        """Parse schema response from LLM."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in schema response")
                # Return a minimal schema structure
                return {
                    "columns": {},
                    "primary_keys": [],
                    "foreign_keys": {},
                    "business_rules": [],
                    "data_quality_rules": []
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse schema response: {str(e)}")
            # Return a minimal schema structure
            return {
                "columns": {},
                "primary_keys": [],
                "foreign_keys": {},
                "business_rules": [],
                "data_quality_rules": []
            }

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """Parse validation response from LLM."""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                self.logger.warning("No JSON found in validation response")
                return {
                    "is_valid": False,
                    "errors": ["Failed to parse validation response"],
                    "warnings": [],
                    "metadata": {}
                }
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse validation response: {str(e)}")
            return {
                "is_valid": False,
                "errors": ["Failed to parse validation response"],
                "warnings": [],
                "metadata": {}
            }

    def _parse_coordination_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse coordination response from LLM."""
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Try to find a single object instead of array
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = response[start_idx:end_idx]
                    return [json.loads(json_str)]
                else:
                    self.logger.warning("No JSON found in coordination response")
                    return []
                    
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse coordination response: {str(e)}")
            return []

    def health_check(self) -> bool:
        """
        Perform a health check on the LLM coordinator.
        
        Returns:
            bool: True if the coordinator is healthy, False otherwise
        """
        try:
            test_prompt = "Respond with 'OK' if you can process this message."
            response = self._make_api_call_with_retry(test_prompt)
            
            is_healthy = "OK" in response.upper()
            self.logger.info(f"LLM Coordinator health check: {'PASSED' if is_healthy else 'FAILED'}")
            return is_healthy
            
        except Exception as e:
            self.logger.error(f"LLM Coordinator health check failed: {str(e)}")
            return False