"""
Collector Agent for the multi-agent data analyzer system.

This module implements the CollectorAgent class responsible for:
- CSV file validation and loading
- Data type detection and inference
- Missing value detection and handling
- Data normalization capabilities
- Initial data profiling and quality assessment
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import re
from datetime import datetime

from .base_agent import BaseAgent, AgentResult
from ..core.models import DataProfile, ValidationResult, DataSchema, ColumnDefinition
from ..core.enums import DataType, Priority
from ..core.shared_context import SharedContext
from ..core.llm_coordinator import LLMCoordinator


class MissingValueReport:
    """Report on missing values in the dataset."""
    
    def __init__(self, missing_counts: Dict[str, int], missing_percentages: Dict[str, float],
                 total_rows: int):
        self.missing_counts = missing_counts
        self.missing_percentages = missing_percentages
        self.total_rows = total_rows
        self.columns_with_missing = [col for col, count in missing_counts.items() if count > 0]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "missing_counts": self.missing_counts,
            "missing_percentages": self.missing_percentages,
            "total_rows": self.total_rows,
            "columns_with_missing": self.columns_with_missing
        }


class CollectorAgent(BaseAgent):
    """
    Agent responsible for data collection, validation, and preprocessing.
    
    The CollectorAgent handles:
    - CSV file loading and structure validation
    - Data type detection and inference
    - Missing value detection and reporting
    - Data normalization and cleaning
    - Initial statistical profiling
    - Data dictionary integration
    """
    
    def __init__(self, llm_coordinator: LLMCoordinator, shared_context: SharedContext):
        """
        Initialize the Collector Agent.
        
        Args:
            llm_coordinator: LLM coordinator for cognitive assistance
            shared_context: Shared context for inter-agent communication
        """
        super().__init__("CollectorAgent", llm_coordinator, shared_context)
        
        # Configuration for data processing
        self.max_sample_rows = 1000  # Maximum rows for sample data
        self.missing_value_threshold = 0.5  # Threshold for flagging columns with too many missing values
        self.outlier_detection_methods = ['iqr', 'zscore']
        
        # Data type inference patterns
        self.date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
        ]
        
        # Current dataset state
        self.current_dataframe: Optional[pd.DataFrame] = None
        self.current_profile: Optional[DataProfile] = None
        self.current_schema: Optional[DataSchema] = None
    
    def _initialize_agent(self) -> None:
        """Initialize the Collector Agent."""
        self.logger.info("Collector Agent initialized successfully")
    
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process input data for collection and validation.
        
        Args:
            input_data: Dictionary containing:
                - file_path: Path to CSV file to process
                - data_dictionary: Optional data dictionary for schema validation
                - processing_options: Optional processing configuration
        
        Returns:
            AgentResult: Processing results with data profile and validation info
        """
        self._log_processing_start("data_collection", input_data)
        
        # Validate input
        validation = self.validate_input(input_data, ["file_path"])
        if not validation.is_valid:
            result = AgentResult(success=False, errors=validation.errors, warnings=validation.warnings)
            self._log_processing_end("data_collection", result)
            return result
        
        try:
            file_path = input_data["file_path"]
            data_dictionary = input_data.get("data_dictionary")
            processing_options = input_data.get("processing_options", {})
            
            # Step 1: Validate CSV file structure
            csv_validation = self.validate_csv_structure(file_path)
            if not csv_validation.is_valid:
                result = AgentResult(success=False, errors=csv_validation.errors, 
                                   warnings=csv_validation.warnings)
                self._log_processing_end("data_collection", result)
                return result
            
            # Step 2: Load the CSV file
            dataframe = self._load_csv_file(file_path, processing_options)
            if dataframe is None:
                result = AgentResult(success=False, errors=["Failed to load CSV file"])
                self._log_processing_end("data_collection", result)
                return result
            
            self.current_dataframe = dataframe
            
            # Step 3: Detect data types
            detected_types = self.detect_data_types(dataframe)
            
            # Step 4: Identify missing values
            missing_report = self.identify_missing_values(dataframe)
            
            # Step 5: Generate data profile
            data_profile = self.generate_data_profile(dataframe)
            self.current_profile = data_profile
            
            # Step 6: Process data dictionary if provided
            if data_dictionary:
                schema = self._process_data_dictionary(data_dictionary, detected_types)
                self.current_schema = schema
                
                # Validate data against schema
                schema_validation = self._validate_against_schema(dataframe, schema)
                if not schema_validation.is_valid:
                    data_profile.to_dict()["schema_validation"] = schema_validation.to_dict()
            
            # Step 7: Normalize columns if needed
            normalized_df = self.normalize_columns(dataframe, self.current_schema)
            if normalized_df is not None:
                self.current_dataframe = normalized_df
                # Update profile with normalized data
                data_profile = self.generate_data_profile(normalized_df)
                self.current_profile = data_profile
            
            # Store results in shared context
            self.shared_context.store_data("raw_dataframe", self.current_dataframe, self.name)
            self.shared_context.store_data("data_profile", data_profile, self.name)
            self.shared_context.store_data("missing_value_report", missing_report, self.name)
            self.shared_context.store_data("detected_data_types", detected_types, self.name)
            
            if self.current_schema:
                self.shared_context.store_data("data_schema", self.current_schema, self.name)
            
            # Prepare result data
            result_data = {
                "data_profile": data_profile.to_dict(),
                "missing_value_report": missing_report.to_dict(),
                "detected_data_types": detected_types,
                "row_count": len(dataframe),
                "column_count": len(dataframe.columns),
                "columns": list(dataframe.columns)
            }
            
            if self.current_schema:
                result_data["data_schema"] = self.current_schema.to_dict()
            
            # Generate warnings for data quality issues
            warnings = []
            if missing_report.columns_with_missing:
                warnings.append(f"Missing values detected in {len(missing_report.columns_with_missing)} columns")
            
            # Check for high missing value percentages
            high_missing_cols = [col for col, pct in missing_report.missing_percentages.items() 
                               if pct > self.missing_value_threshold]
            if high_missing_cols:
                warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
            
            result = AgentResult(success=True, data=result_data, warnings=warnings)
            self._log_processing_end("data_collection", result)
            return result
            
        except Exception as e:
            error_msg = f"Data collection failed: {str(e)}"
            self.logger.error(error_msg)
            result = AgentResult(success=False, errors=[error_msg])
            self._log_processing_end("data_collection", result)
            return result
    
    def validate_csv_structure(self, file_path: str) -> ValidationResult:
        """
        Validate CSV file structure and accessibility.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            ValidationResult: Validation result with any errors or warnings
        """
        errors = []
        warnings = []
        
        try:
            # Check if file exists
            path = Path(file_path)
            if not path.exists():
                errors.append(f"File does not exist: {file_path}")
                return ValidationResult(is_valid=False, errors=errors)
            
            # Check file extension
            if path.suffix.lower() not in ['.csv', '.txt']:
                warnings.append(f"File extension '{path.suffix}' is not .csv - attempting to parse anyway")
            
            # Check file size
            file_size = path.stat().st_size
            if file_size == 0:
                errors.append("File is empty")
                return ValidationResult(is_valid=False, errors=errors)
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                warnings.append(f"Large file detected ({file_size / (1024*1024):.1f}MB) - processing may be slow")
            
            # Try to read first few lines to validate CSV structure
            try:
                sample_df = pd.read_csv(file_path, nrows=5)
                if len(sample_df.columns) == 0:
                    errors.append("No columns detected in CSV file")
                elif len(sample_df.columns) == 1:
                    warnings.append("Only one column detected - check delimiter settings")
                
            except pd.errors.EmptyDataError:
                errors.append("CSV file appears to be empty or has no data rows")
            except pd.errors.ParserError as e:
                errors.append(f"CSV parsing error: {str(e)}")
            except Exception as e:
                errors.append(f"Error reading CSV file: {str(e)}")
            
            is_valid = len(errors) == 0
            return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
            
        except Exception as e:
            errors.append(f"File validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors)
    
    def detect_data_types(self, dataframe: pd.DataFrame) -> Dict[str, DataType]:
        """
        Detect and infer data types for each column.
        
        Args:
            dataframe: DataFrame to analyze
            
        Returns:
            Dict[str, DataType]: Mapping of column names to detected data types
        """
        detected_types = {}
        
        for column in dataframe.columns:
            series = dataframe[column].dropna()  # Remove NaN values for type detection
            
            if len(series) == 0:
                detected_types[column] = DataType.TEXT  # Default for empty columns
                continue
            
            # Check for temporal data
            if self._is_temporal_column(series):
                detected_types[column] = DataType.TEMPORAL
            # Check for numerical data
            elif pd.api.types.is_numeric_dtype(series):
                detected_types[column] = DataType.NUMERICAL
            # Check for boolean data
            elif self._is_boolean_column(series):
                detected_types[column] = DataType.BOOLEAN
            # Check for categorical data
            elif self._is_categorical_column(series):
                detected_types[column] = DataType.CATEGORICAL
            else:
                detected_types[column] = DataType.TEXT
        
        return detected_types
    
    def identify_missing_values(self, dataframe: pd.DataFrame) -> MissingValueReport:
        """
        Identify and report missing values in the dataset.
        
        Args:
            dataframe: DataFrame to analyze
            
        Returns:
            MissingValueReport: Report on missing values
        """
        total_rows = len(dataframe)
        missing_counts = {}
        missing_percentages = {}
        
        for column in dataframe.columns:
            missing_count = dataframe[column].isnull().sum()
            missing_counts[column] = int(missing_count)
            missing_percentages[column] = float(missing_count / total_rows * 100) if total_rows > 0 else 0.0
        
        return MissingValueReport(missing_counts, missing_percentages, total_rows)
    
    def generate_data_profile(self, dataframe: pd.DataFrame) -> DataProfile:
        """
        Generate a comprehensive data profile for the dataset.
        
        Args:
            dataframe: DataFrame to profile
            
        Returns:
            DataProfile: Comprehensive data profile
        """
        # Basic statistics
        row_count = len(dataframe)
        column_count = len(dataframe.columns)
        
        # Data types
        data_types = {col: str(dataframe[col].dtype) for col in dataframe.columns}
        
        # Missing values
        missing_values = {col: int(dataframe[col].isnull().sum()) for col in dataframe.columns}
        
        # Unique values
        unique_values = {}
        for col in dataframe.columns:
            try:
                unique_count = dataframe[col].nunique()
                unique_values[col] = int(unique_count)
            except Exception:
                unique_values[col] = 0
        
        # Memory usage
        memory_usage = float(dataframe.memory_usage(deep=True).sum() / (1024 * 1024))  # MB
        
        # Sample data (first few rows)
        sample_data = dataframe.head(min(self.max_sample_rows, len(dataframe))).copy()
        
        return DataProfile(
            row_count=row_count,
            column_count=column_count,
            data_types=data_types,
            missing_values=missing_values,
            unique_values=unique_values,
            memory_usage=memory_usage,
            sample_data=sample_data
        )
    
    def normalize_columns(self, dataframe: pd.DataFrame, 
                         schema: Optional[DataSchema] = None) -> Optional[pd.DataFrame]:
        """
        Normalize column formats and data types.
        
        Args:
            dataframe: DataFrame to normalize
            schema: Optional data schema for guided normalization
            
        Returns:
            Optional[pd.DataFrame]: Normalized DataFrame or None if no changes needed
        """
        try:
            normalized_df = dataframe.copy()
            changes_made = False
            
            for column in normalized_df.columns:
                original_series = normalized_df[column]
                
                # Skip if column is all NaN
                if original_series.isnull().all():
                    continue
                
                # Apply schema-based normalization if available
                if schema and column in schema.columns:
                    col_def = schema.columns[column]
                    normalized_series = self._normalize_by_schema(original_series, col_def)
                    if not normalized_series.equals(original_series):
                        normalized_df[column] = normalized_series
                        changes_made = True
                else:
                    # Apply general normalization
                    normalized_series = self._apply_general_normalization(original_series)
                    if not normalized_series.equals(original_series):
                        normalized_df[column] = normalized_series
                        changes_made = True
            
            return normalized_df if changes_made else None
            
        except Exception as e:
            self.logger.error(f"Column normalization failed: {str(e)}")
            return None
    
    def _load_csv_file(self, file_path: str, options: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load CSV file with error handling and options.
        
        Args:
            file_path: Path to CSV file
            options: Loading options (delimiter, encoding, etc.)
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if failed
        """
        try:
            # Default options
            load_options = {
                'delimiter': options.get('delimiter', ','),
                'encoding': options.get('encoding', 'utf-8'),
                'low_memory': False,
                'na_values': ['', 'NA', 'N/A', 'NULL', 'null', 'None', 'none', '-', '?']
            }
            
            # Try loading with specified options
            try:
                df = pd.read_csv(file_path, **load_options)
                self.logger.info(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                return df
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        load_options['encoding'] = encoding
                        df = pd.read_csv(file_path, **load_options)
                        self.logger.warning(f"Loaded CSV with encoding: {encoding}")
                        return df
                    except UnicodeDecodeError:
                        continue
                raise UnicodeDecodeError("Could not decode file with any common encoding")
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV file: {str(e)}")
            return None
    
    def _is_temporal_column(self, series: pd.Series) -> bool:
        """Check if a series contains temporal data."""
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Check string patterns for dates
        if series.dtype == 'object':
            sample_values = series.dropna().head(10).astype(str)
            date_matches = 0
            
            for value in sample_values:
                for pattern in self.date_patterns:
                    if re.search(pattern, value):
                        date_matches += 1
                        break
            
            # If more than 50% match date patterns, consider it temporal
            return date_matches / len(sample_values) > 0.5 if len(sample_values) > 0 else False
        
        return False
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if a series contains boolean data."""
        if pd.api.types.is_bool_dtype(series):
            return True
        
        # Check for common boolean representations
        if series.dtype == 'object':
            unique_values = set(series.dropna().astype(str).str.lower().unique())
            boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
            return unique_values.issubset(boolean_values) and len(unique_values) <= 2
        
        return False
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if a series should be treated as categorical."""
        # If already categorical
        if pd.api.types.is_categorical_dtype(series):
            return True
        
        # Check ratio of unique values to total values
        if len(series) > 0:
            unique_ratio = series.nunique() / len(series)
            # Consider categorical if less than 50% unique values and not too many categories
            return unique_ratio < 0.5 and series.nunique() <= 50
        
        return False
    
    def _process_data_dictionary(self, data_dictionary: Dict[str, Any], 
                                detected_types: Dict[str, DataType]) -> DataSchema:
        """
        Process data dictionary and create schema.
        
        Args:
            data_dictionary: Raw data dictionary
            detected_types: Detected data types from analysis
            
        Returns:
            DataSchema: Processed data schema
        """
        columns = {}
        
        # Process each column definition
        for col_name, col_info in data_dictionary.get('columns', {}).items():
            # Determine data type (use dictionary if specified, otherwise use detected)
            if 'data_type' in col_info:
                try:
                    data_type = DataType[col_info['data_type'].upper()]
                except KeyError:
                    data_type = detected_types.get(col_name, DataType.TEXT)
            else:
                data_type = detected_types.get(col_name, DataType.TEXT)
            
            # Create column definition
            col_def = ColumnDefinition(
                name=col_name,
                data_type=data_type,
                description=col_info.get('description', ''),
                unit=col_info.get('unit'),
                valid_range=col_info.get('valid_range'),
                categorical_values=col_info.get('categorical_values'),
                is_nullable=col_info.get('is_nullable', True),
                relationships=col_info.get('relationships', [])
            )
            
            columns[col_name] = col_def
        
        # Create schema
        schema = DataSchema(
            columns=columns,
            primary_keys=data_dictionary.get('primary_keys', []),
            foreign_keys=data_dictionary.get('foreign_keys', {}),
            business_rules=data_dictionary.get('business_rules', []),
            data_quality_rules=data_dictionary.get('data_quality_rules', [])
        )
        
        return schema
    
    def _validate_against_schema(self, dataframe: pd.DataFrame, 
                                schema: DataSchema) -> ValidationResult:
        """
        Validate dataframe against schema.
        
        Args:
            dataframe: DataFrame to validate
            schema: Schema to validate against
            
        Returns:
            ValidationResult: Validation result
        """
        errors = []
        warnings = []
        
        # Check for missing columns
        schema_columns = set(schema.columns.keys())
        df_columns = set(dataframe.columns)
        
        missing_columns = schema_columns - df_columns
        if missing_columns:
            errors.append(f"Missing columns from schema: {missing_columns}")
        
        extra_columns = df_columns - schema_columns
        if extra_columns:
            warnings.append(f"Extra columns not in schema: {extra_columns}")
        
        # Validate each column against its definition
        for col_name, col_def in schema.columns.items():
            if col_name not in dataframe.columns:
                continue
            
            series = dataframe[col_name]
            
            # Check nullable constraint
            if not col_def.is_nullable and series.isnull().any():
                errors.append(f"Column '{col_name}' contains null values but is marked as non-nullable")
            
            # Check valid range for numerical columns
            if col_def.valid_range and col_def.data_type == DataType.NUMERICAL:
                min_val, max_val = col_def.valid_range
                numeric_series = pd.to_numeric(series, errors='coerce')
                out_of_range = numeric_series[(numeric_series < min_val) | (numeric_series > max_val)]
                if not out_of_range.empty:
                    warnings.append(f"Column '{col_name}' has {len(out_of_range)} values outside valid range [{min_val}, {max_val}]")
            
            # Check categorical values
            if col_def.categorical_values and col_def.data_type == DataType.CATEGORICAL:
                valid_values = set(col_def.categorical_values)
                actual_values = set(series.dropna().unique())
                invalid_values = actual_values - valid_values
                if invalid_values:
                    warnings.append(f"Column '{col_name}' has invalid categorical values: {invalid_values}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)
    
    def _normalize_by_schema(self, series: pd.Series, col_def: ColumnDefinition) -> pd.Series:
        """
        Normalize a series based on schema definition.
        
        Args:
            series: Series to normalize
            col_def: Column definition from schema
            
        Returns:
            pd.Series: Normalized series
        """
        normalized = series.copy()
        
        try:
            # Normalize based on data type
            if col_def.data_type == DataType.NUMERICAL:
                normalized = pd.to_numeric(normalized, errors='coerce')
            elif col_def.data_type == DataType.TEMPORAL:
                normalized = pd.to_datetime(normalized, errors='coerce')
            elif col_def.data_type == DataType.BOOLEAN:
                # Convert common boolean representations
                bool_map = {
                    'true': True, 'false': False,
                    'yes': True, 'no': False,
                    '1': True, '0': False,
                    't': True, 'f': False,
                    'y': True, 'n': False
                }
                if series.dtype == 'object':
                    normalized = series.str.lower().map(bool_map)
            elif col_def.data_type == DataType.CATEGORICAL:
                if col_def.categorical_values:
                    # Only keep valid categorical values
                    normalized = series.where(series.isin(col_def.categorical_values))
                normalized = normalized.astype('category')
            
        except Exception as e:
            self.logger.warning(f"Failed to normalize column '{col_def.name}': {str(e)}")
        
        return normalized
    
    def _apply_general_normalization(self, series: pd.Series) -> pd.Series:
        """
        Apply general normalization to a series.
        
        Args:
            series: Series to normalize
            
        Returns:
            pd.Series: Normalized series
        """
        normalized = series.copy()
        
        try:
            # String normalization
            if series.dtype == 'object':
                # Strip whitespace
                normalized = normalized.astype(str).str.strip()
                
                # Replace empty strings with NaN
                normalized = normalized.replace('', np.nan)
                
                # Try to convert to numeric if it looks numeric
                if normalized.str.match(r'^-?\d+\.?\d*$').all():
                    numeric_version = pd.to_numeric(normalized, errors='coerce')
                    if not numeric_version.isnull().all():
                        normalized = numeric_version
        
        except Exception as e:
            self.logger.warning(f"General normalization failed: {str(e)}")
        
        return normalized