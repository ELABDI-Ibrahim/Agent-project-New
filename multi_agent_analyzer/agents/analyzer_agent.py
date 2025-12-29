"""
Analyzer Agent for the multi-agent data analyzer system.

This module provides the AnalyzerAgent class that performs statistical analysis,
correlation analysis, trend detection, and outlier detection on datasets.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

from .base_agent import BaseAgent, AgentResult
from ..core.models import StatisticalSummary, ValidationResult
from ..core.enums import MessageType, Priority
from ..core.shared_context import SharedContext
from ..core.llm_coordinator import LLMCoordinator


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for statistical analysis and trend detection.
    
    This agent performs:
    - Descriptive statistics computation (mean, median, mode, std)
    - Correlation analysis between columns
    - Multiple aggregation methods (sum, count, percentiles, quartiles)
    - Time-series analysis (trends, seasonal patterns)
    - Multi-method outlier detection (IQR, Z-score, Isolation Forest)
    """
    
    def __init__(self, llm_coordinator: LLMCoordinator, shared_context: SharedContext):
        """
        Initialize the Analyzer Agent.
        
        Args:
            llm_coordinator: LLM coordinator for cognitive assistance
            shared_context: Shared context for inter-agent communication
        """
        super().__init__("AnalyzerAgent", llm_coordinator, shared_context)
        
        # Configuration for analysis
        self.outlier_methods = ['iqr', 'zscore', 'isolation_forest']
        self.correlation_methods = ['pearson', 'spearman', 'kendall']
        self.aggregation_methods = ['sum', 'count', 'mean', 'median', 'std', 'min', 'max', 'quantile']
        
        # Thresholds and parameters
        self.zscore_threshold = 3.0
        self.iqr_multiplier = 1.5
        self.isolation_forest_contamination = 0.1
        
        # Time-series analysis parameters
        self.trend_window_sizes = [7, 30, 90]  # Days for moving averages
        self.seasonal_periods = [7, 30, 365]   # Weekly, monthly, yearly patterns
        
        self.logger.info("AnalyzerAgent initialized with statistical and time-series capabilities")
    
    def _initialize_agent(self) -> None:
        """Initialize agent-specific components."""
        # Verify required libraries are available
        try:
            import scipy.stats
            import sklearn.ensemble
            self.logger.info("Statistical libraries verified and available")
        except ImportError as e:
            raise RuntimeError(f"Required statistical libraries not available: {e}")
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process data and perform statistical analysis.
        
        Args:
            input_data: Dictionary containing:
                - 'dataframe': pandas DataFrame to analyze
                - 'analysis_config': Configuration with targeted_analyses
        
        Returns:
            AgentResult: Analysis results including statistical summary
        """
        self._log_processing_start("statistical_analysis", input_data)
        
        # Validate input
        validation = self.validate_input(input_data, ['dataframe'])
        if not validation.is_valid:
            result = AgentResult(success=False, errors=validation.errors)
            self._log_processing_end("statistical_analysis", result)
            return result
        
        try:
            dataframe = input_data['dataframe']
            analysis_config = input_data.get('analysis_config', {})
            
            # Check if we have a targeted analysis plan
            targeted_analyses = analysis_config.get('targeted_analyses', [])
            priority_columns = analysis_config.get('priority_columns', None)
            skip_full = analysis_config.get('skip_full_analysis', False)
            
            if targeted_analyses and skip_full:
                # Run only the targeted analyses specified by LLM
                self.logger.info(f"Running {len(targeted_analyses)} targeted analyses")
                results = self._run_targeted_analyses(dataframe, targeted_analyses, priority_columns)
            else:
                # Run full analysis (old behavior)
                self.logger.info("Running full analysis suite")
                results = self._perform_full_analysis(dataframe, priority_columns, {})
            
            # Store results in shared context
            self.shared_context.store_data("analysis_results", results, self.name)
            
            result = AgentResult(success=True, data=results)
            self._log_processing_end("statistical_analysis", result)
            return result
            
        except Exception as e:
            error_msg = f"Statistical analysis failed: {str(e)}"
            self.logger.error(error_msg)
            result = AgentResult(success=False, errors=[error_msg])
            self._log_processing_end("statistical_analysis", result)
            return result

    def _run_targeted_analyses(
        self,
        dataframe: pd.DataFrame,
        targeted_analyses: List[Dict[str, Any]],
        priority_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """
        Run only the analyses specified in the plan.
        
        Args:
            dataframe: Data to analyze
            targeted_analyses: List of specific analyses to run
            priority_columns: Columns to focus on
        
        Returns:
            Dict with only the requested analysis results
        """
        results = {
            'analysis_type': 'targeted',
            'analyses_performed': []
        }
        
        for analysis in targeted_analyses:
            analysis_type = analysis.get('type')
            columns = analysis.get('columns', priority_columns)
            
            self.logger.info(f"  Running {analysis_type} analysis: {analysis.get('reason', '')}")
            
            try:
                if analysis_type == 'descriptive':
                    results['descriptive_stats'] = self.compute_descriptive_stats(dataframe, columns)
                    results['analyses_performed'].append('descriptive')
                    
                elif analysis_type == 'correlation':
                    correlations = self.calculate_correlations(dataframe, columns)
                    results['correlations'] = {
                        method: corr_df.to_dict() for method, corr_df in correlations.items()
                    }
                    results['analyses_performed'].append('correlation')
                    
                elif analysis_type == 'outliers':
                    outlier_results = self.detect_outliers_multi_method(dataframe, columns)
                    results['outliers'] = outlier_results
                    results['analyses_performed'].append('outliers')
                    
                elif analysis_type == 'timeseries':
                    ts_results = self._perform_timeseries_analysis(dataframe, columns, {})
                    results['timeseries'] = ts_results
                    results['analyses_performed'].append('timeseries')
                    
            except Exception as e:
                self.logger.error(f"Failed to run {analysis_type} analysis: {e}")
        
        return results
    
    def compute_descriptive_stats(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Compute descriptive statistics for numerical columns.
        
        Args:
            data: DataFrame to analyze
            columns: Optional list of columns to analyze (defaults to all numerical)
        
        Returns:
            Dict mapping column names to their descriptive statistics
        """
        if columns is None:
            # Select only numerical columns
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        descriptive_stats = {}
        
        for col in columns:
            if col not in data.columns:
                self.logger.warning(f"Column '{col}' not found in data")
                continue
            
            if not pd.api.types.is_numeric_dtype(data[col]):
                self.logger.warning(f"Column '{col}' is not numerical, skipping")
                continue
            
            col_data = data[col].dropna()
            
            if len(col_data) == 0:
                self.logger.warning(f"Column '{col}' has no valid data")
                continue
            
            stats_dict = {
                'count': len(col_data),
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()),
                'var': float(col_data.var()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
            }
            
            # Calculate mode (most frequent value)
            try:
                mode_result = stats.mode(col_data, keepdims=True)
                stats_dict['mode'] = float(mode_result.mode[0])
                stats_dict['mode_count'] = int(mode_result.count[0])
            except Exception:
                stats_dict['mode'] = None
                stats_dict['mode_count'] = 0
            
            descriptive_stats[col] = stats_dict
        
        return descriptive_stats
    
    def calculate_correlations(self, data: pd.DataFrame, columns: Optional[List[str]] = None, 
                             methods: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlations between columns using multiple methods.
        
        Args:
            data: DataFrame to analyze
            columns: Optional list of columns to analyze
            methods: Optional list of correlation methods to use
        
        Returns:
            Dict mapping method names to correlation matrices
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = self.correlation_methods
        
        correlations = {}
        
        # Filter to only include specified columns that exist and are numerical
        valid_columns = []
        for col in columns:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                valid_columns.append(col)
        
        if len(valid_columns) < 2:
            self.logger.warning("Need at least 2 numerical columns for correlation analysis")
            return correlations
        
        subset_data = data[valid_columns]
        
        for method in methods:
            try:
                if method == 'pearson':
                    corr_matrix = subset_data.corr(method='pearson')
                elif method == 'spearman':
                    corr_matrix = subset_data.corr(method='spearman')
                elif method == 'kendall':
                    corr_matrix = subset_data.corr(method='kendall')
                else:
                    self.logger.warning(f"Unknown correlation method: {method}")
                    continue
                
                correlations[method] = corr_matrix
                
            except Exception as e:
                self.logger.error(f"Failed to calculate {method} correlation: {str(e)}")
        
        return correlations
    
    def perform_aggregations(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                           methods: Optional[List[str]] = None, group_by: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform multiple aggregation methods on the data.
        
        Args:
            data: DataFrame to analyze
            columns: Optional list of columns to aggregate
            methods: Optional list of aggregation methods
            group_by: Optional column to group by before aggregation
        
        Returns:
            Dict containing aggregation results
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = self.aggregation_methods
        
        aggregations = {}
        
        # Filter to valid numerical columns
        valid_columns = [col for col in columns if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]
        
        if not valid_columns:
            self.logger.warning("No valid numerical columns found for aggregation")
            return aggregations
        
        try:
            if group_by and group_by in data.columns:
                # Grouped aggregations
                grouped_data = data.groupby(group_by)[valid_columns]
                
                for method in methods:
                    if method == 'sum':
                        aggregations[f'{method}_grouped'] = grouped_data.sum().to_dict()
                    elif method == 'count':
                        aggregations[f'{method}_grouped'] = grouped_data.count().to_dict()
                    elif method == 'mean':
                        aggregations[f'{method}_grouped'] = grouped_data.mean().to_dict()
                    elif method == 'median':
                        aggregations[f'{method}_grouped'] = grouped_data.median().to_dict()
                    elif method == 'std':
                        aggregations[f'{method}_grouped'] = grouped_data.std().to_dict()
                    elif method == 'min':
                        aggregations[f'{method}_grouped'] = grouped_data.min().to_dict()
                    elif method == 'max':
                        aggregations[f'{method}_grouped'] = grouped_data.max().to_dict()
                    elif method == 'quantile':
                        # Calculate multiple quantiles
                        for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
                            aggregations[f'q{int(q*100)}_grouped'] = grouped_data.quantile(q).to_dict()
            
            # Overall aggregations (without grouping)
            subset_data = data[valid_columns]
            
            for method in methods:
                if method == 'sum':
                    aggregations[method] = subset_data.sum().to_dict()
                elif method == 'count':
                    aggregations[method] = subset_data.count().to_dict()
                elif method == 'mean':
                    aggregations[method] = subset_data.mean().to_dict()
                elif method == 'median':
                    aggregations[method] = subset_data.median().to_dict()
                elif method == 'std':
                    aggregations[method] = subset_data.std().to_dict()
                elif method == 'min':
                    aggregations[method] = subset_data.min().to_dict()
                elif method == 'max':
                    aggregations[method] = subset_data.max().to_dict()
                elif method == 'quantile':
                    # Calculate multiple quantiles
                    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
                        aggregations[f'q{int(q*100)}'] = subset_data.quantile(q).to_dict()
        
        except Exception as e:
            self.logger.error(f"Aggregation failed: {str(e)}")
            raise
        
        return aggregations
    
    def detect_outliers_multi_method(self, data: pd.DataFrame, columns: Optional[List[str]] = None,
                                   methods: Optional[List[str]] = None, 
                                   thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Detect outliers using multiple configurable methods.
        
        Args:
            data: DataFrame to analyze
            columns: Optional list of columns to analyze
            methods: Optional list of outlier detection methods to use
            thresholds: Optional dictionary of method-specific thresholds
        
        Returns:
            Dict containing outlier detection results for each method
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if methods is None:
            methods = self.outlier_methods
        
        # Set up thresholds
        default_thresholds = {
            'zscore': self.zscore_threshold,
            'iqr': self.iqr_multiplier,
            'isolation_forest': self.isolation_forest_contamination,
            'modified_zscore': 3.5,  # Modified Z-score threshold
            'lof': 0.1,  # Local Outlier Factor contamination
            'percentile': 0.05  # Percentile-based outlier threshold (5% on each tail)
        }
        
        if thresholds:
            default_thresholds.update(thresholds)
        
        outlier_results = {
            'methods_used': methods,
            'thresholds': default_thresholds,
            'column_results': {},
            'summary': {}
        }
        
        for col in columns:
            if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            col_data = data[col].dropna()
            if len(col_data) == 0:
                continue
            
            col_results = {}
            
            # IQR method
            if 'iqr' in methods:
                col_results['iqr'] = self._detect_outliers_iqr(col_data, default_thresholds['iqr'])
            
            # Z-score method
            if 'zscore' in methods:
                col_results['zscore'] = self._detect_outliers_zscore(col_data, default_thresholds['zscore'])
            
            # Modified Z-score method (using median absolute deviation)
            if 'modified_zscore' in methods:
                col_results['modified_zscore'] = self._detect_outliers_modified_zscore(col_data, default_thresholds['modified_zscore'])
            
            # Isolation Forest method
            if 'isolation_forest' in methods and len(col_data) > 10:
                col_results['isolation_forest'] = self._detect_outliers_isolation_forest(col_data, default_thresholds['isolation_forest'])
            
            # Local Outlier Factor method
            if 'lof' in methods and len(col_data) > 20:
                col_results['lof'] = self._detect_outliers_lof(col_data, default_thresholds['lof'])
            
            # Percentile-based method
            if 'percentile' in methods:
                col_results['percentile'] = self._detect_outliers_percentile(col_data, default_thresholds['percentile'])
            
            outlier_results['column_results'][col] = col_results
        
        # Generate summary statistics
        outlier_results['summary'] = self._generate_outlier_summary(outlier_results['column_results'])
        
        return outlier_results
    
    def _detect_outliers_iqr(self, data: pd.Series, multiplier: float) -> Dict[str, Any]:
        """Detect outliers using Interquartile Range method."""
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        
        return {
            'method': 'iqr',
            'outlier_values': outliers.tolist(),
            'outlier_indices': outliers.index.tolist(),
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
            'parameters': {'multiplier': multiplier, 'q1': float(q1), 'q3': float(q3), 'iqr': float(iqr)}
        }
    
    def _detect_outliers_zscore(self, data: pd.Series, threshold: float) -> Dict[str, Any]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = z_scores > threshold
        outliers = data[outlier_mask]
        
        return {
            'method': 'zscore',
            'outlier_values': outliers.tolist(),
            'outlier_indices': outliers.index.tolist(),
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'z_scores': z_scores[outlier_mask].tolist(),
            'parameters': {'threshold': threshold, 'mean': float(data.mean()), 'std': float(data.std())}
        }
    
    def _detect_outliers_modified_zscore(self, data: pd.Series, threshold: float) -> Dict[str, Any]:
        """Detect outliers using Modified Z-score method (using median absolute deviation)."""
        median = data.median()
        mad = np.median(np.abs(data - median))
        
        # Modified Z-score formula
        modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)
        outlier_mask = np.abs(modified_z_scores) > threshold
        outliers = data[outlier_mask]
        
        return {
            'method': 'modified_zscore',
            'outlier_values': outliers.tolist(),
            'outlier_indices': outliers.index.tolist(),
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'modified_z_scores': modified_z_scores[outlier_mask].tolist(),
            'parameters': {'threshold': threshold, 'median': float(median), 'mad': float(mad)}
        }
    
    def _detect_outliers_isolation_forest(self, data: pd.Series, contamination: float) -> Dict[str, Any]:
        """Detect outliers using Isolation Forest method."""
        try:
            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            
            outlier_mask = outlier_labels == -1
            outliers = data[outlier_mask]
            
            # Get anomaly scores
            anomaly_scores = iso_forest.decision_function(data.values.reshape(-1, 1))
            outlier_scores = anomaly_scores[outlier_mask]
            
            return {
                'method': 'isolation_forest',
                'outlier_values': outliers.tolist(),
                'outlier_indices': outliers.index.tolist(),
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'anomaly_scores': outlier_scores.tolist(),
                'parameters': {'contamination': contamination, 'n_estimators': 100}
            }
        except Exception as e:
            return {
                'method': 'isolation_forest',
                'error': str(e),
                'count': 0,
                'percentage': 0.0
            }
    
    def _detect_outliers_lof(self, data: pd.Series, contamination: float) -> Dict[str, Any]:
        """Detect outliers using Local Outlier Factor method."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            # Determine number of neighbors (rule of thumb: sqrt(n))
            n_neighbors = max(2, min(20, int(np.sqrt(len(data)))))
            
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(data.values.reshape(-1, 1))
            
            outlier_mask = outlier_labels == -1
            outliers = data[outlier_mask]
            
            # Get negative outlier factor scores
            lof_scores = lof.negative_outlier_factor_
            outlier_scores = lof_scores[outlier_mask]
            
            return {
                'method': 'lof',
                'outlier_values': outliers.tolist(),
                'outlier_indices': outliers.index.tolist(),
                'count': len(outliers),
                'percentage': (len(outliers) / len(data)) * 100,
                'lof_scores': outlier_scores.tolist(),
                'parameters': {'contamination': contamination, 'n_neighbors': n_neighbors}
            }
        except Exception as e:
            return {
                'method': 'lof',
                'error': str(e),
                'count': 0,
                'percentage': 0.0
            }
    
    def _detect_outliers_percentile(self, data: pd.Series, tail_percentage: float) -> Dict[str, Any]:
        """Detect outliers using percentile-based method."""
        lower_percentile = tail_percentage * 100
        upper_percentile = (1 - tail_percentage) * 100
        
        lower_bound = data.quantile(tail_percentage)
        upper_bound = data.quantile(1 - tail_percentage)
        
        outlier_mask = (data < lower_bound) | (data > upper_bound)
        outliers = data[outlier_mask]
        
        return {
            'method': 'percentile',
            'outlier_values': outliers.tolist(),
            'outlier_indices': outliers.index.tolist(),
            'count': len(outliers),
            'percentage': (len(outliers) / len(data)) * 100,
            'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)},
            'parameters': {
                'tail_percentage': tail_percentage,
                'lower_percentile': lower_percentile,
                'upper_percentile': upper_percentile
            }
        }
    
    def _generate_outlier_summary(self, column_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics across all outlier detection methods."""
        summary = {
            'total_columns_analyzed': len(column_results),
            'methods_comparison': {},
            'consensus_outliers': {},
            'method_agreement': {}
        }
        
        # Compare methods across columns
        method_stats = {}
        for col, methods in column_results.items():
            for method_name, method_result in methods.items():
                if 'error' in method_result:
                    continue
                
                if method_name not in method_stats:
                    method_stats[method_name] = {
                        'total_outliers': 0,
                        'columns_processed': 0,
                        'avg_percentage': 0.0
                    }
                
                method_stats[method_name]['total_outliers'] += method_result.get('count', 0)
                method_stats[method_name]['columns_processed'] += 1
                method_stats[method_name]['avg_percentage'] += method_result.get('percentage', 0.0)
        
        # Calculate averages
        for method_name, stats in method_stats.items():
            if stats['columns_processed'] > 0:
                stats['avg_percentage'] /= stats['columns_processed']
        
        summary['methods_comparison'] = method_stats
        
        # Find consensus outliers (outliers detected by multiple methods)
        for col, methods in column_results.items():
            outlier_indices_by_method = {}
            
            for method_name, method_result in methods.items():
                if 'error' not in method_result and 'outlier_indices' in method_result:
                    outlier_indices_by_method[method_name] = set(method_result['outlier_indices'])
            
            if len(outlier_indices_by_method) > 1:
                # Find indices that appear in multiple methods
                all_indices = set()
                for indices in outlier_indices_by_method.values():
                    all_indices.update(indices)
                
                consensus_counts = {}
                for idx in all_indices:
                    count = sum(1 for indices in outlier_indices_by_method.values() if idx in indices)
                    consensus_counts[idx] = count
                
                # Outliers detected by at least 2 methods
                consensus_outliers = {idx: count for idx, count in consensus_counts.items() if count >= 2}
                
                if consensus_outliers:
                    summary['consensus_outliers'][col] = {
                        'indices': list(consensus_outliers.keys()),
                        'agreement_counts': consensus_outliers,
                        'total_consensus': len(consensus_outliers)
                    }
        
        return summary
    
    def _perform_descriptive_analysis(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform descriptive statistical analysis."""
        results = {
            'analysis_type': 'descriptive',
            'descriptive_stats': self.compute_descriptive_stats(dataframe, columns)
        }
        
        # Add distribution tests if requested
        if parameters.get('include_distribution_tests', False):
            results['distribution_tests'] = self._perform_distribution_tests(dataframe, columns)
        
        return results
    
    def _perform_correlation_analysis(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform correlation analysis."""
        methods = parameters.get('methods', self.correlation_methods)
        correlations = self.calculate_correlations(dataframe, columns, methods)
        
        # Convert DataFrames to dictionaries for serialization
        correlation_dicts = {}
        for method, corr_df in correlations.items():
            correlation_dicts[method] = corr_df.to_dict()
        
        return {
            'analysis_type': 'correlation',
            'correlations': correlation_dicts,
            'methods_used': list(correlations.keys())
        }
    
    def _perform_outlier_detection(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform outlier detection using multiple methods."""
        methods = parameters.get('methods', self.outlier_methods)
        thresholds = parameters.get('thresholds', {})
        
        # Use the enhanced multi-method outlier detection
        outlier_results = self.detect_outliers_multi_method(dataframe, columns, methods, thresholds)
        
        return {
            'analysis_type': 'outliers',
            'outliers': outlier_results['column_results'],
            'methods_used': outlier_results['methods_used'],
            'summary': outlier_results['summary'],
            'thresholds_used': outlier_results['thresholds']
        }
    
    def _perform_aggregation_analysis(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                                    parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform aggregation analysis."""
        methods = parameters.get('methods', self.aggregation_methods)
        group_by = parameters.get('group_by', None)
        
        aggregations = self.perform_aggregations(dataframe, columns, methods, group_by)
        
        return {
            'analysis_type': 'aggregation',
            'aggregations': aggregations,
            'methods_used': methods,
            'grouped_by': group_by
        }
    
    def _perform_full_analysis(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                             parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis."""
        results = {
            'analysis_type': 'full',
            'descriptive_stats': self.compute_descriptive_stats(dataframe, columns),
            'correlations': {},
            'outliers': {},
            'aggregations': {}
        }
        
        # Correlation analysis
        try:
            correlations = self.calculate_correlations(dataframe, columns)
            for method, corr_df in correlations.items():
                results['correlations'][method] = corr_df.to_dict()
        except Exception as e:
            self.logger.warning(f"Correlation analysis failed: {str(e)}")
        
        # Outlier detection
        try:
            outlier_results = self._perform_outlier_detection(dataframe, columns, parameters)
            results['outliers'] = outlier_results['outliers']
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {str(e)}")
        
        # Aggregation analysis
        try:
            agg_results = self._perform_aggregation_analysis(dataframe, columns, parameters)
            results['aggregations'] = agg_results['aggregations']
        except Exception as e:
            self.logger.warning(f"Aggregation analysis failed: {str(e)}")
        
        # Time-series analysis (if applicable)
        try:
            ts_results = self._perform_timeseries_analysis(dataframe, columns, parameters)
            results['timeseries'] = {
                'trend_analysis': ts_results.get('trend_analysis', {}),
                'seasonal_analysis': ts_results.get('seasonal_analysis', {}),
                'time_metrics': ts_results.get('time_metrics', {})
            }
        except Exception as e:
            self.logger.warning(f"Time-series analysis failed: {str(e)}")
        
        return results
    
    def detect_trends(self, time_series: pd.Series, date_column: Optional[pd.Series] = None, 
                     window_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Detect trends in time-series data.
        
        Args:
            time_series: Time series data to analyze
            date_column: Optional datetime column (uses index if not provided)
            window_sizes: Optional list of window sizes for moving averages
        
        Returns:
            Dict containing trend analysis results
        """
        if window_sizes is None:
            window_sizes = self.trend_window_sizes
        
        # Ensure we have a datetime index or column
        if date_column is not None:
            # Create a new series with datetime index
            ts_data = pd.Series(time_series.values, index=pd.to_datetime(date_column))
        else:
            ts_data = time_series.copy()
            if not isinstance(ts_data.index, pd.DatetimeIndex):
                try:
                    ts_data.index = pd.to_datetime(ts_data.index)
                except Exception:
                    # If we can't convert to datetime, create a simple range
                    ts_data.index = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D')
        
        # Sort by date
        ts_data = ts_data.sort_index()
        
        trend_results = {
            'data_points': len(ts_data),
            'date_range': {
                'start': ts_data.index.min().isoformat(),
                'end': ts_data.index.max().isoformat()
            },
            'moving_averages': {},
            'growth_rates': {},
            'trend_direction': None,
            'trend_strength': None
        }
        
        # Calculate moving averages
        for window in window_sizes:
            if len(ts_data) >= window:
                ma = ts_data.rolling(window=window, min_periods=1).mean()
                trend_results['moving_averages'][f'ma_{window}'] = {
                    'values': ma.tolist(),
                    'dates': [d.isoformat() for d in ma.index],
                    'current_value': float(ma.iloc[-1]) if len(ma) > 0 else None,
                    'change_from_start': float(ma.iloc[-1] - ma.iloc[0]) if len(ma) > 0 else None
                }
        
        # Calculate growth rates
        if len(ts_data) > 1:
            # Period-over-period growth rates
            pct_change = ts_data.pct_change().dropna()
            trend_results['growth_rates']['period_over_period'] = {
                'mean': float(pct_change.mean()),
                'std': float(pct_change.std()),
                'min': float(pct_change.min()),
                'max': float(pct_change.max())
            }
            
            # Year-over-year growth (if we have enough data)
            if len(ts_data) >= 365:
                yoy_change = ts_data.pct_change(periods=365).dropna()
                if len(yoy_change) > 0:
                    trend_results['growth_rates']['year_over_year'] = {
                        'mean': float(yoy_change.mean()),
                        'std': float(yoy_change.std()),
                        'current': float(yoy_change.iloc[-1])
                    }
        
        # Overall trend analysis using linear regression
        if len(ts_data) >= 3:
            x = np.arange(len(ts_data))
            y = ts_data.values
            
            # Remove NaN values
            valid_mask = ~np.isnan(y)
            if np.sum(valid_mask) >= 3:
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                
                trend_results['trend_direction'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                trend_results['trend_strength'] = abs(r_value)  # Correlation coefficient as strength measure
                trend_results['linear_regression'] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'p_value': float(p_value),
                    'std_error': float(std_err)
                }
        
        return trend_results
    
    def identify_seasonal_patterns(self, time_series: pd.Series, date_column: Optional[pd.Series] = None,
                                 periods: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Identify seasonal patterns in time-series data.
        
        Args:
            time_series: Time series data to analyze
            date_column: Optional datetime column (uses index if not provided)
            periods: Optional list of seasonal periods to check
        
        Returns:
            Dict containing seasonal pattern analysis
        """
        if periods is None:
            periods = self.seasonal_periods
        
        # Prepare time series with datetime index
        if date_column is not None:
            ts_data = pd.Series(time_series.values, index=pd.to_datetime(date_column))
        else:
            ts_data = time_series.copy()
            if not isinstance(ts_data.index, pd.DatetimeIndex):
                try:
                    ts_data.index = pd.to_datetime(ts_data.index)
                except Exception:
                    ts_data.index = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D')
        
        ts_data = ts_data.sort_index()
        
        seasonal_results = {
            'seasonal_patterns': {},
            'cyclical_components': {},
            'seasonal_strength': {}
        }
        
        # Analyze different seasonal periods
        for period in periods:
            if len(ts_data) >= period * 2:  # Need at least 2 cycles
                try:
                    # Group by seasonal period and calculate statistics
                    ts_data_reset = ts_data.reset_index()
                    ts_data_reset['period_index'] = ts_data_reset.index % period
                    
                    seasonal_stats = ts_data_reset.groupby('period_index')[ts_data.name or 0].agg([
                        'mean', 'std', 'min', 'max', 'count'
                    ])
                    
                    seasonal_results['seasonal_patterns'][f'period_{period}'] = {
                        'mean_by_period': seasonal_stats['mean'].to_dict(),
                        'std_by_period': seasonal_stats['std'].to_dict(),
                        'coefficient_of_variation': (seasonal_stats['std'] / seasonal_stats['mean']).to_dict()
                    }
                    
                    # Calculate seasonal strength (variance between periods vs within periods)
                    between_var = seasonal_stats['mean'].var()
                    within_var = seasonal_stats['std'].mean() ** 2
                    seasonal_strength = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
                    
                    seasonal_results['seasonal_strength'][f'period_{period}'] = float(seasonal_strength)
                    
                except Exception as e:
                    self.logger.warning(f"Seasonal analysis failed for period {period}: {str(e)}")
        
        # Day of week analysis (if we have daily data)
        if isinstance(ts_data.index, pd.DatetimeIndex) and len(ts_data) >= 14:
            try:
                dow_stats = ts_data.groupby(ts_data.index.dayofweek).agg(['mean', 'std', 'count'])
                dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                seasonal_results['day_of_week'] = {
                    dow_names[i]: {
                        'mean': float(dow_stats.iloc[i]['mean']) if i < len(dow_stats) else None,
                        'std': float(dow_stats.iloc[i]['std']) if i < len(dow_stats) else None,
                        'count': int(dow_stats.iloc[i]['count']) if i < len(dow_stats) else 0
                    }
                    for i in range(7)
                }
            except Exception as e:
                self.logger.warning(f"Day of week analysis failed: {str(e)}")
        
        # Month analysis (if we have monthly data)
        if isinstance(ts_data.index, pd.DatetimeIndex) and len(ts_data) >= 24:
            try:
                month_stats = ts_data.groupby(ts_data.index.month).agg(['mean', 'std', 'count'])
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                seasonal_results['month_of_year'] = {
                    month_names[i-1]: {
                        'mean': float(month_stats.iloc[i-1]['mean']) if i <= len(month_stats) else None,
                        'std': float(month_stats.iloc[i-1]['std']) if i <= len(month_stats) else None,
                        'count': int(month_stats.iloc[i-1]['count']) if i <= len(month_stats) else 0
                    }
                    for i in range(1, 13)
                }
            except Exception as e:
                self.logger.warning(f"Month analysis failed: {str(e)}")
        
        return seasonal_results
    
    def calculate_time_based_metrics(self, time_series: pd.Series, date_column: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate various time-based metrics for the series.
        
        Args:
            time_series: Time series data to analyze
            date_column: Optional datetime column (uses index if not provided)
        
        Returns:
            Dict containing time-based metrics
        """
        # Prepare time series with datetime index
        if date_column is not None:
            ts_data = pd.Series(time_series.values, index=pd.to_datetime(date_column))
        else:
            ts_data = time_series.copy()
            if not isinstance(ts_data.index, pd.DatetimeIndex):
                try:
                    ts_data.index = pd.to_datetime(ts_data.index)
                except Exception:
                    ts_data.index = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D')
        
        ts_data = ts_data.sort_index()
        
        metrics = {
            'basic_metrics': {
                'total_observations': len(ts_data),
                'non_null_observations': ts_data.count(),
                'date_range_days': (ts_data.index.max() - ts_data.index.min()).days,
                'frequency_estimate': self._estimate_frequency(ts_data.index)
            },
            'volatility_metrics': {},
            'momentum_metrics': {},
            'autocorrelation': {}
        }
        
        # Volatility metrics
        if len(ts_data) > 1:
            returns = ts_data.pct_change().dropna()
            if len(returns) > 0:
                metrics['volatility_metrics'] = {
                    'volatility': float(returns.std()),
                    'annualized_volatility': float(returns.std() * np.sqrt(252)),  # Assuming daily data
                    'max_drawdown': self._calculate_max_drawdown(ts_data),
                    'value_at_risk_5pct': float(returns.quantile(0.05)),
                    'value_at_risk_1pct': float(returns.quantile(0.01))
                }
        
        # Momentum metrics
        if len(ts_data) >= 10:
            # Calculate momentum over different periods
            for period in [5, 10, 20]:
                if len(ts_data) >= period:
                    momentum = (ts_data.iloc[-1] - ts_data.iloc[-period]) / ts_data.iloc[-period]
                    metrics['momentum_metrics'][f'momentum_{period}d'] = float(momentum)
        
        # Autocorrelation analysis
        if len(ts_data) >= 10:
            try:
                # Calculate autocorrelation for different lags
                autocorr_results = {}
                max_lag = min(20, len(ts_data) // 4)
                
                for lag in range(1, max_lag + 1):
                    autocorr = ts_data.autocorr(lag=lag)
                    if not np.isnan(autocorr):
                        autocorr_results[f'lag_{lag}'] = float(autocorr)
                
                metrics['autocorrelation'] = autocorr_results
                
            except Exception as e:
                self.logger.warning(f"Autocorrelation analysis failed: {str(e)}")
        
        return metrics
    
    def _perform_timeseries_analysis(self, dataframe: pd.DataFrame, columns: Optional[List[str]], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive time-series analysis."""
        date_column = parameters.get('date_column', None)
        
        if columns is None:
            columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            'analysis_type': 'timeseries',
            'columns_analyzed': [],
            'trend_analysis': {},
            'seasonal_analysis': {},
            'time_metrics': {}
        }
        
        # Determine date column
        date_series = None
        if date_column and date_column in dataframe.columns:
            date_series = dataframe[date_column]
        elif dataframe.index.dtype.kind in ['M', 'datetime64']:  # DateTime index
            date_series = None  # Will use index
        else:
            # Try to find a datetime column automatically
            for col in dataframe.columns:
                if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                    date_series = dataframe[col]
                    self.logger.info(f"Automatically detected datetime column: {col}")
                    break
        
        # Analyze each numerical column
        for col in columns:
            if col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[col]):
                continue
            
            col_data = dataframe[col].dropna()
            if len(col_data) < 3:  # Need minimum data for time series analysis
                continue
            
            results['columns_analyzed'].append(col)
            
            try:
                # Trend analysis
                trend_results = self.detect_trends(col_data, date_series)
                results['trend_analysis'][col] = trend_results
                
                # Seasonal analysis
                seasonal_results = self.identify_seasonal_patterns(col_data, date_series)
                results['seasonal_analysis'][col] = seasonal_results
                
                # Time-based metrics
                time_metrics = self.calculate_time_based_metrics(col_data, date_series)
                results['time_metrics'][col] = time_metrics
                
            except Exception as e:
                self.logger.error(f"Time-series analysis failed for column {col}: {str(e)}")
        
        return results
    
    def _estimate_frequency(self, datetime_index: pd.DatetimeIndex) -> str:
        """Estimate the frequency of a datetime index."""
        if len(datetime_index) < 2:
            return 'unknown'
        
        try:
            # Calculate the most common time difference
            diffs = datetime_index[1:] - datetime_index[:-1]
            mode_diff = diffs.mode()[0] if len(diffs.mode()) > 0 else diffs[0]
            
            # Convert to frequency string
            if mode_diff <= pd.Timedelta(hours=1):
                return 'hourly_or_less'
            elif mode_diff <= pd.Timedelta(days=1):
                return 'daily'
            elif mode_diff <= pd.Timedelta(days=7):
                return 'weekly'
            elif mode_diff <= pd.Timedelta(days=31):
                return 'monthly'
            elif mode_diff <= pd.Timedelta(days=365):
                return 'yearly'
            else:
                return 'irregular'
        except Exception:
            return 'unknown'
    
    def _calculate_max_drawdown(self, time_series: pd.Series) -> float:
        """Calculate maximum drawdown for a time series."""
        try:
            # Calculate cumulative maximum
            cummax = time_series.expanding().max()
            # Calculate drawdown
            drawdown = (time_series - cummax) / cummax
            # Return maximum drawdown (most negative value)
            return float(drawdown.min())
        except Exception:
            return 0.0
    
    def _perform_distribution_tests(self, dataframe: pd.DataFrame, columns: Optional[List[str]]) -> Dict[str, Dict[str, float]]:
        """Perform statistical distribution tests."""
        if columns is None:
            columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
        
        distribution_tests = {}
        
        for col in columns:
            if col not in dataframe.columns or not pd.api.types.is_numeric_dtype(dataframe[col]):
                continue
            
            col_data = dataframe[col].dropna()
            if len(col_data) < 8:  # Need sufficient data for tests
                continue
            
            tests = {}
            
            try:
                # Shapiro-Wilk test for normality
                if len(col_data) <= 5000:  # Shapiro-Wilk has sample size limitations
                    shapiro_stat, shapiro_p = stats.shapiro(col_data)
                    tests['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                
                # Kolmogorov-Smirnov test for normality
                ks_stat, ks_p = stats.kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                tests['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p}
                
                # D'Agostino's normality test
                if len(col_data) >= 20:
                    dagostino_stat, dagostino_p = stats.normaltest(col_data)
                    tests['dagostino'] = {'statistic': dagostino_stat, 'p_value': dagostino_p}
                
                distribution_tests[col] = tests
                
            except Exception as e:
                self.logger.warning(f"Distribution tests failed for column {col}: {str(e)}")
        
        return distribution_tests