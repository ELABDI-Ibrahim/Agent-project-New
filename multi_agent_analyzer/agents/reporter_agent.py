"""
Reporter agent for the multi-agent data analyzer system.

This module provides the ReporterAgent class that generates comprehensive reports
by synthesizing findings from other agents into structured, human-readable formats.
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

import pandas as pd
import numpy as np

from .base_agent import BaseAgent, AgentResult
from ..core.shared_context import SharedContext
from ..core.llm_coordinator import LLMCoordinator


class ReportSection(Enum):
    """Enumeration of report sections."""
    EXECUTIVE_SUMMARY = "executive_summary"
    METHODOLOGY = "methodology"
    DATA_OVERVIEW = "data_overview"
    KEY_FINDINGS = "key_findings"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    RECOMMENDATIONS = "recommendations"
    RISK_ASSESSMENT = "risk_assessment"
    VISUALIZATIONS = "visualizations"
    TECHNICAL_DETAILS = "technical_details"
    APPENDICES = "appendices"


class ExportFormat(Enum):
    """Supported export formats."""
    TEXT = "text"
    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"


class ReporterAgent(BaseAgent):
    """
    Agent responsible for generating comprehensive analytical reports.
    
    This agent:
    - Synthesizes findings from Collector, Analyzer, and Decision agents
    - Generates executive summaries with key insights
    - Creates structured reports with multiple sections
    - Translates technical terminology into business language
    - Generates terminal-friendly visualizations
    - Exports reports in multiple formats
    """
    
    def __init__(self, name: str, llm_coordinator: LLMCoordinator, shared_context: SharedContext):
        """
        Initialize the Reporter Agent.
        
        Args:
            name: Agent name/identifier
            llm_coordinator: LLM coordinator for language generation
            shared_context: Shared context for data access
        """
        super().__init__(name, llm_coordinator, shared_context)
        
        # Report generation settings
        self.max_summary_length = 500
        self.max_findings_count = 10
        self.visualization_width = 80
        
        # Technical glossary for term translation
        self.technical_glossary = {
            "correlation": "statistical relationship between two variables",
            "outlier": "data point that differs significantly from other observations",
            "standard deviation": "measure of variability or spread in data",
            "percentile": "value below which a percentage of data falls",
            "trend": "general direction in which data is moving over time",
            "seasonality": "regular pattern of changes that repeats over time",
            "anomaly": "unusual pattern that does not conform to expected behavior",
            "feature importance": "measure of how much each variable contributes to predictions",
            "clustering": "grouping similar data points together",
            "confidence interval": "range of values likely to contain the true value"
        }
        
        self.logger.info(f"ReporterAgent {name} initialized")
    
    def _initialize_agent(self) -> None:
        """Initialize agent-specific components."""
        self.logger.info(f"ReporterAgent {self.name} performing initialization")
        # No additional initialization needed
    
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Generate comprehensive report from analysis results.
        
        Args:
            input_data: Dictionary containing:
                - collector_results: Results from Collector Agent
                - analyzer_results: Results from Analyzer Agent
                - decision_results: Results from Decision Agent
                - report_config: Optional configuration for report generation
        
        Returns:
            AgentResult: Generated report and metadata
        """
        self._log_processing_start("report generation", input_data)
        
        try:
            # Validate input data
            validation = self.validate_input(
                input_data,
                required_keys=["collector_results", "analyzer_results", "decision_results"]
            )
            
            if not validation.is_valid:
                return AgentResult(
                    success=False,
                    errors=[f"Invalid input: {', '.join(validation.errors)}"]
                )
            
            # Extract results from other agents
            collector_results = input_data["collector_results"]
            analyzer_results = input_data["analyzer_results"]
            decision_results = input_data["decision_results"]
            report_config = input_data.get("report_config", {})
            
            # Generate report components
            executive_summary = self._generate_executive_summary(
                collector_results, analyzer_results, decision_results
            )
            
            methodology = self._generate_methodology_section(
                collector_results, analyzer_results
            )
            
            data_overview = self._generate_data_overview(collector_results)
            
            key_findings = self._generate_key_findings(
                analyzer_results, decision_results
            )
            
            statistical_analysis = self._generate_statistical_analysis(
                analyzer_results
            )
            
            recommendations = self._generate_recommendations_section(
                decision_results
            )
            
            risk_assessment = self._generate_risk_assessment(decision_results)
            
            visualizations = self._generate_visualizations(
                collector_results, analyzer_results
            )
            
            technical_details = self._generate_technical_details(
                analyzer_results, decision_results
            )
            
            # Assemble complete report
            report = {
                "title": report_config.get("title", "Data Analysis Report"),
                "generated_at": datetime.now().isoformat(),
                "executive_summary": executive_summary,
                "methodology": methodology,
                "data_overview": data_overview,
                "key_findings": key_findings,
                "statistical_analysis": statistical_analysis,
                "recommendations": recommendations,
                "risk_assessment": risk_assessment,
                "visualizations": visualizations,
                "technical_details": technical_details,
                "metadata": {
                    "data_source": collector_results.get("file_path", "Unknown"),
                    "rows_analyzed": collector_results.get("row_count", 0),
                    "columns_analyzed": collector_results.get("column_count", 0),
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            
            # Store report in shared context
            self.shared_context.store_data("final_report", report, self.name)
            
            # Generate formatted outputs
            formatted_reports = self._generate_formatted_outputs(
                report, report_config
            )
            
            result = AgentResult(
                success=True,
                data={
                    "report": report,
                    "formatted_reports": formatted_reports,
                    "sections_generated": len(report) - 2  # Exclude title and metadata
                }
            )
            
            self._log_processing_end("report generation", result)
            return result
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            return AgentResult(
                success=False,
                errors=[f"Report generation failed: {str(e)}"]
            )
    
    def _generate_executive_summary(
        self,
        collector_results: Dict[str, Any],
        analyzer_results: Dict[str, Any],
        decision_results: Dict[str, Any]
    ) -> str:
        """
        Generate executive summary of analysis.
        
        Args:
            collector_results: Data collection results
            analyzer_results: Statistical analysis results
            decision_results: Recommendations and decisions
        
        Returns:
            str: Executive summary text
        """
        try:
            # Extract key information
            row_count = collector_results.get("row_count", 0)
            column_count = collector_results.get("column_count", 0)
            
            # Get top findings
            anomaly_count = len(analyzer_results.get("outliers", {}).get("all_outliers", []))
            
            # Get top recommendations
            recommendations = decision_results.get("recommendations", [])
            high_priority_count = sum(
                1 for rec in recommendations 
                if rec.get("priority") == "HIGH"
            )
            
            # Use LLM to generate natural language summary
            context = {
                "row_count": row_count,
                "column_count": column_count,
                "anomaly_count": anomaly_count,
                "high_priority_recommendations": high_priority_count,
                "total_recommendations": len(recommendations)
            }
            
            summary_prompt = f"""
            Generate a concise executive summary (max 3 paragraphs) for a data analysis report with:
            - Dataset: {row_count} rows, {column_count} columns
            - {anomaly_count} anomalies detected
            - {len(recommendations)} recommendations generated ({high_priority_count} high priority)
            
            Focus on business impact and actionable insights.
            """
            
            summary = self.request_llm_assistance(summary_prompt, context)
            
            # Ensure summary is not too long
            if len(summary) > self.max_summary_length:
                summary = summary[:self.max_summary_length] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate executive summary: {str(e)}")
            return "Executive summary generation failed. Please refer to detailed sections below."
    
    def _generate_methodology_section(
        self,
        collector_results: Dict[str, Any],
        analyzer_results: Dict[str, Any]
    ) -> str:
        """
        Generate methodology section describing analysis approach.
        
        Args:
            collector_results: Data collection results
            analyzer_results: Statistical analysis results
        
        Returns:
            str: Methodology description
        """
        methodology_parts = []
        
        # Data collection methodology
        methodology_parts.append("Data Collection and Preprocessing:")
        methodology_parts.append(
            f"- Data validation and quality assessment performed on CSV input"
        )
        
        if collector_results.get("missing_values"):
            methodology_parts.append(
                f"- Missing value detection and handling applied"
            )
        
        methodology_parts.append("")
        
        # Statistical analysis methodology
        methodology_parts.append("Statistical Analysis:")
        
        if analyzer_results.get("descriptive_stats"):
            methodology_parts.append(
                "- Descriptive statistics computed (mean, median, std, etc.)"
            )
        
        if analyzer_results.get("correlations"):
            methodology_parts.append(
                "- Correlation analysis performed between numerical variables"
            )
        
        if analyzer_results.get("time_series_analysis"):
            methodology_parts.append(
                "- Time-series trend detection and seasonal pattern analysis"
            )
        
        outlier_methods = analyzer_results.get("outlier_methods", [])
        if outlier_methods:
            methods_str = ", ".join(outlier_methods)
            methodology_parts.append(
                f"- Outlier detection using methods: {methods_str}"
            )
        
        if analyzer_results.get("clustering_results"):
            methodology_parts.append(
                "- Clustering analysis to identify data groupings"
            )
        
        return "\n".join(methodology_parts)
    
    def _generate_data_overview(self, collector_results: Dict[str, Any]) -> str:
        """
        Generate data overview section.
        
        Args:
            collector_results: Data collection results
        
        Returns:
            str: Data overview text
        """
        overview_parts = []
        
        # Basic statistics
        row_count = collector_results.get("row_count", 0)
        column_count = collector_results.get("column_count", 0)
        
        overview_parts.append(f"Dataset Dimensions: {row_count:,} rows × {column_count} columns")
        overview_parts.append("")
        
        # Data types
        data_types = collector_results.get("data_types", {})
        if data_types:
            overview_parts.append("Column Data Types:")
            type_counts = {}
            for col, dtype in data_types.items():
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            for dtype, count in sorted(type_counts.items()):
                overview_parts.append(f"  - {dtype}: {count} columns")
            overview_parts.append("")
        
        # Data quality
        missing_values = collector_results.get("missing_values", {})
        if missing_values:
            total_missing = sum(missing_values.values())
            overview_parts.append(f"Data Quality: {total_missing:,} missing values detected")
            
            # Show columns with most missing values
            sorted_missing = sorted(
                missing_values.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            if sorted_missing:
                overview_parts.append("  Columns with most missing values:")
                for col, count in sorted_missing:
                    pct = (count / row_count * 100) if row_count > 0 else 0
                    overview_parts.append(f"    - {col}: {count} ({pct:.1f}%)")
        
        return "\n".join(overview_parts)
    
    def _generate_key_findings(
        self,
        analyzer_results: Dict[str, Any],
        decision_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate list of key findings from analysis.
        
        Args:
            analyzer_results: Statistical analysis results
            decision_results: Recommendations and decisions
        
        Returns:
            List[str]: Key findings
        """
        findings = []
        
        # Outlier findings
        outliers = analyzer_results.get("outliers", {})
        if outliers:
            all_outliers = outliers.get("all_outliers", [])
            if all_outliers:
                findings.append(
                    f"Detected {len(all_outliers)} anomalous data points requiring investigation"
                )
        
        # Correlation findings
        correlations = analyzer_results.get("correlations", {})
        if correlations and "strong_correlations" in correlations:
            strong_corrs = correlations["strong_correlations"]
            if strong_corrs:
                findings.append(
                    f"Identified {len(strong_corrs)} strong correlations between variables"
                )
        
        # Trend findings
        time_series = analyzer_results.get("time_series_analysis", {})
        if time_series:
            for metric, analysis in time_series.items():
                if analysis.get("trend_direction"):
                    direction = analysis["trend_direction"]
                    findings.append(
                        f"{metric} shows {direction} trend over time"
                    )
        
        # Clustering findings
        clustering = analyzer_results.get("clustering_results", {})
        if clustering:
            n_clusters = clustering.get("n_clusters", 0)
            if n_clusters > 0:
                findings.append(
                    f"Data naturally groups into {n_clusters} distinct clusters"
                )
        
        # Feature importance findings
        feature_importance = analyzer_results.get("feature_importance", {})
        if feature_importance:
            top_features = feature_importance.get("top_features", [])
            if top_features:
                top_feature_name = top_features[0].get("feature", "Unknown")
                findings.append(
                    f"{top_feature_name} identified as most significant predictor"
                )
        
        # Priority findings from decisions
        recommendations = decision_results.get("recommendations", [])
        high_priority = [r for r in recommendations if r.get("priority") == "HIGH"]
        if high_priority:
            findings.append(
                f"{len(high_priority)} high-priority maintenance actions recommended"
            )
        
        # Limit to max findings
        return findings[:self.max_findings_count]
    
    def _generate_statistical_analysis(self, analyzer_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed statistical analysis section.
        
        Args:
            analyzer_results: Statistical analysis results
        
        Returns:
            Dict[str, Any]: Structured statistical analysis
        """
        analysis = {}
        
        # Descriptive statistics
        desc_stats = analyzer_results.get("descriptive_stats", {})
        if desc_stats:
            analysis["descriptive_statistics"] = self._format_descriptive_stats(desc_stats)
        
        # Correlation analysis
        correlations = analyzer_results.get("correlations", {})
        if correlations:
            analysis["correlation_analysis"] = self._format_correlations(correlations)
        
        # Outlier analysis
        outliers = analyzer_results.get("outliers", {})
        if outliers:
            analysis["outlier_analysis"] = self._format_outliers(outliers)
        
        # Time-series analysis
        time_series = analyzer_results.get("time_series_analysis", {})
        if time_series:
            analysis["time_series_analysis"] = self._format_time_series(time_series)
        
        # Clustering analysis
        clustering = analyzer_results.get("clustering_results", {})
        if clustering:
            analysis["clustering_analysis"] = self._format_clustering(clustering)
        
        return analysis
    
    def _format_descriptive_stats(self, desc_stats: Dict[str, Any]) -> str:
        """Format descriptive statistics for report."""
        lines = []
        
        for column, stats in desc_stats.items():
            lines.append(f"\n{column}:")
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {stat_name}: {value:.4f}")
                else:
                    lines.append(f"  {stat_name}: {value}")
        
        return "\n".join(lines)
    
    def _format_correlations(self, correlations: Dict[str, Any]) -> str:
        """Format correlation analysis for report."""
        lines = []
        
        strong_corrs = correlations.get("strong_correlations", [])
        if strong_corrs:
            lines.append("Strong Correlations Detected:")
            for corr in strong_corrs[:10]:  # Top 10
                col1 = corr.get("column1", "")
                col2 = corr.get("column2", "")
                value = corr.get("correlation", 0)
                lines.append(f"  {col1} ↔ {col2}: {value:.3f}")
        
        return "\n".join(lines) if lines else "No strong correlations detected"
    
    def _format_outliers(self, outliers: Dict[str, Any]) -> str:
        """Format outlier analysis for report."""
        lines = []
        
        all_outliers = outliers.get("all_outliers", [])
        if all_outliers:
            lines.append(f"Total Outliers Detected: {len(all_outliers)}")
            
            # Group by column
            by_column = {}
            for outlier in all_outliers:
                col = outlier.get("column", "Unknown")
                by_column[col] = by_column.get(col, 0) + 1
            
            lines.append("\nOutliers by Column:")
            for col, count in sorted(by_column.items(), key=lambda x: x[1], reverse=True)[:10]:
                lines.append(f"  {col}: {count} outliers")
        
        return "\n".join(lines) if lines else "No outliers detected"
    
    def _format_time_series(self, time_series: Dict[str, Any]) -> str:
        """Format time-series analysis for report."""
        lines = []
        
        for metric, analysis in time_series.items():
            lines.append(f"\n{metric}:")
            
            if "trend_direction" in analysis:
                lines.append(f"  Trend: {analysis['trend_direction']}")
            
            if "seasonality_detected" in analysis:
                if analysis["seasonality_detected"]:
                    lines.append("  Seasonality: Detected")
                else:
                    lines.append("  Seasonality: Not detected")
            
            if "growth_rate" in analysis:
                rate = analysis["growth_rate"]
                lines.append(f"  Growth Rate: {rate:.2f}%")
        
        return "\n".join(lines) if lines else "No time-series analysis performed"
    
    def _format_clustering(self, clustering: Dict[str, Any]) -> str:
        """Format clustering analysis for report."""
        lines = []
        
        n_clusters = clustering.get("n_clusters", 0)
        lines.append(f"Number of Clusters: {n_clusters}")
        
        cluster_sizes = clustering.get("cluster_sizes", {})
        if cluster_sizes:
            lines.append("\nCluster Distribution:")
            for cluster_id, size in cluster_sizes.items():
                lines.append(f"  Cluster {cluster_id}: {size} data points")
        
        return "\n".join(lines)
    
    def _generate_recommendations_section(self, decision_results: Dict[str, Any]) -> str:
        """
        Generate recommendations section.
        
        Args:
            decision_results: Recommendations and decisions
        
        Returns:
            str: Formatted recommendations
        """
        lines = []
        
        recommendations = decision_results.get("recommendations", [])
        
        if not recommendations:
            return "No specific recommendations generated."
        
        # Group by priority (using string values instead of enum)
        by_priority = {"HIGH": [], "MEDIUM": [], "LOW": []}
        for rec in recommendations:
            priority = rec.get("priority", "MEDIUM")
            by_priority[priority].append(rec)
        
        # Format each priority group
        for priority in ["HIGH", "MEDIUM", "LOW"]:
            recs = by_priority[priority]
            if recs:
                lines.append(f"\n{priority} Priority Recommendations:")
                for i, rec in enumerate(recs, 1):
                    lines.append(f"\n{i}. {rec.get('title', 'Untitled')}")
                    lines.append(f"   {rec.get('description', 'No description')}")
                    
                    if "estimated_impact" in rec:
                        lines.append(f"   Expected Impact: {rec['estimated_impact']}")
                    
                    if "timeline" in rec:
                        lines.append(f"   Timeline: {rec['timeline']}")
        
        return "\n".join(lines)
    
    def _generate_risk_assessment(self, decision_results: Dict[str, Any]) -> str:
        """
        Generate risk assessment section.
        
        Args:
            decision_results: Recommendations and decisions
        
        Returns:
            str: Risk assessment text
        """
        lines = []
        
        recommendations = decision_results.get("recommendations", [])
        
        # Count by risk level
        risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for rec in recommendations:
            risk_level = rec.get("risk_level", "MEDIUM")
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        lines.append("Risk Distribution:")
        lines.append(f"  High Risk: {risk_counts['HIGH']} items")
        lines.append(f"  Medium Risk: {risk_counts['MEDIUM']} items")
        lines.append(f"  Low Risk: {risk_counts['LOW']} items")
        
        # Highlight high risk items
        high_risk = [r for r in recommendations if r.get("risk_level") == "HIGH"]
        if high_risk:
            lines.append("\nHigh Risk Items Requiring Immediate Attention:")
            for rec in high_risk[:5]:  # Top 5
                lines.append(f"  - {rec.get('title', 'Untitled')}")
        
        return "\n".join(lines)
    
    def _generate_visualizations(
        self,
        collector_results: Dict[str, Any],
        analyzer_results: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate terminal-friendly visualizations.
        
        Args:
            collector_results: Data collection results
            analyzer_results: Statistical analysis results
        
        Returns:
            Dict[str, str]: Visualization strings
        """
        visualizations = {}
        
        # Distribution histogram (ASCII)
        desc_stats = analyzer_results.get("descriptive_stats", {})
        if desc_stats:
            for column, stats in list(desc_stats.items())[:3]:  # First 3 columns
                if all(k in stats for k in ["min", "max", "mean"]):
                    hist = self._create_ascii_histogram(
                        column,
                        stats["min"],
                        stats["max"],
                        stats["mean"]
                    )
                    visualizations[f"distribution_{column}"] = hist
        
        # Correlation heatmap (ASCII)
        correlations = analyzer_results.get("correlations", {})
        if correlations and "matrix" in correlations:
            heatmap = self._create_correlation_heatmap(correlations["matrix"])
            visualizations["correlation_heatmap"] = heatmap
        
        # Trend chart (ASCII)
        time_series = analyzer_results.get("time_series_analysis", {})
        if time_series:
            for metric, analysis in list(time_series.items())[:2]:  # First 2 metrics
                if "values" in analysis:
                    chart = self._create_trend_chart(metric, analysis["values"])
                    visualizations[f"trend_{metric}"] = chart
        
        return visualizations
    
    def _create_ascii_histogram(
        self,
        column: str,
        min_val: float,
        max_val: float,
        mean_val: float
    ) -> str:
        """Create ASCII histogram visualization."""
        width = self.visualization_width
        
        lines = [f"\nDistribution of {column}:"]
        lines.append("=" * width)
        
        # Create simple bar chart
        range_val = max_val - min_val
        if range_val > 0:
            mean_pos = int(((mean_val - min_val) / range_val) * (width - 10))
            
            lines.append(f"Min: {min_val:.2f}")
            lines.append(" " * mean_pos + "▼ Mean")
            lines.append("|" + "-" * (width - 2) + "|")
            lines.append(f"Max: {max_val:.2f}")
        
        return "\n".join(lines)
    
    def _create_correlation_heatmap(self, matrix: Dict[str, Any]) -> str:
        """Create ASCII correlation heatmap."""
        lines = ["\nCorrelation Heatmap:"]
        lines.append("=" * self.visualization_width)
        
        # Simple text representation
        lines.append("Legend: ++ (strong positive), + (positive), 0 (neutral), - (negative), -- (strong negative)")
        
        if isinstance(matrix, dict):
            for col1, correlations in list(matrix.items())[:5]:  # First 5 columns
                line_parts = [f"{col1[:20]:20}"]
                for col2, value in list(correlations.items())[:5]:
                    if isinstance(value, (int, float)):
                        if value > 0.7:
                            symbol = "++"
                        elif value > 0.3:
                            symbol = "+ "
                        elif value < -0.7:
                            symbol = "--"
                        elif value < -0.3:
                            symbol = "- "
                        else:
                            symbol = "0 "
                        line_parts.append(symbol)
                lines.append(" ".join(line_parts))
        
        return "\n".join(lines)
    
    def _create_trend_chart(self, metric: str, values: List[float]) -> str:
        """Create ASCII trend chart."""
        lines = [f"\nTrend Chart: {metric}"]
        lines.append("=" * self.visualization_width)
        
        if not values or len(values) < 2:
            lines.append("Insufficient data for trend visualization")
            return "\n".join(lines)
        
        # Normalize values to chart height
        chart_height = 10
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val
        
        if range_val == 0:
            lines.append("No variation in data")
            return "\n".join(lines)
        
        # Create simple line chart
        normalized = [
            int(((v - min_val) / range_val) * (chart_height - 1))
            for v in values[:self.visualization_width]
        ]
        
        # Draw chart
        for row in range(chart_height - 1, -1, -1):
            line = ""
            for val in normalized:
                if val >= row:
                    line += "█"
                else:
                    line += " "
            lines.append(line)
        
        lines.append("-" * len(normalized))
        lines.append(f"Min: {min_val:.2f}  Max: {max_val:.2f}")
        
        return "\n".join(lines)
    
    def _generate_technical_details(
        self,
        analyzer_results: Dict[str, Any],
        decision_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate technical details appendix.
        
        Args:
            analyzer_results: Statistical analysis results
            decision_results: Recommendations and decisions
        
        Returns:
            Dict[str, Any]: Technical details
        """
        details = {}
        
        # Analysis parameters
        details["analysis_parameters"] = {
            "outlier_methods": analyzer_results.get("outlier_methods", []),
            "correlation_method": analyzer_results.get("correlation_method", "pearson"),
            "clustering_algorithm": analyzer_results.get("clustering_algorithm", "kmeans")
        }
        
        # Raw statistics
        details["raw_statistics"] = analyzer_results.get("descriptive_stats", {})
        
        # Decision criteria
        details["decision_criteria"] = {
            "priority_thresholds": decision_results.get("priority_thresholds", {}),
            "risk_assessment_method": decision_results.get("risk_method", "rule_based")
        }
        
        return details
    
    def _generate_formatted_outputs(
        self,
        report: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate formatted report outputs in multiple formats.
        
        Args:
            report: Complete report data
            config: Report configuration
        
        Returns:
            Dict[str, str]: Formatted reports by format
        """
        formats = config.get("export_formats", ["text", "markdown"])
        outputs = {}
        
        for fmt in formats:
            if fmt == "text":
                outputs["text"] = self._format_as_text(report)
            elif fmt == "markdown":
                outputs["markdown"] = self._format_as_markdown(report)
            elif fmt == "json":
                outputs["json"] = json.dumps(report, indent=2, default=str)
            elif fmt == "html":
                outputs["html"] = self._format_as_html(report)
        
        return outputs
    
    def _format_as_text(self, report: Dict[str, Any]) -> str:
        """Format report as plain text."""
        lines = []
        
        # Title and header
        title = report.get("title", "Data Analysis Report")
        lines.append("=" * 80)
        lines.append(title.center(80))
        lines.append("=" * 80)
        lines.append(f"\nGenerated: {report.get('generated_at', 'N/A')}")
        lines.append("\n")
        
        # Executive Summary
        lines.append("-" * 80)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(report.get("executive_summary", "N/A"))
        lines.append("\n")
        
        # Data Overview
        lines.append("-" * 80)
        lines.append("DATA OVERVIEW")
        lines.append("-" * 80)
        lines.append(report.get("data_overview", "N/A"))
        lines.append("\n")
        
        # Key Findings
        lines.append("-" * 80)
        lines.append("KEY FINDINGS")
        lines.append("-" * 80)
        findings = report.get("key_findings", [])
        for i, finding in enumerate(findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("\n")
        
        # Recommendations
        lines.append("-" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        lines.append(report.get("recommendations", "N/A"))
        lines.append("\n")
        
        # Risk Assessment
        lines.append("-" * 80)
        lines.append("RISK ASSESSMENT")
        lines.append("-" * 80)
        lines.append(report.get("risk_assessment", "N/A"))
        lines.append("\n")
        
        # Visualizations
        visualizations = report.get("visualizations", {})
        if visualizations:
            lines.append("-" * 80)
            lines.append("VISUALIZATIONS")
            lines.append("-" * 80)
            for viz_name, viz_content in visualizations.items():
                lines.append(viz_content)
                lines.append("\n")
        
        # Methodology
        lines.append("-" * 80)
        lines.append("METHODOLOGY")
        lines.append("-" * 80)
        lines.append(report.get("methodology", "N/A"))
        lines.append("\n")
        
        # Metadata
        metadata = report.get("metadata", {})
        lines.append("-" * 80)
        lines.append("REPORT METADATA")
        lines.append("-" * 80)
        for key, value in metadata.items():
            lines.append(f"{key}: {value}")
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _format_as_markdown(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown."""
        lines = []
        
        # Title
        title = report.get("title", "Data Analysis Report")
        lines.append(f"# {title}\n")
        lines.append(f"*Generated: {report.get('generated_at', 'N/A')}*\n")
        lines.append("---\n")
        
        # Executive Summary
        lines.append("## Executive Summary\n")
        lines.append(report.get("executive_summary", "N/A"))
        lines.append("\n")
        
        # Data Overview
        lines.append("## Data Overview\n")
        lines.append(report.get("data_overview", "N/A"))
        lines.append("\n")
        
        # Key Findings
        lines.append("## Key Findings\n")
        findings = report.get("key_findings", [])
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("\n")
        
        # Statistical Analysis
        lines.append("## Statistical Analysis\n")
        stat_analysis = report.get("statistical_analysis", {})
        for section, content in stat_analysis.items():
            lines.append(f"### {section.replace('_', ' ').title()}\n")
            if isinstance(content, str):
                lines.append(f"```\n{content}\n```\n")
            else:
                lines.append(f"{content}\n")
        
        # Recommendations
        lines.append("## Recommendations\n")
        lines.append(report.get("recommendations", "N/A"))
        lines.append("\n")
        
        # Risk Assessment
        lines.append("## Risk Assessment\n")
        lines.append(report.get("risk_assessment", "N/A"))
        lines.append("\n")
        
        # Visualizations
        visualizations = report.get("visualizations", {})
        if visualizations:
            lines.append("## Visualizations\n")
            for viz_name, viz_content in visualizations.items():
                lines.append(f"### {viz_name.replace('_', ' ').title()}\n")
                lines.append(f"```\n{viz_content}\n```\n")
        
        # Methodology
        lines.append("## Methodology\n")
        lines.append(report.get("methodology", "N/A"))
        lines.append("\n")
        
        # Technical Details
        lines.append("## Technical Details\n")
        tech_details = report.get("technical_details", {})
        lines.append("```json\n")
        lines.append(json.dumps(tech_details, indent=2, default=str))
        lines.append("\n```\n")
        
        # Metadata
        metadata = report.get("metadata", {})
        lines.append("---\n")
        lines.append("### Report Metadata\n")
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        
        return "\n".join(lines)
    
    def _format_as_html(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        title = report.get("title", "Data Analysis Report")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .metadata {{
            font-size: 0.9em;
            color: #7f8c8d;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        ul {{
            padding-left: 20px;
        }}
        .finding {{
            padding: 10px;
            margin: 5px 0;
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
        }}
        .recommendation {{
            padding: 10px;
            margin: 10px 0;
            background-color: #e8f8f5;
            border-left: 4px solid #2ecc71;
        }}
        .risk-high {{
            background-color: #fadbd8;
            border-left: 4px solid #e74c3c;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p class="metadata">Generated: {report.get('generated_at', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <p>{report.get('executive_summary', 'N/A')}</p>
    </div>
    
    <div class="section">
        <h2>Data Overview</h2>
        <pre>{report.get('data_overview', 'N/A')}</pre>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
"""
        
        # Add findings
        findings = report.get("key_findings", [])
        for finding in findings:
            html += f'        <div class="finding">{finding}</div>\n'
        
        html += """    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <pre>"""
        html += report.get("recommendations", "N/A")
        html += """</pre>
    </div>
    
    <div class="section">
        <h2>Risk Assessment</h2>
        <pre>"""
        html += report.get("risk_assessment", "N/A")
        html += """</pre>
    </div>
    
    <div class="section">
        <h2>Methodology</h2>
        <pre>"""
        html += report.get("methodology", "N/A")
        html += """</pre>
    </div>
    
    <div class="section">
        <h2>Report Metadata</h2>
        <ul>
"""
        
        # Add metadata
        metadata = report.get("metadata", {})
        for key, value in metadata.items():
            html += f"            <li><strong>{key}:</strong> {value}</li>\n"
        
        html += """        </ul>
    </div>
</body>
</html>"""
        
        return html
    
    def translate_technical_terms(
        self,
        content: str,
        additional_glossary: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Translate technical terms in content to accessible language.
        
        Args:
            content: Text content with technical terms
            additional_glossary: Additional term definitions
        
        Returns:
            str: Content with technical terms explained
        """
        glossary = self.technical_glossary.copy()
        if additional_glossary:
            glossary.update(additional_glossary)
        
        # Use LLM to help with translation
        translation_prompt = f"""
        Translate the following technical content into accessible business language.
        Replace or explain these technical terms: {', '.join(glossary.keys())}
        
        Content: {content}
        
        Provide a clear, non-technical version that maintains accuracy.
        """
        
        try:
            translated = self.request_llm_assistance(
                translation_prompt,
                {"glossary": glossary}
            )
            return translated
        except Exception as e:
            self.logger.error(f"Translation failed: {str(e)}")
            return content
    
    def export_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: ExportFormat = ExportFormat.TEXT
    ) -> bool:
        """
        Export report to file.
        
        Args:
            report: Report data
            output_path: Output file path
            format: Export format
        
        Returns:
            bool: Success status
        """
        try:
            formatted_reports = self._generate_formatted_outputs(
                report,
                {"export_formats": [format.value]}
            )
            
            content = formatted_reports.get(format.value, "")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            self.logger.info(f"Report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Export failed: {str(e)}")
            return False