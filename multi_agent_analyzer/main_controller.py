"""
Main controller for orchestrating the multi-agent data analyzer system.

This module manages agent lifecycle, workflow coordination, and error handling.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

from .core.shared_context import SharedContext
from .core.llm_coordinator import LLMCoordinator
from .core.enums import AnalysisState, DataType
from .config import Config
from .agents.collector_agent import CollectorAgent
from .agents.analyzer_agent import AnalyzerAgent
from .agents.decision_agent import DecisionAgent
from .agents.reporter_agent import ReporterAgent


class WorkflowError(Exception):
    """Exception raised for workflow execution errors."""
    pass


class MainController:
    """
    Main controller that orchestrates the multi-agent analysis workflow.
    
    Responsibilities:
    - Agent lifecycle management (initialization, execution, cleanup)
    - Workflow coordination (Collector → Analyzer → Decision → Reporter)
    - Error handling and recovery
    - Progress tracking and state management
    """
    
    def __init__(self, gemini_api_key: str, log_level: int = logging.INFO):
        """
        Initialize the main controller.
        
        Args:
            gemini_api_key: API key for Gemini LLM
            log_level: Logging level (default: INFO)
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Create config object for LLM coordinator
        self.config = Config()
        self.config.gemini_api_key = gemini_api_key
        
        # Initialize core components
        self.shared_context = SharedContext()
        self.llm_coordinator = LLMCoordinator(config=self.config)
        
        # Initialize agents
        self.collector_agent: Optional[CollectorAgent] = None
        self.analyzer_agent: Optional[AnalyzerAgent] = None
        self.decision_agent: Optional[DecisionAgent] = None
        self.reporter_agent: Optional[ReporterAgent] = None
        
        # Workflow state
        self.is_initialized = False
        self.current_step = 0
        self.total_steps = 4
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        self.logger.info("MainController initialized")
    
    def initialize_agents(self) -> bool:
        """
        Initialize all agents in the system.
        
        Returns:
            bool: True if all agents initialized successfully
        """
        try:
            self.logger.info("Initializing agents...")
            
            # Create agents (without name parameter - they set it internally)
            self.collector_agent = CollectorAgent(
                llm_coordinator=self.llm_coordinator,
                shared_context=self.shared_context
            )
            
            self.analyzer_agent = AnalyzerAgent(
                llm_coordinator=self.llm_coordinator,
                shared_context=self.shared_context
            )
            
            self.decision_agent = DecisionAgent(
                llm_coordinator=self.llm_coordinator,
                shared_context=self.shared_context
            )
            
            self.reporter_agent = ReporterAgent(
                name="reporter",
                llm_coordinator=self.llm_coordinator,
                shared_context=self.shared_context
            )
            
            # Initialize each agent
            agents = [
                self.collector_agent,
                self.analyzer_agent,
                self.decision_agent,
                self.reporter_agent
            ]
            
            for agent in agents:
                if not agent.initialize():
                    self.logger.error(f"Failed to initialize {agent.name}")
                    return False
                self.logger.info(f"✓ {agent.name.capitalize()} agent initialized")
            
            self.is_initialized = True
            self.shared_context.update_analysis_state(AnalysisState.INITIALIZED)
            return True
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {str(e)}", exc_info=True)
            self.errors.append(f"Initialization error: {str(e)}")
            return False
    
    def execute_workflow(
        self,
        csv_path: str,
        data_dict: Optional[Dict[str, Any]] = None,
        report_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete analysis workflow with LLM-driven planning.
        
        Args:
            csv_path: Path to CSV file to analyze
            data_dict: Optional data dictionary with column metadata
            report_config: Optional report generation configuration
        
        Returns:
            Dict containing workflow results and final report
        """
        if not self.is_initialized:
            raise WorkflowError("Agents not initialized. Call initialize_agents() first.")
        
        self.logger.info("=" * 80)
        self.logger.info("Starting Multi-Agent Analysis Workflow")
        self.logger.info("=" * 80)
        
        workflow_start_time = time.time()
        results = {
            "success": False,
            "collector_results": None,
            "analyzer_results": None,
            "decision_results": None,
            "report": None,
            "errors": [],
            "warnings": [],
            "execution_time": 0,
            "analysis_plan": None
        }
        
        try:
            # Step 1: Data Collection
            self.current_step = 1
            collector_results = self._execute_collector(csv_path, data_dict)
            if not collector_results.success:
                raise WorkflowError(f"Collector failed: {', '.join(collector_results.errors)}")
            results["collector_results"] = collector_results.data
            
            # Step 1.5: LLM creates intelligent analysis plan
            self.logger.info(f"\n[Planning] Creating Intelligent Analysis Strategy")
            self.logger.info("-" * 80)
            analysis_plan = self._create_analysis_plan(collector_results.data, data_dict)
            results["analysis_plan"] = analysis_plan
            self.logger.info(f"✓ Analysis plan created: {len(analysis_plan.get('analyses', []))} analyses planned")
            
            # Step 2: Targeted Statistical Analysis (based on plan)
            self.current_step = 2
            analyzer_results = self._execute_analyzer_with_plan(collector_results.data, analysis_plan)
            if not analyzer_results.success:
                raise WorkflowError(f"Analyzer failed: {', '.join(analyzer_results.errors)}")
            results["analyzer_results"] = analyzer_results.data
            
            # Step 3: Decision Making
            self.current_step = 3
            decision_results = self._execute_decision(
                collector_results.data,
                analyzer_results.data,
                analysis_plan
            )
            if not decision_results.success:
                raise WorkflowError(f"Decision failed: {', '.join(decision_results.errors)}")
            results["decision_results"] = decision_results.data
            
            # Step 4: Report Generation
            self.current_step = 4
            report_results = self._execute_reporter(
                collector_results.data,
                analyzer_results.data,
                decision_results.data,
                report_config
            )
            if not report_results.success:
                raise WorkflowError(f"Reporter failed: {', '.join(report_results.errors)}")
            results["report"] = report_results.data
            
            # Workflow completed successfully
            results["success"] = True
            self.shared_context.update_analysis_state(AnalysisState.COMPLETED)
            
            # Collect warnings from all agents
            for agent_result in [collector_results, analyzer_results, decision_results, report_results]:
                results["warnings"].extend(agent_result.warnings)
            
        except WorkflowError as e:
            self.logger.error(f"Workflow failed at step {self.current_step}: {str(e)}")
            results["errors"].append(str(e))
            self.shared_context.update_analysis_state(AnalysisState.ERROR)
            
            # Save partial results
            self._save_partial_results(results)
            
        except Exception as e:
            self.logger.error(f"Unexpected error in workflow: {str(e)}", exc_info=True)
            results["errors"].append(f"Unexpected error: {str(e)}")
            self.shared_context.update_analysis_state(AnalysisState.ERROR)
            
            # Save partial results
            self._save_partial_results(results)
        
        finally:
            results["execution_time"] = time.time() - workflow_start_time
            self.logger.info(f"Workflow completed in {results['execution_time']:.2f} seconds")
        
        return results

    def _execute_collector(
        self,
        csv_path: str,
        data_dict: Optional[Dict[str, Any]]
    ):
        """Execute the Collector Agent."""
        self.logger.info(f"\n[Step 1/{self.total_steps}] Data Collection & Preprocessing")
        self.logger.info("-" * 80)
        self.shared_context.update_analysis_state(AnalysisState.COLLECTING)
        
        input_data = {
            "file_path": csv_path,
            "data_dictionary": data_dict
        }
        
        step_start = time.time()
        result = self.collector_agent.process(input_data)
        step_duration = time.time() - step_start
        
        self.logger.info(f"✓ Collection completed in {step_duration:.2f}s")
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"  ⚠ {warning}")
        
        return result
    
    def _create_analysis_plan(
        self,
        collector_data: Dict[str, Any],
        data_dict: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use LLM to create an intelligent analysis plan based on data characteristics.
        
        Args:
            collector_data: Results from data collection
            data_dict: Optional data dictionary
        
        Returns:
            Dict containing analysis plan with targeted analyses
        """
        # Prepare context for LLM
        data_profile = collector_data.get("data_profile", {})
        missing_report = collector_data.get("missing_value_report", {})
        detected_types = collector_data.get("detected_data_types", {})
        
        # Build concise summary for LLM
        summary = {
            "rows": data_profile.get("row_count", 0),
            "columns": data_profile.get("column_count", 0),
            "column_names": collector_data.get("columns", []),
            "data_types": {k: v.name for k, v in detected_types.items()},
            "missing_values": {
                col: pct for col, pct in missing_report.get("missing_percentages", {}).items()
                if pct > 0
            },
            "high_missing_columns": [
                col for col, pct in missing_report.get("missing_percentages", {}).items()
                if pct > 50
            ]
        }
        
        # Create prompt for LLM
        prompt = f"""
You are an expert data analyst. Based on this dataset summary, create a focused analysis plan.

Dataset Summary:
- {summary['rows']:,} rows, {summary['columns']} columns
- Columns: {', '.join(summary['column_names'][:20])}{'...' if len(summary['column_names']) > 20 else ''}
- Data types: {summary['data_types']}
- Missing data: {len(summary['missing_values'])} columns with missing values
- High missing (>50%): {summary['high_missing_columns']}

Create an analysis plan that:
1. Identifies 3-5 MOST IMPORTANT analyses to perform (not everything!)
2. Focuses on columns that are likely to have insights
3. Identifies potential data quality issues to investigate
4. Suggests relationships to explore

Respond in JSON format:
{{
  "priority_columns": ["col1", "col2", ...],
  "analyses": [
    {{
      "type": "descriptive|correlation|outliers|timeseries",
      "columns": ["col1", "col2"],
      "reason": "Why this analysis is important",
      "priority": "high|medium|low"
    }}
  ],
  "data_quality_checks": [
    {{
      "issue": "Description of potential issue",
      "columns": ["affected_columns"],
      "severity": "high|medium|low"
    }}
  ],
  "key_questions": ["Question 1", "Question 2"]
}}

Keep it focused - quality over quantity. Aim for 3-5 targeted analyses.
"""
        
        try:
            # Get LLM response
            import json
            response = self.llm_coordinator.resolve_ambiguity(
                prompt,
                {"data_summary": summary, "data_dict": data_dict}
            )
            
            # Parse JSON response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                plan = json.loads(json_str)
                
                self.logger.info(f"LLM suggested {len(plan.get('analyses', []))} targeted analyses")
                self.logger.info(f"Priority columns: {', '.join(plan.get('priority_columns', [])[:5])}")
                
                return plan
            else:
                self.logger.warning("Could not parse LLM analysis plan, using fallback")
                return self._create_fallback_plan(summary, detected_types)
                
        except Exception as e:
            self.logger.error(f"LLM analysis planning failed: {e}")
            return self._create_fallback_plan(summary, detected_types)
    
    def _create_fallback_plan(
        self,
        summary: Dict[str, Any],
        detected_types: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a simple fallback analysis plan if LLM fails.
        
        Args:
            summary: Data summary
            detected_types: Detected column types
        
        Returns:
            Basic analysis plan
        """
        # Get numerical columns
        numerical_cols = [
            col for col, dtype in detected_types.items()
            if dtype == DataType.NUMERICAL
        ][:10]  # Limit to 10
        
        return {
            "priority_columns": numerical_cols,
            "analyses": [
                {
                    "type": "descriptive",
                    "columns": numerical_cols,
                    "reason": "Basic statistical profiling",
                    "priority": "high"
                },
                {
                    "type": "correlation",
                    "columns": numerical_cols,
                    "reason": "Find relationships between variables",
                    "priority": "medium"
                }
            ],
            "data_quality_checks": [],
            "key_questions": ["What are the basic statistics of this data?"]
        }
    
    def _execute_analyzer_with_plan(
        self,
        collector_data: Dict[str, Any],
        analysis_plan: Dict[str, Any]
    ):
        """Execute the Analyzer Agent with targeted analysis plan."""
        self.logger.info(f"\n[Step 2/{self.total_steps}] Targeted Statistical Analysis")
        self.logger.info("-" * 80)
        self.shared_context.update_analysis_state(AnalysisState.ANALYZING)
        
        # Retrieve the actual dataframe from shared context
        dataframe = self.shared_context.retrieve_data("raw_dataframe")
        
        # Build targeted analysis config from plan
        analyses_to_run = analysis_plan.get("analyses", [])
        priority_columns = analysis_plan.get("priority_columns", [])
        
        # Log what we're analyzing
        self.logger.info(f"Running {len(analyses_to_run)} targeted analyses")
        for analysis in analyses_to_run:
            self.logger.info(f"  - {analysis['type']}: {analysis.get('reason', 'N/A')}")
        
        input_data = {
            "dataframe": dataframe,
            "data_profile": collector_data.get("data_profile"),
            "data_dictionary": collector_data.get("data_schema"),
            "analysis_config": {
                "targeted_analyses": analyses_to_run,
                "priority_columns": priority_columns,
                "skip_full_analysis": True  # Don't do everything!
            }
        }
        
        step_start = time.time()
        result = self.analyzer_agent.process(input_data)
        step_duration = time.time() - step_start
        
        self.logger.info(f"✓ Analysis completed in {step_duration:.2f}s")
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"  ⚠ {warning}")
        
        return result
    
    def _execute_decision(
        self,
        collector_data: Dict[str, Any],
        analyzer_data: Dict[str, Any],
        analysis_plan: Dict[str, Any]
    ):
        """Execute the Decision Agent."""
        self.logger.info(f"\n[Step 3/{self.total_steps}] Decision Making & Recommendations")
        self.logger.info("-" * 80)
        self.shared_context.update_analysis_state(AnalysisState.DECIDING)
        
        input_data = {
            "analysis_results": analyzer_data,
            "data_profile": collector_data.get("data_profile"),
            "analysis_plan": analysis_plan,  # Pass the plan
            "decision_config": {
                "generate_recommendations": True,
                "simulate_scenarios": False,  # Skip if not needed
                "assess_risks": True
            }
        }
        
        step_start = time.time()
        result = self.decision_agent.process(input_data)
        step_duration = time.time() - step_start
        
        self.logger.info(f"✓ Decisions generated in {step_duration:.2f}s")
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"  ⚠ {warning}")
        
        return result
    
    def _execute_reporter(
        self,
        collector_data: Dict[str, Any],
        analyzer_data: Dict[str, Any],
        decision_data: Dict[str, Any],
        report_config: Optional[Dict[str, Any]]
    ):
        """Execute the Reporter Agent."""
        self.logger.info(f"\n[Step 4/{self.total_steps}] Report Generation")
        self.logger.info("-" * 80)
        self.shared_context.update_analysis_state(AnalysisState.REPORTING)
        
        input_data = {
            "collector_results": collector_data,
            "analyzer_results": analyzer_data,
            "decision_results": decision_data,
            "report_config": report_config or {}
        }
        
        step_start = time.time()
        result = self.reporter_agent.process(input_data)
        step_duration = time.time() - step_start
        
        self.logger.info(f"✓ Report generated in {step_duration:.2f}s")
        if result.warnings:
            for warning in result.warnings:
                self.logger.warning(f"  ⚠ {warning}")
        
        return result
    
    def _save_partial_results(self, results: Dict[str, Any]) -> None:
        """
        Save partial results when workflow fails.
        
        Args:
            results: Partial results to save
        """
        try:
            import json
            from datetime import datetime
            
            # Create results directory if it doesn't exist
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Save partial results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = results_dir / f"partial_results_{timestamp}.json"
            
            # Convert to JSON-serializable format
            json_results = {
                "success": results["success"],
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "errors": results["errors"],
                "warnings": results["warnings"],
                "execution_time": results["execution_time"],
                "has_collector_results": results["collector_results"] is not None,
                "has_analyzer_results": results["analyzer_results"] is not None,
                "has_decision_results": results["decision_results"] is not None,
                "has_report": results["report"] is not None
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Partial results saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save partial results: {str(e)}")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """
        Get current workflow status.
        
        Returns:
            Dict with status information
        """
        return {
            "initialized": self.is_initialized,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "progress_percentage": (self.current_step / self.total_steps) * 100,
            "current_state": self.shared_context.get_analysis_state().name,
            "errors": self.errors.copy(),
            "warnings": self.warnings.copy(),
            "agent_statuses": {
                "collector": self.collector_agent.get_agent_status() if self.collector_agent else None,
                "analyzer": self.analyzer_agent.get_agent_status() if self.analyzer_agent else None,
                "decision": self.decision_agent.get_agent_status() if self.decision_agent else None,
                "reporter": self.reporter_agent.get_agent_status() if self.reporter_agent else None
            }
        }
    
    def reset(self) -> None:
        """Reset the controller to initial state."""
        self.logger.info("Resetting controller...")
        
        # Clear shared context
        self.shared_context.clear_all_data()
        
        # Reset workflow state
        self.current_step = 0
        self.errors.clear()
        self.warnings.clear()
        
        # Update state
        self.shared_context.update_analysis_state(AnalysisState.INITIALIZED)
        
        self.logger.info("Controller reset complete")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.logger.info("Cleaning up resources...")
        self.reset()
        self.is_initialized = False