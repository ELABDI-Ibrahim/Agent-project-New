"""
Decision Agent for the multi-agent data analyzer system.

This module implements the DecisionAgent class responsible for:
- Recommendation generation based on analysis results
- Priority ranking by criticality and failure probability
- Resource constraint consideration in scheduling
- Scenario simulation and impact analysis
- Risk assessment and mitigation strategies
"""

import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentResult
from ..core.models import Recommendation, ValidationResult
from ..core.enums import Priority, RiskLevel
from ..core.shared_context import SharedContext
from ..core.llm_coordinator import LLMCoordinator


@dataclass
class ResourceConstraints:
    """Resource constraints for decision making."""
    budget_limit: Optional[float] = None
    time_limit_days: Optional[int] = None
    personnel_available: Optional[int] = None
    equipment_available: List[str] = None
    maintenance_windows: List[Tuple[datetime, datetime]] = None
    
    def __post_init__(self):
        if self.equipment_available is None:
            self.equipment_available = []
        if self.maintenance_windows is None:
            self.maintenance_windows = []


@dataclass
class Scenario:
    """Scenario definition for what-if analysis."""
    id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    expected_outcomes: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "expected_outcomes": self.expected_outcomes
        }


@dataclass
class SimulationResult:
    """Result of scenario simulation."""
    scenario_id: str
    scenario_name: str
    predicted_outcomes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    risk_assessment: Dict[str, Any]
    resource_utilization: Dict[str, float]
    timeline_impact: Dict[str, int]  # Days
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "predicted_outcomes": self.predicted_outcomes,
            "confidence_intervals": self.confidence_intervals,
            "risk_assessment": self.risk_assessment,
            "resource_utilization": self.resource_utilization,
            "timeline_impact": self.timeline_impact
        }


@dataclass
class PriorityMatrix:
    """Priority matrix for ranking recommendations."""
    recommendations: List[Recommendation]
    priority_scores: Dict[str, float]
    ranking_criteria: Dict[str, float]  # Weights for different criteria
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "priority_scores": self.priority_scores,
            "ranking_criteria": self.ranking_criteria
        }


class DecisionAgent(BaseAgent):
    """
    Agent responsible for decision making and recommendation generation.
    
    This agent performs:
    - Recommendation generation based on analysis results
    - Priority ranking by criticality and failure probability
    - Resource constraint consideration in scheduling
    - Scenario simulation and impact analysis
    - Risk assessment and mitigation strategies
    """
    
    def __init__(self, llm_coordinator: LLMCoordinator, shared_context: SharedContext):
        """
        Initialize the Decision Agent.
        
        Args:
            llm_coordinator: LLM coordinator for cognitive assistance
            shared_context: Shared context for inter-agent communication
        """
        super().__init__("DecisionAgent", llm_coordinator, shared_context)
        
        # Configuration for decision making
        self.priority_weights = {
            'criticality': 0.4,
            'failure_probability': 0.3,
            'impact_severity': 0.2,
            'resource_efficiency': 0.1
        }
        
        # Risk assessment thresholds
        self.risk_thresholds = {
            'critical': 0.8,
            'high': 0.6,
            'medium': 0.4,
            'low': 0.2
        }
        
        # Default resource constraints
        self.default_constraints = ResourceConstraints()
        
        self.logger.info("DecisionAgent initialized with recommendation and simulation capabilities")
    
    def _initialize_agent(self) -> None:
        """Initialize agent-specific components."""
        self.logger.info("Decision Agent initialization completed")
    
    def process(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Process analysis results and generate decisions/recommendations.
        
        Args:
            input_data: Dictionary containing:
                - 'analysis_results': Results from analyzer agent
                - 'decision_type': Type of decision to make ('recommendations', 'scenarios', 'prioritization')
                - 'resource_constraints': Optional resource constraints
                - 'scenarios': Optional scenarios for simulation
                - 'parameters': Optional decision parameters
        
        Returns:
            AgentResult: Decision results including recommendations and simulations
        """
        self._log_processing_start("decision_making", input_data)
        
        # Validate input
        validation = self.validate_input(input_data, ['analysis_results'])
        if not validation.is_valid:
            result = AgentResult(success=False, errors=validation.errors)
            self._log_processing_end("decision_making", result)
            return result
        
        try:
            analysis_results = input_data['analysis_results']
            decision_type = input_data.get('decision_type', 'recommendations')
            resource_constraints = input_data.get('resource_constraints', self.default_constraints)
            scenarios = input_data.get('scenarios', [])
            parameters = input_data.get('parameters', {})
            
            # Convert resource constraints if it's a dict
            if isinstance(resource_constraints, dict):
                resource_constraints = ResourceConstraints(**resource_constraints)
            
            # Perform decision making based on type
            if decision_type == 'recommendations':
                results = self._generate_recommendations_workflow(analysis_results, resource_constraints, parameters)
            elif decision_type == 'scenarios':
                results = self._simulate_scenarios_workflow(analysis_results, scenarios, parameters)
            elif decision_type == 'prioritization':
                existing_recommendations = input_data.get('recommendations', [])
                results = self._prioritize_recommendations_workflow(existing_recommendations, analysis_results, parameters)
            else:  # 'full' decision analysis
                results = self._perform_full_decision_analysis(analysis_results, resource_constraints, scenarios, parameters)
            
            # Store results in shared context
            self.shared_context.store_data("decision_results", results, self.name)
            
            result = AgentResult(success=True, data=results)
            self._log_processing_end("decision_making", result)
            return result
            
        except Exception as e:
            error_msg = f"Decision making failed: {str(e)}"
            self.logger.error(error_msg)
            result = AgentResult(success=False, errors=[error_msg])
            self._log_processing_end("decision_making", result)
            return result
    
    def generate_recommendations(self, analysis_results: Dict[str, Any], 
                               resource_constraints: Optional[ResourceConstraints] = None) -> List[Recommendation]:
        """
        Generate maintenance recommendations based on analysis results.
        
        Args:
            analysis_results: Results from statistical analysis
            resource_constraints: Optional resource constraints to consider
        
        Returns:
            List[Recommendation]: Generated recommendations
        """
        if resource_constraints is None:
            resource_constraints = self.default_constraints
        
        recommendations = []
        
        try:
            # Extract key insights from analysis results
            insights = self._extract_key_insights(analysis_results)
            
            # Generate recommendations based on different analysis types
            if 'outliers' in analysis_results:
                outlier_recommendations = self._generate_outlier_recommendations(
                    analysis_results['outliers'], insights, resource_constraints
                )
                recommendations.extend(outlier_recommendations)
            
            if 'correlations' in analysis_results:
                correlation_recommendations = self._generate_correlation_recommendations(
                    analysis_results['correlations'], insights, resource_constraints
                )
                recommendations.extend(correlation_recommendations)
            
            if 'timeseries' in analysis_results:
                trend_recommendations = self._generate_trend_recommendations(
                    analysis_results['timeseries'], insights, resource_constraints
                )
                recommendations.extend(trend_recommendations)
            
            if 'descriptive_stats' in analysis_results:
                statistical_recommendations = self._generate_statistical_recommendations(
                    analysis_results['descriptive_stats'], insights, resource_constraints
                )
                recommendations.extend(statistical_recommendations)
            
            # Use LLM to enhance recommendations with domain knowledge
            enhanced_recommendations = self._enhance_recommendations_with_llm(recommendations, analysis_results)
            
            self.logger.info(f"Generated {len(enhanced_recommendations)} recommendations")
            return enhanced_recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            raise
    
    def prioritize_actions(self, recommendations: List[Recommendation], 
                          analysis_context: Optional[Dict[str, Any]] = None) -> PriorityMatrix:
        """
        Prioritize recommendations by criticality and failure probability.
        
        Args:
            recommendations: List of recommendations to prioritize
            analysis_context: Optional analysis context for informed prioritization
        
        Returns:
            PriorityMatrix: Prioritized recommendations with scores
        """
        if not recommendations:
            return PriorityMatrix(recommendations=[], priority_scores={}, ranking_criteria=self.priority_weights)
        
        priority_scores = {}
        
        for rec in recommendations:
            # Calculate priority score based on multiple criteria
            score = self._calculate_priority_score(rec, analysis_context)
            priority_scores[rec.id] = score
        
        # Sort recommendations by priority score (descending)
        sorted_recommendations = sorted(recommendations, key=lambda r: priority_scores[r.id], reverse=True)
        
        # Update priority levels based on scores
        for i, rec in enumerate(sorted_recommendations):
            if i < len(sorted_recommendations) * 0.2:  # Top 20%
                rec.priority = Priority.HIGH
            elif i < len(sorted_recommendations) * 0.6:  # Next 40%
                rec.priority = Priority.MEDIUM
            else:  # Bottom 40%
                rec.priority = Priority.LOW
        
        return PriorityMatrix(
            recommendations=sorted_recommendations,
            priority_scores=priority_scores,
            ranking_criteria=self.priority_weights
        )
    
    def simulate_scenarios(self, base_data: pd.DataFrame, scenarios: List[Scenario], 
                          analysis_context: Optional[Dict[str, Any]] = None) -> List[SimulationResult]:
        """
        Create "what-if" scenarios and simulate their impacts.
        
        Args:
            base_data: Base dataset for simulation
            scenarios: List of scenarios to simulate
            analysis_context: Optional analysis context
        
        Returns:
            List[SimulationResult]: Simulation results for each scenario
        """
        simulation_results = []
        
        for scenario in scenarios:
            try:
                # Simulate the scenario
                result = self._simulate_single_scenario(base_data, scenario, analysis_context)
                simulation_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Scenario simulation failed for {scenario.name}: {str(e)}")
                # Create error result
                error_result = SimulationResult(
                    scenario_id=scenario.id,
                    scenario_name=scenario.name,
                    predicted_outcomes={"error": 1.0},
                    confidence_intervals={},
                    risk_assessment={"error": str(e)},
                    resource_utilization={},
                    timeline_impact={}
                )
                simulation_results.append(error_result)
        
        self.logger.info(f"Completed simulation of {len(scenarios)} scenarios")
        return simulation_results
    
    def compare_strategies(self, simulation_results: List[SimulationResult], 
                          comparison_criteria: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple strategies based on simulation results.
        
        Args:
            simulation_results: Results from scenario simulations
            comparison_criteria: Optional criteria for comparison
        
        Returns:
            Dict containing strategy comparison analysis
        """
        if not simulation_results:
            return {"error": "No simulation results to compare"}
        
        if comparison_criteria is None:
            comparison_criteria = ['cost_effectiveness', 'risk_reduction', 'timeline_efficiency', 'resource_utilization']
        
        comparison = {
            'scenarios_compared': len(simulation_results),
            'comparison_criteria': comparison_criteria,
            'rankings': {},
            'trade_offs': {},
            'recommendations': {}
        }
        
        # Rank scenarios by each criterion
        for criterion in comparison_criteria:
            rankings = self._rank_scenarios_by_criterion(simulation_results, criterion)
            comparison['rankings'][criterion] = rankings
        
        # Analyze trade-offs between scenarios
        comparison['trade_offs'] = self._analyze_strategy_tradeoffs(simulation_results, comparison_criteria)
        
        # Generate strategy recommendations
        comparison['recommendations'] = self._generate_strategy_recommendations(simulation_results, comparison)
        
        return comparison
    
    def assess_risks(self, recommendations: List[Recommendation], 
                    analysis_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess risks associated with recommendations.
        
        Args:
            recommendations: List of recommendations to assess
            analysis_context: Optional analysis context
        
        Returns:
            Dict containing risk assessment results
        """
        risk_assessment = {
            'total_recommendations': len(recommendations),
            'risk_distribution': {},
            'high_risk_recommendations': [],
            'mitigation_strategies': {},
            'overall_risk_score': 0.0
        }
        
        risk_scores = []
        risk_distribution = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'MINIMAL': 0}
        
        for rec in recommendations:
            # Calculate risk score for recommendation
            risk_score = self._calculate_risk_score(rec, analysis_context)
            risk_scores.append(risk_score)
            
            # Update risk level based on score
            if risk_score >= self.risk_thresholds['critical']:
                rec.risk_level = RiskLevel.CRITICAL
                risk_distribution['CRITICAL'] += 1
            elif risk_score >= self.risk_thresholds['high']:
                rec.risk_level = RiskLevel.HIGH
                risk_distribution['HIGH'] += 1
            elif risk_score >= self.risk_thresholds['medium']:
                rec.risk_level = RiskLevel.MEDIUM
                risk_distribution['MEDIUM'] += 1
            elif risk_score >= self.risk_thresholds['low']:
                rec.risk_level = RiskLevel.LOW
                risk_distribution['LOW'] += 1
            else:
                rec.risk_level = RiskLevel.MINIMAL
                risk_distribution['MINIMAL'] += 1
            
            # Track high-risk recommendations
            if rec.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                risk_assessment['high_risk_recommendations'].append({
                    'id': rec.id,
                    'title': rec.title,
                    'risk_level': rec.risk_level.name,
                    'risk_score': risk_score
                })
        
        risk_assessment['risk_distribution'] = risk_distribution
        risk_assessment['overall_risk_score'] = np.mean(risk_scores) if risk_scores else 0.0
        
        # Generate mitigation strategies for high-risk recommendations
        risk_assessment['mitigation_strategies'] = self._generate_mitigation_strategies(
            risk_assessment['high_risk_recommendations'], analysis_context
        )
        
        return risk_assessment
    
    def _extract_key_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights from analysis results."""
        insights = {
            'critical_issues': [],
            'trends': [],
            'anomalies': [],
            'correlations': [],
            'performance_metrics': {}
        }
        
        # Extract outlier insights
        if 'outliers' in analysis_results:
            for column, outlier_data in analysis_results['outliers'].items():
                if isinstance(outlier_data, dict):
                    for method, method_results in outlier_data.items():
                        if isinstance(method_results, dict) and method_results.get('count', 0) > 0:
                            insights['anomalies'].append({
                                'column': column,
                                'method': method,
                                'count': method_results['count'],
                                'percentage': method_results.get('percentage', 0)
                            })
        
        # Extract correlation insights
        if 'correlations' in analysis_results:
            for method, corr_matrix in analysis_results['correlations'].items():
                if isinstance(corr_matrix, dict):
                    # Find strong correlations
                    for col1, correlations in corr_matrix.items():
                        if isinstance(correlations, dict):
                            for col2, corr_value in correlations.items():
                                if col1 != col2 and abs(corr_value) > 0.7:
                                    insights['correlations'].append({
                                        'column1': col1,
                                        'column2': col2,
                                        'correlation': corr_value,
                                        'method': method
                                    })
        
        # Extract trend insights
        if 'timeseries' in analysis_results:
            trend_analysis = analysis_results['timeseries'].get('trend_analysis', {})
            for column, trend_data in trend_analysis.items():
                if isinstance(trend_data, dict):
                    direction = trend_data.get('trend_direction')
                    strength = trend_data.get('trend_strength', 0)
                    if direction and strength > 0.5:
                        insights['trends'].append({
                            'column': column,
                            'direction': direction,
                            'strength': strength
                        })
        
        return insights
    
    def _generate_outlier_recommendations(self, outlier_data: Dict[str, Any], 
                                        insights: Dict[str, Any], 
                                        constraints: ResourceConstraints) -> List[Recommendation]:
        """Generate recommendations based on outlier detection."""
        recommendations = []
        
        for anomaly in insights['anomalies']:
            if anomaly['percentage'] > 5:  # Significant outlier percentage
                rec_id = str(uuid.uuid4())
                
                # Determine priority based on outlier severity
                if anomaly['percentage'] > 20:
                    priority = Priority.HIGH
                    timeline = "1-2 weeks"
                elif anomaly['percentage'] > 10:
                    priority = Priority.MEDIUM
                    timeline = "2-4 weeks"
                else:
                    priority = Priority.LOW
                    timeline = "1-2 months"
                
                recommendation = Recommendation(
                    id=rec_id,
                    title=f"Investigate {anomaly['column']} Anomalies",
                    description=f"Column '{anomaly['column']}' shows {anomaly['count']} outliers "
                               f"({anomaly['percentage']:.1f}% of data) detected by {anomaly['method']} method. "
                               f"These anomalies may indicate data quality issues or equipment malfunctions.",
                    priority=priority,
                    estimated_impact=anomaly['percentage'] / 100.0,
                    required_resources=["Data analyst", "Domain expert"],
                    timeline=timeline,
                    risk_level=RiskLevel.MEDIUM,
                    supporting_evidence=[
                        f"Outlier detection method: {anomaly['method']}",
                        f"Outlier count: {anomaly['count']}",
                        f"Percentage of data affected: {anomaly['percentage']:.1f}%"
                    ]
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_correlation_recommendations(self, correlation_data: Dict[str, Any], 
                                            insights: Dict[str, Any], 
                                            constraints: ResourceConstraints) -> List[Recommendation]:
        """Generate recommendations based on correlation analysis."""
        recommendations = []
        
        for correlation in insights['correlations']:
            if abs(correlation['correlation']) > 0.8:  # Very strong correlation
                rec_id = str(uuid.uuid4())
                
                correlation_type = "positive" if correlation['correlation'] > 0 else "negative"
                
                recommendation = Recommendation(
                    id=rec_id,
                    title=f"Leverage {correlation_type.title()} Correlation: {correlation['column1']} - {correlation['column2']}",
                    description=f"Strong {correlation_type} correlation ({correlation['correlation']:.3f}) detected "
                               f"between '{correlation['column1']}' and '{correlation['column2']}'. "
                               f"This relationship can be leveraged for predictive maintenance or optimization.",
                    priority=Priority.MEDIUM,
                    estimated_impact=abs(correlation['correlation']),
                    required_resources=["Data scientist", "Process engineer"],
                    timeline="2-6 weeks",
                    risk_level=RiskLevel.LOW,
                    supporting_evidence=[
                        f"Correlation coefficient: {correlation['correlation']:.3f}",
                        f"Analysis method: {correlation['method']}",
                        f"Relationship type: {correlation_type}"
                    ]
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_trend_recommendations(self, timeseries_data: Dict[str, Any], 
                                      insights: Dict[str, Any], 
                                      constraints: ResourceConstraints) -> List[Recommendation]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        for trend in insights['trends']:
            if trend['strength'] > 0.7:  # Strong trend
                rec_id = str(uuid.uuid4())
                
                if trend['direction'] == 'decreasing':
                    priority = Priority.HIGH
                    risk_level = RiskLevel.HIGH
                    timeline = "1-3 weeks"
                    action = "Address declining performance"
                else:
                    priority = Priority.MEDIUM
                    risk_level = RiskLevel.LOW
                    timeline = "1-2 months"
                    action = "Optimize improving performance"
                
                recommendation = Recommendation(
                    id=rec_id,
                    title=f"{action}: {trend['column']}",
                    description=f"Strong {trend['direction']} trend detected in '{trend['column']}' "
                               f"(strength: {trend['strength']:.3f}). "
                               f"{'Immediate attention required to prevent further degradation.' if trend['direction'] == 'decreasing' else 'Opportunity to further optimize performance.'}",
                    priority=priority,
                    estimated_impact=trend['strength'],
                    required_resources=["Maintenance team", "Equipment specialist"],
                    timeline=timeline,
                    risk_level=risk_level,
                    supporting_evidence=[
                        f"Trend direction: {trend['direction']}",
                        f"Trend strength: {trend['strength']:.3f}",
                        f"Column analyzed: {trend['column']}"
                    ]
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_statistical_recommendations(self, stats_data: Dict[str, Any], 
                                            insights: Dict[str, Any], 
                                            constraints: ResourceConstraints) -> List[Recommendation]:
        """Generate recommendations based on descriptive statistics."""
        recommendations = []
        
        for column, stats in stats_data.items():
            if isinstance(stats, dict):
                # Check for high variability
                cv = stats.get('std', 0) / stats.get('mean', 1) if stats.get('mean', 0) != 0 else 0
                
                if cv > 0.5:  # High coefficient of variation
                    rec_id = str(uuid.uuid4())
                    
                    recommendation = Recommendation(
                        id=rec_id,
                        title=f"Reduce Variability in {column}",
                        description=f"High variability detected in '{column}' (CV: {cv:.3f}). "
                                   f"This indicates inconsistent performance that may benefit from standardization or process improvement.",
                        priority=Priority.MEDIUM,
                        estimated_impact=min(cv, 1.0),
                        required_resources=["Process engineer", "Quality control"],
                        timeline="3-6 weeks",
                        risk_level=RiskLevel.MEDIUM,
                        supporting_evidence=[
                            f"Coefficient of variation: {cv:.3f}",
                            f"Standard deviation: {stats.get('std', 0):.3f}",
                            f"Mean value: {stats.get('mean', 0):.3f}"
                        ]
                    )
                    
                    recommendations.append(recommendation)
        
        return recommendations
    
    def _enhance_recommendations_with_llm(self, recommendations: List[Recommendation], 
                                        analysis_results: Dict[str, Any]) -> List[Recommendation]:
        """Enhance recommendations using LLM insights."""
        enhanced_recommendations = []
        
        for rec in recommendations:
            try:
                # Request LLM enhancement
                llm_query = f"""
                Enhance this maintenance recommendation with domain expertise:
                
                Title: {rec.title}
                Description: {rec.description}
                Priority: {rec.priority.name}
                Timeline: {rec.timeline}
                
                Analysis Context: {str(analysis_results)[:500]}...
                
                Please provide:
                1. Enhanced description with technical details
                2. Additional required resources if needed
                3. Potential risks and mitigation strategies
                4. Success metrics for implementation
                """
                
                llm_response = self.request_llm_assistance(llm_query, {
                    "recommendation_id": rec.id,
                    "analysis_summary": analysis_results
                })
                
                # Parse LLM response and enhance recommendation
                enhanced_rec = self._parse_llm_enhancement(rec, llm_response)
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed for recommendation {rec.id}: {str(e)}")
                enhanced_recommendations.append(rec)  # Use original if enhancement fails
        
        return enhanced_recommendations
    
    def _parse_llm_enhancement(self, original_rec: Recommendation, llm_response: str) -> Recommendation:
        """Parse LLM response and enhance recommendation."""
        # For now, append LLM insights to description
        # In a more sophisticated implementation, this would parse structured LLM output
        
        enhanced_description = original_rec.description
        if llm_response and "LLM assistance unavailable" not in llm_response:
            enhanced_description += f"\n\nAI Insights: {llm_response[:200]}..."
        
        # Create enhanced recommendation
        enhanced_rec = Recommendation(
            id=original_rec.id,
            title=original_rec.title,
            description=enhanced_description,
            priority=original_rec.priority,
            estimated_impact=original_rec.estimated_impact,
            required_resources=original_rec.required_resources,
            timeline=original_rec.timeline,
            risk_level=original_rec.risk_level,
            supporting_evidence=original_rec.supporting_evidence
        )
        
        return enhanced_rec
    
    def _calculate_priority_score(self, recommendation: Recommendation, 
                                context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate priority score for a recommendation."""
        score = 0.0
        
        # Base score from estimated impact
        score += recommendation.estimated_impact * self.priority_weights['impact_severity']
        
        # Risk level contribution
        risk_scores = {
            RiskLevel.CRITICAL: 1.0,
            RiskLevel.HIGH: 0.8,
            RiskLevel.MEDIUM: 0.6,
            RiskLevel.LOW: 0.4,
            RiskLevel.MINIMAL: 0.2
        }
        score += risk_scores.get(recommendation.risk_level, 0.5) * self.priority_weights['criticality']
        
        # Timeline urgency (shorter timeline = higher priority)
        timeline_scores = {
            "immediate": 1.0,
            "1-2 weeks": 0.9,
            "2-4 weeks": 0.7,
            "1-2 months": 0.5,
            "3-6 weeks": 0.6,
            "2-6 weeks": 0.7
        }
        
        timeline_score = 0.5  # Default
        for timeline_key, score_value in timeline_scores.items():
            if timeline_key.lower() in recommendation.timeline.lower():
                timeline_score = score_value
                break
        
        score += timeline_score * self.priority_weights['failure_probability']
        
        # Resource efficiency (fewer resources = higher efficiency)
        resource_efficiency = max(0.1, 1.0 - (len(recommendation.required_resources) * 0.1))
        score += resource_efficiency * self.priority_weights['resource_efficiency']
        
        return min(1.0, score)  # Cap at 1.0
    
    def _calculate_risk_score(self, recommendation: Recommendation, 
                            context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate risk score for a recommendation."""
        risk_score = 0.0
        
        # Base risk from estimated impact
        risk_score += recommendation.estimated_impact * 0.4
        
        # Timeline risk (longer timeline = higher risk)
        timeline_risk = 0.5  # Default
        if "immediate" in recommendation.timeline.lower():
            timeline_risk = 0.9
        elif "week" in recommendation.timeline.lower():
            timeline_risk = 0.6
        elif "month" in recommendation.timeline.lower():
            timeline_risk = 0.3
        
        risk_score += timeline_risk * 0.3
        
        # Resource complexity risk
        resource_risk = min(1.0, len(recommendation.required_resources) * 0.15)
        risk_score += resource_risk * 0.3
        
        return min(1.0, risk_score)
    
    def _simulate_single_scenario(self, base_data: pd.DataFrame, scenario: Scenario, 
                                context: Optional[Dict[str, Any]] = None) -> SimulationResult:
        """Simulate a single scenario."""
        # This is a simplified simulation - in practice, this would involve
        # complex modeling based on the scenario parameters
        
        predicted_outcomes = {}
        confidence_intervals = {}
        resource_utilization = {}
        timeline_impact = {}
        
        # Simulate based on scenario parameters
        for param_name, param_value in scenario.parameters.items():
            if isinstance(param_value, (int, float)):
                # Simple linear impact model
                base_value = scenario.expected_outcomes.get(param_name, 0)
                predicted_value = base_value * (1 + param_value * 0.1)  # 10% impact per unit
                predicted_outcomes[param_name] = predicted_value
                
                # Add confidence intervals (±20% for demonstration)
                confidence_intervals[param_name] = (
                    predicted_value * 0.8,
                    predicted_value * 1.2
                )
        
        # Simulate resource utilization
        for resource in ["budget", "time", "personnel"]:
            utilization = np.random.uniform(0.3, 0.9)  # Random utilization for demo
            resource_utilization[resource] = utilization
        
        # Simulate timeline impact
        timeline_impact = {
            "implementation_days": np.random.randint(7, 90),
            "payback_days": np.random.randint(30, 365)
        }
        
        # Risk assessment
        risk_assessment = {
            "overall_risk": np.random.uniform(0.1, 0.8),
            "technical_risk": np.random.uniform(0.1, 0.6),
            "financial_risk": np.random.uniform(0.1, 0.7),
            "operational_risk": np.random.uniform(0.1, 0.5)
        }
        
        return SimulationResult(
            scenario_id=scenario.id,
            scenario_name=scenario.name,
            predicted_outcomes=predicted_outcomes,
            confidence_intervals=confidence_intervals,
            risk_assessment=risk_assessment,
            resource_utilization=resource_utilization,
            timeline_impact=timeline_impact
        )
    
    def _rank_scenarios_by_criterion(self, simulation_results: List[SimulationResult], 
                                   criterion: str) -> List[Dict[str, Any]]:
        """Rank scenarios by a specific criterion."""
        rankings = []
        
        for result in simulation_results:
            score = 0.0
            
            if criterion == 'cost_effectiveness':
                # Higher predicted outcomes with lower resource utilization
                avg_outcome = np.mean(list(result.predicted_outcomes.values())) if result.predicted_outcomes else 0
                avg_utilization = np.mean(list(result.resource_utilization.values())) if result.resource_utilization else 1
                score = avg_outcome / max(avg_utilization, 0.1)
            
            elif criterion == 'risk_reduction':
                # Lower overall risk is better
                overall_risk = result.risk_assessment.get('overall_risk', 0.5)
                score = 1.0 - overall_risk
            
            elif criterion == 'timeline_efficiency':
                # Shorter implementation time is better
                impl_days = result.timeline_impact.get('implementation_days', 90)
                score = 1.0 - (impl_days / 365)  # Normalize to year
            
            elif criterion == 'resource_utilization':
                # Moderate utilization is optimal (not too low, not too high)
                avg_utilization = np.mean(list(result.resource_utilization.values())) if result.resource_utilization else 0.5
                score = 1.0 - abs(avg_utilization - 0.7)  # Optimal around 70%
            
            rankings.append({
                'scenario_id': result.scenario_id,
                'scenario_name': result.scenario_name,
                'score': score,
                'criterion': criterion
            })
        
        # Sort by score (descending)
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return rankings
    
    def _analyze_strategy_tradeoffs(self, simulation_results: List[SimulationResult], 
                                  criteria: List[str]) -> Dict[str, Any]:
        """Analyze trade-offs between different strategies."""
        tradeoffs = {
            'pareto_efficient': [],
            'dominated_scenarios': [],
            'trade_off_analysis': {}
        }
        
        # Simple Pareto efficiency analysis
        for i, result1 in enumerate(simulation_results):
            is_dominated = False
            
            for j, result2 in enumerate(simulation_results):
                if i != j:
                    # Check if result1 is dominated by result2
                    if self._is_dominated(result1, result2, criteria):
                        is_dominated = True
                        break
            
            if is_dominated:
                tradeoffs['dominated_scenarios'].append(result1.scenario_id)
            else:
                tradeoffs['pareto_efficient'].append(result1.scenario_id)
        
        return tradeoffs
    
    def _is_dominated(self, result1: SimulationResult, result2: SimulationResult, 
                     criteria: List[str]) -> bool:
        """Check if result1 is dominated by result2."""
        # Simplified dominance check
        result2_better_count = 0
        result1_better_count = 0
        
        for criterion in criteria:
            score1 = self._get_criterion_score(result1, criterion)
            score2 = self._get_criterion_score(result2, criterion)
            
            if score2 > score1:
                result2_better_count += 1
            elif score1 > score2:
                result1_better_count += 1
        
        # result1 is dominated if result2 is better in all criteria or equal in some and better in others
        return result2_better_count > 0 and result1_better_count == 0
    
    def _get_criterion_score(self, result: SimulationResult, criterion: str) -> float:
        """Get score for a specific criterion."""
        if criterion == 'cost_effectiveness':
            avg_outcome = np.mean(list(result.predicted_outcomes.values())) if result.predicted_outcomes else 0
            avg_utilization = np.mean(list(result.resource_utilization.values())) if result.resource_utilization else 1
            return avg_outcome / max(avg_utilization, 0.1)
        elif criterion == 'risk_reduction':
            return 1.0 - result.risk_assessment.get('overall_risk', 0.5)
        elif criterion == 'timeline_efficiency':
            impl_days = result.timeline_impact.get('implementation_days', 90)
            return 1.0 - (impl_days / 365)
        elif criterion == 'resource_utilization':
            avg_utilization = np.mean(list(result.resource_utilization.values())) if result.resource_utilization else 0.5
            return 1.0 - abs(avg_utilization - 0.7)
        else:
            return 0.5
    
    def _generate_strategy_recommendations(self, simulation_results: List[SimulationResult], 
                                         comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy recommendations based on comparison."""
        recommendations = {
            'best_overall': None,
            'best_by_criterion': {},
            'recommended_approach': '',
            'implementation_order': []
        }
        
        # Find best overall strategy (highest average rank across criteria)
        scenario_scores = {}
        for criterion, rankings in comparison['rankings'].items():
            for i, ranking in enumerate(rankings):
                scenario_id = ranking['scenario_id']
                if scenario_id not in scenario_scores:
                    scenario_scores[scenario_id] = []
                scenario_scores[scenario_id].append(len(rankings) - i)  # Higher rank = higher score
        
        # Calculate average scores
        avg_scores = {}
        for scenario_id, scores in scenario_scores.items():
            avg_scores[scenario_id] = np.mean(scores)
        
        # Best overall
        if avg_scores:
            best_scenario_id = max(avg_scores, key=avg_scores.get)
            best_result = next((r for r in simulation_results if r.scenario_id == best_scenario_id), None)
            if best_result:
                recommendations['best_overall'] = {
                    'scenario_id': best_result.scenario_id,
                    'scenario_name': best_result.scenario_name,
                    'average_score': avg_scores[best_scenario_id]
                }
        
        # Best by each criterion
        for criterion, rankings in comparison['rankings'].items():
            if rankings:
                recommendations['best_by_criterion'][criterion] = {
                    'scenario_id': rankings[0]['scenario_id'],
                    'scenario_name': rankings[0]['scenario_name'],
                    'score': rankings[0]['score']
                }
        
        return recommendations
    
    def _generate_mitigation_strategies(self, high_risk_recommendations: List[Dict[str, Any]], 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Generate mitigation strategies for high-risk recommendations."""
        mitigation_strategies = {}
        
        for rec in high_risk_recommendations:
            strategies = []
            
            if rec['risk_level'] == 'CRITICAL':
                strategies.extend([
                    "Implement phased rollout with pilot testing",
                    "Establish dedicated monitoring and rollback procedures",
                    "Assign senior technical lead for oversight",
                    "Create detailed contingency plans"
                ])
            elif rec['risk_level'] == 'HIGH':
                strategies.extend([
                    "Conduct thorough impact assessment before implementation",
                    "Establish clear success metrics and monitoring",
                    "Plan for additional resource allocation if needed",
                    "Schedule regular progress reviews"
                ])
            
            # Add general mitigation strategies
            strategies.extend([
                "Document all assumptions and dependencies",
                "Establish clear communication channels",
                "Plan for stakeholder training and change management"
            ])
            
            mitigation_strategies[rec['id']] = strategies
        
        return mitigation_strategies
    
    def _generate_recommendations_workflow(self, analysis_results: Dict[str, Any], 
                                         constraints: ResourceConstraints, 
                                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations workflow."""
        recommendations = self.generate_recommendations(analysis_results, constraints)
        priority_matrix = self.prioritize_actions(recommendations, analysis_results)
        risk_assessment = self.assess_risks(recommendations, analysis_results)
        
        return {
            'decision_type': 'recommendations',
            'recommendations': [rec.to_dict() for rec in recommendations],
            'priority_matrix': priority_matrix.to_dict(),
            'risk_assessment': risk_assessment,
            'total_recommendations': len(recommendations)
        }
    
    def _simulate_scenarios_workflow(self, analysis_results: Dict[str, Any], 
                                   scenarios: List[Dict[str, Any]], 
                                   parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate scenarios workflow."""
        # Convert scenario dicts to Scenario objects
        scenario_objects = []
        for scenario_dict in scenarios:
            scenario = Scenario(
                id=scenario_dict.get('id', str(uuid.uuid4())),
                name=scenario_dict.get('name', 'Unnamed Scenario'),
                description=scenario_dict.get('description', ''),
                parameters=scenario_dict.get('parameters', {}),
                expected_outcomes=scenario_dict.get('expected_outcomes', {})
            )
            scenario_objects.append(scenario)
        
        # Create dummy base data for simulation
        base_data = pd.DataFrame({'dummy': [1, 2, 3]})  # Placeholder
        
        simulation_results = self.simulate_scenarios(base_data, scenario_objects, analysis_results)
        strategy_comparison = self.compare_strategies(simulation_results)
        
        return {
            'decision_type': 'scenarios',
            'simulation_results': [result.to_dict() for result in simulation_results],
            'strategy_comparison': strategy_comparison,
            'total_scenarios': len(scenario_objects)
        }
    
    def _prioritize_recommendations_workflow(self, recommendations: List[Dict[str, Any]], 
                                           analysis_results: Dict[str, Any], 
                                           parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize recommendations workflow."""
        # Convert recommendation dicts to Recommendation objects
        rec_objects = []
        for rec_dict in recommendations:
            rec = Recommendation(
                id=rec_dict.get('id', str(uuid.uuid4())),
                title=rec_dict.get('title', 'Untitled'),
                description=rec_dict.get('description', ''),
                priority=Priority[rec_dict.get('priority', 'MEDIUM')],
                estimated_impact=rec_dict.get('estimated_impact', 0.5),
                required_resources=rec_dict.get('required_resources', []),
                timeline=rec_dict.get('timeline', '1-2 months'),
                risk_level=RiskLevel[rec_dict.get('risk_level', 'MEDIUM')],
                supporting_evidence=rec_dict.get('supporting_evidence', [])
            )
            rec_objects.append(rec)
        
        priority_matrix = self.prioritize_actions(rec_objects, analysis_results)
        risk_assessment = self.assess_risks(rec_objects, analysis_results)
        
        return {
            'decision_type': 'prioritization',
            'priority_matrix': priority_matrix.to_dict(),
            'risk_assessment': risk_assessment,
            'total_recommendations': len(rec_objects)
        }
    
    def _perform_full_decision_analysis(self, analysis_results: Dict[str, Any], 
                                      constraints: ResourceConstraints, 
                                      scenarios: List[Dict[str, Any]], 
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive decision analysis."""
        # Generate recommendations
        recommendations = self.generate_recommendations(analysis_results, constraints)
        
        # Prioritize recommendations
        priority_matrix = self.prioritize_actions(recommendations, analysis_results)
        
        # Assess risks
        risk_assessment = self.assess_risks(recommendations, analysis_results)
        
        # Simulate scenarios if provided
        simulation_results = []
        strategy_comparison = {}
        if scenarios:
            scenario_objects = []
            for scenario_dict in scenarios:
                scenario = Scenario(
                    id=scenario_dict.get('id', str(uuid.uuid4())),
                    name=scenario_dict.get('name', 'Unnamed Scenario'),
                    description=scenario_dict.get('description', ''),
                    parameters=scenario_dict.get('parameters', {}),
                    expected_outcomes=scenario_dict.get('expected_outcomes', {})
                )
                scenario_objects.append(scenario)
            
            base_data = pd.DataFrame({'dummy': [1, 2, 3]})  # Placeholder
            simulation_results = self.simulate_scenarios(base_data, scenario_objects, analysis_results)
            strategy_comparison = self.compare_strategies(simulation_results)
        
        return {
            'decision_type': 'full',
            'recommendations': [rec.to_dict() for rec in recommendations],
            'priority_matrix': priority_matrix.to_dict(),
            'risk_assessment': risk_assessment,
            'simulation_results': [result.to_dict() for result in simulation_results],
            'strategy_comparison': strategy_comparison,
            'summary': {
                'total_recommendations': len(recommendations),
                'high_priority_count': sum(1 for rec in recommendations if rec.priority == Priority.HIGH),
                'high_risk_count': len(risk_assessment.get('high_risk_recommendations', [])),
                'scenarios_analyzed': len(scenarios)
            }
        }