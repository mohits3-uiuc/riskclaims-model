"""
Model Comparison Framework for Claims Risk Classification Pipeline

This module provides comprehensive comparison capabilities for multiple ML models
including statistical testing, ranking, and automated model selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare
import itertools

# Sklearn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from .model_evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """
    Comprehensive model comparison for claims risk classification
    
    Features:
    - Statistical significance testing between models
    - Model ranking based on multiple criteria
    - Cross-validation comparison
    - Business impact comparison
    - Automated model selection
    - Detailed comparison reports
    - Visualization of model performance differences
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model comparison framework
        
        Args:
            config: Configuration dictionary with comparison parameters
        """
        self.config = config or self._get_default_config()
        self.evaluator = ModelEvaluator(config.get('evaluator_config') if config else None)
        self.comparison_results = {}
        self.model_rankings = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default comparison configuration"""
        return {
            # Statistical testing
            'significance_level': 0.05,
            'statistical_test': 'paired_ttest',  # 'paired_ttest', 'wilcoxon', 'mcnemar'
            'bootstrap_samples': 1000,
            'confidence_interval': 0.95,
            
            # Cross-validation for comparison
            'cv_folds': 5,
            'cv_repeats': 3,  # Number of times to repeat CV for robust comparison
            'stratified_cv': True,
            'random_state': 42,
            
            # Ranking criteria and weights
            'ranking_metrics': {
                'accuracy': 0.15,
                'f1_score': 0.20,
                'roc_auc': 0.25,
                'precision_positive': 0.15,
                'recall_positive': 0.15,
                'cost_savings_percentage': 0.10  # Business impact
            },
            
            # Model selection criteria
            'selection_primary_metric': 'f1_score',  # Primary metric for selection
            'selection_business_weight': 0.3,  # Weight for business metrics in selection
            'min_performance_threshold': {
                'accuracy': 0.7,
                'f1_score': 0.6,
                'roc_auc': 0.7
            },
            
            # Comparison thresholds
            'practical_difference_threshold': 0.02,  # 2% difference considered practical
            'statistical_power_threshold': 0.8,
            
            # Plotting settings
            'figure_size': (14, 10),
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300
        }
    
    def compare_models(self, 
                      models: Dict[str, Any], 
                      X: pd.DataFrame, 
                      y: pd.Series,
                      X_test: Optional[pd.DataFrame] = None,
                      y_test: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Compare multiple models comprehensively
        
        Args:
            models: Dictionary of model_name -> fitted model
            X: Training features (used for cross-validation comparison)
            y: Training targets
            X_test: Test features (optional, for holdout evaluation)
            y_test: Test targets (optional)
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        logger.info(f"Starting comparison of {len(models)} models")
        
        comparison_results = {
            'comparison_timestamp': datetime.now().isoformat(),
            'models_compared': list(models.keys()),
            'comparison_config': self.config.copy()
        }
        
        # 1. Individual model evaluation (if test data provided)
        individual_results = {}
        if X_test is not None and y_test is not None:
            logger.info("Evaluating individual models on test set...")
            for model_name, model in models.items():
                try:
                    results = self.evaluator.evaluate_single_model(
                        model, X_test, y_test, model_name
                    )
                    individual_results[model_name] = results
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    individual_results[model_name] = {'error': str(e)}
        
        comparison_results['individual_evaluations'] = individual_results
        
        # 2. Cross-validation comparison
        logger.info("Performing cross-validation comparison...")
        cv_comparison = self._cross_validation_comparison(models, X, y)
        comparison_results['cv_comparison'] = cv_comparison
        
        # 3. Statistical significance testing
        logger.info("Performing statistical significance tests...")
        if len(models) >= 2:
            statistical_tests = self._statistical_testing(models, X, y)
            comparison_results['statistical_tests'] = statistical_tests
        
        # 4. Model ranking
        logger.info("Computing model rankings...")
        rankings = self._compute_model_rankings(individual_results, cv_comparison)
        comparison_results['rankings'] = rankings
        
        # 5. Model selection recommendation
        logger.info("Generating model selection recommendation...")
        recommendation = self._recommend_best_model(
            individual_results, cv_comparison, rankings
        )
        comparison_results['recommendation'] = recommendation
        
        # 6. Business impact comparison
        if individual_results:
            logger.info("Comparing business impact...")
            business_comparison = self._compare_business_impact(individual_results)
            comparison_results['business_comparison'] = business_comparison
        
        # Store results
        self.comparison_results = comparison_results
        self.model_rankings = rankings
        
        logger.info("Model comparison completed")
        return comparison_results
    
    def _cross_validation_comparison(self, 
                                    models: Dict[str, Any], 
                                    X: pd.DataFrame, 
                                    y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation comparison between models"""
        
        # Set up cross-validation
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        cv_results = {}
        cv_scores_all = {}  # For statistical testing
        
        # Define metrics to compare
        metrics = ['accuracy', 'f1_weighted', 'roc_auc', 'precision_weighted', 'recall_weighted']
        
        for model_name, model in models.items():
            logger.info(f"Cross-validating {model_name}...")
            cv_results[model_name] = {}
            cv_scores_all[model_name] = {}
            
            try:
                for metric in metrics:
                    # Perform multiple rounds of CV for robustness
                    all_scores = []
                    for round_num in range(self.config['cv_repeats']):
                        scores = cross_val_score(
                            model, X, y,
                            cv=cv,
                            scoring=metric,
                            n_jobs=-1
                        )
                        all_scores.extend(scores)
                    
                    cv_results[model_name][metric] = {
                        'mean': np.mean(all_scores),
                        'std': np.std(all_scores),
                        'min': np.min(all_scores),
                        'max': np.max(all_scores),
                        'median': np.median(all_scores),
                        'scores': all_scores
                    }
                    cv_scores_all[model_name][metric] = all_scores
                    
            except Exception as e:
                logger.error(f"Error in cross-validation for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        # Add comparison statistics
        cv_results['comparison_stats'] = self._compute_cv_comparison_stats(cv_scores_all)
        
        return cv_results
    
    def _compute_cv_comparison_stats(self, cv_scores: Dict[str, Dict[str, List]]) -> Dict[str, Any]:
        """Compute cross-validation comparison statistics"""
        
        stats_results = {}
        model_names = list(cv_scores.keys())
        
        if len(model_names) < 2:
            return stats_results
        
        # Get metrics
        metrics = list(next(iter(cv_scores.values())).keys())
        
        for metric in metrics:
            stats_results[metric] = {}
            
            # Get scores for all models for this metric
            metric_scores = {name: cv_scores[name][metric] for name in model_names}
            
            # Pairwise comparison
            pairwise_comparisons = {}
            for model1, model2 in itertools.combinations(model_names, 2):
                scores1 = metric_scores[model1]
                scores2 = metric_scores[model2]
                
                # Paired t-test
                try:
                    t_stat, p_value = ttest_rel(scores1, scores2)
                    effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                        (np.std(scores1)**2 + np.std(scores2)**2) / 2
                    )
                    
                    pairwise_comparisons[f"{model1}_vs_{model2}"] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.config['significance_level'],
                        'effect_size': float(effect_size),
                        'mean_difference': np.mean(scores1) - np.mean(scores2),
                        'winner': model1 if np.mean(scores1) > np.mean(scores2) else model2
                    }
                except Exception as e:
                    logger.warning(f"Error in pairwise comparison {model1} vs {model2}: {e}")
            
            stats_results[metric]['pairwise_comparisons'] = pairwise_comparisons
            
            # Overall ranking for this metric
            metric_means = {name: np.mean(scores) for name, scores in metric_scores.items()}
            sorted_models = sorted(metric_means.items(), key=lambda x: x[1], reverse=True)
            stats_results[metric]['ranking'] = [model for model, score in sorted_models]
            stats_results[metric]['scores'] = {model: score for model, score in sorted_models}
        
        return stats_results
    
    def _statistical_testing(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive statistical testing between models"""
        
        test_results = {
            'test_type': self.config['statistical_test'],
            'significance_level': self.config['significance_level']
        }
        
        model_names = list(models.keys())
        
        # Get cross-validation scores for primary metric
        primary_metric = self.config['selection_primary_metric']
        cv = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        model_scores = {}
        for model_name, model in models.items():
            try:
                scores = cross_val_score(
                    model, X, y,
                    cv=cv,
                    scoring=primary_metric,
                    n_jobs=-1
                )
                model_scores[model_name] = scores
            except Exception as e:
                logger.error(f"Error getting scores for {model_name}: {e}")
                continue
        
        # Pairwise statistical tests
        pairwise_tests = {}
        for model1, model2 in itertools.combinations(model_names, 2):
            if model1 not in model_scores or model2 not in model_scores:
                continue
                
            scores1 = model_scores[model1]
            scores2 = model_scores[model2]
            
            try:
                if self.config['statistical_test'] == 'paired_ttest':
                    statistic, p_value = ttest_rel(scores1, scores2)
                    test_name = "Paired t-test"
                elif self.config['statistical_test'] == 'wilcoxon':
                    statistic, p_value = wilcoxon(scores1, scores2)
                    test_name = "Wilcoxon signed-rank test"
                else:
                    statistic, p_value = ttest_rel(scores1, scores2)
                    test_name = "Paired t-test (default)"
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.std(scores1)**2 + np.std(scores2)**2) / 2)
                cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std if pooled_std > 0 else 0
                
                # Confidence interval for difference
                diff = scores1 - scores2
                ci_lower, ci_upper = stats.t.interval(
                    self.config['confidence_interval'], 
                    len(diff)-1, 
                    loc=np.mean(diff), 
                    scale=stats.sem(diff)
                )
                
                pairwise_tests[f"{model1}_vs_{model2}"] = {
                    'test_name': test_name,
                    'statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < self.config['significance_level'],
                    'cohens_d': float(cohens_d),
                    'effect_size_interpretation': self._interpret_effect_size(cohens_d),
                    'mean_difference': float(np.mean(scores1) - np.mean(scores2)),
                    'confidence_interval': [float(ci_lower), float(ci_upper)],
                    'practical_difference': abs(np.mean(scores1) - np.mean(scores2)) > self.config['practical_difference_threshold'],
                    'winner': model1 if np.mean(scores1) > np.mean(scores2) else model2,
                    'winner_probability': self._calculate_winner_probability(scores1, scores2)
                }
                
            except Exception as e:
                logger.error(f"Error in statistical test {model1} vs {model2}: {e}")
        
        test_results['pairwise_tests'] = pairwise_tests
        
        # Overall test (if more than 2 models)
        if len(model_names) > 2:
            try:
                all_scores = [model_scores[name] for name in model_names if name in model_scores]
                if len(all_scores) > 2:
                    statistic, p_value = friedmanchisquare(*all_scores)
                    test_results['overall_test'] = {
                        'test_name': 'Friedman Chi-square test',
                        'statistic': float(statistic),
                        'p_value': float(p_value),
                        'significant': p_value < self.config['significance_level'],
                        'interpretation': 'Significant differences exist between models' if p_value < self.config['significance_level'] else 'No significant differences between models'
                    }
            except Exception as e:
                logger.error(f"Error in overall statistical test: {e}")
        
        return test_results
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _calculate_winner_probability(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Calculate probability that model1 beats model2"""
        return float(np.mean(scores1 > scores2))
    
    def _compute_model_rankings(self, 
                               individual_results: Dict[str, Any], 
                               cv_comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Compute comprehensive model rankings"""
        
        rankings = {
            'ranking_timestamp': datetime.now().isoformat(),
            'ranking_criteria': self.config['ranking_metrics']
        }
        
        model_names = list(individual_results.keys()) if individual_results else list(cv_comparison.keys())
        model_names = [name for name in model_names if name != 'comparison_stats']
        
        if not model_names:
            return rankings
        
        # Compute weighted scores
        model_scores = {}
        detailed_scores = {}
        
        for model_name in model_names:
            if individual_results and model_name in individual_results and 'error' not in individual_results[model_name]:
                # Use individual evaluation results if available
                results = individual_results[model_name]
                detailed_scores[model_name] = {}
                
                total_score = 0
                valid_metrics = 0
                
                for metric, weight in self.config['ranking_metrics'].items():
                    if metric in results:
                        score = results[metric]
                        if isinstance(score, dict) and 'cost_savings_percentage' == metric:
                            score = results['business_metrics']['cost_savings_percentage'] / 100
                        
                        if not pd.isna(score) and score is not None:
                            normalized_score = min(max(score, 0), 1)  # Ensure [0,1] range
                            weighted_score = normalized_score * weight
                            total_score += weighted_score
                            valid_metrics += 1
                            detailed_scores[model_name][metric] = {
                                'raw_score': float(score),
                                'normalized_score': float(normalized_score),
                                'weighted_score': float(weighted_score),
                                'weight': float(weight)
                            }
                
                model_scores[model_name] = total_score / max(valid_metrics, 1)
                
            elif model_name in cv_comparison and 'error' not in cv_comparison[model_name]:
                # Fall back to CV results
                cv_results = cv_comparison[model_name]
                detailed_scores[model_name] = {}
                
                total_score = 0
                valid_metrics = 0
                
                for metric, weight in self.config['ranking_metrics'].items():
                    # Map ranking metric to CV metric
                    cv_metric = self._map_ranking_to_cv_metric(metric)
                    
                    if cv_metric in cv_results:
                        score = cv_results[cv_metric]['mean']
                        normalized_score = min(max(score, 0), 1)
                        weighted_score = normalized_score * weight
                        total_score += weighted_score
                        valid_metrics += 1
                        
                        detailed_scores[model_name][metric] = {
                            'raw_score': float(score),
                            'normalized_score': float(normalized_score),
                            'weighted_score': float(weighted_score),
                            'weight': float(weight)
                        }
                
                model_scores[model_name] = total_score / max(valid_metrics, 1)
        
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        rankings['overall_ranking'] = [
            {
                'rank': i + 1,
                'model_name': model,
                'score': float(score),
                'normalized_score': float(score * 100)
            }
            for i, (model, score) in enumerate(sorted_models)
        ]
        
        rankings['detailed_scores'] = detailed_scores
        rankings['metric_rankings'] = {}
        
        # Individual metric rankings
        for metric in self.config['ranking_metrics'].keys():
            metric_scores = {}
            for model_name in model_names:
                if (model_name in detailed_scores and 
                    metric in detailed_scores[model_name]):
                    metric_scores[model_name] = detailed_scores[model_name][metric]['raw_score']
            
            if metric_scores:
                sorted_metric = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
                rankings['metric_rankings'][metric] = [
                    {'rank': i + 1, 'model_name': model, 'score': float(score)}
                    for i, (model, score) in enumerate(sorted_metric)
                ]
        
        return rankings
    
    def _map_ranking_to_cv_metric(self, ranking_metric: str) -> str:
        """Map ranking metric names to CV metric names"""
        mapping = {
            'accuracy': 'accuracy',
            'f1_score': 'f1_weighted',
            'roc_auc': 'roc_auc',
            'precision_positive': 'precision_weighted',
            'recall_positive': 'recall_weighted'
        }
        return mapping.get(ranking_metric, ranking_metric)
    
    def _recommend_best_model(self,
                             individual_results: Dict[str, Any],
                             cv_comparison: Dict[str, Any], 
                             rankings: Dict[str, Any]) -> Dict[str, Any]:
        """Generate automated model selection recommendation"""
        
        recommendation = {
            'recommendation_timestamp': datetime.now().isoformat(),
            'selection_criteria': {
                'primary_metric': self.config['selection_primary_metric'],
                'business_weight': self.config['selection_business_weight'],
                'min_thresholds': self.config['min_performance_threshold']
            }
        }
        
        if not rankings.get('overall_ranking'):
            recommendation['status'] = 'error'
            recommendation['message'] = 'No valid rankings available'
            return recommendation
        
        # Get top-ranked models
        top_models = rankings['overall_ranking'][:3]  # Top 3 models
        
        # Filter by minimum performance thresholds
        qualified_models = []
        for model_info in top_models:
            model_name = model_info['model_name']
            meets_threshold = True
            
            if individual_results and model_name in individual_results:
                results = individual_results[model_name]
                for metric, threshold in self.config['min_performance_threshold'].items():
                    if metric in results and results[metric] < threshold:
                        meets_threshold = False
                        break
            
            if meets_threshold:
                qualified_models.append(model_info)
        
        if not qualified_models:
            recommendation['status'] = 'warning'
            recommendation['message'] = 'No models meet minimum performance thresholds'
            recommendation['recommended_model'] = top_models[0]['model_name'] if top_models else None
        else:
            recommendation['status'] = 'success'
            recommendation['recommended_model'] = qualified_models[0]['model_name']
            recommendation['confidence'] = self._calculate_recommendation_confidence(
                qualified_models, individual_results
            )
        
        # Detailed recommendation reasoning
        recommendation['reasoning'] = self._generate_recommendation_reasoning(
            qualified_models if qualified_models else top_models,
            individual_results, cv_comparison
        )
        
        recommendation['alternative_models'] = qualified_models[1:] if len(qualified_models) > 1 else []
        
        return recommendation
    
    def _calculate_recommendation_confidence(self, 
                                           qualified_models: List[Dict], 
                                           individual_results: Dict[str, Any]) -> float:
        """Calculate confidence in the recommendation"""
        if len(qualified_models) < 2:
            return 0.5  # Low confidence if only one model
        
        # Calculate score gap between top models
        top_score = qualified_models[0]['score']
        second_score = qualified_models[1]['score']
        score_gap = top_score - second_score
        
        # Normalize confidence based on score gap (larger gap = higher confidence)
        confidence = min(0.5 + (score_gap * 10), 1.0)  # Scale and cap at 1.0
        
        return float(confidence)
    
    def _generate_recommendation_reasoning(self,
                                         top_models: List[Dict],
                                         individual_results: Dict[str, Any],
                                         cv_comparison: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""
        
        reasoning = []
        
        if not top_models:
            return ["No models available for comparison"]
        
        recommended_model = top_models[0]['model_name']
        
        # Overall performance reasoning
        reasoning.append(f"'{recommended_model}' achieved the highest overall score ({top_models[0]['score']:.3f}) based on weighted performance metrics")
        
        # Specific metric strengths
        if individual_results and recommended_model in individual_results:
            results = individual_results[recommended_model]
            
            strengths = []
            if results.get('roc_auc', 0) > 0.8:
                strengths.append(f"excellent discrimination ability (AUC: {results['roc_auc']:.3f})")
            if results.get('f1_score', 0) > 0.7:
                strengths.append(f"balanced precision-recall performance (F1: {results['f1_score']:.3f})")
            if results.get('business_metrics', {}).get('cost_savings_percentage', 0) > 10:
                cost_savings = results['business_metrics']['cost_savings_percentage']
                strengths.append(f"strong business impact ({cost_savings:.1f}% cost savings)")
            
            if strengths:
                reasoning.append(f"Key strengths: {', '.join(strengths)}")
        
        # Comparison with alternatives
        if len(top_models) > 1:
            second_model = top_models[1]['model_name']
            score_diff = (top_models[0]['score'] - top_models[1]['score']) * 100
            reasoning.append(f"Outperformed '{second_model}' by {score_diff:.1f} percentage points in overall scoring")
        
        # Statistical significance (if available in comparison results)
        if hasattr(self, 'comparison_results') and 'statistical_tests' in self.comparison_results:
            stat_tests = self.comparison_results['statistical_tests'].get('pairwise_tests', {})
            significant_wins = 0
            for test_key, test_result in stat_tests.items():
                if recommended_model in test_key and test_result.get('significant', False) and test_result.get('winner') == recommended_model:
                    significant_wins += 1
            
            if significant_wins > 0:
                reasoning.append(f"Demonstrated statistically significant superiority in {significant_wins} pairwise comparison(s)")
        
        return reasoning
    
    def _compare_business_impact(self, individual_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare business impact across models"""
        
        business_comparison = {
            'comparison_timestamp': datetime.now().isoformat()
        }
        
        # Extract business metrics
        business_metrics = {}
        for model_name, results in individual_results.items():
            if 'business_metrics' in results:
                business_metrics[model_name] = results['business_metrics']
        
        if not business_metrics:
            business_comparison['status'] = 'no_business_metrics'
            return business_comparison
        
        # Compare cost savings
        cost_savings_ranking = sorted(
            [(name, metrics['cost_savings']) for name, metrics in business_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        business_comparison['cost_savings_ranking'] = [
            {
                'rank': i + 1,
                'model_name': model,
                'cost_savings': float(savings),
                'cost_savings_percentage': float(business_metrics[model]['cost_savings_percentage'])
            }
            for i, (model, savings) in enumerate(cost_savings_ranking)
        ]
        
        # Compare risk detection rates
        detection_ranking = sorted(
            [(name, metrics['high_risk_detection_rate']) for name, metrics in business_metrics.items()],
            key=lambda x: x[1], reverse=True
        )
        
        business_comparison['risk_detection_ranking'] = [
            {
                'rank': i + 1,
                'model_name': model,
                'detection_rate': float(rate)
            }
            for i, (model, rate) in enumerate(detection_ranking)
        ]
        
        # Overall business impact score
        business_scores = {}
        for model_name, metrics in business_metrics.items():
            # Weighted combination of cost savings and risk detection
            normalized_cost_savings = metrics['cost_savings_percentage'] / 100
            detection_rate = metrics['high_risk_detection_rate']
            
            # Business score (60% cost savings, 40% detection rate)
            business_score = (normalized_cost_savings * 0.6) + (detection_rate * 0.4)
            business_scores[model_name] = business_score
        
        business_impact_ranking = sorted(business_scores.items(), key=lambda x: x[1], reverse=True)
        
        business_comparison['business_impact_ranking'] = [
            {
                'rank': i + 1,
                'model_name': model,
                'business_score': float(score)
            }
            for i, (model, score) in enumerate(business_impact_ranking)
        ]
        
        return business_comparison
    
    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive comparison report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        if not self.comparison_results:
            return "No comparison results available. Run compare_models() first."
        
        results = self.comparison_results
        
        report = f"""
# Model Comparison Report

## Summary
- **Comparison Date**: {results.get('comparison_timestamp', 'N/A')}
- **Models Compared**: {', '.join(results.get('models_compared', []))}
- **Primary Metric**: {self.config['selection_primary_metric']}

## Recommendation
"""
        
        if 'recommendation' in results:
            rec = results['recommendation']
            report += f"""
### Recommended Model: **{rec.get('recommended_model', 'N/A')}**
- **Confidence**: {rec.get('confidence', 'N/A'):.2%}
- **Status**: {rec.get('status', 'N/A')}

#### Reasoning:
"""
            for reason in rec.get('reasoning', []):
                report += f"- {reason}\n"
        
        # Model Rankings
        if 'rankings' in results and results['rankings'].get('overall_ranking'):
            report += f"""
## Model Rankings

### Overall Performance Ranking
"""
            for rank_info in results['rankings']['overall_ranking']:
                report += f"{rank_info['rank']}. **{rank_info['model_name']}** - Score: {rank_info['normalized_score']:.1f}%\n"
        
        # Statistical Tests
        if 'statistical_tests' in results and 'pairwise_tests' in results['statistical_tests']:
            report += f"""
## Statistical Significance Analysis

### Pairwise Comparisons ({results['statistical_tests']['test_type']})
"""
            for test_name, test_result in results['statistical_tests']['pairwise_tests'].items():
                models = test_name.replace('_vs_', ' vs ')
                significance = "✓ Significant" if test_result['significant'] else "✗ Not Significant"
                report += f"""
#### {models}
- **Result**: {significance} (p = {test_result['p_value']:.4f})
- **Effect Size**: {test_result['effect_size_interpretation']} (Cohen's d = {test_result['cohens_d']:.3f})
- **Winner**: {test_result['winner']}
- **Mean Difference**: {test_result['mean_difference']:.4f}
"""
        
        # Business Impact
        if 'business_comparison' in results:
            bc = results['business_comparison']
            report += f"""
## Business Impact Analysis
"""
            
            if 'cost_savings_ranking' in bc:
                report += f"""
### Cost Savings Ranking
"""
                for item in bc['cost_savings_ranking']:
                    report += f"{item['rank']}. **{item['model_name']}** - ${item['cost_savings']:,.0f} ({item['cost_savings_percentage']:.1f}%)\n"
            
            if 'risk_detection_ranking' in bc:
                report += f"""
### Risk Detection Rate Ranking  
"""
                for item in bc['risk_detection_ranking']:
                    report += f"{item['rank']}. **{item['model_name']}** - {item['detection_rate']:.1%}\n"
        
        # Cross-Validation Results
        if 'cv_comparison' in results:
            report += f"""
## Cross-Validation Performance

### Mean Performance Scores
"""
            cv_results = results['cv_comparison']
            for model_name in results['models_compared']:
                if model_name in cv_results and 'error' not in cv_results[model_name]:
                    model_cv = cv_results[model_name]
                    report += f"""
#### {model_name}
- **Accuracy**: {model_cv.get('accuracy', {}).get('mean', 'N/A'):.4f} ± {model_cv.get('accuracy', {}).get('std', 'N/A'):.4f}
- **F1-Score**: {model_cv.get('f1_weighted', {}).get('mean', 'N/A'):.4f} ± {model_cv.get('f1_weighted', {}).get('std', 'N/A'):.4f}  
- **ROC-AUC**: {model_cv.get('roc_auc', {}).get('mean', 'N/A'):.4f} ± {model_cv.get('roc_auc', {}).get('std', 'N/A'):.4f}
"""
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Comparison report saved to {output_path}")
        
        return report
    
    def plot_comparison_results(self, save_dir: Optional[str] = None):
        """
        Generate comparison plots
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        if not self.comparison_results:
            logger.warning("No comparison results available")
            return
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle('Model Comparison Results', fontsize=16)
        
        # Plot 1: Overall Rankings
        if 'rankings' in self.comparison_results:
            rankings = self.comparison_results['rankings']['overall_ranking']
            models = [r['model_name'] for r in rankings]
            scores = [r['normalized_score'] for r in rankings]
            
            bars = axes[0, 0].barh(models, scores)
            axes[0, 0].set_title('Overall Model Rankings')
            axes[0, 0].set_xlabel('Overall Score (%)')
            
            # Color bars by performance
            for bar, score in zip(bars, scores):
                if score > 80:
                    bar.set_color('green')
                elif score > 60:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # Plot 2: Cross-Validation Performance
        if 'cv_comparison' in self.comparison_results:
            cv_data = self.comparison_results['cv_comparison']
            models = [name for name in self.comparison_results['models_compared'] 
                     if name in cv_data and 'error' not in cv_data[name]]
            
            if models:
                accuracy_means = [cv_data[model]['accuracy']['mean'] for model in models]
                accuracy_stds = [cv_data[model]['accuracy']['std'] for model in models]
                
                axes[0, 1].bar(models, accuracy_means, yerr=accuracy_stds, capsize=5)
                axes[0, 1].set_title('Cross-Validation Accuracy')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Business Impact
        if 'business_comparison' in self.comparison_results:
            bc = self.comparison_results['business_comparison']
            if 'cost_savings_ranking' in bc:
                business_data = bc['cost_savings_ranking']
                models = [item['model_name'] for item in business_data]
                savings = [item['cost_savings_percentage'] for item in business_data]
                
                axes[1, 0].bar(models, savings)
                axes[1, 0].set_title('Cost Savings Impact')
                axes[1, 0].set_ylabel('Cost Savings (%)')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Metric Comparison Heatmap
        if 'individual_evaluations' in self.comparison_results:
            individual_results = self.comparison_results['individual_evaluations']
            
            # Create comparison matrix
            metrics = ['accuracy', 'f1_score', 'roc_auc', 'precision_positive', 'recall_positive']
            models = list(individual_results.keys())
            
            comparison_matrix = []
            for model in models:
                if 'error' not in individual_results[model]:
                    row = [individual_results[model].get(metric, 0) for metric in metrics]
                    comparison_matrix.append(row)
                else:
                    comparison_matrix.append([0] * len(metrics))
            
            if comparison_matrix:
                sns.heatmap(comparison_matrix, 
                           xticklabels=[m.replace('_', ' ').title() for m in metrics],
                           yticklabels=models,
                           annot=True, fmt='.3f', ax=axes[1, 1], cmap='YlOrRd')
                axes[1, 1].set_title('Performance Metrics Heatmap')
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir and self.config['save_plots']:
            import os
            os.makedirs(save_dir, exist_ok=True)
            filename = f"model_comparison_plots.{self.config['plot_format']}"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Comparison plots saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def export_comparison_results(self, output_path: str, format: str = 'json'):
        """
        Export comparison results
        
        Args:
            output_path: Path to save results
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(self.comparison_results, f, indent=2)
        
        elif format == 'csv':
            # Create flattened DataFrame for CSV export
            if 'individual_evaluations' in self.comparison_results:
                flattened_results = []
                
                for model_name, results in self.comparison_results['individual_evaluations'].items():
                    if 'error' not in results:
                        flat_result = {'model_name': model_name}
                        
                        # Flatten nested dictionaries
                        def flatten_dict(d, parent_key='', sep='_'):
                            items = []
                            for k, v in d.items():
                                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                                if isinstance(v, dict):
                                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                                elif isinstance(v, list):
                                    items.append((new_key, str(v)))
                                else:
                                    items.append((new_key, v))
                            return dict(items)
                        
                        flat_result.update(flatten_dict(results))
                        flattened_results.append(flat_result)
                
                df = pd.DataFrame(flattened_results)
                df.to_csv(output_path, index=False)
        
        logger.info(f"Comparison results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    print("ModelComparison example usage:")
    print("1. Initialize comparison framework")
    print("2. Use compare_models() with multiple fitted models")
    print("3. Generate reports with generate_comparison_report()")
    print("4. Create visualizations with plot_comparison_results()")
    print("5. Export results with export_comparison_results()")