"""
Example Script: Model Evaluation and Comparison Pipeline

This script demonstrates how to use the complete evaluation framework
for comparing multiple models in the claims risk classification pipeline.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime

# Add the src directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import our models and evaluation components
from models import RandomForestClaimsModel, XGBoostClaimsModel, NeuralNetworkClaimsModel
from evaluation import ModelEvaluator, ModelComparison
from preprocessing import StructuredDataPreprocessor, UnstructuredDataPreprocessor, FeatureEngineer
from data_validation import SchemaValidator, DataQualityChecker
import config.config as config_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sample_data():
    """
    Load or generate sample claims data for demonstration
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Loading sample claims data...")
    
    # In a real scenario, this would load from database/files
    # For demonstration, we'll create realistic sample data
    
    np.random.seed(42)
    n_samples = 1000
    n_test = 200
    
    # Generate structured features
    data = {
        'claim_amount': np.random.lognormal(8, 1, n_samples + n_test),
        'claimant_age': np.random.normal(45, 15, n_samples + n_test),
        'policy_duration_months': np.random.randint(1, 120, n_samples + n_test),
        'previous_claims': np.random.poisson(0.5, n_samples + n_test),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples + n_test),
        'claim_type': np.random.choice(['Auto', 'Home', 'Health', 'Life'], n_samples + n_test),
        'day_of_week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], n_samples + n_test),
        'weather_condition': np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_samples + n_test)
    }
    
    # Generate unstructured text features
    claim_descriptions = []
    for i in range(n_samples + n_test):
        # Generate realistic claim descriptions
        templates = [
            "Vehicle collision on highway resulted in significant damage to front end",
            "House fire caused by electrical fault, extensive smoke and water damage",
            "Slip and fall incident at commercial property, injured back and knee",
            "Storm damage to roof, multiple shingles missing, water intrusion",
            "Theft of personal belongings from vehicle, window broken"
        ]
        desc = np.random.choice(templates)
        if np.random.random() > 0.5:  # Add variation
            desc += f" Claim filed on {np.random.choice(['Monday', 'Friday', 'weekend'])}"
        claim_descriptions.append(desc)
    
    data['claim_description'] = claim_descriptions
    
    # Generate target variable (risk level)
    # High risk criteria: high amount, young/old claimants, many previous claims
    risk_score = (
        0.3 * (data['claim_amount'] > np.percentile(data['claim_amount'], 75)) +
        0.2 * ((data['claimant_age'] < 25) | (data['claimant_age'] > 65)) +
        0.3 * (data['previous_claims'] >= 2) +
        0.2 * np.random.random(n_samples + n_test)  # Add randomness
    )
    
    y = ['high' if score > 0.6 else 'low' for score in risk_score]
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['risk_level'] = y
    
    # Split into train/test
    train_df = df.iloc[:n_samples]
    test_df = df.iloc[n_samples:]
    
    # Separate features and targets
    feature_cols = [col for col in df.columns if col != 'risk_level']
    
    X_train = train_df[feature_cols]
    y_train = train_df['risk_level']
    X_test = test_df[feature_cols]
    y_test = test_df['risk_level']
    
    logger.info(f"Loaded data - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution - Test: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Preprocess the data using our preprocessing pipeline
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_train: Training targets
        
    Returns:
        Tuple of (X_train_processed, X_test_processed)
    """
    logger.info("Preprocessing data...")
    
    # Separate structured and unstructured data
    structured_cols = ['claim_amount', 'claimant_age', 'policy_duration_months', 
                      'previous_claims', 'region', 'claim_type', 'day_of_week', 
                      'weather_condition']
    unstructured_cols = ['claim_description']
    
    X_train_structured = X_train[structured_cols]
    X_test_structured = X_test[structured_cols]
    X_train_unstructured = X_train[unstructured_cols]
    X_test_unstructured = X_test[unstructured_cols]
    
    # Initialize preprocessors
    structured_preprocessor = StructuredDataPreprocessor()
    unstructured_preprocessor = UnstructuredDataPreprocessor()
    feature_engineer = FeatureEngineer()
    
    # Fit and transform structured data
    X_train_struct_processed = structured_preprocessor.fit_transform(X_train_structured, y_train)
    X_test_struct_processed = structured_preprocessor.transform(X_test_structured)
    
    # Fit and transform unstructured data
    X_train_unstruct_processed = unstructured_preprocessor.fit_transform(X_train_unstructured, y_train)
    X_test_unstruct_processed = unstructured_preprocessor.transform(X_test_unstructured)
    
    # Combine and engineer features
    X_train_processed = feature_engineer.fit_transform(
        X_train_struct_processed, X_train_unstruct_processed, y_train
    )
    X_test_processed = feature_engineer.transform(
        X_test_struct_processed, X_test_unstruct_processed
    )
    
    logger.info(f"Preprocessing complete - Train shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")
    
    return X_train_processed, X_test_processed


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    """
    Train multiple models for comparison
    
    Args:
        X_train: Training features
        y_train: Training targets
        
    Returns:
        Dictionary of trained models
    """
    logger.info("Training models...")
    
    models = {}
    
    # 1. Random Forest Model
    logger.info("Training Random Forest model...")
    rf_model = RandomForestClaimsModel()
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    
    # 2. XGBoost Model
    logger.info("Training XGBoost model...")
    xgb_model = XGBoostClaimsModel()
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    # 3. Neural Network Model
    logger.info("Training Neural Network model...")
    nn_model = NeuralNetworkClaimsModel()
    nn_model.fit(X_train, y_train)
    models['Neural Network'] = nn_model
    
    logger.info(f"Training complete - {len(models)} models trained")
    return models


def evaluate_and_compare_models(models: Dict[str, Any], 
                              X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              X_test: pd.DataFrame, 
                              y_test: pd.Series,
                              output_dir: str = "evaluation_results"):
    """
    Perform comprehensive evaluation and comparison of models
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        output_dir: Directory to save results
    """
    logger.info("Starting comprehensive model evaluation and comparison...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator and comparison framework
    evaluator = ModelEvaluator()
    comparison = ModelComparison()
    
    # 1. Individual Model Evaluation
    logger.info("=" * 50)
    logger.info("INDIVIDUAL MODEL EVALUATION")
    logger.info("=" * 50)
    
    individual_results = {}
    for model_name, model in models.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        try:
            # Evaluate model
            results = evaluator.evaluate_single_model(model, X_test, y_test, model_name)
            individual_results[model_name] = results
            
            # Generate individual report
            report = evaluator.generate_evaluation_report(results)
            report_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_report.md")
            with open(report_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Report saved: {report_path}")
            
            # Generate plots
            evaluator.plot_evaluation_results(results, output_dir)
            
            # Print summary
            logger.info(f"{model_name} Summary:")
            logger.info(f"  Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  F1-Score: {results.get('f1_score', 'N/A'):.4f}")
            logger.info(f"  ROC-AUC: {results.get('roc_auc', 'N/A'):.4f}")
            if 'business_metrics' in results:
                cost_savings = results['business_metrics'].get('cost_savings_percentage', 0)
                logger.info(f"  Cost Savings: {cost_savings:.1f}%")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
            individual_results[model_name] = {'error': str(e)}
    
    # 2. Model Comparison
    logger.info("\n" + "=" * 50)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 50)
    
    try:
        # Perform comprehensive comparison
        comparison_results = comparison.compare_models(models, X_train, y_train, X_test, y_test)
        
        # Generate comparison report
        comparison_report = comparison.generate_comparison_report()
        comparison_report_path = os.path.join(output_dir, "model_comparison_report.md")
        with open(comparison_report_path, 'w') as f:
            f.write(comparison_report)
        
        logger.info(f"Comparison report saved: {comparison_report_path}")
        
        # Generate comparison plots
        comparison.plot_comparison_results(output_dir)
        
        # Print comparison summary
        logger.info("\nComparison Summary:")
        if 'recommendation' in comparison_results:
            rec = comparison_results['recommendation']
            logger.info(f"Recommended Model: {rec.get('recommended_model', 'N/A')}")
            logger.info(f"Confidence: {rec.get('confidence', 'N/A'):.1%}")
            logger.info(f"Status: {rec.get('status', 'N/A')}")
        
        if 'rankings' in comparison_results and comparison_results['rankings'].get('overall_ranking'):
            logger.info("\nOverall Rankings:")
            for rank_info in comparison_results['rankings']['overall_ranking']:
                logger.info(f"  {rank_info['rank']}. {rank_info['model_name']} - {rank_info['normalized_score']:.1f}%")
        
        # Statistical significance summary
        if 'statistical_tests' in comparison_results:
            logger.info("\nStatistical Significance:")
            stat_tests = comparison_results['statistical_tests'].get('pairwise_tests', {})
            significant_results = [
                f"{test_name.replace('_vs_', ' vs ')}: {result['winner']} wins (p={result['p_value']:.4f})"
                for test_name, result in stat_tests.items()
                if result.get('significant', False)
            ]
            
            if significant_results:
                for result in significant_results:
                    logger.info(f"  {result}")
            else:
                logger.info("  No statistically significant differences found")
        
        # Export results
        comparison.export_comparison_results(
            os.path.join(output_dir, "comparison_results.json"), 'json'
        )
        comparison.export_comparison_results(
            os.path.join(output_dir, "comparison_results.csv"), 'csv'
        )
        
    except Exception as e:
        logger.error(f"Error in model comparison: {e}")
    
    # 3. Summary and Recommendations
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY AND RECOMMENDATIONS")
    logger.info("=" * 50)
    
    try:
        # Create final summary
        summary = create_final_summary(individual_results, comparison_results if 'comparison_results' in locals() else {})
        summary_path = os.path.join(output_dir, "evaluation_summary.md")
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Final summary saved: {summary_path}")
        logger.info("\nEvaluation completed successfully!")
        logger.info(f"All results saved to: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        logger.error(f"Error creating summary: {e}")


def create_final_summary(individual_results: Dict[str, Any], 
                        comparison_results: Dict[str, Any]) -> str:
    """
    Create final evaluation summary
    
    Args:
        individual_results: Individual model evaluation results
        comparison_results: Model comparison results
        
    Returns:
        Summary report as string
    """
    summary = f"""
# Claims Risk Classification Model Evaluation Summary

## Evaluation Overview
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Models Evaluated**: {', '.join(individual_results.keys())}
- **Evaluation Framework**: Comprehensive ML model evaluation with statistical testing

## Key Findings

### Best Performing Model
"""
    
    if comparison_results and 'recommendation' in comparison_results:
        rec = comparison_results['recommendation']
        summary += f"""
**Recommended Model**: {rec.get('recommended_model', 'N/A')}
- **Confidence Level**: {rec.get('confidence', 0):.1%}
- **Selection Status**: {rec.get('status', 'N/A')}

**Key Reasons**:
"""
        for reason in rec.get('reasoning', []):
            summary += f"- {reason}\n"
    
    summary += f"""
### Performance Comparison
"""
    
    # Add performance table
    if individual_results:
        summary += """
| Model | Accuracy | F1-Score | ROC-AUC | Business Impact |
|-------|----------|----------|---------|-----------------|
"""
        
        for model_name, results in individual_results.items():
            if 'error' not in results:
                accuracy = results.get('accuracy', 0)
                f1_score = results.get('f1_score', 0)
                roc_auc = results.get('roc_auc', 0)
                
                business_impact = "N/A"
                if 'business_metrics' in results:
                    cost_savings = results['business_metrics'].get('cost_savings_percentage', 0)
                    business_impact = f"{cost_savings:.1f}% cost savings"
                
                summary += f"| {model_name} | {accuracy:.3f} | {f1_score:.3f} | {roc_auc:.3f} | {business_impact} |\n"
    
    summary += f"""
### Statistical Analysis
"""
    
    if comparison_results and 'statistical_tests' in comparison_results:
        stat_tests = comparison_results['statistical_tests'].get('pairwise_tests', {})
        significant_count = sum(1 for result in stat_tests.values() if result.get('significant', False))
        
        summary += f"""
- **Total Pairwise Comparisons**: {len(stat_tests)}
- **Statistically Significant Differences**: {significant_count}
- **Statistical Test Used**: {comparison_results['statistical_tests'].get('test_type', 'N/A')}
"""
        
        if significant_count > 0:
            summary += "\n**Significant Comparisons**:\n"
            for test_name, result in stat_tests.items():
                if result.get('significant', False):
                    models = test_name.replace('_vs_', ' vs ')
                    summary += f"- {models}: {result['winner']} significantly better (p={result['p_value']:.4f})\n"
    
    summary += f"""
### Business Impact Analysis
"""
    
    if comparison_results and 'business_comparison' in comparison_results:
        bc = comparison_results['business_comparison']
        if 'cost_savings_ranking' in bc:
            best_cost_model = bc['cost_savings_ranking'][0]
            summary += f"""
- **Best Cost Performance**: {best_cost_model['model_name']}
  - Cost Savings: ${best_cost_model['cost_savings']:,.0f} ({best_cost_model['cost_savings_percentage']:.1f}%)
  
- **Cost Ranking**:
"""
            for item in bc['cost_savings_ranking']:
                summary += f"  {item['rank']}. {item['model_name']}: {item['cost_savings_percentage']:.1f}% savings\n"
    
    summary += f"""
## Recommendations

### Production Model Selection
Based on the comprehensive evaluation, the recommended approach is:

1. **Deploy the recommended model** ({comparison_results.get('recommendation', {}).get('recommended_model', 'TBD')}) for production use
2. **Monitor model performance** continuously using the evaluation framework
3. **Retrain models** quarterly or when performance degrades
4. **Implement A/B testing** to validate model improvements

### Next Steps
1. **Model Deployment**: Implement the selected model in the production pipeline
2. **Monitoring Setup**: Establish automated model performance monitoring
3. **Data Pipeline**: Ensure robust data preprocessing and validation
4. **Business Integration**: Align model outputs with business processes

---
*Report generated by Claims Risk Classification Evaluation Framework*
"""
    
    return summary


def main():
    """Main execution function"""
    logger.info("Starting Claims Risk Classification Model Evaluation Pipeline")
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_sample_data()
        
        # Preprocess data
        X_train_processed, X_test_processed = preprocess_data(X_train, X_test, y_train)
        
        # Train models
        models = train_models(X_train_processed, y_train)
        
        # Evaluate and compare models
        evaluate_and_compare_models(
            models, 
            X_train_processed, 
            y_train, 
            X_test_processed, 
            y_test
        )
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()