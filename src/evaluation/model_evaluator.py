"""
Model Evaluator for Claims Risk Classification Pipeline

This module provides comprehensive evaluation capabilities for machine learning models
including various metrics, visualization, and detailed performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports for metrics and evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    log_loss, matthews_corrcoef, cohen_kappa_score,
    balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Statistical imports
from scipy import stats
from scipy.stats import chi2_contingency

# Plotting imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for claims risk classification
    
    Features:
    - Multiple classification metrics (accuracy, precision, recall, F1, AUC-ROC, etc.)
    - Cross-validation evaluation
    - Confusion matrix analysis
    - ROC and Precision-Recall curves
    - Calibration analysis
    - Feature importance evaluation
    - Business impact metrics for insurance domain
    - Statistical significance testing
    - Model interpretation and explainability metrics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize model evaluator
        
        Args:
            config: Configuration dictionary with evaluation parameters
        """
        self.config = config or self._get_default_config()
        self.evaluation_results = {}
        self.evaluation_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default evaluation configuration"""
        return {
            # Cross-validation settings
            'cv_folds': 5,
            'cv_scoring': ['accuracy', 'precision_weighted', 'recall_weighted', 
                          'f1_weighted', 'roc_auc'],
            'stratified_cv': True,
            'random_state': 42,
            
            # Metric thresholds for classification
            'prediction_threshold': 0.5,
            'positive_class': 'high',  # For binary classification
            
            # Business metrics (insurance specific)
            'claim_cost_low': 5000,   # Average cost of low-risk claim
            'claim_cost_high': 25000, # Average cost of high-risk claim
            'investigation_cost': 500, # Cost to investigate a claim flagged as high-risk
            
            # Plotting settings
            'figure_size': (12, 8),
            'save_plots': True,
            'plot_format': 'png',
            'dpi': 300,
            
            # Statistical testing
            'significance_level': 0.05,
            'bootstrap_samples': 1000,
            
            # Calibration settings
            'calibration_bins': 10,
            'calibration_strategy': 'uniform'  # 'uniform' or 'quantile'
        }
    
    def evaluate_single_model(self, 
                             model, 
                             X_test: pd.DataFrame, 
                             y_test: pd.Series,
                             model_name: str = None) -> Dict[str, Any]:
        """
        Evaluate a single model comprehensively
        
        Args:
            model: Fitted model instance
            X_test: Test features
            y_test: Test targets  
            model_name: Name for the model (for reporting)
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        model_name = model_name or getattr(model, 'model_name', 'UnknownModel')
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            if len(y_pred_proba.shape) == 2 and y_pred_proba.shape[1] == 2:
                y_pred_proba_positive = y_pred_proba[:, 1]  # Probability of positive class
            else:
                y_pred_proba_positive = y_pred_proba[:, 0]
                
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}
        
        # Calculate all metrics
        results = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': len(X_test),
            'test_features': len(X_test.columns)
        }
        
        # Basic classification metrics
        results.update(self._calculate_basic_metrics(y_test, y_pred, y_pred_proba_positive))
        
        # Confusion matrix analysis
        results.update(self._analyze_confusion_matrix(y_test, y_pred))
        
        # ROC and PR curve metrics
        results.update(self._calculate_curve_metrics(y_test, y_pred_proba_positive))
        
        # Calibration analysis
        results.update(self._analyze_calibration(y_test, y_pred_proba_positive))
        
        # Business impact metrics
        results.update(self._calculate_business_metrics(y_test, y_pred, y_pred_proba_positive))
        
        # Feature importance (if available)
        if hasattr(model, 'get_feature_importance'):
            try:
                importance_df = model.get_feature_importance()
                results['feature_importance'] = {
                    'top_features': importance_df.head(10).to_dict('records'),
                    'importance_sum': importance_df['importance'].sum(),
                    'features_above_threshold': len(importance_df[importance_df['importance'] > 0.01])
                }
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")
        
        # Store results
        self.evaluation_results[model_name] = results
        self.evaluation_history.append({
            'model_name': model_name,
            'timestamp': results['evaluation_timestamp'],
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc']
        })
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
    
    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics"""
        # Convert to binary if needed
        if hasattr(y_true, 'map'):
            y_true_binary = y_true.map({'low': 0, 'high': 1}) if y_true.dtype == 'object' else y_true
        else:
            y_true_binary = y_true
        
        if isinstance(y_pred[0], str):
            y_pred_binary = pd.Series(y_pred).map({'low': 0, 'high': 1})
        else:
            y_pred_binary = y_pred
        
        metrics = {
            # Basic metrics
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            
            # Class-specific metrics (for binary classification)
            'precision_positive': precision_score(y_true, y_pred, pos_label=self.config['positive_class'], zero_division=0),
            'recall_positive': recall_score(y_true, y_pred, pos_label=self.config['positive_class'], zero_division=0),
            'f1_positive': f1_score(y_true, y_pred, pos_label=self.config['positive_class'], zero_division=0),
            
            # Probability-based metrics
            'log_loss': log_loss(y_true_binary, y_pred_proba),
            'roc_auc': roc_auc_score(y_true_binary, y_pred_proba),
            
            # Other metrics
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        return metrics
    
    def _analyze_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze confusion matrix and derive insights"""
        cm = confusion_matrix(y_true, y_pred)
        
        # For binary classification
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Positive/Negative Predictive Values
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            return {
                'confusion_matrix': cm.tolist(),
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'sensitivity': tpr,  # True Positive Rate
                'specificity': tnr,  # True Negative Rate
                'false_positive_rate': fpr,
                'false_negative_rate': fnr,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv
            }
        else:
            return {
                'confusion_matrix': cm.tolist(),
                'matrix_shape': cm.shape
            }
    
    def _calculate_curve_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC and Precision-Recall curve metrics"""
        # Convert to binary for curve calculations
        if hasattr(y_true, 'map'):
            y_true_binary = y_true.map({'low': 0, 'high': 1}) if y_true.dtype == 'object' else y_true
        else:
            y_true_binary = y_true
        
        # ROC Curve
        try:
            fpr, tpr, roc_thresholds = roc_curve(y_true_binary, y_pred_proba)
            roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
            
            # Find optimal threshold using Youden's J statistic
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = roc_thresholds[optimal_idx]
            
        except Exception as e:
            logger.warning(f"Error calculating ROC curve: {e}")
            fpr, tpr, roc_auc, optimal_threshold = [], [], 0.0, 0.5
        
        # Precision-Recall Curve
        try:
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
            # Area under PR curve
            pr_auc = np.trapz(precision_curve, recall_curve)
            
            # Find threshold that maximizes F1 score
            f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (precision_curve[:-1] + recall_curve[:-1])
            f1_scores = np.nan_to_num(f1_scores)
            best_f1_idx = np.argmax(f1_scores)
            best_f1_threshold = pr_thresholds[best_f1_idx] if len(pr_thresholds) > best_f1_idx else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating PR curve: {e}")
            precision_curve, recall_curve, pr_auc, best_f1_threshold = [], [], 0.0, 0.5
        
        return {
            'roc_curve': {
                'fpr': fpr.tolist() if hasattr(fpr, 'tolist') else [],
                'tpr': tpr.tolist() if hasattr(tpr, 'tolist') else [],
                'auc': roc_auc,
                'optimal_threshold': float(optimal_threshold)
            },
            'pr_curve': {
                'precision': precision_curve.tolist() if hasattr(precision_curve, 'tolist') else [],
                'recall': recall_curve.tolist() if hasattr(recall_curve, 'tolist') else [],
                'auc': float(pr_auc),
                'best_f1_threshold': float(best_f1_threshold)
            }
        }
    
    def _analyze_calibration(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Analyze probability calibration"""
        # Convert to binary
        if hasattr(y_true, 'map'):
            y_true_binary = y_true.map({'low': 0, 'high': 1}) if y_true.dtype == 'object' else y_true
        else:
            y_true_binary = y_true
        
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary, y_pred_proba, 
                n_bins=self.config['calibration_bins'],
                strategy=self.config['calibration_strategy']
            )
            
            # Calculate calibration metrics
            # Expected Calibration Error (ECE)
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Maximum Calibration Error (MCE)
            mce = np.max(np.abs(fraction_of_positives - mean_predicted_value))
            
            # Brier Score
            brier_score = np.mean((y_pred_proba - y_true_binary) ** 2)
            
            return {
                'calibration': {
                    'fraction_of_positives': fraction_of_positives.tolist(),
                    'mean_predicted_value': mean_predicted_value.tolist(),
                    'expected_calibration_error': float(ece),
                    'maximum_calibration_error': float(mce),
                    'brier_score': float(brier_score)
                }
            }
        except Exception as e:
            logger.warning(f"Error analyzing calibration: {e}")
            return {'calibration': {'error': str(e)}}
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate business-specific metrics for insurance claims"""
        # Create confusion matrix for cost calculations
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Cost calculations
            # True Negative: Correctly classified low-risk claims (no investigation cost)
            cost_tn = tn * self.config['claim_cost_low']
            
            # False Positive: Low-risk claims classified as high-risk (investigation cost)
            cost_fp = fp * (self.config['claim_cost_low'] + self.config['investigation_cost'])
            
            # False Negative: High-risk claims missed (full high cost)
            cost_fn = fn * self.config['claim_cost_high']
            
            # True Positive: High-risk claims caught (investigation cost + reduced payout)
            # Assume investigation reduces cost by 30%
            cost_tp = tp * (self.config['investigation_cost'] + self.config['claim_cost_high'] * 0.7)
            
            total_cost = cost_tn + cost_fp + cost_fn + cost_tp
            
            # Baseline cost (no model - all claims paid at face value)
            baseline_cost = (len(y_true[y_true == 'low']) * self.config['claim_cost_low'] + 
                           len(y_true[y_true == 'high']) * self.config['claim_cost_high'])
            
            cost_savings = baseline_cost - total_cost
            cost_savings_percentage = (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0
            
            # Risk-adjusted metrics
            high_risk_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_at_threshold = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            return {
                'business_metrics': {
                    'total_cost': float(total_cost),
                    'baseline_cost': float(baseline_cost),
                    'cost_savings': float(cost_savings),
                    'cost_savings_percentage': float(cost_savings_percentage),
                    'cost_per_claim': float(total_cost / len(y_true)),
                    'high_risk_detection_rate': float(high_risk_recall),
                    'investigation_precision': float(precision_at_threshold),
                    'cost_breakdown': {
                        'true_negatives': float(cost_tn),
                        'false_positives': float(cost_fp),
                        'false_negatives': float(cost_fn),
                        'true_positives': float(cost_tp)
                    }
                }
            }
        else:
            return {'business_metrics': {'error': 'Multi-class business metrics not implemented'}}
    
    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation
        
        Args:
            model: Model instance to evaluate
            X: Features
            y: Targets
            
        Returns:
            Dictionary with cross-validation results
        """
        logger.info("Performing cross-validation...")
        
        # Set up cross-validation
        if self.config['stratified_cv']:
            cv = StratifiedKFold(
                n_splits=self.config['cv_folds'],
                shuffle=True,
                random_state=self.config['random_state']
            )
        else:
            cv = self.config['cv_folds']
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y,
            cv=cv,
            scoring=self.config['cv_scoring'],
            return_train_score=True,
            n_jobs=-1
        )
        
        # Process results
        results = {
            'cv_folds': self.config['cv_folds'],
            'cv_metrics': {}
        }
        
        for metric in self.config['cv_scoring']:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['cv_metrics'][metric] = {
                'test_mean': float(np.mean(test_scores)),
                'test_std': float(np.std(test_scores)),
                'test_scores': test_scores.tolist(),
                'train_mean': float(np.mean(train_scores)),
                'train_std': float(np.std(train_scores)),
                'train_scores': train_scores.tolist(),
                'overfitting_score': float(np.mean(train_scores) - np.mean(test_scores))
            }
        
        logger.info("Cross-validation completed")
        return results
    
    def generate_evaluation_report(self, model_results: Dict[str, Any], output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            model_results: Results from evaluate_single_model
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        model_name = model_results.get('model_name', 'Unknown Model')
        
        report = f"""
# Model Evaluation Report: {model_name}

## Summary
- **Evaluation Date**: {model_results.get('evaluation_timestamp', 'N/A')}
- **Test Samples**: {model_results.get('test_samples', 'N/A')}
- **Test Features**: {model_results.get('test_features', 'N/A')}

## Performance Metrics

### Classification Metrics
- **Accuracy**: {model_results.get('accuracy', 'N/A'):.4f}
- **Balanced Accuracy**: {model_results.get('balanced_accuracy', 'N/A'):.4f}
- **Precision (Weighted)**: {model_results.get('precision', 'N/A'):.4f}
- **Recall (Weighted)**: {model_results.get('recall', 'N/A'):.4f}
- **F1-Score (Weighted)**: {model_results.get('f1_score', 'N/A'):.4f}
- **ROC AUC**: {model_results.get('roc_auc', 'N/A'):.4f}

### High-Risk Class Performance
- **Precision (High-Risk)**: {model_results.get('precision_positive', 'N/A'):.4f}
- **Recall (High-Risk)**: {model_results.get('recall_positive', 'N/A'):.4f}
- **F1-Score (High-Risk)**: {model_results.get('f1_positive', 'N/A'):.4f}

### Confusion Matrix Analysis
- **True Positives**: {model_results.get('true_positives', 'N/A')}
- **True Negatives**: {model_results.get('true_negatives', 'N/A')}
- **False Positives**: {model_results.get('false_positives', 'N/A')}
- **False Negatives**: {model_results.get('false_negatives', 'N/A')}
- **Sensitivity**: {model_results.get('sensitivity', 'N/A'):.4f}
- **Specificity**: {model_results.get('specificity', 'N/A'):.4f}

## Business Impact Analysis
"""
        
        if 'business_metrics' in model_results:
            bm = model_results['business_metrics']
            report += f"""
### Cost Analysis
- **Total Cost**: ${bm.get('total_cost', 'N/A'):,.2f}
- **Baseline Cost**: ${bm.get('baseline_cost', 'N/A'):,.2f}
- **Cost Savings**: ${bm.get('cost_savings', 'N/A'):,.2f}
- **Cost Savings %**: {bm.get('cost_savings_percentage', 'N/A'):.2f}%
- **Cost per Claim**: ${bm.get('cost_per_claim', 'N/A'):,.2f}

### Risk Detection
- **High-Risk Detection Rate**: {bm.get('high_risk_detection_rate', 'N/A'):.4f}
- **Investigation Precision**: {bm.get('investigation_precision', 'N/A'):.4f}
"""
        
        report += f"""
## Model Quality Indicators
- **Log Loss**: {model_results.get('log_loss', 'N/A'):.4f}
- **Matthews Correlation Coefficient**: {model_results.get('matthews_corrcoef', 'N/A'):.4f}
- **Cohen's Kappa**: {model_results.get('cohen_kappa', 'N/A'):.4f}
"""
        
        # Add calibration analysis if available
        if 'calibration' in model_results:
            cal = model_results['calibration']
            report += f"""
## Probability Calibration
- **Expected Calibration Error**: {cal.get('expected_calibration_error', 'N/A'):.4f}
- **Maximum Calibration Error**: {cal.get('maximum_calibration_error', 'N/A'):.4f}
- **Brier Score**: {cal.get('brier_score', 'N/A'):.4f}
"""
        
        # Add feature importance if available
        if 'feature_importance' in model_results:
            fi = model_results['feature_importance']
            report += f"""
## Feature Importance Analysis
- **Features Above Threshold**: {fi.get('features_above_threshold', 'N/A')}
- **Total Importance Sum**: {fi.get('importance_sum', 'N/A'):.4f}

### Top Important Features:
"""
            for i, feature_info in enumerate(fi.get('top_features', [])[:5], 1):
                report += f"{i}. **{feature_info.get('feature', 'N/A')}**: {feature_info.get('importance', 'N/A'):.4f}\n"
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Evaluation report saved to {output_path}")
        
        return report
    
    def plot_evaluation_results(self, model_results: Dict[str, Any], save_dir: Optional[str] = None):
        """
        Generate evaluation plots
        
        Args:
            model_results: Results from evaluate_single_model
            save_dir: Directory to save plots (optional)
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        model_name = model_results.get('model_name', 'Model')
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.config['figure_size'])
        fig.suptitle(f'{model_name} - Evaluation Results', fontsize=16)
        
        # Plot 1: Confusion Matrix
        if 'confusion_matrix' in model_results:
            cm = np.array(model_results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], 
                       cmap='Blues', cbar_kws={'label': 'Count'})
            axes[0, 0].set_title('Confusion Matrix')
            axes[0, 0].set_xlabel('Predicted')
            axes[0, 0].set_ylabel('Actual')
        
        # Plot 2: ROC Curve
        if 'roc_curve' in model_results:
            roc_data = model_results['roc_curve']
            if roc_data.get('fpr') and roc_data.get('tpr'):
                axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'], 
                              label=f"ROC Curve (AUC = {roc_data['auc']:.3f})")
                axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[0, 1].set_xlabel('False Positive Rate')
                axes[0, 1].set_ylabel('True Positive Rate')
                axes[0, 1].set_title('ROC Curve')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
        
        # Plot 3: Precision-Recall Curve
        if 'pr_curve' in model_results:
            pr_data = model_results['pr_curve']
            if pr_data.get('precision') and pr_data.get('recall'):
                axes[1, 0].plot(pr_data['recall'], pr_data['precision'],
                              label=f"PR Curve (AUC = {pr_data['auc']:.3f})")
                axes[1, 0].set_xlabel('Recall')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].set_title('Precision-Recall Curve')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
        
        # Plot 4: Calibration Plot
        if 'calibration' in model_results:
            cal_data = model_results['calibration']
            if cal_data.get('fraction_of_positives') and cal_data.get('mean_predicted_value'):
                axes[1, 1].plot(cal_data['mean_predicted_value'], cal_data['fraction_of_positives'],
                              marker='o', label='Model')
                axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
                axes[1, 1].set_xlabel('Mean Predicted Probability')
                axes[1, 1].set_ylabel('Fraction of Positives')
                axes[1, 1].set_title('Calibration Plot')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot if directory provided
        if save_dir and self.config['save_plots']:
            import os
            os.makedirs(save_dir, exist_ok=True)
            filename = f"{model_name.replace(' ', '_')}_evaluation_plots.{self.config['plot_format']}"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=self.config['dpi'], bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {filepath}")
        else:
            plt.show()
        
        plt.close()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all evaluations performed
        
        Returns:
            Dictionary with evaluation summary
        """
        return {
            'total_evaluations': len(self.evaluation_results),
            'evaluated_models': list(self.evaluation_results.keys()),
            'evaluation_history': self.evaluation_history,
            'config': self.config
        }
    
    def export_results(self, output_path: str, format: str = 'json'):
        """
        Export all evaluation results
        
        Args:
            output_path: Path to save results
            format: Export format ('json' or 'csv')
        """
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump({
                    'evaluation_results': self.evaluation_results,
                    'evaluation_history': self.evaluation_history,
                    'config': self.config
                }, f, indent=2)
        
        elif format == 'csv':
            # Create flattened DataFrame for CSV export
            flattened_results = []
            
            for model_name, results in self.evaluation_results.items():
                flat_result = {'model_name': model_name}
                
                # Flatten nested dictionaries
                def flatten_dict(d, parent_key='', sep='_'):
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        elif isinstance(v, list):
                            items.append((new_key, str(v)))  # Convert list to string
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flat_result.update(flatten_dict(results))
                flattened_results.append(flat_result)
            
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Evaluation results exported to {output_path}")


# Example usage
if __name__ == "__main__":
    # This would typically be run with actual models and data
    # Here's a demonstration of how to use the evaluator
    
    print("ModelEvaluator example usage:")
    print("1. Initialize evaluator")
    evaluator = ModelEvaluator()
    
    print("2. Evaluate models using evaluator.evaluate_single_model()")
    print("3. Compare results using evaluator.evaluation_results")
    print("4. Generate reports using evaluator.generate_evaluation_report()")
    print("5. Create plots using evaluator.plot_evaluation_results()")
    
    print("\nEvaluator configuration:")
    for key, value in evaluator.config.items():
        print(f"  {key}: {value}")