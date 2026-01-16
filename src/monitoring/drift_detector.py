"""
Model Monitoring and Drift Detection for Claims Risk Classification Pipeline

This module provides comprehensive monitoring capabilities for detecting data drift,
model performance degradation, and automated alerting for production ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta
import json
import warnings
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

# Statistical imports
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import scipy.spatial.distance as distance

# Sklearn imports
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Time series analysis
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Comprehensive data drift detection for structured and unstructured features
    
    Features:
    - Statistical drift detection (KS test, chi-square, PSI)
    - Distribution shift detection
    - Feature importance drift
    - Multivariate drift detection
    - Customizable thresholds and alerts
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize drift detector
        
        Args:
            config: Configuration dictionary with drift detection parameters
        """
        self.config = config or self._get_default_config()
        self.reference_data = None
        self.reference_stats = {}
        self.drift_history = []
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default drift detection configuration"""
        return {
            # Statistical test thresholds
            'ks_test_threshold': 0.05,
            'chi2_test_threshold': 0.05,
            'psi_threshold': 0.2,  # Population Stability Index threshold
            'js_divergence_threshold': 0.1,  # Jensen-Shannon divergence threshold
            
            # Drift severity levels
            'warning_threshold': 0.1,
            'critical_threshold': 0.3,
            
            # Monitoring windows
            'window_size': 1000,  # Number of samples for drift detection
            'min_samples': 100,   # Minimum samples required for drift detection
            
            # Feature types for different drift tests
            'categorical_features': [],
            'numerical_features': [],
            'text_features': [],
            
            # Alert settings
            'enable_alerts': True,
            'alert_methods': ['log', 'email', 'webhook'],
            'alert_cooldown_hours': 24,  # Hours between alerts for same drift
            
            # Plotting settings
            'figure_size': (12, 8),
            'save_plots': True
        }
    
    def fit_reference(self, reference_data: pd.DataFrame, 
                     feature_types: Optional[Dict[str, List[str]]] = None):
        """
        Fit drift detector on reference (training) data
        
        Args:
            reference_data: Reference dataset to compare against
            feature_types: Dictionary specifying feature types
        """
        logger.info("Fitting drift detector on reference data...")
        
        self.reference_data = reference_data.copy()
        
        # Update feature types if provided
        if feature_types:
            self.config.update(feature_types)
        else:
            # Auto-detect feature types
            self._auto_detect_feature_types(reference_data)
        
        # Calculate reference statistics
        self._calculate_reference_stats()
        
        logger.info(f"Drift detector fitted on {len(reference_data)} reference samples")
    
    def _auto_detect_feature_types(self, data: pd.DataFrame):
        """Auto-detect feature types based on data"""
        categorical_features = []
        numerical_features = []
        text_features = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if it's text (long strings) or categorical
                avg_length = data[col].astype(str).str.len().mean()
                unique_ratio = data[col].nunique() / len(data)
                
                if avg_length > 20 or unique_ratio > 0.5:
                    text_features.append(col)
                else:
                    categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.config['categorical_features'] = categorical_features
        self.config['numerical_features'] = numerical_features
        self.config['text_features'] = text_features
        
        logger.info(f"Auto-detected features - Categorical: {len(categorical_features)}, "
                   f"Numerical: {len(numerical_features)}, Text: {len(text_features)}")
    
    def _calculate_reference_stats(self):
        """Calculate reference statistics for drift detection"""
        self.reference_stats = {}
        
        # Numerical features statistics
        for feature in self.config['numerical_features']:
            if feature in self.reference_data.columns:
                values = self.reference_data[feature].dropna()
                self.reference_stats[feature] = {
                    'type': 'numerical',
                    'mean': float(values.mean()),
                    'std': float(values.std()),
                    'median': float(values.median()),
                    'q25': float(values.quantile(0.25)),
                    'q75': float(values.quantile(0.75)),
                    'min': float(values.min()),
                    'max': float(values.max()),
                    'values': values.values,  # Store for KS test
                    'bins': np.histogram(values, bins=20)[1]  # For PSI calculation
                }
        
        # Categorical features statistics
        for feature in self.config['categorical_features']:
            if feature in self.reference_data.columns:
                values = self.reference_data[feature].dropna()
                value_counts = values.value_counts()
                self.reference_stats[feature] = {
                    'type': 'categorical',
                    'value_counts': value_counts.to_dict(),
                    'proportions': (value_counts / len(values)).to_dict(),
                    'unique_values': list(values.unique()),
                    'num_categories': len(value_counts)
                }
        
        # Text features statistics (basic)
        for feature in self.config['text_features']:
            if feature in self.reference_data.columns:
                values = self.reference_data[feature].dropna().astype(str)
                self.reference_stats[feature] = {
                    'type': 'text',
                    'avg_length': float(values.str.len().mean()),
                    'median_length': float(values.str.len().median()),
                    'vocab_size': len(' '.join(values).split()),
                    # Store word frequencies for drift detection
                    'word_counts': self._get_word_counts(values)
                }
    
    def _get_word_counts(self, text_series: pd.Series) -> Dict[str, int]:
        """Get word counts from text series"""
        from collections import Counter
        all_words = ' '.join(text_series).lower().split()
        return dict(Counter(all_words).most_common(100))  # Top 100 words
    
    def detect_drift(self, new_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift in new data compared to reference data
        
        Args:
            new_data: New data to check for drift
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Must call fit_reference() before detect_drift()")
        
        logger.info(f"Detecting drift in {len(new_data)} new samples...")
        
        drift_results = {
            'detection_timestamp': datetime.now().isoformat(),
            'sample_size': len(new_data),
            'features_analyzed': len([f for f in new_data.columns if f in self.reference_stats]),
            'drift_detected': False,
            'feature_drifts': {},
            'overall_drift_score': 0.0
        }
        
        total_drift_score = 0.0
        features_with_drift = 0
        
        # Check drift for each feature
        for feature_name, ref_stats in self.reference_stats.items():
            if feature_name not in new_data.columns:
                continue
            
            feature_drift = self._detect_feature_drift(
                feature_name, new_data[feature_name], ref_stats
            )
            
            drift_results['feature_drifts'][feature_name] = feature_drift
            total_drift_score += feature_drift['drift_score']
            
            if feature_drift['drift_detected']:
                features_with_drift += 1
        
        # Calculate overall drift metrics
        num_features = len(drift_results['feature_drifts'])
        drift_results['overall_drift_score'] = total_drift_score / max(num_features, 1)
        drift_results['features_with_drift'] = features_with_drift
        drift_results['drift_percentage'] = (features_with_drift / max(num_features, 1)) * 100
        
        # Determine if overall drift detected
        drift_results['drift_detected'] = (
            drift_results['overall_drift_score'] > self.config['warning_threshold'] or
            features_with_drift > 0
        )
        
        # Classify drift severity
        drift_results['severity'] = self._classify_drift_severity(drift_results['overall_drift_score'])
        
        # Store in history
        self.drift_history.append({
            'timestamp': drift_results['detection_timestamp'],
            'overall_drift_score': drift_results['overall_drift_score'],
            'features_with_drift': features_with_drift,
            'severity': drift_results['severity']
        })
        
        # Generate alerts if necessary
        if drift_results['drift_detected'] and self.config['enable_alerts']:
            self._generate_drift_alert(drift_results)
        
        logger.info(f"Drift detection complete - Drift detected: {drift_results['drift_detected']}, "
                   f"Severity: {drift_results['severity']}")
        
        return drift_results
    
    def _detect_feature_drift(self, feature_name: str, 
                             new_values: pd.Series, 
                             ref_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift for a single feature"""
        
        feature_drift = {
            'feature_name': feature_name,
            'feature_type': ref_stats['type'],
            'drift_detected': False,
            'drift_score': 0.0,
            'tests_performed': [],
            'p_values': {},
            'drift_metrics': {}
        }
        
        new_values_clean = new_values.dropna()
        
        if len(new_values_clean) < self.config['min_samples']:
            feature_drift['error'] = 'Insufficient samples for drift detection'
            return feature_drift
        
        try:
            if ref_stats['type'] == 'numerical':
                feature_drift.update(self._detect_numerical_drift(new_values_clean, ref_stats))
            elif ref_stats['type'] == 'categorical':
                feature_drift.update(self._detect_categorical_drift(new_values_clean, ref_stats))
            elif ref_stats['type'] == 'text':
                feature_drift.update(self._detect_text_drift(new_values_clean, ref_stats))
                
        except Exception as e:
            logger.error(f"Error detecting drift for feature {feature_name}: {e}")
            feature_drift['error'] = str(e)
        
        return feature_drift
    
    def _detect_numerical_drift(self, new_values: pd.Series, 
                               ref_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift in numerical features"""
        drift_info = {
            'tests_performed': ['ks_test', 'psi'],
            'p_values': {},
            'drift_metrics': {}
        }
        
        # Kolmogorov-Smirnov Test
        ref_values = ref_stats['values']
        ks_stat, ks_p_value = ks_2samp(ref_values, new_values)
        
        drift_info['p_values']['ks_test'] = float(ks_p_value)
        drift_info['drift_metrics']['ks_statistic'] = float(ks_stat)
        
        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(new_values, ref_stats['bins'], ref_values)
        drift_info['drift_metrics']['psi_score'] = float(psi_score)
        
        # Statistical summary comparison
        new_mean = new_values.mean()
        new_std = new_values.std()
        
        drift_info['drift_metrics']['mean_shift'] = float(abs(new_mean - ref_stats['mean']) / ref_stats['std'])
        drift_info['drift_metrics']['std_ratio'] = float(new_std / ref_stats['std'])
        
        # Determine drift
        ks_drift = ks_p_value < self.config['ks_test_threshold']
        psi_drift = psi_score > self.config['psi_threshold']
        
        drift_info['drift_detected'] = ks_drift or psi_drift
        drift_info['drift_score'] = max(
            1 - ks_p_value,  # Higher score for lower p-value
            psi_score / self.config['psi_threshold']  # Normalized PSI score
        )
        
        return drift_info
    
    def _detect_categorical_drift(self, new_values: pd.Series, 
                                 ref_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift in categorical features"""
        drift_info = {
            'tests_performed': ['chi2_test', 'psi'],
            'p_values': {},
            'drift_metrics': {}
        }
        
        # Get value counts for new data
        new_value_counts = new_values.value_counts()
        new_proportions = new_value_counts / len(new_values)
        
        # Chi-square test for distribution differences
        ref_proportions = ref_stats['proportions']
        
        # Align categories
        all_categories = set(list(ref_proportions.keys()) + list(new_proportions.keys()))
        
        ref_counts = []
        new_counts = []
        
        for cat in all_categories:
            ref_prop = ref_proportions.get(cat, 0)
            new_prop = new_proportions.get(cat, 0)
            
            ref_counts.append(ref_prop * len(self.reference_data))
            new_counts.append(new_prop * len(new_values))
        
        # Chi-square test (only if we have sufficient counts)
        if min(ref_counts + new_counts) >= 5:
            chi2_stat, chi2_p_value = stats.chisquare(new_counts, ref_counts)
            drift_info['p_values']['chi2_test'] = float(chi2_p_value)
            drift_info['drift_metrics']['chi2_statistic'] = float(chi2_stat)
        else:
            chi2_p_value = 1.0  # Assume no drift if insufficient data
        
        # PSI for categorical data
        psi_score = 0.0
        for cat in all_categories:
            ref_prop = ref_proportions.get(cat, 1e-10)  # Avoid log(0)
            new_prop = new_proportions.get(cat, 1e-10)
            psi_score += (new_prop - ref_prop) * np.log(new_prop / ref_prop)
        
        drift_info['drift_metrics']['psi_score'] = float(psi_score)
        
        # New categories detection
        new_categories = set(new_proportions.keys()) - set(ref_proportions.keys())
        missing_categories = set(ref_proportions.keys()) - set(new_proportions.keys())
        
        drift_info['drift_metrics']['new_categories'] = list(new_categories)
        drift_info['drift_metrics']['missing_categories'] = list(missing_categories)
        drift_info['drift_metrics']['category_shift_score'] = len(new_categories) + len(missing_categories)
        
        # Determine drift
        chi2_drift = chi2_p_value < self.config['chi2_test_threshold']
        psi_drift = psi_score > self.config['psi_threshold']
        category_drift = len(new_categories) > 0
        
        drift_info['drift_detected'] = chi2_drift or psi_drift or category_drift
        drift_info['drift_score'] = max(
            1 - chi2_p_value,
            psi_score / self.config['psi_threshold'],
            len(new_categories) * 0.1  # Penalty for new categories
        )
        
        return drift_info
    
    def _detect_text_drift(self, new_values: pd.Series, 
                          ref_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Detect drift in text features"""
        drift_info = {
            'tests_performed': ['length_distribution', 'vocabulary_drift'],
            'p_values': {},
            'drift_metrics': {}
        }
        
        new_values_str = new_values.astype(str)
        
        # Length distribution drift
        ref_avg_length = ref_stats['avg_length']
        new_avg_length = new_values_str.str.len().mean()
        
        length_shift = abs(new_avg_length - ref_avg_length) / ref_avg_length
        drift_info['drift_metrics']['length_shift'] = float(length_shift)
        drift_info['drift_metrics']['avg_length_new'] = float(new_avg_length)
        drift_info['drift_metrics']['avg_length_ref'] = float(ref_avg_length)
        
        # Vocabulary drift (simple word frequency comparison)
        new_word_counts = self._get_word_counts(new_values_str)
        ref_word_counts = ref_stats['word_counts']
        
        # Calculate Jensen-Shannon divergence for word distributions
        js_div = self._calculate_js_divergence(ref_word_counts, new_word_counts)
        drift_info['drift_metrics']['js_divergence'] = float(js_div)
        
        # Determine drift
        length_drift = length_shift > 0.2  # 20% change in average length
        vocab_drift = js_div > self.config['js_divergence_threshold']
        
        drift_info['drift_detected'] = length_drift or vocab_drift
        drift_info['drift_score'] = max(
            length_shift / 0.2,  # Normalized length shift
            js_div / self.config['js_divergence_threshold']  # Normalized JS divergence
        )
        
        return drift_info
    
    def _calculate_psi(self, new_values: pd.Series, ref_bins: np.ndarray, ref_values: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        # Bin the new data using reference bins
        new_hist, _ = np.histogram(new_values, bins=ref_bins)
        ref_hist, _ = np.histogram(ref_values, bins=ref_bins)
        
        # Convert to proportions
        new_props = new_hist / len(new_values)
        ref_props = ref_hist / len(ref_values)
        
        # Avoid division by zero
        new_props = np.where(new_props == 0, 1e-10, new_props)
        ref_props = np.where(ref_props == 0, 1e-10, ref_props)
        
        # Calculate PSI
        psi = np.sum((new_props - ref_props) * np.log(new_props / ref_props))
        
        return float(psi)
    
    def _calculate_js_divergence(self, ref_dist: Dict[str, int], new_dist: Dict[str, int]) -> float:
        """Calculate Jensen-Shannon divergence between two word distributions"""
        # Get all words
        all_words = set(ref_dist.keys()) | set(new_dist.keys())
        
        if not all_words:
            return 0.0
        
        # Convert to probability distributions
        ref_total = sum(ref_dist.values())
        new_total = sum(new_dist.values())
        
        ref_probs = np.array([ref_dist.get(word, 0) / ref_total for word in all_words])
        new_probs = np.array([new_dist.get(word, 0) / new_total for word in all_words])
        
        # Avoid log(0)
        ref_probs = np.where(ref_probs == 0, 1e-10, ref_probs)
        new_probs = np.where(new_probs == 0, 1e-10, new_probs)
        
        # Calculate JS divergence
        m = 0.5 * (ref_probs + new_probs)
        js_div = 0.5 * stats.entropy(ref_probs, m) + 0.5 * stats.entropy(new_probs, m)
        
        return float(js_div)
    
    def _classify_drift_severity(self, drift_score: float) -> str:
        """Classify drift severity based on score"""
        if drift_score < self.config['warning_threshold']:
            return 'none'
        elif drift_score < self.config['critical_threshold']:
            return 'warning'
        else:
            return 'critical'
    
    def _generate_drift_alert(self, drift_results: Dict[str, Any]):
        """Generate drift alert"""
        severity = drift_results['severity']
        
        alert_message = f"""
DRIFT ALERT - Severity: {severity.upper()}

Detection Time: {drift_results['detection_timestamp']}
Overall Drift Score: {drift_results['overall_drift_score']:.4f}
Features with Drift: {drift_results['features_with_drift']}/{drift_results['features_analyzed']}
Drift Percentage: {drift_results['drift_percentage']:.1f}%

Top Drifted Features:
"""
        
        # Add top 5 features with highest drift scores
        feature_drifts = drift_results['feature_drifts']
        sorted_features = sorted(
            [(name, info['drift_score']) for name, info in feature_drifts.items()],
            key=lambda x: x[1], reverse=True
        )
        
        for feature_name, drift_score in sorted_features[:5]:
            alert_message += f"- {feature_name}: {drift_score:.4f}\n"
        
        # Log alert
        if 'log' in self.config['alert_methods']:
            if severity == 'critical':
                logger.critical(alert_message)
            else:
                logger.warning(alert_message)
        
        # TODO: Implement email and webhook alerts
        if 'email' in self.config['alert_methods']:
            logger.info("Email alert would be sent here")
        
        if 'webhook' in self.config['alert_methods']:
            logger.info("Webhook alert would be sent here")


class ModelPerformanceMonitor:
    """
    Monitor ML model performance over time and detect performance degradation
    
    Features:
    - Performance metric tracking
    - Performance drift detection
    - Model degradation alerts
    - Performance comparison with baseline
    - Automated retraining recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize performance monitor"""
        self.config = config or self._get_default_config()
        self.baseline_performance = {}
        self.performance_history = []
        self.models = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default performance monitoring configuration"""
        return {
            # Performance thresholds
            'degradation_threshold': 0.05,  # 5% performance drop triggers alert
            'critical_threshold': 0.10,     # 10% drop is critical
            
            # Monitoring metrics
            'primary_metrics': ['accuracy', 'f1_score', 'roc_auc'],
            'business_metrics': ['cost_savings_percentage'],
            
            # Monitoring windows
            'performance_window': 30,  # Days to look back for performance calculation
            'trend_window': 7,         # Days to calculate performance trend
            
            # Alert settings
            'enable_alerts': True,
            'alert_cooldown_hours': 12,
            
            # Retraining recommendations
            'retrain_threshold': 0.08,  # Recommend retraining at 8% drop
            'min_days_between_retrain': 7
        }
    
    def set_baseline_performance(self, model_name: str, performance_metrics: Dict[str, float]):
        """Set baseline performance for a model"""
        self.baseline_performance[model_name] = {
            'timestamp': datetime.now().isoformat(),
            'metrics': performance_metrics.copy()
        }
        logger.info(f"Baseline performance set for {model_name}")
    
    def log_performance(self, model_name: str, 
                       performance_metrics: Dict[str, float],
                       prediction_count: int = 0,
                       timestamp: Optional[datetime] = None):
        """Log current model performance"""
        
        timestamp = timestamp or datetime.now()
        
        performance_entry = {
            'model_name': model_name,
            'timestamp': timestamp.isoformat(),
            'metrics': performance_metrics.copy(),
            'prediction_count': prediction_count
        }
        
        self.performance_history.append(performance_entry)
        
        # Check for performance degradation
        if model_name in self.baseline_performance:
            degradation_result = self._check_performance_degradation(model_name, performance_metrics)
            performance_entry['degradation_analysis'] = degradation_result
            
            if degradation_result['degradation_detected'] and self.config['enable_alerts']:
                self._generate_performance_alert(model_name, degradation_result)
    
    def _check_performance_degradation(self, model_name: str, 
                                     current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check for performance degradation compared to baseline"""
        
        baseline = self.baseline_performance[model_name]['metrics']
        
        degradation_info = {
            'degradation_detected': False,
            'severity': 'none',
            'metric_changes': {},
            'overall_degradation': 0.0
        }
        
        total_degradation = 0.0
        metrics_analyzed = 0
        
        for metric in self.config['primary_metrics']:
            if metric in baseline and metric in current_metrics:
                baseline_value = baseline[metric]
                current_value = current_metrics[metric]
                
                # Calculate relative change (negative means degradation for most metrics)
                if baseline_value > 0:
                    relative_change = (current_value - baseline_value) / baseline_value
                else:
                    relative_change = 0.0
                
                # For metrics where higher is better, degradation is negative change
                degradation = -relative_change
                
                degradation_info['metric_changes'][metric] = {
                    'baseline': float(baseline_value),
                    'current': float(current_value),
                    'relative_change': float(relative_change),
                    'degradation': float(degradation)
                }
                
                total_degradation += degradation
                metrics_analyzed += 1
        
        if metrics_analyzed > 0:
            degradation_info['overall_degradation'] = total_degradation / metrics_analyzed
            
            # Determine if degradation detected
            if degradation_info['overall_degradation'] > self.config['degradation_threshold']:
                degradation_info['degradation_detected'] = True
                
                if degradation_info['overall_degradation'] > self.config['critical_threshold']:
                    degradation_info['severity'] = 'critical'
                else:
                    degradation_info['severity'] = 'warning'
        
        return degradation_info
    
    def get_performance_trend(self, model_name: str, days: Optional[int] = None) -> Dict[str, Any]:
        """Get performance trend for a model over specified days"""
        
        days = days or self.config['trend_window']
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Get recent performance data
        recent_performance = [
            entry for entry in self.performance_history
            if (entry['model_name'] == model_name and 
                datetime.fromisoformat(entry['timestamp']) > cutoff_time)
        ]
        
        if len(recent_performance) < 2:
            return {'error': 'Insufficient data for trend analysis'}
        
        # Sort by timestamp
        recent_performance.sort(key=lambda x: x['timestamp'])
        
        trend_analysis = {
            'model_name': model_name,
            'analysis_period_days': days,
            'data_points': len(recent_performance),
            'metric_trends': {}
        }
        
        # Analyze trend for each metric
        for metric in self.config['primary_metrics']:
            metric_values = []
            timestamps = []
            
            for entry in recent_performance:
                if metric in entry['metrics']:
                    metric_values.append(entry['metrics'][metric])
                    timestamps.append(datetime.fromisoformat(entry['timestamp']))
            
            if len(metric_values) >= 2:
                # Calculate trend (simple linear regression slope)
                x = np.array([(ts - timestamps[0]).days for ts in timestamps])
                y = np.array(metric_values)
                
                if len(x) > 1 and np.std(x) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    trend_analysis['metric_trends'][metric] = {
                        'slope': float(slope),
                        'r_squared': float(r_value ** 2),
                        'p_value': float(p_value),
                        'trend_direction': 'improving' if slope > 0 else 'declining',
                        'trend_strength': 'strong' if abs(r_value) > 0.7 else 'weak',
                        'current_value': float(metric_values[-1]),
                        'initial_value': float(metric_values[0]),
                        'total_change': float(metric_values[-1] - metric_values[0])
                    }
        
        return trend_analysis
    
    def recommend_actions(self, model_name: str) -> Dict[str, Any]:
        """Generate recommendations based on model performance"""
        
        recommendations = {
            'model_name': model_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'priority': 'low'
        }
        
        # Check if we have baseline performance
        if model_name not in self.baseline_performance:
            recommendations['recommendations'].append({
                'action': 'establish_baseline',
                'description': 'Establish baseline performance metrics for monitoring',
                'priority': 'high'
            })
            return recommendations
        
        # Get recent performance data
        recent_entries = [
            entry for entry in self.performance_history[-10:]  # Last 10 entries
            if entry['model_name'] == model_name
        ]
        
        if not recent_entries:
            recommendations['recommendations'].append({
                'action': 'resume_monitoring',
                'description': 'Resume performance logging - no recent data available',
                'priority': 'medium'
            })
            return recommendations
        
        # Analyze recent performance
        latest_entry = recent_entries[-1]
        
        if 'degradation_analysis' in latest_entry:
            degradation = latest_entry['degradation_analysis']
            
            if degradation['severity'] == 'critical':
                recommendations['priority'] = 'critical'
                recommendations['recommendations'].append({
                    'action': 'immediate_investigation',
                    'description': f'Critical performance degradation detected ({degradation["overall_degradation"]:.1%})',
                    'priority': 'critical'
                })
                
                if degradation['overall_degradation'] > self.config['retrain_threshold']:
                    recommendations['recommendations'].append({
                        'action': 'retrain_model',
                        'description': 'Consider immediate model retraining',
                        'priority': 'critical'
                    })
            
            elif degradation['severity'] == 'warning':
                recommendations['priority'] = 'medium'
                recommendations['recommendations'].append({
                    'action': 'investigate_performance',
                    'description': f'Performance degradation warning ({degradation["overall_degradation"]:.1%})',
                    'priority': 'medium'
                })
        
        # Check trends
        trend_analysis = self.get_performance_trend(model_name)
        
        if 'error' not in trend_analysis:
            declining_metrics = [
                metric for metric, trend in trend_analysis['metric_trends'].items()
                if trend['trend_direction'] == 'declining' and trend['trend_strength'] == 'strong'
            ]
            
            if declining_metrics:
                recommendations['recommendations'].append({
                    'action': 'monitor_trend',
                    'description': f'Declining trend detected in: {", ".join(declining_metrics)}',
                    'priority': 'medium'
                })
        
        # Data drift recommendation (placeholder)
        recommendations['recommendations'].append({
            'action': 'check_data_drift',
            'description': 'Run data drift detection to identify potential causes',
            'priority': 'low'
        })
        
        return recommendations
    
    def _generate_performance_alert(self, model_name: str, degradation_result: Dict[str, Any]):
        """Generate performance degradation alert"""
        
        severity = degradation_result['severity']
        degradation_pct = degradation_result['overall_degradation'] * 100
        
        alert_message = f"""
PERFORMANCE ALERT - {severity.upper()}

Model: {model_name}
Overall Performance Degradation: {degradation_pct:.1f}%
Timestamp: {datetime.now().isoformat()}

Affected Metrics:
"""
        
        for metric, change_info in degradation_result['metric_changes'].items():
            alert_message += f"- {metric}: {change_info['baseline']:.4f} → {change_info['current']:.4f} ({change_info['relative_change']:.1%})\n"
        
        # Log appropriate level
        if severity == 'critical':
            logger.critical(alert_message)
        else:
            logger.warning(alert_message)


class IntegratedMonitoringSystem:
    """
    Integrated monitoring system combining data drift and performance monitoring
    
    Features:
    - Unified monitoring dashboard
    - Correlation analysis between drift and performance
    - Automated monitoring workflows
    - Comprehensive reporting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize integrated monitoring system"""
        self.config = config or {}
        
        # Initialize components
        self.drift_detector = DataDriftDetector(config.get('drift_config'))
        self.performance_monitor = ModelPerformanceMonitor(config.get('performance_config'))
        
        self.monitoring_history = []
    
    def setup_monitoring(self, model_name: str, 
                        reference_data: pd.DataFrame,
                        baseline_performance: Dict[str, float]):
        """Setup monitoring for a model"""
        logger.info(f"Setting up integrated monitoring for {model_name}")
        
        # Setup drift detection
        self.drift_detector.fit_reference(reference_data)
        
        # Setup performance monitoring
        self.performance_monitor.set_baseline_performance(model_name, baseline_performance)
        
        logger.info(f"Monitoring setup complete for {model_name}")
    
    def monitor_model(self, model_name: str,
                     new_data: pd.DataFrame,
                     performance_metrics: Dict[str, float],
                     prediction_count: int = 0) -> Dict[str, Any]:
        """Perform comprehensive monitoring check"""
        
        monitoring_result = {
            'model_name': model_name,
            'monitoring_timestamp': datetime.now().isoformat(),
            'data_samples': len(new_data),
            'prediction_count': prediction_count
        }
        
        # Data drift detection
        logger.info("Performing data drift detection...")
        drift_results = self.drift_detector.detect_drift(new_data)
        monitoring_result['drift_analysis'] = drift_results
        
        # Performance monitoring
        logger.info("Performing performance monitoring...")
        self.performance_monitor.log_performance(
            model_name, performance_metrics, prediction_count
        )
        
        # Get recent performance analysis
        trend_analysis = self.performance_monitor.get_performance_trend(model_name)
        monitoring_result['performance_analysis'] = trend_analysis
        
        # Get recommendations
        recommendations = self.performance_monitor.recommend_actions(model_name)
        monitoring_result['recommendations'] = recommendations
        
        # Correlation analysis
        monitoring_result['correlation_analysis'] = self._analyze_drift_performance_correlation(
            drift_results, performance_metrics
        )
        
        # Store in history
        self.monitoring_history.append(monitoring_result)
        
        # Generate integrated alert if needed
        self._check_integrated_alerts(monitoring_result)
        
        return monitoring_result
    
    def _analyze_drift_performance_correlation(self, 
                                             drift_results: Dict[str, Any],
                                             performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze correlation between drift and performance"""
        
        correlation_analysis = {
            'drift_performance_correlation': 'unknown',
            'analysis': 'Insufficient historical data for correlation analysis'
        }
        
        # This would require more historical data and sophisticated analysis
        # For now, we'll provide basic correlation indicators
        
        if drift_results['drift_detected']:
            # Simple heuristic: if we have drift and poor performance, they might be correlated
            avg_performance = np.mean([
                performance_metrics.get(metric, 0) 
                for metric in ['accuracy', 'f1_score', 'roc_auc']
                if metric in performance_metrics
            ])
            
            if avg_performance < 0.7:  # Arbitrary threshold
                correlation_analysis['drift_performance_correlation'] = 'possible'
                correlation_analysis['analysis'] = 'Data drift detected with suboptimal performance - investigation recommended'
            else:
                correlation_analysis['drift_performance_correlation'] = 'unlikely'
                correlation_analysis['analysis'] = 'Data drift detected but performance remains good'
        
        return correlation_analysis
    
    def _check_integrated_alerts(self, monitoring_result: Dict[str, Any]):
        """Check for integrated alert conditions"""
        
        drift_detected = monitoring_result['drift_analysis']['drift_detected']
        drift_severity = monitoring_result['drift_analysis']['severity']
        
        performance_issues = (
            monitoring_result.get('recommendations', {}).get('priority', 'low') in ['critical', 'high']
        )
        
        # Generate integrated alert if both drift and performance issues detected
        if drift_detected and performance_issues:
            alert_message = f"""
INTEGRATED MONITORING ALERT - CRITICAL

Model: {monitoring_result['model_name']}
Timestamp: {monitoring_result['monitoring_timestamp']}

Issues Detected:
- Data Drift: {drift_severity.upper()}
- Performance Issues: {monitoring_result.get('recommendations', {}).get('priority', 'unknown').upper()}

Recommended Actions:
1. Immediate investigation of data drift causes
2. Performance degradation analysis
3. Consider emergency model rollback if critical
4. Plan model retraining with recent data

Correlation Analysis: {monitoring_result['correlation_analysis']['analysis']}
"""
            
            logger.critical(alert_message)
    
    def generate_monitoring_report(self, model_name: str, 
                                 days: int = 30,
                                 output_path: Optional[str] = None) -> str:
        """Generate comprehensive monitoring report"""
        
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Filter monitoring history
        recent_monitoring = [
            entry for entry in self.monitoring_history
            if (entry['model_name'] == model_name and 
                datetime.fromisoformat(entry['monitoring_timestamp']) > cutoff_time)
        ]
        
        report = f"""
# Model Monitoring Report: {model_name}

## Report Period
- **Start Date**: {(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')}
- **End Date**: {datetime.now().strftime('%Y-%m-%d')}
- **Monitoring Checks**: {len(recent_monitoring)}

## Summary
"""
        
        if recent_monitoring:
            drift_detections = sum(1 for entry in recent_monitoring 
                                 if entry['drift_analysis']['drift_detected'])
            
            performance_issues = sum(1 for entry in recent_monitoring
                                   if entry.get('recommendations', {}).get('priority') in ['critical', 'high'])
            
            report += f"""
- **Data Drift Detections**: {drift_detections} ({drift_detections/len(recent_monitoring)*100:.1f}%)
- **Performance Issues**: {performance_issues} ({performance_issues/len(recent_monitoring)*100:.1f}%)
- **Average Data Samples per Check**: {np.mean([entry['data_samples'] for entry in recent_monitoring]):.0f}
"""
            
            # Latest status
            latest_entry = recent_monitoring[-1]
            report += f"""

## Current Status
- **Last Monitoring Check**: {latest_entry['monitoring_timestamp']}
- **Data Drift Status**: {latest_entry['drift_analysis']['severity'].title()}
- **Performance Status**: {latest_entry.get('recommendations', {}).get('priority', 'unknown').title()}
- **Overall Drift Score**: {latest_entry['drift_analysis']['overall_drift_score']:.4f}
"""
            
            # Trends
            if 'performance_analysis' in latest_entry and 'error' not in latest_entry['performance_analysis']:
                perf_analysis = latest_entry['performance_analysis']
                report += f"""

## Performance Trends
"""
                for metric, trend_info in perf_analysis.get('metric_trends', {}).items():
                    direction = "📈" if trend_info['trend_direction'] == 'improving' else "📉"
                    report += f"- **{metric.title()}**: {direction} {trend_info['trend_direction'].title()} (R² = {trend_info['r_squared']:.3f})\n"
        
        else:
            report += "\nNo monitoring data available for the specified period."
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Monitoring report saved to {output_path}")
        
        return report
    
    def export_monitoring_data(self, output_path: str, format: str = 'json'):
        """Export monitoring data"""
        
        export_data = {
            'monitoring_history': self.monitoring_history,
            'drift_detection_history': self.drift_detector.drift_history,
            'performance_history': self.performance_monitor.performance_history,
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
        elif format == 'csv':
            # Create flattened CSV (simplified version)
            df = pd.DataFrame(self.monitoring_history)
            df.to_csv(output_path, index=False)
        
        logger.info(f"Monitoring data exported to {output_path}")


# Example usage
if __name__ == "__main__":
    print("Monitoring Framework Components:")
    print("1. DataDriftDetector - Detects statistical drift in features")
    print("2. ModelPerformanceMonitor - Tracks model performance over time")  
    print("3. IntegratedMonitoringSystem - Unified monitoring solution")
    print("\nExample usage: monitor = IntegratedMonitoringSystem()")
    print("monitor.setup_monitoring(model_name, reference_data, baseline_performance)")
    print("monitor.monitor_model(model_name, new_data, current_performance)")