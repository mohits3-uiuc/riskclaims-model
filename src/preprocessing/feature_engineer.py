"""
Feature Engineer for Claims Risk Classification Pipeline

This module provides advanced feature engineering capabilities that combine
structured and unstructured data to create powerful predictive features
for risk classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Advanced feature engineering for claims risk classification
    
    Features:
    - Cross-feature interactions
    - Domain-specific risk indicators
    - Statistical aggregations
    - Time-based features
    - Risk scoring algorithms
    - Feature selection and dimensionality reduction
    - Automated feature generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.is_fitted = False
        self.feature_generators = {}
        self.feature_selectors = {}
        self.risk_models = {}
        
    def _get_default_config(self) -> Dict:
        """Get default feature engineering configuration"""
        return {
            # Feature creation
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'polynomial_degree': 2,
            'create_ratio_features': True,
            'create_binning_features': True,
            
            # Domain-specific features
            'create_risk_scores': True,
            'create_fraud_indicators': True,
            'create_severity_indicators': True,
            
            # Statistical features
            'create_statistical_features': True,
            'rolling_window_days': 30,
            
            # Feature selection
            'perform_feature_selection': True,
            'selection_methods': ['mutual_info', 'f_classif'],
            'max_features_ratio': 0.8,
            'min_feature_importance': 0.001,
            
            # Dimensionality reduction
            'perform_pca': False,
            'pca_variance_threshold': 0.95,
            
            # Feature validation
            'check_feature_correlation': True,
            'correlation_threshold': 0.95,
            'remove_low_variance_features': True,
            'variance_threshold': 0.01
        }
    
    def fit(self, 
            structured_df: pd.DataFrame, 
            unstructured_df: Optional[pd.DataFrame] = None,
            target: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit feature engineer on training data
        
        Args:
            structured_df: Structured features DataFrame
            unstructured_df: Unstructured features DataFrame (optional)
            target: Target variable for supervised feature engineering
            
        Returns:
            Self
        """
        logger.info("Fitting feature engineer...")
        
        # Combine structured and unstructured features
        if unstructured_df is not None:
            combined_df = pd.concat([structured_df, unstructured_df], axis=1)
        else:
            combined_df = structured_df.copy()
        
        self.base_features = combined_df.columns.tolist()
        
        # Create engineered features
        engineered_df = self._create_engineered_features(combined_df, target)
        
        # Fit feature selection methods
        if self.config['perform_feature_selection'] and target is not None:
            self._fit_feature_selection(engineered_df, target)
        
        # Fit dimensionality reduction
        if self.config['perform_pca']:
            self._fit_pca(engineered_df)
        
        self.is_fitted = True
        logger.info("Feature engineer fitted successfully")
        return self
    
    def transform(self, 
                  structured_df: pd.DataFrame, 
                  unstructured_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Transform data using fitted feature engineer
        
        Args:
            structured_df: Structured features DataFrame
            unstructured_df: Unstructured features DataFrame (optional)
            
        Returns:
            Engineered features DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer must be fitted before transform")
        
        logger.info("Engineering features...")
        
        # Combine features
        if unstructured_df is not None:
            combined_df = pd.concat([structured_df, unstructured_df], axis=1)
        else:
            combined_df = structured_df.copy()
        
        # Create engineered features
        engineered_df = self._create_engineered_features(combined_df)
        
        # Apply feature selection
        if hasattr(self, 'selected_features'):
            available_features = [f for f in self.selected_features if f in engineered_df.columns]
            engineered_df = engineered_df[available_features]
        
        # Apply PCA if fitted
        if hasattr(self, 'pca'):
            pca_features = self.pca.transform(engineered_df)
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'pca_{i}' for i in range(pca_features.shape[1])],
                index=engineered_df.index
            )
            engineered_df = pd.concat([engineered_df, pca_df], axis=1)
        
        logger.info(f"Feature engineering completed. Final shape: {engineered_df.shape}")
        return engineered_df
    
    def fit_transform(self, 
                     structured_df: pd.DataFrame, 
                     unstructured_df: Optional[pd.DataFrame] = None,
                     target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step
        
        Args:
            structured_df: Structured features DataFrame
            unstructured_df: Unstructured features DataFrame (optional)
            target: Target variable
            
        Returns:
            Engineered features DataFrame
        """
        return self.fit(structured_df, unstructured_df, target).transform(structured_df, unstructured_df)
    
    def _create_engineered_features(self, df: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """Create all engineered features"""
        engineered_df = df.copy()
        
        # Basic mathematical operations
        if self.config['create_ratio_features']:
            engineered_df = self._create_ratio_features(engineered_df)
        
        # Interaction features
        if self.config['create_interaction_features']:
            engineered_df = self._create_interaction_features(engineered_df)
        
        # Polynomial features (use cautiously)
        if self.config['create_polynomial_features']:
            engineered_df = self._create_polynomial_features(engineered_df)
        
        # Binning features
        if self.config['create_binning_features']:
            engineered_df = self._create_binning_features(engineered_df)
        
        # Domain-specific features
        if self.config['create_risk_scores']:
            engineered_df = self._create_risk_scores(engineered_df)
        
        if self.config['create_fraud_indicators']:
            engineered_df = self._create_fraud_indicators(engineered_df)
        
        if self.config['create_severity_indicators']:
            engineered_df = self._create_severity_indicators(engineered_df)
        
        # Statistical features
        if self.config['create_statistical_features']:
            engineered_df = self._create_statistical_features(engineered_df)
        
        # Clean up infinite and NaN values
        engineered_df = engineered_df.replace([np.inf, -np.inf], np.nan)
        engineered_df = engineered_df.fillna(0)
        
        return engineered_df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and division features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Important ratios for insurance claims
        ratio_pairs = [
            ('claim_amount', 'customer_age'),
            ('claim_amount', 'policy_duration'),
            ('customer_age', 'policy_duration'),
            ('text_length', 'word_count'),  # If text features exist
            ('sentiment_positive', 'sentiment_negative')  # If sentiment features exist
        ]
        
        for col1, col2 in ratio_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Avoid division by zero
                denominator = df[col2].replace(0, 1e-6)
                df[f'{col1}_to_{col2}_ratio'] = df[col1] / denominator
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Specific interactions that make business sense
        interactions = [
            ('claim_amount', 'customer_age', 'multiply'),
            ('claim_amount', 'policy_duration', 'multiply'),
            ('customer_age', 'policy_duration', 'multiply'),
            ('sentiment_compound', 'text_length', 'multiply'),  # If available
        ]
        
        for col1, col2, operation in interactions:
            if col1 in df.columns and col2 in df.columns:
                if operation == 'multiply':
                    df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                elif operation == 'add':
                    df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for selected numerical columns"""
        # Only apply to most important numerical features to avoid explosion
        important_numerical_cols = []
        
        # Select important features
        for col in ['claim_amount', 'customer_age', 'policy_duration']:
            if col in df.columns:
                important_numerical_cols.append(col)
        
        if len(important_numerical_cols) > 0 and len(important_numerical_cols) <= 5:
            poly = PolynomialFeatures(
                degree=self.config['polynomial_degree'], 
                include_bias=False,
                interaction_only=True  # Only interactions, not pure polynomials
            )
            
            poly_features = poly.fit_transform(df[important_numerical_cols])
            feature_names = poly.get_feature_names_out(important_numerical_cols)
            
            # Only keep new features (not the original ones)
            new_features = poly_features[:, len(important_numerical_cols):]
            new_feature_names = feature_names[len(important_numerical_cols):]
            
            poly_df = pd.DataFrame(
                new_features,
                columns=[f'poly_{name}' for name in new_feature_names],
                index=df.index
            )
            
            df = pd.concat([df, poly_df], axis=1)
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned categorical features from numerical ones"""
        # Claim amount bins
        if 'claim_amount' in df.columns:
            df['claim_amount_bin'] = pd.cut(
                df['claim_amount'],
                bins=[0, 500, 2000, 10000, 50000, np.inf],
                labels=['very_low', 'low', 'medium', 'high', 'very_high']
            ).astype(str)
        
        # Customer age bins
        if 'customer_age' in df.columns:
            df['age_bin'] = pd.cut(
                df['customer_age'],
                bins=[0, 25, 35, 50, 65, 100],
                labels=['young', 'adult', 'middle', 'senior', 'elderly']
            ).astype(str)
        
        # Policy duration bins
        if 'policy_duration' in df.columns:
            df['policy_duration_bin'] = pd.cut(
                df['policy_duration'],
                bins=[0, 6, 12, 24, 60, np.inf],
                labels=['new', 'short', 'medium', 'long', 'very_long']
            ).astype(str)
        
        return df
    
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        # Basic risk score based on claim amount and customer profile
        risk_components = []
        
        if 'claim_amount' in df.columns:
            # Normalize claim amount (higher = more risk)
            claim_risk = (df['claim_amount'] - df['claim_amount'].min()) / (df['claim_amount'].max() - df['claim_amount'].min())
            risk_components.append(claim_risk)
        
        if 'customer_age' in df.columns:
            # Age risk (very young and very old = higher risk)
            age_risk = np.abs(df['customer_age'] - 45) / 45  # 45 as optimal age
            risk_components.append(age_risk)
        
        if 'policy_duration' in df.columns:
            # Duration risk (very new policies = higher risk)
            duration_risk = 1 / (1 + df['policy_duration'] / 12)  # Convert to years
            risk_components.append(duration_risk)
        
        # Sentiment-based risk (if available)
        if 'sentiment_negative' in df.columns:
            risk_components.append(df['sentiment_negative'])
        
        if risk_components:
            df['composite_risk_score'] = np.mean(risk_components, axis=0)
        
        # Fraud risk indicators
        fraud_indicators = []
        
        # High claim amounts
        if 'claim_amount' in df.columns:
            claim_95th = df['claim_amount'].quantile(0.95)
            fraud_indicators.append((df['claim_amount'] > claim_95th).astype(int))
        
        # Negative sentiment
        if 'sentiment_negative' in df.columns:
            fraud_indicators.append((df['sentiment_negative'] > 0.5).astype(int))
        
        # Multiple damage types mentioned (if text features available)
        for damage_type in ['collision', 'theft', 'fire', 'flood']:
            col_name = f'damage_types_mentions'  # From text processing
            if col_name in df.columns:
                fraud_indicators.append((df[col_name] > 2).astype(int))
                break
        
        if fraud_indicators:
            df['fraud_risk_score'] = np.sum(fraud_indicators, axis=0) / len(fraud_indicators)
        
        return df
    
    def _create_fraud_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create specific fraud detection indicators"""
        # Round number claims (often suspicious)
        if 'claim_amount' in df.columns:
            df['round_number_claim'] = (df['claim_amount'] % 100 == 0).astype(int)
            df['very_round_claim'] = (df['claim_amount'] % 1000 == 0).astype(int)
        
        # Weekend claims (potentially higher fraud risk)
        if 'claim_date' in df.columns:
            try:
                claim_dates = pd.to_datetime(df['claim_date'])
                df['weekend_claim'] = (claim_dates.dt.weekday >= 5).astype(int)
                df['late_night_claim'] = (claim_dates.dt.hour >= 22).astype(int)
            except:
                pass
        
        # Text-based fraud indicators
        fraud_keywords = ['staged', 'suspicious', 'fraudulent', 'fake', 'false']
        if 'fraud_indicators_mentions' in df.columns:
            df['fraud_keywords_present'] = (df['fraud_indicators_mentions'] > 0).astype(int)
        
        # Inconsistency indicators
        if 'text_length' in df.columns and 'claim_amount' in df.columns:
            # Very short descriptions for high-value claims
            df['short_desc_high_claim'] = ((df['text_length'] < 50) & (df['claim_amount'] > df['claim_amount'].quantile(0.8))).astype(int)
        
        return df
    
    def _create_severity_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create claim severity indicators"""
        # Medical severity indicators
        medical_keywords = ['hospital', 'surgery', 'ambulance', 'emergency', 'severe']
        if 'medical_terms_mentions' in df.columns:
            df['high_medical_severity'] = (df['medical_terms_mentions'] >= 2).astype(int)
        
        # Damage severity
        severity_keywords = ['total', 'destroyed', 'demolished', 'extensive', 'major']
        if 'severity_indicators_mentions' in df.columns:
            df['high_damage_severity'] = (df['severity_indicators_mentions'] >= 1).astype(int)
        
        # Multi-party involvement (higher complexity/severity)
        if 'person_mentions' in df.columns:
            df['multi_party_claim'] = (df['person_mentions'] > 2).astype(int)
        
        # Legal involvement indicators
        legal_keywords = ['lawyer', 'attorney', 'legal', 'court', 'lawsuit']
        # This would need to be extracted from text separately
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Create rolling statistics if we have date information
        if 'claim_date' in df.columns:
            try:
                df_sorted = df.sort_values('claim_date')
                
                # Rolling mean of claim amounts
                if 'claim_amount' in df.columns:
                    df['claim_amount_rolling_mean'] = df_sorted['claim_amount'].rolling(window=30, min_periods=1).mean()
                
                # Count of recent claims by customer (if customer_id available)
                if 'customer_id' in df.columns:
                    df['recent_claims_count'] = df_sorted.groupby('customer_id')['claim_date'].rolling(window='30D').count().values
                
            except Exception as e:
                logger.warning(f"Could not create time-based features: {e}")
        
        # Percentile features
        for col in ['claim_amount', 'customer_age', 'policy_duration']:
            if col in df.columns:
                df[f'{col}_percentile'] = df[col].rank(pct=True)
                df[f'{col}_is_outlier'] = ((df[col] < df[col].quantile(0.05)) | (df[col] > df[col].quantile(0.95))).astype(int)
        
        return df
    
    def _fit_feature_selection(self, df: pd.DataFrame, target: pd.Series):
        """Fit feature selection methods"""
        self.feature_selectors = {}
        
        # Mutual information selector
        if 'mutual_info' in self.config['selection_methods']:
            max_features = min(int(len(df.columns) * self.config['max_features_ratio']), len(df.columns))
            self.feature_selectors['mutual_info'] = SelectKBest(
                score_func=mutual_info_classif,
                k=max_features
            )
            self.feature_selectors['mutual_info'].fit(df, target)
        
        # F-score selector
        if 'f_classif' in self.config['selection_methods']:
            max_features = min(int(len(df.columns) * self.config['max_features_ratio']), len(df.columns))
            self.feature_selectors['f_classif'] = SelectKBest(
                score_func=f_classif,
                k=max_features
            )
            self.feature_selectors['f_classif'].fit(df, target)
        
        # Combine selection results
        selected_features_sets = []
        for selector_name, selector in self.feature_selectors.items():
            selected_mask = selector.get_support()
            selected_features = df.columns[selected_mask].tolist()
            selected_features_sets.append(set(selected_features))
        
        # Take intersection of selected features
        if selected_features_sets:
            self.selected_features = list(set.intersection(*selected_features_sets))
        else:
            self.selected_features = df.columns.tolist()
        
        logger.info(f"Feature selection: {len(self.selected_features)} features selected from {len(df.columns)}")
    
    def _fit_pca(self, df: pd.DataFrame):
        """Fit PCA for dimensionality reduction"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 10:  # Only apply PCA if we have many features
            self.pca = PCA(n_components=self.config['pca_variance_threshold'])
            self.pca.fit(df[numerical_cols])
            logger.info(f"PCA fitted: {self.pca.n_components_} components explain {self.config['pca_variance_threshold']*100}% variance")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from selection methods"""
        importance_dfs = []
        
        for selector_name, selector in self.feature_selectors.items():
            if hasattr(selector, 'scores_'):
                importance_df = pd.DataFrame({
                    'feature': selector.feature_names_in_,
                    'importance': selector.scores_,
                    'method': selector_name
                })
                importance_dfs.append(importance_df)
        
        if importance_dfs:
            combined_importance = pd.concat(importance_dfs, ignore_index=True)
            return combined_importance.sort_values('importance', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_feature_engineer(self, file_path: str):
        """Save fitted feature engineer to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted feature engineer")
        
        joblib.dump({
            'config': self.config,
            'is_fitted': self.is_fitted,
            'base_features': self.base_features,
            'feature_selectors': self.feature_selectors,
            'selected_features': getattr(self, 'selected_features', []),
            'pca': getattr(self, 'pca', None)
        }, file_path)
        
        logger.info(f"Feature engineer saved to {file_path}")
    
    @classmethod
    def load_feature_engineer(cls, file_path: str) -> 'FeatureEngineer':
        """Load fitted feature engineer from disk"""
        data = joblib.load(file_path)
        
        feature_engineer = cls(data['config'])
        feature_engineer.is_fitted = data['is_fitted']
        feature_engineer.base_features = data['base_features']
        feature_engineer.feature_selectors = data['feature_selectors']
        
        if 'selected_features' in data:
            feature_engineer.selected_features = data['selected_features']
        
        if data.get('pca') is not None:
            feature_engineer.pca = data['pca']
        
        logger.info(f"Feature engineer loaded from {file_path}")
        return feature_engineer
    
    def get_engineering_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process"""
        summary = {
            'is_fitted': self.is_fitted,
            'config': self.config
        }
        
        if self.is_fitted:
            summary.update({
                'base_features_count': len(self.base_features),
                'selected_features_count': len(getattr(self, 'selected_features', [])),
                'feature_selection_methods': list(self.feature_selectors.keys()),
                'pca_enabled': hasattr(self, 'pca')
            })
            
            if hasattr(self, 'pca'):
                summary['pca_components'] = self.pca.n_components_
                summary['pca_explained_variance'] = self.pca.explained_variance_ratio_.sum()
        
        return summary


# Example usage
if __name__ == "__main__":
    # Create sample structured data
    np.random.seed(42)
    structured_data = pd.DataFrame({
        'claim_amount': np.random.lognormal(7, 1, 1000),
        'customer_age': np.random.randint(18, 80, 1000),
        'policy_duration': np.random.randint(1, 60, 1000),
        'claim_date': pd.date_range('2020-01-01', periods=1000)
    })
    
    # Create sample unstructured data (text features)
    unstructured_data = pd.DataFrame({
        'text_length': np.random.randint(10, 500, 1000),
        'word_count': np.random.randint(5, 100, 1000),
        'sentiment_positive': np.random.uniform(0, 1, 1000),
        'sentiment_negative': np.random.uniform(0, 1, 1000),
        'medical_terms_mentions': np.random.randint(0, 5, 1000)
    })
    
    # Target variable
    target = pd.Series(np.random.choice(['low', 'high'], 1000, p=[0.7, 0.3]), name='risk_level')
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Fit and transform
    engineered_features = feature_engineer.fit_transform(
        structured_data, 
        unstructured_data, 
        target
    )
    
    print("Feature Engineering Summary:")
    summary = feature_engineer.get_engineering_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nOriginal features: {structured_data.shape[1] + unstructured_data.shape[1]}")
    print(f"Engineered features: {engineered_features.shape[1]}")
    print(f"Sample feature names: {engineered_features.columns[:10].tolist()}")
    
    # Get feature importance
    importance_df = feature_engineer.get_feature_importance()
    if not importance_df.empty:
        print(f"\nTop 10 most important features:")
        print(importance_df.head(10))
    
    # Save feature engineer
    feature_engineer.save_feature_engineer("feature_engineer.pkl")