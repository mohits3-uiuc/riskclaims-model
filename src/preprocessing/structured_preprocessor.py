"""
Structured Data Preprocessor for Claims Risk Classification Pipeline

This module handles preprocessing of structured data including:
- Data cleaning and normalization
- Feature engineering
- Categorical encoding
- Numerical feature scaling
- Missing value imputation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredDataPreprocessor:
    """
    Comprehensive preprocessor for structured claims data
    
    Features:
    - Automated data cleaning and validation
    - Missing value imputation strategies
    - Feature engineering and transformation
    - Categorical encoding (one-hot, label, target encoding)
    - Numerical feature scaling and normalization
    - Feature selection and dimensionality reduction
    - Data leakage prevention
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize structured data preprocessor
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._get_default_config()
        self.fitted_transformers = {}
        self.feature_names = []
        self.target_encoders = {}
        self.is_fitted = False
        
    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration"""
        return {
            # Missing value handling
            'numerical_imputation': 'median',  # 'mean', 'median', 'mode', 'knn'
            'categorical_imputation': 'mode',   # 'mode', 'constant', 'knn'
            'missing_value_threshold': 0.7,    # Drop columns with >70% missing
            
            # Feature scaling
            'scaling_method': 'standard',       # 'standard', 'minmax', 'robust'
            
            # Categorical encoding
            'categorical_encoding': 'onehot',   # 'onehot', 'label', 'target'
            'max_categories': 10,               # Max categories for one-hot encoding
            
            # Feature engineering
            'create_interaction_features': True,
            'create_polynomial_features': False,
            'polynomial_degree': 2,
            
            # Outlier handling
            'outlier_method': 'IQR',           # 'IQR', 'zscore', 'isolation'
            'outlier_action': 'cap',           # 'remove', 'cap', 'transform'
            
            # Feature selection
            'feature_selection': True,
            'selection_method': 'mutual_info', # 'f_classif', 'mutual_info', 'rfe'
            'max_features': 50,
            
            # Date/time features
            'extract_date_features': True,
            
            # Data validation
            'validate_data_types': True,
            'remove_constant_features': True,
            'remove_duplicate_features': True
        }
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'StructuredDataPreprocessor':
        """
        Fit the preprocessor to training data
        
        Args:
            X: Training features
            y: Target variable (optional, needed for some encoding methods)
            
        Returns:
            Self
        """
        logger.info("Fitting structured data preprocessor...")
        
        X = X.copy()
        self.original_features = X.columns.tolist()
        
        # Data validation and cleaning
        X = self._validate_and_clean_data(X)
        
        # Handle missing values
        X = self._fit_missing_value_handlers(X)
        
        # Engineer features
        X = self._fit_feature_engineering(X, y)
        
        # Fit encoders and scalers
        self._fit_transformers(X, y)
        
        # Feature selection
        if self.config['feature_selection'] and y is not None:
            self._fit_feature_selector(X, y)
        
        self.is_fitted = True
        logger.info("Structured data preprocessor fitted successfully")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        logger.info("Transforming structured data...")
        
        X = X.copy()
        
        # Apply same cleaning steps
        X = self._validate_and_clean_data(X)
        
        # Handle missing values
        X = self._transform_missing_values(X)
        
        # Engineer features
        X = self._transform_feature_engineering(X)
        
        # Apply transformations
        X = self._transform_features(X)
        
        # Apply feature selection
        if hasattr(self, 'feature_selector'):
            X = self._transform_feature_selection(X)
        
        logger.info(f"Data transformed. Final shape: {X.shape}")
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit preprocessor and transform data in one step
        
        Args:
            X: Training features
            y: Target variable (optional)
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _validate_and_clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean input data"""
        # Remove columns with too many missing values
        missing_threshold = self.config['missing_value_threshold']
        missing_ratios = X.isnull().mean()
        cols_to_drop = missing_ratios[missing_ratios > missing_threshold].index
        
        if len(cols_to_drop) > 0:
            logger.warning(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            X = X.drop(columns=cols_to_drop)
        
        # Remove constant features
        if self.config['remove_constant_features']:
            constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
            if len(constant_cols) > 0:
                logger.info(f"Removing {len(constant_cols)} constant features")
                X = X.drop(columns=constant_cols)
        
        # Remove duplicate columns
        if self.config['remove_duplicate_features']:
            X = X.loc[:, ~X.columns.duplicated()]
        
        return X
    
    def _fit_missing_value_handlers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit missing value imputers"""
        self.imputers = {}
        
        # Numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            if self.config['numerical_imputation'] == 'knn':
                self.imputers['numerical'] = KNNImputer(n_neighbors=5)
            else:
                strategy = self.config['numerical_imputation']
                self.imputers['numerical'] = SimpleImputer(strategy=strategy)
            
            self.imputers['numerical'].fit(X[numerical_cols])
        
        # Categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            if self.config['categorical_imputation'] == 'knn':
                # Convert to numerical for KNN, then back
                le_temp = LabelEncoder()
                X_cat_encoded = X[categorical_cols].apply(lambda x: le_temp.fit_transform(x.astype(str)))
                self.imputers['categorical'] = KNNImputer(n_neighbors=5)
                self.imputers['categorical'].fit(X_cat_encoded)
            else:
                strategy = self.config['categorical_imputation']
                fill_value = 'missing' if strategy == 'constant' else None
                self.imputers['categorical'] = SimpleImputer(
                    strategy=strategy, 
                    fill_value=fill_value
                )
                self.imputers['categorical'].fit(X[categorical_cols])
        
        return X
    
    def _transform_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform missing values using fitted imputers"""
        # Numerical imputation
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0 and 'numerical' in self.imputers:
            X[numerical_cols] = self.imputers['numerical'].transform(X[numerical_cols])
        
        # Categorical imputation
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0 and 'categorical' in self.imputers:
            X[categorical_cols] = self.imputers['categorical'].transform(X[categorical_cols])
        
        return X
    
    def _fit_feature_engineering(self, X: pd.DataFrame, y: Optional[pd.Series]) -> pd.DataFrame:
        """Fit feature engineering transformations"""
        self.engineered_features = []
        
        # Date feature extraction
        if self.config['extract_date_features']:
            date_cols = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_cols:
                try:
                    X[col] = pd.to_datetime(X[col])
                    # Extract date components
                    X[f'{col}_year'] = X[col].dt.year
                    X[f'{col}_month'] = X[col].dt.month
                    X[f'{col}_day'] = X[col].dt.day
                    X[f'{col}_weekday'] = X[col].dt.dayofweek
                    X[f'{col}_is_weekend'] = X[col].dt.weekday >= 5
                    X[f'{col}_days_since'] = (datetime.now() - X[col]).dt.days
                    
                    self.engineered_features.extend([
                        f'{col}_year', f'{col}_month', f'{col}_day', 
                        f'{col}_weekday', f'{col}_is_weekend', f'{col}_days_since'
                    ])
                except:
                    logger.warning(f"Could not parse {col} as date column")
        
        # Interaction features
        if self.config['create_interaction_features']:
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                # Create some key interactions (limit to avoid explosion)
                important_pairs = [
                    ('claim_amount', 'customer_age'),
                    ('claim_amount', 'policy_duration'),
                    ('customer_age', 'policy_duration')
                ]
                
                for col1, col2 in important_pairs:
                    if col1 in X.columns and col2 in X.columns:
                        interaction_name = f'{col1}_x_{col2}'
                        X[interaction_name] = X[col1] * X[col2]
                        self.engineered_features.append(interaction_name)
        
        # Domain-specific features
        X = self._create_domain_features(X)
        
        return X
    
    def _transform_feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        # Date features
        if self.config['extract_date_features']:
            date_cols = [col for col in X.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_cols:
                try:
                    X[col] = pd.to_datetime(X[col])
                    X[f'{col}_year'] = X[col].dt.year
                    X[f'{col}_month'] = X[col].dt.month
                    X[f'{col}_day'] = X[col].dt.day
                    X[f'{col}_weekday'] = X[col].dt.dayofweek
                    X[f'{col}_is_weekend'] = X[col].dt.weekday >= 5
                    X[f'{col}_days_since'] = (datetime.now() - X[col]).dt.days
                except:
                    pass
        
        # Interaction features
        if self.config['create_interaction_features']:
            important_pairs = [
                ('claim_amount', 'customer_age'),
                ('claim_amount', 'policy_duration'),
                ('customer_age', 'policy_duration')
            ]
            
            for col1, col2 in important_pairs:
                if col1 in X.columns and col2 in X.columns:
                    interaction_name = f'{col1}_x_{col2}'
                    X[interaction_name] = X[col1] * X[col2]
        
        # Domain features
        X = self._create_domain_features(X)
        
        return X
    
    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for claims data"""
        # Claim amount categories
        if 'claim_amount' in X.columns:
            X['claim_amount_category'] = pd.cut(
                X['claim_amount'], 
                bins=[0, 1000, 5000, 20000, np.inf],
                labels=['small', 'medium', 'large', 'very_large']
            )
            self.engineered_features.append('claim_amount_category')
        
        # Customer age groups
        if 'customer_age' in X.columns:
            X['age_group'] = pd.cut(
                X['customer_age'],
                bins=[0, 25, 35, 50, 65, np.inf],
                labels=['young', 'adult', 'middle_aged', 'senior', 'elderly']
            )
            self.engineered_features.append('age_group')
        
        # Policy duration categories
        if 'policy_duration' in X.columns:
            X['policy_duration_category'] = pd.cut(
                X['policy_duration'],
                bins=[0, 6, 12, 24, np.inf],
                labels=['new', 'short_term', 'medium_term', 'long_term']
            )
            self.engineered_features.append('policy_duration_category')
        
        return X
    
    def _fit_transformers(self, X: pd.DataFrame, y: Optional[pd.Series]):
        """Fit encoders and scalers"""
        # Separate numerical and categorical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.column_types = {
            'numerical': numerical_cols,
            'categorical': categorical_cols
        }
        
        # Fit categorical encoders
        if len(categorical_cols) > 0:
            self.encoders = {}
            
            for col in categorical_cols:
                unique_values = X[col].nunique()
                
                if self.config['categorical_encoding'] == 'onehot' and unique_values <= self.config['max_categories']:
                    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                    encoder.fit(X[[col]])
                    self.encoders[col] = {'type': 'onehot', 'encoder': encoder}
                    
                elif self.config['categorical_encoding'] == 'target' and y is not None:
                    # Target encoding
                    target_means = X.groupby(col)[y.name].mean() if hasattr(y, 'name') else X.assign(target=y).groupby(col)['target'].mean()
                    global_mean = y.mean()
                    self.encoders[col] = {'type': 'target', 'mapping': target_means.to_dict(), 'global_mean': global_mean}
                    
                else:
                    # Label encoding
                    encoder = LabelEncoder()
                    encoder.fit(X[col].astype(str))
                    self.encoders[col] = {'type': 'label', 'encoder': encoder}
        
        # Fit numerical scalers
        if len(numerical_cols) > 0:
            if self.config['scaling_method'] == 'standard':
                self.scaler = StandardScaler()
            elif self.config['scaling_method'] == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                self.scaler = StandardScaler()  # Default
            
            self.scaler.fit(X[numerical_cols])
    
    def _transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply encodings and scaling"""
        # Apply categorical encoding
        if hasattr(self, 'encoders'):
            for col, encoder_info in self.encoders.items():
                if col not in X.columns:
                    continue
                    
                if encoder_info['type'] == 'onehot':
                    encoded = encoder_info['encoder'].transform(X[[col]])
                    feature_names = encoder_info['encoder'].get_feature_names_out([col])
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
                    X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)
                    
                elif encoder_info['type'] == 'target':
                    mapping = encoder_info['mapping']
                    global_mean = encoder_info['global_mean']
                    X[col] = X[col].map(mapping).fillna(global_mean)
                    
                elif encoder_info['type'] == 'label':
                    # Handle unknown categories
                    unknown_mask = ~X[col].astype(str).isin(encoder_info['encoder'].classes_)
                    X.loc[unknown_mask, col] = encoder_info['encoder'].classes_[0]  # Use first class for unknown
                    X[col] = encoder_info['encoder'].transform(X[col].astype(str))
        
        # Apply numerical scaling
        if hasattr(self, 'scaler'):
            numerical_cols = [col for col in self.column_types['numerical'] if col in X.columns]
            if len(numerical_cols) > 0:
                X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X
    
    def _fit_feature_selector(self, X: pd.DataFrame, y: pd.Series):
        """Fit feature selector"""
        if self.config['selection_method'] == 'f_classif':
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(self.config['max_features'], X.shape[1]))
        else:
            self.feature_selector = SelectKBest(score_func=mutual_info_classif, k=min(self.config['max_features'], X.shape[1]))
        
        self.feature_selector.fit(X, y)
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        logger.info(f"Selected {len(self.selected_features)} features out of {X.shape[1]}")
    
    def _transform_feature_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection"""
        if hasattr(self, 'selected_features'):
            available_features = [col for col in self.selected_features if col in X.columns]
            X = X[available_features]
        
        return X
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from selector"""
        if not hasattr(self, 'feature_selector'):
            return pd.DataFrame()
        
        scores = self.feature_selector.scores_
        feature_names = self.column_types['numerical'] + self.column_types['categorical']
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(scores)],
            'importance': scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_preprocessor(self, file_path: str):
        """Save fitted preprocessor to disk"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        joblib.dump({
            'config': self.config,
            'fitted_transformers': self.fitted_transformers,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'imputers': getattr(self, 'imputers', {}),
            'encoders': getattr(self, 'encoders', {}),
            'scaler': getattr(self, 'scaler', None),
            'feature_selector': getattr(self, 'feature_selector', None),
            'selected_features': getattr(self, 'selected_features', []),
            'column_types': getattr(self, 'column_types', {}),
            'original_features': getattr(self, 'original_features', []),
            'engineered_features': getattr(self, 'engineered_features', [])
        }, file_path)
        
        logger.info(f"Preprocessor saved to {file_path}")
    
    @classmethod
    def load_preprocessor(cls, file_path: str) -> 'StructuredDataPreprocessor':
        """Load fitted preprocessor from disk"""
        data = joblib.load(file_path)
        
        preprocessor = cls(data['config'])
        preprocessor.fitted_transformers = data['fitted_transformers']
        preprocessor.feature_names = data['feature_names']
        preprocessor.is_fitted = data['is_fitted']
        
        # Restore all fitted components
        for attr, value in data.items():
            if attr not in ['config']:
                setattr(preprocessor, attr, value)
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of preprocessing steps applied"""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "original_features": len(getattr(self, 'original_features', [])),
            "final_features": len(getattr(self, 'selected_features', [])),
            "engineered_features": len(getattr(self, 'engineered_features', [])),
            "numerical_features": len(self.column_types.get('numerical', [])),
            "categorical_features": len(self.column_types.get('categorical', [])),
            "scaling_method": self.config['scaling_method'],
            "encoding_method": self.config['categorical_encoding'],
            "feature_selection": self.config['feature_selection']
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'claim_id': [f'C{i:03d}' for i in range(1000)],
        'claim_amount': np.random.lognormal(7, 1, 1000),
        'customer_age': np.random.randint(18, 80, 1000),
        'policy_duration': np.random.randint(1, 60, 1000),
        'claim_type': np.random.choice(['auto', 'home', 'health'], 1000),
        'region': np.random.choice(['north', 'south', 'east', 'west'], 1000),
        'claim_date': pd.date_range('2020-01-01', periods=1000, freq='D')[:1000],
        'customer_segment': np.random.choice(['premium', 'standard', 'basic'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(1000, 50, False), 'customer_age'] = np.nan
    sample_data.loc[np.random.choice(1000, 30, False), 'claim_type'] = np.nan
    
    # Target variable
    y = np.random.choice(['low', 'high'], 1000)
    y_series = pd.Series(y, name='risk_level')
    
    # Initialize and fit preprocessor
    preprocessor = StructuredDataPreprocessor()
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(sample_data, y_series)
    
    print("Preprocessing Summary:")
    print(preprocessor.get_preprocessing_summary())
    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"\nFirst few transformed features: {X_transformed.columns[:10].tolist()}")
    
    # Save preprocessor
    preprocessor.save_preprocessor("structured_preprocessor.pkl")