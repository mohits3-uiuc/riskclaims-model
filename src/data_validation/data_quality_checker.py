"""
Data Quality Checker for Claims Risk Classification Pipeline

This module provides comprehensive data quality assessment for both
structured and unstructured data, including statistical analysis,
anomaly detection, and data profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import re
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataQualityIssue:
    """Represents a data quality issue"""
    severity: str  # 'critical', 'high', 'medium', 'low'
    category: str  # 'missing_data', 'outliers', 'duplicates', 'inconsistency', 'format'
    description: str
    affected_fields: List[str]
    affected_records: int
    recommendation: str


@dataclass
class DataQualityReport:
    """Comprehensive data quality assessment report"""
    dataset_name: str
    total_records: int
    total_fields: int
    quality_score: float  # 0-100
    issues: List[DataQualityIssue] = field(default_factory=list)
    field_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class DataQualityChecker:
    """
    Comprehensive data quality checker for claims data
    
    Performs various data quality checks including:
    - Missing data analysis
    - Duplicate detection
    - Outlier detection
    - Data consistency checks
    - Statistical profiling
    - Text quality assessment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize data quality checker
        
        Args:
            config: Configuration dictionary with thresholds and settings
        """
        self.config = config or self._get_default_config()
        self.quality_history = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for data quality checks"""
        return {
            'missing_data_threshold': 0.15,  # 15% missing data threshold
            'outlier_method': 'IQR',  # 'IQR', 'zscore', 'isolation_forest'
            'outlier_threshold': 1.5,
            'duplicate_threshold': 0.05,  # 5% duplicates threshold
            'text_min_length': 5,
            'text_max_length': 1000,
            'date_range_years': 10,
            'categorical_max_cardinality': 100,
            'numerical_zero_threshold': 0.50  # 50% zeros in numerical field
        }
    
    def assess_data_quality(self, 
                           df: pd.DataFrame, 
                           dataset_name: str = "Claims Dataset") -> DataQualityReport:
        """
        Perform comprehensive data quality assessment
        
        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset
            
        Returns:
            DataQualityReport with detailed assessment
        """
        logger.info(f"Starting data quality assessment for {dataset_name}")
        
        report = DataQualityReport(
            dataset_name=dataset_name,
            total_records=len(df),
            total_fields=len(df.columns),
            quality_score=0.0
        )
        
        # Perform various quality checks
        self._check_missing_data(df, report)
        self._check_duplicates(df, report)
        self._check_outliers(df, report)
        self._check_data_types(df, report)
        self._check_data_consistency(df, report)
        self._profile_fields(df, report)
        self._generate_statistics(df, report)
        
        # Calculate overall quality score
        report.quality_score = self._calculate_quality_score(report)
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)
        
        # Store in history
        self.quality_history.append({
            'timestamp': report.timestamp,
            'dataset_name': dataset_name,
            'quality_score': report.quality_score,
            'total_issues': len(report.issues)
        })
        
        logger.info(f"Data quality assessment completed. Score: {report.quality_score:.2f}/100")
        return report
    
    def _check_missing_data(self, df: pd.DataFrame, report: DataQualityReport):
        """Check for missing data issues"""
        missing_stats = df.isnull().sum()
        missing_percentage = (missing_stats / len(df)) * 100
        
        for field, missing_count in missing_stats.items():
            if missing_count > 0:
                missing_pct = missing_percentage[field]
                
                if missing_pct > self.config['missing_data_threshold'] * 100:
                    severity = 'critical' if missing_pct > 50 else 'high'
                    report.issues.append(DataQualityIssue(
                        severity=severity,
                        category='missing_data',
                        description=f"Field '{field}' has {missing_pct:.1f}% missing values",
                        affected_fields=[field],
                        affected_records=missing_count,
                        recommendation=f"Investigate missing data pattern and consider imputation strategies"
                    ))
        
        # Overall missing data statistics
        report.statistics['missing_data'] = {
            'fields_with_missing': len(missing_stats[missing_stats > 0]),
            'total_missing_values': missing_stats.sum(),
            'average_missing_percentage': missing_percentage.mean(),
            'fields_missing_details': missing_percentage[missing_percentage > 0].to_dict()
        }
    
    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport):
        """Check for duplicate records"""
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = (duplicate_count / len(df)) * 100
        
        if duplicate_count > 0:
            severity = 'high' if duplicate_percentage > self.config['duplicate_threshold'] * 100 else 'medium'
            report.issues.append(DataQualityIssue(
                severity=severity,
                category='duplicates',
                description=f"Found {duplicate_count} duplicate records ({duplicate_percentage:.1f}%)",
                affected_fields=list(df.columns),
                affected_records=duplicate_count,
                recommendation="Remove duplicate records or investigate if they are legitimate"
            ))
        
        # Check for duplicate IDs if ID columns exist
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for id_col in id_columns:
            id_duplicates = df[id_col].duplicated().sum()
            if id_duplicates > 0:
                report.issues.append(DataQualityIssue(
                    severity='critical',
                    category='duplicates',
                    description=f"Found {id_duplicates} duplicate IDs in '{id_col}'",
                    affected_fields=[id_col],
                    affected_records=id_duplicates,
                    recommendation=f"Ensure '{id_col}' contains unique identifiers"
                ))
        
        report.statistics['duplicates'] = {
            'duplicate_records': duplicate_count,
            'duplicate_percentage': duplicate_percentage,
            'duplicate_id_fields': {col: df[col].duplicated().sum() for col in id_columns}
        }
    
    def _check_outliers(self, df: pd.DataFrame, report: DataQualityReport):
        """Check for outliers in numerical fields"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_stats = {}
        for col in numerical_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) < 10:  # Skip if too few values
                continue
            
            outliers = self._detect_outliers(series, self.config['outlier_method'])
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(series)) * 100
            
            outlier_stats[col] = {
                'count': outlier_count,
                'percentage': outlier_percentage,
                'values': outliers.tolist() if len(outliers) < 10 else f"{len(outliers)} outliers detected"
            }
            
            if outlier_percentage > 10:  # More than 10% outliers
                severity = 'high' if outlier_percentage > 20 else 'medium'
                report.issues.append(DataQualityIssue(
                    severity=severity,
                    category='outliers',
                    description=f"Field '{col}' has {outlier_percentage:.1f}% outliers",
                    affected_fields=[col],
                    affected_records=outlier_count,
                    recommendation="Investigate outliers and consider data transformation or cleaning"
                ))
        
        report.statistics['outliers'] = outlier_stats
    
    def _detect_outliers(self, series: pd.Series, method: str) -> pd.Series:
        """Detect outliers using specified method"""
        if method == 'IQR':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.config['outlier_threshold'] * IQR
            upper_bound = Q3 + self.config['outlier_threshold'] * IQR
            return series[(series < lower_bound) | (series > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            return series[z_scores > 3]
        
        else:  # Default to IQR
            return self._detect_outliers(series, 'IQR')
    
    def _check_data_types(self, df: pd.DataFrame, report: DataQualityReport):
        """Check data types and format consistency"""
        type_issues = []
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            # Check for mixed types
            if series.dtype == 'object':
                # Check if should be numeric
                numeric_count = sum(1 for val in series if self._is_numeric(str(val)))
                if numeric_count > 0.8 * len(series):
                    type_issues.append({
                        'field': col,
                        'issue': 'should_be_numeric',
                        'description': f"Field '{col}' appears to be numeric but stored as text"
                    })
                
                # Check date fields
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        pd.to_datetime(series.head(100), errors='raise')
                    except:
                        type_issues.append({
                            'field': col,
                            'issue': 'invalid_date_format',
                            'description': f"Field '{col}' contains invalid date formats"
                        })
        
        if type_issues:
            for issue in type_issues:
                report.issues.append(DataQualityIssue(
                    severity='medium',
                    category='format',
                    description=issue['description'],
                    affected_fields=[issue['field']],
                    affected_records=0,  # Would need to calculate based on specific issue
                    recommendation="Convert to appropriate data type or fix format inconsistencies"
                ))
        
        report.statistics['data_types'] = {
            'type_distribution': df.dtypes.value_counts().to_dict(),
            'object_fields': df.select_dtypes(include=['object']).columns.tolist(),
            'numeric_fields': df.select_dtypes(include=[np.number]).columns.tolist()
        }
    
    def _check_data_consistency(self, df: pd.DataFrame, report: DataQualityReport):
        """Check for data consistency issues"""
        consistency_issues = []
        
        # Check for negative values where they shouldn't be
        negative_fields = ['amount', 'age', 'duration', 'count', 'quantity']
        for col in df.columns:
            if any(field in col.lower() for field in negative_fields):
                if col in df.select_dtypes(include=[np.number]).columns:
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        consistency_issues.append({
                            'field': col,
                            'issue': 'negative_values',
                            'count': negative_count,
                            'description': f"Field '{col}' has {negative_count} negative values"
                        })
        
        # Check for excessive zeros
        for col in df.select_dtypes(include=[np.number]).columns:
            zero_count = (df[col] == 0).sum()
            zero_percentage = zero_count / len(df)
            if zero_percentage > self.config['numerical_zero_threshold']:
                consistency_issues.append({
                    'field': col,
                    'issue': 'excessive_zeros',
                    'count': zero_count,
                    'description': f"Field '{col}' has {zero_percentage:.1%} zero values"
                })
        
        for issue in consistency_issues:
            severity = 'high' if issue['count'] > len(df) * 0.1 else 'medium'
            report.issues.append(DataQualityIssue(
                severity=severity,
                category='inconsistency',
                description=issue['description'],
                affected_fields=[issue['field']],
                affected_records=issue['count'],
                recommendation="Investigate data collection process and validate business logic"
            ))
    
    def _profile_fields(self, df: pd.DataFrame, report: DataQualityReport):
        """Generate detailed profile for each field"""
        for col in df.columns:
            series = df[col]
            profile = {
                'type': str(series.dtype),
                'non_null_count': series.count(),
                'null_count': series.isnull().sum(),
                'unique_count': series.nunique(),
                'cardinality': series.nunique() / len(series) if len(series) > 0 else 0
            }
            
            if series.dtype in ['int64', 'float64']:
                # Numerical field profile
                desc_stats = series.describe()
                profile.update({
                    'mean': desc_stats['mean'],
                    'std': desc_stats['std'],
                    'min': desc_stats['min'],
                    'max': desc_stats['max'],
                    'q25': desc_stats['25%'],
                    'q50': desc_stats['50%'],
                    'q75': desc_stats['75%'],
                    'zeros_count': (series == 0).sum(),
                    'negative_count': (series < 0).sum()
                })
            
            elif series.dtype == 'object':
                # Text/categorical field profile
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    profile.update({
                        'avg_length': non_null_series.astype(str).str.len().mean(),
                        'min_length': non_null_series.astype(str).str.len().min(),
                        'max_length': non_null_series.astype(str).str.len().max(),
                        'most_frequent': non_null_series.value_counts().index[0] if len(non_null_series) > 0 else None,
                        'frequency_of_most_frequent': non_null_series.value_counts().iloc[0] if len(non_null_series) > 0 else 0
                    })
                    
                    # Check if it's a categorical field (low cardinality)
                    if series.nunique() < self.config['categorical_max_cardinality']:
                        profile['value_counts'] = series.value_counts().to_dict()
            
            report.field_profiles[col] = profile
    
    def _generate_statistics(self, df: pd.DataFrame, report: DataQualityReport):
        """Generate overall dataset statistics"""
        report.statistics['dataset'] = {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'completeness': ((df.count().sum()) / (df.shape[0] * df.shape[1])) * 100,
            'field_types': df.dtypes.value_counts().to_dict()
        }
    
    def _calculate_quality_score(self, report: DataQualityReport) -> float:
        """Calculate overall data quality score (0-100)"""
        base_score = 100.0
        
        # Deduct points for issues based on severity
        severity_weights = {
            'critical': 20,
            'high': 10,
            'medium': 5,
            'low': 2
        }
        
        for issue in report.issues:
            deduction = severity_weights.get(issue.severity, 2)
            # Scale deduction by affected records percentage
            if report.total_records > 0:
                affected_percentage = issue.affected_records / report.total_records
                scaled_deduction = deduction * min(affected_percentage * 2, 1.0)  # Cap at 100% impact
                base_score -= scaled_deduction
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, report: DataQualityReport) -> List[str]:
        """Generate actionable recommendations based on issues"""
        recommendations = []
        
        issue_categories = {}
        for issue in report.issues:
            category = issue.category
            if category not in issue_categories:
                issue_categories[category] = []
            issue_categories[category].append(issue)
        
        # Generate category-specific recommendations
        if 'missing_data' in issue_categories:
            recommendations.append("Implement missing data handling strategies (imputation, removal, or flagging)")
        
        if 'duplicates' in issue_categories:
            recommendations.append("Establish data deduplication processes and unique constraint validations")
        
        if 'outliers' in issue_categories:
            recommendations.append("Investigate outlier patterns and implement outlier detection in data pipeline")
        
        if 'inconsistency' in issue_categories:
            recommendations.append("Review data collection processes and implement business rule validations")
        
        if 'format' in issue_categories:
            recommendations.append("Standardize data formats and implement data type validations")
        
        # Overall recommendations based on quality score
        if report.quality_score < 70:
            recommendations.append("Consider comprehensive data cleaning before model training")
        
        if report.quality_score < 50:
            recommendations.append("Review data sources and collection processes - significant quality issues detected")
        
        return recommendations
    
    def _is_numeric(self, value: str) -> bool:
        """Check if string value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def generate_quality_report_html(self, report: DataQualityReport, output_path: str):
        """Generate HTML report for data quality assessment"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {report.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; color: {'green' if report.quality_score > 80 else 'orange' if report.quality_score > 60 else 'red'}; }}
                .section {{ margin: 20px 0; }}
                .issue {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-left: 5px solid red; }}
                .recommendation {{ background-color: #e6f3ff; padding: 10px; margin: 5px 0; border-left: 5px solid blue; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {report.dataset_name}</p>
                <p><strong>Timestamp:</strong> {report.timestamp}</p>
                <p><strong>Records:</strong> {report.total_records:,}</p>
                <p><strong>Fields:</strong> {report.total_fields}</p>
                <p class="score">Quality Score: {report.quality_score:.1f}/100</p>
            </div>
            
            <div class="section">
                <h2>Issues Found ({len(report.issues)})</h2>
                {''.join([f'<div class="issue"><strong>{issue.severity.upper()}:</strong> {issue.description}<br><em>Recommendation: {issue.recommendation}</em></div>' for issue in report.issues])}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {''.join([f'<div class="recommendation">{rec}</div>' for rec in report.recommendations])}
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Quality report saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Create sample data with quality issues
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'claim_id': ['C001', 'C002', 'C003', 'C002', 'C004'],  # Duplicate
        'claim_amount': [1500.0, -500.0, 50000.0, 2500.0, np.nan],  # Negative, outlier, missing
        'customer_age': [30, 200, 35, 45, 25],  # Outlier
        'claim_description': ['Valid claim', '', 'A' * 2000, 'Normal claim', np.nan],  # Empty, too long, missing
        'region': ['north', 'south', 'EAST', 'west', 'unknown']  # Case inconsistency
    })
    
    # Initialize checker
    checker = DataQualityChecker()
    
    # Assess quality
    report = checker.assess_data_quality(sample_data, "Sample Claims Data")
    
    # Print results
    print(f"Quality Score: {report.quality_score:.2f}/100")
    print(f"Issues Found: {len(report.issues)}")
    for issue in report.issues:
        print(f"- {issue.severity.upper()}: {issue.description}")
    
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")
    
    # Generate HTML report
    checker.generate_quality_report_html(report, "quality_report.html")