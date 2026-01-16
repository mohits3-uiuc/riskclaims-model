"""
Schema Validator for Claims Risk Classification Pipeline

This module provides comprehensive schema validation for structured data
coming from various sources including databases, CSV files, and JSON data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import yaml
import logging
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataType(Enum):
    """Supported data types for schema validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"


@dataclass
class FieldSchema:
    """Schema definition for a single field"""
    name: str
    data_type: DataType
    required: bool = True
    nullable: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of schema validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field_validation_results: Dict[str, bool]
    summary: Dict[str, Any]


class SchemaValidator:
    """
    Comprehensive schema validator for claims data
    
    Supports validation of structured data against predefined schemas
    with detailed error reporting and data quality insights.
    """
    
    def __init__(self, schema_config: Optional[Union[str, Dict]] = None):
        """
        Initialize schema validator
        
        Args:
            schema_config: Path to schema config file or schema dictionary
        """
        self.schema = self._load_schema(schema_config)
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0
        }
        
    def _load_schema(self, schema_config: Optional[Union[str, Dict]]) -> Dict[str, FieldSchema]:
        """Load schema from config file or dictionary"""
        if schema_config is None:
            return self._get_default_claims_schema()
        
        if isinstance(schema_config, str):
            with open(schema_config, 'r') as f:
                schema_dict = yaml.safe_load(f)
        else:
            schema_dict = schema_config
            
        schema = {}
        for field_name, field_config in schema_dict.items():
            schema[field_name] = FieldSchema(
                name=field_name,
                data_type=DataType(field_config.get('type', 'string')),
                required=field_config.get('required', True),
                nullable=field_config.get('nullable', False),
                min_value=field_config.get('min_value'),
                max_value=field_config.get('max_value'),
                allowed_values=field_config.get('allowed_values'),
                pattern=field_config.get('pattern'),
                description=field_config.get('description')
            )
        
        return schema
    
    def _get_default_claims_schema(self) -> Dict[str, FieldSchema]:
        """Get default schema for claims data"""
        return {
            'claim_id': FieldSchema('claim_id', DataType.STRING, required=True),
            'claim_amount': FieldSchema('claim_amount', DataType.FLOAT, required=True, min_value=0),
            'claim_type': FieldSchema('claim_type', DataType.CATEGORICAL, required=True, 
                                    allowed_values=['auto', 'home', 'health', 'life']),
            'customer_id': FieldSchema('customer_id', DataType.STRING, required=True),
            'customer_age': FieldSchema('customer_age', DataType.INTEGER, required=True, 
                                      min_value=18, max_value=120),
            'policy_duration': FieldSchema('policy_duration', DataType.INTEGER, required=True, 
                                         min_value=0),
            'region': FieldSchema('region', DataType.CATEGORICAL, required=True,
                                allowed_values=['north', 'south', 'east', 'west', 'central']),
            'claim_date': FieldSchema('claim_date', DataType.DATE, required=True),
            'claim_description': FieldSchema('claim_description', DataType.STRING, required=False),
            'customer_segment': FieldSchema('customer_segment', DataType.CATEGORICAL, 
                                          required=False, allowed_values=['premium', 'standard', 'basic']),
            'risk_level': FieldSchema('risk_level', DataType.CATEGORICAL, required=False,
                                    allowed_values=['low', 'high'])
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate pandas DataFrame against schema
        
        Args:
            df: DataFrame to validate
            
        Returns:
            ValidationResult object with validation details
        """
        self.validation_stats['total_validations'] += 1
        errors = []
        warnings = []
        field_results = {}
        
        # Check for missing required columns
        required_fields = [field.name for field in self.schema.values() if field.required]
        missing_fields = set(required_fields) - set(df.columns)
        
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
            for field in missing_fields:
                field_results[field] = False
        
        # Check for unexpected columns
        unexpected_fields = set(df.columns) - set(self.schema.keys())
        if unexpected_fields:
            warnings.append(f"Unexpected fields found: {unexpected_fields}")
        
        # Validate each field
        for field_name, field_schema in self.schema.items():
            if field_name in df.columns:
                field_errors = self._validate_field(df[field_name], field_schema)
                if field_errors:
                    errors.extend(field_errors)
                    field_results[field_name] = False
                else:
                    field_results[field_name] = True
            elif not field_schema.required:
                field_results[field_name] = True
        
        # Generate summary
        summary = {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'missing_fields': list(missing_fields),
            'unexpected_fields': list(unexpected_fields),
            'validation_timestamp': datetime.now().isoformat()
        }
        
        is_valid = len(errors) == 0
        if is_valid:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            field_validation_results=field_results,
            summary=summary
        )
        
        logger.info(f"Schema validation completed. Valid: {is_valid}, Errors: {len(errors)}, Warnings: {len(warnings)}")
        return result
    
    def _validate_field(self, series: pd.Series, field_schema: FieldSchema) -> List[str]:
        """Validate a single field against its schema"""
        errors = []
        
        # Check for null values
        null_count = series.isnull().sum()
        if null_count > 0 and not field_schema.nullable:
            errors.append(f"Field '{field_schema.name}' contains {null_count} null values but is not nullable")
        
        # Skip further validation for null values
        non_null_series = series.dropna()
        if len(non_null_series) == 0:
            return errors
        
        # Data type validation
        if field_schema.data_type == DataType.INTEGER:
            if not pd.api.types.is_integer_dtype(non_null_series):
                errors.append(f"Field '{field_schema.name}' should be integer type")
                
        elif field_schema.data_type == DataType.FLOAT:
            if not pd.api.types.is_numeric_dtype(non_null_series):
                errors.append(f"Field '{field_schema.name}' should be numeric type")
                
        elif field_schema.data_type == DataType.DATE:
            try:
                pd.to_datetime(non_null_series, errors='raise')
            except:
                errors.append(f"Field '{field_schema.name}' contains invalid date values")
                
        elif field_schema.data_type == DataType.DATETIME:
            try:
                pd.to_datetime(non_null_series, errors='raise')
            except:
                errors.append(f"Field '{field_schema.name}' contains invalid datetime values")
        
        # Range validation
        if field_schema.min_value is not None and pd.api.types.is_numeric_dtype(non_null_series):
            if non_null_series.min() < field_schema.min_value:
                errors.append(f"Field '{field_schema.name}' has values below minimum {field_schema.min_value}")
        
        if field_schema.max_value is not None and pd.api.types.is_numeric_dtype(non_null_series):
            if non_null_series.max() > field_schema.max_value:
                errors.append(f"Field '{field_schema.name}' has values above maximum {field_schema.max_value}")
        
        # Allowed values validation
        if field_schema.allowed_values is not None:
            invalid_values = set(non_null_series) - set(field_schema.allowed_values)
            if invalid_values:
                errors.append(f"Field '{field_schema.name}' contains invalid values: {invalid_values}")
        
        # Pattern validation
        if field_schema.pattern is not None:
            pattern = re.compile(field_schema.pattern)
            invalid_count = sum(1 for val in non_null_series if not pattern.match(str(val)))
            if invalid_count > 0:
                errors.append(f"Field '{field_schema.name}' has {invalid_count} values not matching pattern '{field_schema.pattern}'")
        
        return errors
    
    def validate_json(self, json_data: Union[str, Dict, List[Dict]]) -> ValidationResult:
        """
        Validate JSON data against schema
        
        Args:
            json_data: JSON string, dict, or list of dicts
            
        Returns:
            ValidationResult object
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Convert to DataFrame for validation
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        return self.validate_dataframe(df)
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get summary of current schema"""
        return {
            'total_fields': len(self.schema),
            'required_fields': [f.name for f in self.schema.values() if f.required],
            'optional_fields': [f.name for f in self.schema.values() if not f.required],
            'nullable_fields': [f.name for f in self.schema.values() if f.nullable],
            'categorical_fields': [f.name for f in self.schema.values() if f.data_type == DataType.CATEGORICAL],
            'numeric_fields': [f.name for f in self.schema.values() 
                             if f.data_type in [DataType.INTEGER, DataType.FLOAT]],
            'validation_stats': self.validation_stats
        }
    
    def export_schema(self, file_path: str) -> None:
        """Export current schema to YAML file"""
        schema_dict = {}
        for field_name, field_schema in self.schema.items():
            schema_dict[field_name] = {
                'type': field_schema.data_type.value,
                'required': field_schema.required,
                'nullable': field_schema.nullable
            }
            
            if field_schema.min_value is not None:
                schema_dict[field_name]['min_value'] = field_schema.min_value
            if field_schema.max_value is not None:
                schema_dict[field_name]['max_value'] = field_schema.max_value
            if field_schema.allowed_values is not None:
                schema_dict[field_name]['allowed_values'] = field_schema.allowed_values
            if field_schema.pattern is not None:
                schema_dict[field_name]['pattern'] = field_schema.pattern
            if field_schema.description is not None:
                schema_dict[field_name]['description'] = field_schema.description
        
        with open(file_path, 'w') as f:
            yaml.dump(schema_dict, f, default_flow_style=False)
        
        logger.info(f"Schema exported to {file_path}")


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_data = pd.DataFrame({
        'claim_id': ['C001', 'C002', 'C003'],
        'claim_amount': [1500.0, 2500.0, 3000.0],
        'claim_type': ['auto', 'home', 'health'],
        'customer_id': ['CU001', 'CU002', 'CU003'],
        'customer_age': [30, 45, 35],
        'policy_duration': [12, 24, 18],
        'region': ['north', 'south', 'east'],
        'claim_date': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'claim_description': ['Car accident', 'Water damage', 'Medical emergency'],
        'risk_level': ['low', 'high', 'low']
    })
    
    # Initialize validator
    validator = SchemaValidator()
    
    # Validate data
    result = validator.validate_dataframe(sample_data)
    
    print("Validation Results:")
    print(f"Is Valid: {result.is_valid}")
    print(f"Errors: {result.errors}")
    print(f"Warnings: {result.warnings}")
    print(f"Summary: {result.summary}")
    
    # Export schema
    validator.export_schema("claim_schema.yaml")