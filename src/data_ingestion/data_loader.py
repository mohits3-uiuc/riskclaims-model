"""
Unified Data Loader for Claims Risk Classification Pipeline

This module orchestrates data ingestion from multiple sources and provides
a unified interface for loading claims data for ML pipeline processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import concurrent.futures
from dataclasses import dataclass

from .database_connector import DatabaseConnector, AWSRDSConnector
from .s3_connector import S3DataConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    source_type: str  # 'database', 's3', 'api', 'stream'
    connection_params: Dict[str, Any]
    data_format: str  # 'structured', 'unstructured', 'mixed'
    enabled: bool = True
    priority: int = 1  # Lower number = higher priority
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DataLoader:
    """
    Unified data loader for claims risk classification
    
    Features:
    - Multi-source data ingestion (databases, S3, APIs)
    - Parallel data loading
    - Data validation and quality checks
    - Incremental loading support
    - Data lineage tracking
    - Error handling and retry mechanisms
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.data_sources: Dict[str, DataSource] = {}
        self.connectors = {}
        self.load_history = []
        
        # Initialize connectors
        self._initialize_connectors()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'parallel_loading': True,
            'max_workers': 4,
            'batch_size': 10000,
            'enable_data_validation': True,
            'enable_quality_checks': True,
            'retry_attempts': 3,
            'retry_delay': 5,
            'incremental_loading': True,
            'data_cache_enabled': False,
            'cache_ttl_hours': 24,
            'output_formats': ['pandas', 'parquet', 'json'],
            'temp_storage_path': '/tmp/claims_data_loader'
        }
    
    def _initialize_connectors(self):
        """Initialize data connectors"""
        try:
            self.connectors['database'] = DatabaseConnector(
                self.config.get('database_config')
            )
            self.connectors['rds'] = AWSRDSConnector(
                self.config.get('rds_config')
            )
            self.connectors['s3'] = S3DataConnector(
                self.config.get('s3_config')
            )
            logger.info("Data connectors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connectors: {e}")
            raise
    
    def register_data_source(self, data_source: DataSource):
        """
        Register a new data source
        
        Args:
            data_source: DataSource configuration
        """
        self.data_sources[data_source.name] = data_source
        logger.info(f"Registered data source: {data_source.name} ({data_source.source_type})")
        
        # Initialize connection if needed
        if data_source.source_type in ['database', 'rds'] and data_source.enabled:
            self._initialize_database_connection(data_source)
    
    def _initialize_database_connection(self, data_source: DataSource):
        """Initialize database connection for a data source"""
        try:
            connector = self.connectors.get(data_source.source_type)
            if connector:
                success = connector.connect(
                    data_source.name, 
                    data_source.connection_params
                )
                if success:
                    logger.info(f"Connected to {data_source.name}")
                else:
                    logger.error(f"Failed to connect to {data_source.name}")
        except Exception as e:
            logger.error(f"Error initializing connection for {data_source.name}: {e}")
    
    def load_data(self, 
                 source_names: Optional[List[str]] = None,
                 date_range: Optional[Tuple[datetime, datetime]] = None,
                 filters: Optional[Dict[str, Any]] = None,
                 output_format: str = 'pandas') -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load data from specified sources
        
        Args:
            source_names: List of data source names (all if None)
            date_range: Tuple of (start_date, end_date) for filtering
            filters: Additional filters to apply
            output_format: Output format ('pandas', 'dict', 'combined')
            
        Returns:
            Loaded data in specified format
        """
        if not source_names:
            source_names = list(self.data_sources.keys())
        
        # Filter enabled sources
        active_sources = [
            name for name in source_names 
            if name in self.data_sources and self.data_sources[name].enabled
        ]
        
        if not active_sources:
            logger.warning("No active data sources found")
            return pd.DataFrame() if output_format == 'pandas' else {}
        
        logger.info(f"Loading data from {len(active_sources)} sources: {active_sources}")
        
        # Load data from sources
        if self.config['parallel_loading'] and len(active_sources) > 1:
            loaded_data = self._load_data_parallel(active_sources, date_range, filters)
        else:
            loaded_data = self._load_data_sequential(active_sources, date_range, filters)
        
        # Process output format
        return self._format_output(loaded_data, output_format)
    
    def _load_data_parallel(self, 
                           source_names: List[str],
                           date_range: Optional[Tuple[datetime, datetime]],
                           filters: Optional[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Load data from multiple sources in parallel"""
        loaded_data = {}
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config['max_workers']
        ) as executor:
            
            # Submit load tasks
            future_to_source = {
                executor.submit(
                    self._load_from_single_source, 
                    source_name, 
                    date_range, 
                    filters
                ): source_name 
                for source_name in source_names
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        loaded_data[source_name] = data
                        logger.info(f"Loaded {len(data)} records from {source_name}")
                    else:
                        logger.warning(f"No data loaded from {source_name}")
                        
                except Exception as e:
                    logger.error(f"Failed to load data from {source_name}: {e}")
        
        return loaded_data
    
    def _load_data_sequential(self, 
                             source_names: List[str],
                             date_range: Optional[Tuple[datetime, datetime]],
                             filters: Optional[Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """Load data from sources sequentially"""
        loaded_data = {}
        
        for source_name in source_names:
            try:
                data = self._load_from_single_source(source_name, date_range, filters)
                if data is not None and not data.empty:
                    loaded_data[source_name] = data
                    logger.info(f"Loaded {len(data)} records from {source_name}")
                else:
                    logger.warning(f"No data loaded from {source_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load data from {source_name}: {e}")
        
        return loaded_data
    
    def _load_from_single_source(self, 
                               source_name: str,
                               date_range: Optional[Tuple[datetime, datetime]],
                               filters: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Load data from a single source"""
        data_source = self.data_sources[source_name]
        
        try:
            if data_source.source_type in ['database', 'rds']:
                return self._load_from_database(data_source, date_range, filters)
                
            elif data_source.source_type == 's3':
                return self._load_from_s3(data_source, date_range, filters)
                
            else:
                logger.error(f"Unsupported source type: {data_source.source_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading from {source_name}: {e}")
            
            # Retry logic
            for attempt in range(self.config['retry_attempts']):
                logger.info(f"Retry attempt {attempt + 1} for {source_name}")
                try:
                    import time
                    time.sleep(self.config['retry_delay'])
                    
                    if data_source.source_type in ['database', 'rds']:
                        return self._load_from_database(data_source, date_range, filters)
                    elif data_source.source_type == 's3':
                        return self._load_from_s3(data_source, date_range, filters)
                        
                except Exception as retry_e:
                    logger.error(f"Retry {attempt + 1} failed: {retry_e}")
            
            logger.error(f"All retry attempts failed for {source_name}")
            return None
    
    def _load_from_database(self, 
                          data_source: DataSource,
                          date_range: Optional[Tuple[datetime, datetime]],
                          filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load data from database source"""
        connector = self.connectors[data_source.source_type]
        
        # Build query based on data source configuration
        if 'query' in data_source.connection_params:
            # Use custom query
            query = data_source.connection_params['query']
            params = {}
            
            # Add date range parameters
            if date_range:
                start_date, end_date = date_range
                query += " AND claim_date >= :start_date AND claim_date <= :end_date"
                params.update({'start_date': start_date, 'end_date': end_date})
            
            # Add additional filters
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ','.join([f":filter_{key}_{i}" for i in range(len(value))])
                        query += f" AND {key} IN ({placeholders})"
                        for i, v in enumerate(value):
                            params[f'filter_{key}_{i}'] = v
                    else:
                        query += f" AND {key} = :filter_{key}"
                        params[f'filter_{key}'] = value
            
            return connector.execute_query(data_source.name, query, params)
        
        else:
            # Use standard claims data method
            start_date = date_range[0] if date_range else None
            end_date = date_range[1] if date_range else None
            
            claim_status = None
            limit = None
            
            if filters:
                claim_status = filters.get('claim_status')
                limit = filters.get('limit', self.config['batch_size'])
            
            return connector.get_claims_data(
                data_source.name,
                start_date=start_date,
                end_date=end_date,
                claim_status=claim_status,
                limit=limit
            )
    
    def _load_from_s3(self, 
                     data_source: DataSource,
                     date_range: Optional[Tuple[datetime, datetime]],
                     filters: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """Load data from S3 source"""
        connector = self.connectors['s3']
        
        bucket = data_source.connection_params.get('bucket')
        prefix = data_source.connection_params.get('prefix', '')
        file_format = data_source.connection_params.get('file_format', 'csv')
        
        if not bucket:
            raise ValueError(f"Bucket not specified for S3 source {data_source.name}")
        
        # List objects with date filtering if specified
        if date_range:
            start_date, end_date = date_range
            # Implement date-based filtering for S3 objects
            # This is a simplified approach - in practice, you'd use S3 object keys with date patterns
            objects = connector.list_objects(
                bucket, 
                prefix,
                max_keys=self.config['batch_size']
            )
            
            # Filter objects by date (assuming date is in the key or metadata)
            filtered_objects = []
            for obj in objects:
                obj_date = obj['last_modified'].replace(tzinfo=None)
                if start_date <= obj_date <= end_date:
                    filtered_objects.append(obj)
            
            objects = filtered_objects
        else:
            objects = connector.list_objects(
                bucket, 
                prefix,
                max_keys=self.config['batch_size']
            )
        
        if not objects:
            logger.warning(f"No objects found in s3://{bucket}/{prefix}")
            return pd.DataFrame()
        
        # Load and combine data from multiple S3 objects
        dataframes = []
        
        for obj in objects:
            try:
                if data_source.data_format == 'structured':
                    df = connector.read_structured_data(bucket, obj['key'])
                    
                    # Add metadata columns
                    df['_source_file'] = obj['key']
                    df['_source_bucket'] = bucket
                    df['_load_timestamp'] = datetime.utcnow()
                    
                    dataframes.append(df)
                    
                elif data_source.data_format == 'unstructured':
                    # Process unstructured data and create DataFrame
                    content_data = connector.read_unstructured_data(bucket, obj['key'])
                    
                    if content_data['processing_status'] == 'success':
                        # Create DataFrame from unstructured content
                        df = pd.DataFrame([{
                            'file_key': obj['key'],
                            'content': content_data['content'],
                            'file_format': content_data['file_format'],
                            'metadata': json.dumps(content_data['metadata']),
                            '_source_bucket': bucket,
                            '_load_timestamp': datetime.utcnow()
                        }])
                        dataframes.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load S3 object {obj['key']}: {e}")
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Apply filters if specified
        if filters and not combined_df.empty:
            combined_df = self._apply_filters(combined_df, filters)
        
        return combined_df
    
    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to DataFrame"""
        filtered_df = df.copy()
        
        for column, value in filters.items():
            if column in filtered_df.columns:
                if isinstance(value, list):
                    filtered_df = filtered_df[filtered_df[column].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        return filtered_df
    
    def _format_output(self, 
                      loaded_data: Dict[str, pd.DataFrame],
                      output_format: str) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Format output according to specified format"""
        
        if output_format == 'dict':
            return loaded_data
        
        elif output_format == 'pandas' or output_format == 'combined':
            if len(loaded_data) == 0:
                return pd.DataFrame()
            
            elif len(loaded_data) == 1:
                return list(loaded_data.values())[0]
            
            else:
                # Combine all dataframes
                combined_dataframes = []
                
                for source_name, df in loaded_data.items():
                    df = df.copy()
                    df['_data_source'] = source_name
                    combined_dataframes.append(df)
                
                return pd.concat(combined_dataframes, ignore_index=True)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def load_incremental_data(self, 
                             source_name: str,
                             last_update_timestamp: datetime,
                             output_format: str = 'pandas') -> pd.DataFrame:
        """
        Load incremental data since last update
        
        Args:
            source_name: Name of the data source
            last_update_timestamp: Timestamp of last data load
            output_format: Output format
            
        Returns:
            DataFrame with incremental data
        """
        if source_name not in self.data_sources:
            raise ValueError(f"Unknown data source: {source_name}")
        
        current_time = datetime.utcnow()
        
        # Load data with date filter
        incremental_data = self.load_data(
            source_names=[source_name],
            date_range=(last_update_timestamp, current_time),
            output_format=output_format
        )
        
        logger.info(f"Loaded incremental data from {source_name}: {len(incremental_data) if hasattr(incremental_data, '__len__') else 'N/A'} records")
        
        return incremental_data
    
    def validate_data_sources(self) -> Dict[str, bool]:
        """
        Validate all registered data sources
        
        Returns:
            Dictionary mapping source names to validation status
        """
        validation_results = {}
        
        for source_name, data_source in self.data_sources.items():
            if not data_source.enabled:
                validation_results[source_name] = False
                continue
            
            try:
                if data_source.source_type in ['database', 'rds']:
                    connector = self.connectors[data_source.source_type]
                    validation_results[source_name] = connector.test_connection(source_name)
                
                elif data_source.source_type == 's3':
                    bucket = data_source.connection_params.get('bucket')
                    prefix = data_source.connection_params.get('prefix', '')
                    
                    connector = self.connectors['s3']
                    objects = connector.list_objects(bucket, prefix, max_keys=1)
                    validation_results[source_name] = len(objects) > 0
                
                else:
                    validation_results[source_name] = False
                    
            except Exception as e:
                logger.error(f"Validation failed for {source_name}: {e}")
                validation_results[source_name] = False
        
        return validation_results
    
    def get_data_profile(self, source_name: str, 
                        sample_size: int = 1000) -> Dict[str, Any]:
        """
        Get data profile for a source
        
        Args:
            source_name: Name of the data source
            sample_size: Number of records to sample for profiling
            
        Returns:
            Data profile dictionary
        """
        try:
            # Load sample data
            sample_data = self.load_data(
                source_names=[source_name],
                filters={'limit': sample_size}
            )
            
            if sample_data.empty:
                return {'error': 'No data available for profiling'}
            
            # Generate profile
            profile = {
                'source_name': source_name,
                'sample_size': len(sample_data),
                'columns': list(sample_data.columns),
                'data_types': sample_data.dtypes.to_dict(),
                'null_counts': sample_data.isnull().sum().to_dict(),
                'unique_counts': sample_data.nunique().to_dict(),
                'memory_usage': sample_data.memory_usage().sum(),
                'profiling_timestamp': datetime.utcnow().isoformat()
            }
            
            # Add numeric statistics for numeric columns
            numeric_columns = sample_data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                profile['numeric_stats'] = sample_data[numeric_columns].describe().to_dict()
            
            return profile
            
        except Exception as e:
            logger.error(f"Data profiling failed for {source_name}: {e}")
            return {'error': str(e)}
    
    def close_all_connections(self):
        """Close all active connections"""
        for connector in self.connectors.values():
            if hasattr(connector, 'close_all_connections'):
                connector.close_all_connections()
        
        logger.info("All data loader connections closed")


# Example configuration for different data sources
def create_example_data_sources() -> List[DataSource]:
    """Create example data source configurations"""
    
    # Database source
    db_source = DataSource(
        name='claims_database',
        source_type='database',
        connection_params={
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'claims_db',
            'username': 'claims_user',
            'password': 'password123'
        },
        data_format='structured',
        priority=1,
        metadata={'description': 'Primary claims database'}
    )
    
    # S3 structured data source
    s3_structured_source = DataSource(
        name='s3_claims_structured',
        source_type='s3',
        connection_params={
            'bucket': 'claims-data-lake',
            'prefix': 'structured/claims/',
            'file_format': 'parquet'
        },
        data_format='structured',
        priority=2,
        metadata={'description': 'Structured claims data in S3'}
    )
    
    # S3 unstructured data source
    s3_unstructured_source = DataSource(
        name='s3_claims_documents',
        source_type='s3',
        connection_params={
            'bucket': 'claims-data-lake',
            'prefix': 'documents/',
            'file_format': 'mixed'
        },
        data_format='unstructured',
        priority=3,
        metadata={'description': 'Claims documents and images in S3'}
    )
    
    return [db_source, s3_structured_source, s3_unstructured_source]


# Example usage
if __name__ == "__main__":
    print("Data Loader Usage Examples:")
    print("1. Initialize: loader = DataLoader(config)")
    print("2. Register source: loader.register_data_source(data_source)")
    print("3. Load data: df = loader.load_data(['source1', 'source2'])")
    print("4. Incremental load: df = loader.load_incremental_data('source1', last_timestamp)")
    print("5. Validate sources: results = loader.validate_data_sources()")
    print("6. Data profiling: profile = loader.get_data_profile('source1')")
    
    # Show example data source creation
    example_sources = create_example_data_sources()
    print(f"\nExample data sources created: {len(example_sources)}")
    for source in example_sources:
        print(f"- {source.name}: {source.source_type} ({source.data_format})")