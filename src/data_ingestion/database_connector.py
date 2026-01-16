"""
Database Connector for Claims Data Ingestion

This module provides connectors for various database sources to extract
claims data for the ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import os
from urllib.parse import urlparse
import yaml

# Database connectors
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pymysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import cx_Oracle
    ORACLE_AVAILABLE = True
except ImportError:
    ORACLE_AVAILABLE = False

try:
    from sqlalchemy import create_engine, text
    import sqlalchemy as sa
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# AWS services
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Unified database connector for claims data extraction
    
    Supports:
    - PostgreSQL
    - MySQL
    - Oracle
    - AWS RDS
    - AWS Redshift
    - SQLAlchemy-compatible databases
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database connector
        
        Args:
            config: Database configuration dictionary
        """
        self.config = config or self._load_default_config()
        self.connections = {}
        self.engines = {}
        
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default database configuration"""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../../config/database_config.yaml'
        )
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            'default_batch_size': 10000,
            'connection_timeout': 30,
            'query_timeout': 300,
            'retry_attempts': 3,
            'retry_delay': 5
        }
    
    def connect(self, connection_name: str, 
                connection_params: Dict[str, Any]) -> bool:
        """
        Establish database connection
        
        Args:
            connection_name: Name for the connection
            connection_params: Connection parameters
            
        Returns:
            Boolean indicating success
        """
        try:
            db_type = connection_params.get('type', 'postgresql').lower()
            
            if SQLALCHEMY_AVAILABLE:
                # Use SQLAlchemy for unified connection handling
                engine = self._create_sqlalchemy_engine(connection_params)
                self.engines[connection_name] = engine
                
                # Test connection
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                logger.info(f"Connected to {db_type} database: {connection_name}")
                return True
                
            else:
                # Fallback to direct database drivers
                if db_type == 'postgresql' and POSTGRES_AVAILABLE:
                    return self._connect_postgresql(connection_name, connection_params)
                elif db_type == 'mysql' and MYSQL_AVAILABLE:
                    return self._connect_mysql(connection_name, connection_params)
                else:
                    logger.error(f"Database type {db_type} not supported or driver not available")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to {connection_name}: {e}")
            return False
    
    def _create_sqlalchemy_engine(self, params: Dict[str, Any]):
        """Create SQLAlchemy engine from parameters"""
        db_type = params.get('type', 'postgresql').lower()
        host = params.get('host', 'localhost')
        port = params.get('port')
        database = params.get('database')
        username = params.get('username')
        password = params.get('password')
        
        # Set default ports
        if not port:
            port = {
                'postgresql': 5432,
                'mysql': 3306,
                'oracle': 1521
            }.get(db_type, 5432)
        
        # Build connection string
        if db_type == 'postgresql':
            connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'mysql':
            connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == 'oracle':
            connection_string = f"oracle+cx_oracle://{username}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        # Additional parameters
        engine_params = {
            'pool_size': params.get('pool_size', 5),
            'max_overflow': params.get('max_overflow', 10),
            'pool_pre_ping': True,
            'connect_args': params.get('connect_args', {})
        }
        
        return create_engine(connection_string, **engine_params)
    
    def _connect_postgresql(self, connection_name: str, params: Dict[str, Any]) -> bool:
        """Connect to PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=params.get('host', 'localhost'),
                port=params.get('port', 5432),
                database=params['database'],
                user=params['username'],
                password=params['password'],
                connect_timeout=self.config['connection_timeout']
            )
            self.connections[connection_name] = conn
            return True
        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            return False
    
    def _connect_mysql(self, connection_name: str, params: Dict[str, Any]) -> bool:
        """Connect to MySQL database"""
        try:
            conn = pymysql.connect(
                host=params.get('host', 'localhost'),
                port=params.get('port', 3306),
                database=params['database'],
                user=params['username'],
                password=params['password'],
                connect_timeout=self.config['connection_timeout'],
                cursorclass=pymysql.cursors.DictCursor
            )
            self.connections[connection_name] = conn
            return True
        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            return False
    
    def execute_query(self, connection_name: str, query: str, 
                     params: Optional[Dict] = None,
                     fetch_size: Optional[int] = None) -> pd.DataFrame:
        """
        Execute query and return results as DataFrame
        
        Args:
            connection_name: Name of the connection to use
            query: SQL query to execute
            params: Query parameters
            fetch_size: Number of rows to fetch at once
            
        Returns:
            DataFrame with query results
        """
        if connection_name not in self.engines and connection_name not in self.connections:
            raise ValueError(f"Connection '{connection_name}' not found")
        
        try:
            fetch_size = fetch_size or self.config['default_batch_size']
            
            if connection_name in self.engines:
                # Use SQLAlchemy
                with self.engines[connection_name].connect() as conn:
                    if params:
                        result = conn.execute(text(query), params)
                    else:
                        result = conn.execute(text(query))
                    
                    # Fetch data in chunks
                    data = []
                    while True:
                        chunk = result.fetchmany(fetch_size)
                        if not chunk:
                            break
                        data.extend(chunk)
                    
                    # Convert to DataFrame
                    if data:
                        df = pd.DataFrame(data)
                        logger.info(f"Query executed successfully. Retrieved {len(df)} rows.")
                        return df
                    else:
                        return pd.DataFrame()
            
            else:
                # Use direct connection
                conn = self.connections[connection_name]
                df = pd.read_sql_query(query, conn, params=params)
                logger.info(f"Query executed successfully. Retrieved {len(df)} rows.")
                return df
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_claims_data(self, connection_name: str,
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       claim_status: Optional[List[str]] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve claims data with common filters
        
        Args:
            connection_name: Database connection name
            start_date: Start date for claim filtering
            end_date: End date for claim filtering
            claim_status: List of claim statuses to filter
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame with claims data
        """
        # Build base query
        base_query = """
        SELECT 
            claim_id,
            policy_number,
            claim_amount,
            claim_date,
            claim_status,
            claim_type,
            claimant_age,
            policy_duration_months,
            previous_claims_count,
            region,
            claim_description,
            investigation_flag,
            settlement_amount,
            processing_time_days,
            adjuster_notes
        FROM claims_data
        WHERE 1=1
        """
        
        params = {}
        
        # Add date filters
        if start_date:
            base_query += " AND claim_date >= :start_date"
            params['start_date'] = start_date
            
        if end_date:
            base_query += " AND claim_date <= :end_date"
            params['end_date'] = end_date
        
        # Add status filter
        if claim_status:
            placeholders = ','.join([f":status_{i}" for i in range(len(claim_status))])
            base_query += f" AND claim_status IN ({placeholders})"
            for i, status in enumerate(claim_status):
                params[f'status_{i}'] = status
        
        # Add limit
        if limit:
            base_query += f" LIMIT {limit}"
        
        # Order by claim date
        base_query += " ORDER BY claim_date DESC"
        
        return self.execute_query(connection_name, base_query, params)
    
    def get_table_schema(self, connection_name: str, table_name: str) -> Dict[str, Any]:
        """
        Get table schema information
        
        Args:
            connection_name: Database connection name
            table_name: Name of the table
            
        Returns:
            Dictionary with schema information
        """
        try:
            if connection_name in self.engines:
                engine = self.engines[connection_name]
                inspector = sa.inspect(engine)
                
                columns = inspector.get_columns(table_name)
                indexes = inspector.get_indexes(table_name)
                
                return {
                    'table_name': table_name,
                    'columns': [
                        {
                            'name': col['name'],
                            'type': str(col['type']),
                            'nullable': col['nullable'],
                            'default': col.get('default')
                        }
                        for col in columns
                    ],
                    'indexes': [
                        {
                            'name': idx['name'],
                            'columns': idx['column_names'],
                            'unique': idx['unique']
                        }
                        for idx in indexes
                    ]
                }
            else:
                # Fallback for direct connections
                query = f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                """
                
                schema_df = self.execute_query(connection_name, query)
                
                return {
                    'table_name': table_name,
                    'columns': schema_df.to_dict('records')
                }
                
        except Exception as e:
            logger.error(f"Failed to get schema for table {table_name}: {e}")
            return {}
    
    def test_connection(self, connection_name: str) -> bool:
        """
        Test database connection
        
        Args:
            connection_name: Name of connection to test
            
        Returns:
            Boolean indicating connection health
        """
        try:
            test_df = self.execute_query(connection_name, "SELECT 1 as test_column")
            return len(test_df) == 1 and test_df.iloc[0]['test_column'] == 1
        except Exception as e:
            logger.error(f"Connection test failed for {connection_name}: {e}")
            return False
    
    def close_connection(self, connection_name: str):
        """Close database connection"""
        try:
            if connection_name in self.engines:
                self.engines[connection_name].dispose()
                del self.engines[connection_name]
                
            if connection_name in self.connections:
                self.connections[connection_name].close()
                del self.connections[connection_name]
                
            logger.info(f"Connection {connection_name} closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_name}: {e}")
    
    def close_all_connections(self):
        """Close all database connections"""
        for conn_name in list(self.engines.keys()):
            self.close_connection(conn_name)
        
        for conn_name in list(self.connections.keys()):
            self.close_connection(conn_name)


class AWSRDSConnector(DatabaseConnector):
    """
    Specialized connector for AWS RDS databases
    
    Features:
    - IAM database authentication
    - SSL connection support
    - Parameter Store integration for credentials
    - CloudWatch metrics integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AWS RDS connector"""
        super().__init__(config)
        
        if AWS_AVAILABLE:
            self.rds_client = boto3.client('rds')
            self.ssm_client = boto3.client('ssm')
        else:
            logger.warning("AWS SDK not available. Some features may not work.")
    
    def connect_with_iam(self, connection_name: str, 
                        rds_params: Dict[str, Any]) -> bool:
        """
        Connect to RDS using IAM database authentication
        
        Args:
            connection_name: Name for the connection
            rds_params: RDS connection parameters
            
        Returns:
            Boolean indicating success
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available for IAM authentication")
            return False
        
        try:
            # Generate IAM auth token
            hostname = rds_params['hostname']
            port = rds_params.get('port', 5432)
            username = rds_params['username']
            region = rds_params.get('region', 'us-east-1')
            
            auth_token = self.rds_client.generate_db_auth_token(
                DBHostname=hostname,
                Port=port,
                DBUsername=username,
                Region=region
            )
            
            # Create connection with IAM token
            connection_params = {
                'type': rds_params.get('type', 'postgresql'),
                'host': hostname,
                'port': port,
                'database': rds_params['database'],
                'username': username,
                'password': auth_token,
                'connect_args': {
                    'sslmode': 'require'
                }
            }
            
            return self.connect(connection_name, connection_params)
            
        except Exception as e:
            logger.error(f"IAM authentication failed: {e}")
            return False
    
    def get_credentials_from_parameter_store(self, parameter_name: str) -> Dict[str, str]:
        """
        Retrieve database credentials from AWS Parameter Store
        
        Args:
            parameter_name: Name of the parameter in Parameter Store
            
        Returns:
            Dictionary with credentials
        """
        if not AWS_AVAILABLE:
            logger.error("AWS SDK not available")
            return {}
        
        try:
            response = self.ssm_client.get_parameter(
                Name=parameter_name,
                WithDecryption=True
            )
            
            # Assume parameter value is JSON
            import json
            credentials = json.loads(response['Parameter']['Value'])
            
            return credentials
            
        except Exception as e:
            logger.error(f"Failed to retrieve credentials from Parameter Store: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'default_batch_size': 5000,
        'connection_timeout': 30
    }
    
    # Initialize connector
    db_connector = DatabaseConnector(config)
    
    # Example connection parameters
    postgres_params = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'claims_db',
        'username': 'claims_user',
        'password': 'password123'
    }
    
    print("Database Connector Usage Example:")
    print("1. Connect to database: db_connector.connect('claims_db', postgres_params)")
    print("2. Execute query: df = db_connector.execute_query('claims_db', 'SELECT * FROM claims LIMIT 10')")
    print("3. Get claims data: claims_df = db_connector.get_claims_data('claims_db', limit=1000)")
    print("4. Test connection: health = db_connector.test_connection('claims_db')")
    print("5. Close connection: db_connector.close_connection('claims_db')")