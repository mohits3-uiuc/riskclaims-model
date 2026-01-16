"""
S3 Data Connector for Claims Risk Classification Pipeline

This module provides comprehensive S3 integration for ingesting structured 
and unstructured claims data from AWS S3 data lakes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import os
import io
import json
from pathlib import Path
import mimetypes
from urllib.parse import urlparse

# AWS SDK
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# File processing libraries
try:
    import PyPDF2
    PDF_PROCESSING = True
except ImportError:
    PDF_PROCESSING = False

try:
    from PIL import Image
    IMAGE_PROCESSING = True
except ImportError:
    IMAGE_PROCESSING = False

try:
    import openpyxl
    EXCEL_PROCESSING = True
except ImportError:
    EXCEL_PROCESSING = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class S3DataConnector:
    """
    Comprehensive S3 data connector for claims data ingestion
    
    Features:
    - Structured data ingestion (CSV, JSON, Parquet, Excel)
    - Unstructured data processing (PDF, images, text files)
    - Batch and streaming data processing
    - Data cataloging and metadata extraction
    - AWS Glue integration
    - Lifecycle management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize S3 data connector
        
        Args:
            config: S3 configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        if AWS_AVAILABLE:
            self.s3_client = boto3.client('s3')
            self.s3_resource = boto3.resource('s3')
        else:
            logger.error("AWS SDK not available. S3 operations will fail.")
            raise ImportError("boto3 is required for S3 operations")
        
        self.supported_formats = {
            'structured': ['.csv', '.json', '.parquet', '.xlsx', '.xls'],
            'unstructured': ['.pdf', '.txt', '.doc', '.docx', '.jpg', '.jpeg', '.png', '.tiff'],
            'compressed': ['.zip', '.gz', '.tar', '.tar.gz']
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default S3 configuration"""
        return {
            'default_bucket': 'claims-data-lake',
            'batch_size': 1000,
            'max_file_size_mb': 500,
            'enable_multipart_upload': True,
            'multipart_threshold': 100 * 1024 * 1024,  # 100MB
            'max_concurrency': 10,
            'enable_metadata_extraction': True,
            'temp_dir': '/tmp/s3_processing',
            'supported_regions': ['us-east-1', 'us-west-2', 'eu-west-1']
        }
    
    def list_objects(self, bucket: str, prefix: str = '', 
                    max_keys: int = 1000,
                    file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List objects in S3 bucket with filtering
        
        Args:
            bucket: S3 bucket name
            prefix: Object prefix to filter by
            max_keys: Maximum number of objects to return
            file_extensions: List of file extensions to filter by
            
        Returns:
            List of object metadata dictionaries
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_keys}
            )
            
            objects = []
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Filter by file extension if specified
                        if file_extensions:
                            _, ext = os.path.splitext(obj['Key'])
                            if ext.lower() not in [e.lower() for e in file_extensions]:
                                continue
                        
                        # Extract metadata
                        obj_metadata = {
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag'].strip('"'),
                            'storage_class': obj.get('StorageClass', 'STANDARD'),
                            'file_extension': os.path.splitext(obj['Key'])[1],
                            'file_name': os.path.basename(obj['Key'])
                        }
                        
                        objects.append(obj_metadata)
            
            logger.info(f"Found {len(objects)} objects in s3://{bucket}/{prefix}")
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list objects in s3://{bucket}/{prefix}: {e}")
            return []
    
    def download_file(self, bucket: str, key: str, 
                     local_path: Optional[str] = None) -> str:
        """
        Download file from S3 to local storage
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local path to save file (optional)
            
        Returns:
            Local file path
        """
        if not local_path:
            # Create temp directory if it doesn't exist
            temp_dir = self.config['temp_dir']
            os.makedirs(temp_dir, exist_ok=True)
            local_path = os.path.join(temp_dir, os.path.basename(key))
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            return local_path
            
        except ClientError as e:
            logger.error(f"Failed to download s3://{bucket}/{key}: {e}")
            raise
    
    def read_structured_data(self, bucket: str, key: str, 
                           file_format: Optional[str] = None,
                           **kwargs) -> pd.DataFrame:
        """
        Read structured data directly from S3 into pandas DataFrame
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            file_format: File format (auto-detected if not provided)
            **kwargs: Additional parameters for pandas read functions
            
        Returns:
            DataFrame with the data
        """
        if not file_format:
            _, ext = os.path.splitext(key)
            file_format = ext.lower()
        
        try:
            s3_url = f"s3://{bucket}/{key}"
            
            if file_format in ['.csv']:
                df = pd.read_csv(s3_url, **kwargs)
                
            elif file_format in ['.json']:
                df = pd.read_json(s3_url, **kwargs)
                
            elif file_format in ['.parquet']:
                df = pd.read_parquet(s3_url, **kwargs)
                
            elif file_format in ['.xlsx', '.xls']:
                if not EXCEL_PROCESSING:
                    raise ImportError("openpyxl is required for Excel file processing")
                
                # Download file first as pandas doesn't support S3 URLs for Excel
                local_path = self.download_file(bucket, key)
                df = pd.read_excel(local_path, **kwargs)
                
                # Clean up temp file
                try:
                    os.remove(local_path)
                except:
                    pass
                    
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Successfully read {len(df)} rows from s3://{bucket}/{key}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read structured data from s3://{bucket}/{key}: {e}")
            raise
    
    def read_unstructured_data(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Read and process unstructured data from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Dictionary with extracted content and metadata
        """
        _, ext = os.path.splitext(key)
        file_format = ext.lower()
        
        try:
            # Download file to local temp directory
            local_path = self.download_file(bucket, key)
            
            result = {
                'key': key,
                'file_format': file_format,
                'content': '',
                'metadata': {},
                'processing_status': 'success'
            }
            
            try:
                if file_format == '.pdf':
                    result.update(self._process_pdf(local_path))
                    
                elif file_format == '.txt':
                    result.update(self._process_text_file(local_path))
                    
                elif file_format in ['.jpg', '.jpeg', '.png', '.tiff']:
                    result.update(self._process_image(local_path))
                    
                elif file_format in ['.doc', '.docx']:
                    result.update(self._process_word_document(local_path))
                    
                else:
                    logger.warning(f"Unsupported unstructured format: {file_format}")
                    result['processing_status'] = 'unsupported_format'
                    
            finally:
                # Clean up temp file
                try:
                    os.remove(local_path)
                except:
                    pass
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process unstructured data from s3://{bucket}/{key}: {e}")
            return {
                'key': key,
                'file_format': file_format,
                'content': '',
                'metadata': {},
                'processing_status': 'error',
                'error_message': str(e)
            }
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file and extract text content"""
        if not PDF_PROCESSING:
            return {
                'content': '',
                'metadata': {},
                'processing_status': 'pdf_library_not_available'
            }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages
                text_content = ''
                for page_num, page in enumerate(pdf_reader.pages):
                    text_content += page.extract_text() + '\n'
                
                # Extract metadata
                metadata = {
                    'num_pages': len(pdf_reader.pages),
                    'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                    'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                    'creation_date': str(pdf_reader.metadata.get('/CreationDate', '')) if pdf_reader.metadata else '',
                    'text_length': len(text_content.strip())
                }
                
                return {
                    'content': text_content.strip(),
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return {
                'content': '',
                'metadata': {},
                'processing_status': 'pdf_processing_error',
                'error_message': str(e)
            }
    
    def _process_text_file(self, file_path: str) -> Dict[str, Any]:
        """Process plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Basic text statistics
            metadata = {
                'char_count': len(content),
                'word_count': len(content.split()),
                'line_count': len(content.split('\n'))
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    content = file.read()
                
                metadata = {
                    'char_count': len(content),
                    'word_count': len(content.split()),
                    'line_count': len(content.split('\n')),
                    'encoding_used': 'latin-1'
                }
                
                return {
                    'content': content,
                    'metadata': metadata
                }
                
            except Exception as e:
                logger.error(f"Text file processing failed: {e}")
                return {
                    'content': '',
                    'metadata': {},
                    'processing_status': 'text_processing_error',
                    'error_message': str(e)
                }
    
    def _process_image(self, file_path: str) -> Dict[str, Any]:
        """Process image file and extract metadata"""
        if not IMAGE_PROCESSING:
            return {
                'content': f'[Image file: {os.path.basename(file_path)}]',
                'metadata': {},
                'processing_status': 'image_library_not_available'
            }
        
        try:
            with Image.open(file_path) as img:
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.size[0],
                    'height': img.size[1]
                }
                
                # Extract EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    exif = img._getexif()
                    metadata['exif_available'] = True
                    metadata['exif_data'] = {str(k): str(v) for k, v in exif.items() if v}
                else:
                    metadata['exif_available'] = False
                
                return {
                    'content': f'[Image: {img.format} {img.size[0]}x{img.size[1]}]',
                    'metadata': metadata
                }
                
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'content': f'[Image file: {os.path.basename(file_path)}]',
                'metadata': {},
                'processing_status': 'image_processing_error',
                'error_message': str(e)
            }
    
    def _process_word_document(self, file_path: str) -> Dict[str, Any]:
        """Process Word document (placeholder - would need python-docx)"""
        return {
            'content': f'[Word document: {os.path.basename(file_path)}]',
            'metadata': {
                'note': 'Word document processing requires python-docx library'
            },
            'processing_status': 'word_processing_not_implemented'
        }
    
    def batch_process_claims_data(self, bucket: str, prefix: str = 'claims/',
                                 file_extensions: Optional[List[str]] = None,
                                 processing_function: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Batch process claims data from S3
        
        Args:
            bucket: S3 bucket name
            prefix: Prefix to filter objects
            file_extensions: File extensions to process
            processing_function: Custom processing function
            
        Returns:
            List of processing results
        """
        # List objects to process
        objects = self.list_objects(
            bucket, 
            prefix, 
            max_keys=self.config['batch_size'],
            file_extensions=file_extensions
        )
        
        if not objects:
            logger.info("No objects found for processing")
            return []
        
        results = []
        
        for obj_metadata in objects:
            try:
                key = obj_metadata['key']
                logger.info(f"Processing {key}...")
                
                # Determine if structured or unstructured
                file_ext = obj_metadata['file_extension'].lower()
                
                if file_ext in self.supported_formats['structured']:
                    # Process structured data
                    df = self.read_structured_data(bucket, key)
                    
                    if processing_function:
                        processed_result = processing_function(df, obj_metadata)
                    else:
                        processed_result = {
                            'key': key,
                            'type': 'structured',
                            'rows': len(df),
                            'columns': list(df.columns),
                            'data_types': df.dtypes.to_dict(),
                            'processing_status': 'success'
                        }
                    
                elif file_ext in self.supported_formats['unstructured']:
                    # Process unstructured data
                    processed_result = self.read_unstructured_data(bucket, key)
                    processed_result['type'] = 'unstructured'
                    
                else:
                    logger.warning(f"Unsupported file format: {file_ext}")
                    processed_result = {
                        'key': key,
                        'type': 'unsupported',
                        'processing_status': 'unsupported_format',
                        'file_extension': file_ext
                    }
                
                # Add original metadata
                processed_result['s3_metadata'] = obj_metadata
                results.append(processed_result)
                
            except Exception as e:
                logger.error(f"Error processing {obj_metadata['key']}: {e}")
                results.append({
                    'key': obj_metadata['key'],
                    'type': 'error',
                    'processing_status': 'error',
                    'error_message': str(e),
                    's3_metadata': obj_metadata
                })
        
        logger.info(f"Processed {len(results)} objects from s3://{bucket}/{prefix}")
        return results
    
    def upload_processed_data(self, data: pd.DataFrame, 
                            bucket: str, key: str,
                            file_format: str = 'parquet',
                            metadata: Optional[Dict[str, str]] = None) -> bool:
        """
        Upload processed data back to S3
        
        Args:
            data: DataFrame to upload
            bucket: S3 bucket name
            key: S3 object key
            file_format: File format for upload
            metadata: Object metadata
            
        Returns:
            Boolean indicating success
        """
        try:
            if file_format.lower() == 'parquet':
                buffer = io.BytesIO()
                data.to_parquet(buffer, index=False)
                buffer.seek(0)
                content_type = 'application/octet-stream'
                
            elif file_format.lower() == 'csv':
                buffer = io.StringIO()
                data.to_csv(buffer, index=False)
                buffer = io.BytesIO(buffer.getvalue().encode('utf-8'))
                content_type = 'text/csv'
                
            elif file_format.lower() == 'json':
                buffer = io.BytesIO()
                data.to_json(buffer, orient='records', lines=True)
                buffer.seek(0)
                content_type = 'application/json'
                
            else:
                raise ValueError(f"Unsupported upload format: {file_format}")
            
            # Upload to S3
            extra_args = {'ContentType': content_type}
            if metadata:
                extra_args['Metadata'] = metadata
            
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=buffer.getvalue(),
                **extra_args
            )
            
            logger.info(f"Successfully uploaded data to s3://{bucket}/{key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload data to s3://{bucket}/{key}: {e}")
            return False
    
    def create_data_catalog_entry(self, bucket: str, key: str, 
                                schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create data catalog entry for processed claims data
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            schema_info: Schema information for the data
            
        Returns:
            Catalog entry dictionary
        """
        catalog_entry = {
            'dataset_id': f"{bucket}_{key.replace('/', '_')}",
            'location': f"s3://{bucket}/{key}",
            'created_at': datetime.utcnow().isoformat(),
            'schema': schema_info,
            'data_source': 'claims_s3_ingestion',
            'tags': {
                'source': 'claims_data',
                'ingestion_method': 's3_connector',
                'bucket': bucket
            }
        }
        
        return catalog_entry
    
    def get_object_metadata(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Get detailed object metadata from S3
        
        Args:
            bucket: S3 bucket name  
            key: S3 object key
            
        Returns:
            Metadata dictionary
        """
        try:
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            
            metadata = {
                'content_length': response.get('ContentLength', 0),
                'content_type': response.get('ContentType', ''),
                'last_modified': response.get('LastModified'),
                'etag': response.get('ETag', '').strip('"'),
                'storage_class': response.get('StorageClass', 'STANDARD'),
                'metadata': response.get('Metadata', {}),
                'server_side_encryption': response.get('ServerSideEncryption'),
                'version_id': response.get('VersionId')
            }
            
            return metadata
            
        except ClientError as e:
            logger.error(f"Failed to get metadata for s3://{bucket}/{key}: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'default_bucket': 'claims-data-lake',
        'batch_size': 100,
        'enable_metadata_extraction': True
    }
    
    print("S3 Data Connector Usage Examples:")
    print("1. Initialize: connector = S3DataConnector(config)")
    print("2. List objects: objects = connector.list_objects('bucket', 'claims/')")
    print("3. Read CSV: df = connector.read_structured_data('bucket', 'claims.csv')")
    print("4. Read PDF: content = connector.read_unstructured_data('bucket', 'claim_doc.pdf')")
    print("5. Batch process: results = connector.batch_process_claims_data('bucket', 'claims/')")
    print("6. Upload results: connector.upload_processed_data(df, 'bucket', 'processed/claims.parquet')")