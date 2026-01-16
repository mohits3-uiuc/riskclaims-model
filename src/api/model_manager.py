"""
Model Manager for Claims Risk Classification API

This module manages ML model loading, caching, and lifecycle
for efficient API operations.
"""

import asyncio
import pickle
import joblib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import gc
import sys
from threading import Lock

# Import our model classes
sys.path.append(str(Path(__file__).parent.parent))
from models import RandomForestClaimsModel, XGBoostClaimsModel, NeuralNetworkClaimsModel
from preprocessing import StructuredDataPreprocessor, UnstructuredDataPreprocessor, FeatureEngineer

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Error raised when model loading fails"""
    pass


class ModelCache:
    """
    Thread-safe model cache with LRU eviction
    
    Features:
    - Thread-safe model storage
    - LRU eviction policy
    - Memory usage tracking
    - Model metadata management
    """
    
    def __init__(self, max_size: int = 3):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.model_metadata = {}
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if key in self.cache:
                self.access_times[key] = datetime.now()
                return self.cache[key]
            return None
    
    def put(self, key: str, model: Any, metadata: Dict[str, Any] = None):
        """Put model in cache with LRU eviction"""
        with self.lock:
            # If cache is full, remove least recently used
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = model
            self.access_times[key] = datetime.now()
            
            if metadata:
                self.model_metadata[key] = metadata
            
            logger.info(f"Model '{key}' cached (cache size: {len(self.cache)})")
    
    def remove(self, key: str):
        """Remove model from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                self.model_metadata.pop(key, None)
                
                # Force garbage collection
                gc.collect()
                
                logger.info(f"Model '{key}' removed from cache")
    
    def _evict_lru(self):
        """Evict least recently used model"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.remove(lru_key)
        logger.info(f"Evicted LRU model: {lru_key}")
    
    def clear(self):
        """Clear entire cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.model_metadata.clear()
            gc.collect()
            logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "cached_models": list(self.cache.keys()),
                "memory_usage_estimate": self._estimate_memory_usage()
            }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage of cached models"""
        try:
            total_size = sum(sys.getsizeof(model) for model in self.cache.values())
            return f"{total_size / (1024*1024):.1f} MB"
        except Exception:
            return "Unknown"


class ModelManager:
    """
    Manages ML models for the API
    
    Features:
    - Async model loading
    - Model caching and lifecycle management
    - Preprocessing pipeline management
    - Model performance tracking
    - Health monitoring
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.cache = ModelCache(max_size=self.config.get('model_cache_size', 3))
        self.models_dir = Path(self.config.get('models_dir', 'models'))
        self.preprocessors_dir = Path(self.config.get('preprocessors_dir', 'preprocessors'))
        
        # Model class mapping
        self.model_classes = {
            'Random Forest': RandomForestClaimsModel,
            'XGBoost': XGBoostClaimsModel,
            'Neural Network': NeuralNetworkClaimsModel
        }
        
        # Model usage statistics
        self.usage_stats = {
            'prediction_counts': {},
            'load_times': {},
            'error_counts': {},
            'last_used': {}
        }
        
        # Preprocessing components
        self.preprocessors = {}
        self._preprocessors_loaded = False
    
    async def load_models(self):
        """Load all available models asynchronously"""
        logger.info("Starting model loading process...")
        
        try:
            # Load preprocessors first
            await self.load_preprocessors()
            
            # Load models in parallel
            load_tasks = []
            for model_name in self.config.get('available_models', ['Random Forest']):
                task = asyncio.create_task(self.load_specific_model(model_name))
                load_tasks.append(task)
            
            # Wait for all models to load
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Check results
            loaded_count = 0
            for model_name, result in zip(self.config.get('available_models', []), results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to load {model_name}: {result}")
                    self.usage_stats['error_counts'][model_name] = 1
                else:
                    loaded_count += 1
                    logger.info(f"Successfully loaded {model_name}")
            
            logger.info(f"Model loading completed: {loaded_count}/{len(results)} models loaded")
            
        except Exception as e:
            logger.error(f"Model loading process failed: {e}", exc_info=True)
            raise ModelLoadError(f"Failed to load models: {e}")
    
    async def load_specific_model(self, model_name: str):
        """Load a specific model asynchronously"""
        if model_name in self.cache.cache:
            logger.info(f"Model {model_name} already cached")
            return
        
        start_time = datetime.now()
        
        try:
            # Check if we have a pre-trained model file
            model_file = self.models_dir / f"{model_name.lower().replace(' ', '_')}_model.pkl"
            
            if model_file.exists():
                # Load pre-trained model
                logger.info(f"Loading pre-trained model from {model_file}")
                model = await self._load_model_file(model_file)
            else:
                # Create and train new model
                logger.info(f"Creating new {model_name} model")
                model = await self._create_new_model(model_name)
            
            # Cache the model
            load_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                'load_time': load_time,
                'loaded_at': datetime.now().isoformat(),
                'model_type': model_name,
                'version': getattr(model, 'version', '1.0.0')
            }
            
            self.cache.put(model_name, model, metadata)
            self.usage_stats['load_times'][model_name] = load_time
            
            logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
            self.usage_stats['error_counts'][model_name] = \
                self.usage_stats['error_counts'].get(model_name, 0) + 1
            raise ModelLoadError(f"Failed to load {model_name}: {e}")
    
    async def _load_model_file(self, model_file: Path):
        """Load model from file asynchronously"""
        
        def load_sync():
            try:
                # Try joblib first (for sklearn models)
                return joblib.load(model_file)
            except Exception:
                try:
                    # Try pickle as fallback
                    with open(model_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    raise ModelLoadError(f"Could not load model file {model_file}: {e}")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, load_sync)
    
    async def _create_new_model(self, model_name: str):
        """Create and train a new model"""
        
        def create_sync():
            if model_name not in self.model_classes:
                raise ModelLoadError(f"Unknown model type: {model_name}")
            
            # Create model instance
            model_class = self.model_classes[model_name]
            model_config = self.config.get('model_config', {}).get(model_name, {})
            
            model = model_class(**model_config)
            
            # For demo purposes, create a simple trained model
            # In production, this would load actual training data
            logger.warning(f"Creating untrained {model_name} model - training data required")
            
            return model
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, create_sync)
    
    async def load_preprocessors(self):
        """Load preprocessing components"""
        if self._preprocessors_loaded:
            return
        
        logger.info("Loading preprocessing components...")
        
        try:
            # Check for saved preprocessors
            struct_preprocessor_file = self.preprocessors_dir / "structured_preprocessor.pkl"
            unstruct_preprocessor_file = self.preprocessors_dir / "unstructured_preprocessor.pkl"
            feature_engineer_file = self.preprocessors_dir / "feature_engineer.pkl"
            
            if all(f.exists() for f in [struct_preprocessor_file, unstruct_preprocessor_file, feature_engineer_file]):
                # Load saved preprocessors
                self.preprocessors['structured'] = joblib.load(struct_preprocessor_file)
                self.preprocessors['unstructured'] = joblib.load(unstruct_preprocessor_file)
                self.preprocessors['feature_engineer'] = joblib.load(feature_engineer_file)
                
                logger.info("Loaded saved preprocessing components")
            else:
                # Create new preprocessors
                logger.warning("Creating new preprocessing components - training required")
                
                self.preprocessors['structured'] = StructuredDataPreprocessor()
                self.preprocessors['unstructured'] = UnstructuredDataPreprocessor()
                self.preprocessors['feature_engineer'] = FeatureEngineer()
            
            self._preprocessors_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load preprocessors: {e}")
            raise ModelLoadError(f"Preprocessing components failed to load: {e}")
    
    async def get_model(self, model_name: str):
        """Get a model (load if not cached)"""
        model = self.cache.get(model_name)
        
        if model is None:
            logger.info(f"Model {model_name} not cached, loading...")
            await self.load_specific_model(model_name)
            model = self.cache.get(model_name)
        
        if model is None:
            raise ModelLoadError(f"Failed to load model: {model_name}")
        
        # Update usage statistics
        self.usage_stats['last_used'][model_name] = datetime.now().isoformat()
        
        return model
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.cache.cache.keys())
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name not in self.cache.cache:
            return None
        
        metadata = self.cache.model_metadata.get(model_name, {})
        
        return {
            "name": model_name,
            "status": "loaded",
            "version": metadata.get("version", "unknown"),
            "loaded_at": metadata.get("loaded_at"),
            "load_time": metadata.get("load_time"),
            "last_used": self.usage_stats['last_used'].get(model_name),
            "prediction_count": self.usage_stats['prediction_counts'].get(model_name, 0),
            "error_count": self.usage_stats['error_counts'].get(model_name, 0)
        }
    
    async def update_model_usage(self, model_name: str):
        """Update model usage statistics"""
        self.usage_stats['prediction_counts'][model_name] = \
            self.usage_stats['prediction_counts'].get(model_name, 0) + 1
        self.usage_stats['last_used'][model_name] = datetime.now().isoformat()
    
    async def update_batch_usage(self, model_name: str, batch_size: int):
        """Update model usage for batch predictions"""
        self.usage_stats['prediction_counts'][model_name] = \
            self.usage_stats['prediction_counts'].get(model_name, 0) + batch_size
        self.usage_stats['last_used'][model_name] = datetime.now().isoformat()
    
    async def reload_model(self, model_name: str):
        """Reload a specific model"""
        logger.info(f"Reloading model: {model_name}")
        
        # Remove from cache
        self.cache.remove(model_name)
        
        # Load again
        await self.load_specific_model(model_name)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on model manager"""
        cache_info = self.cache.get_cache_info()
        
        return {
            "status": "healthy",
            "cache_info": cache_info,
            "preprocessors_loaded": self._preprocessors_loaded,
            "usage_stats": {
                "total_predictions": sum(self.usage_stats['prediction_counts'].values()),
                "models_with_errors": len([k for k, v in self.usage_stats['error_counts'].items() if v > 0]),
                "average_load_time": sum(self.usage_stats['load_times'].values()) / max(len(self.usage_stats['load_times']), 1)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get detailed usage statistics"""
        return {
            "prediction_counts": self.usage_stats['prediction_counts'].copy(),
            "load_times": self.usage_stats['load_times'].copy(),
            "error_counts": self.usage_stats['error_counts'].copy(),
            "last_used": self.usage_stats['last_used'].copy(),
            "cache_info": self.cache.get_cache_info()
        }
    
    async def save_models(self, output_dir: Path = None):
        """Save all cached models to disk"""
        output_dir = output_dir or self.models_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving models to {output_dir}...")
        
        for model_name, model in self.cache.cache.items():
            try:
                model_file = output_dir / f"{model_name.lower().replace(' ', '_')}_model.pkl"
                
                # Save using joblib (preferred for sklearn models)
                joblib.dump(model, model_file)
                
                logger.info(f"Saved {model_name} to {model_file}")
                
            except Exception as e:
                logger.error(f"Failed to save {model_name}: {e}")
        
        # Save preprocessors
        if self.preprocessors:
            preproc_dir = output_dir.parent / "preprocessors"
            preproc_dir.mkdir(exist_ok=True)
            
            for name, preprocessor in self.preprocessors.items():
                try:
                    preproc_file = preproc_dir / f"{name}_preprocessor.pkl"
                    joblib.dump(preprocessor, preproc_file)
                    logger.info(f"Saved {name} preprocessor to {preproc_file}")
                except Exception as e:
                    logger.error(f"Failed to save {name} preprocessor: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model manager...")
        self.cache.clear()
        self.preprocessors.clear()
        self._preprocessors_loaded = False


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def test_model_manager():
        config = {
            'model_cache_size': 2,
            'available_models': ['Random Forest', 'XGBoost'],
            'models_dir': 'test_models'
        }
        
        manager = ModelManager(config)
        
        try:
            print("Loading models...")
            await manager.load_models()
            
            print("Available models:", await manager.get_available_models())
            
            print("Getting model info...")
            for model_name in await manager.get_available_models():
                info = await manager.get_model_info(model_name)
                print(f"{model_name}: {info}")
            
            print("Health check:", await manager.health_check())
            
        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            manager.cleanup()
    
    asyncio.run(test_model_manager())
