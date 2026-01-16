#!/usr/bin/env python3
"""
Production Server for Claims Risk Classification API

This script provides various ways to run the API server including
development mode, production deployment, and Docker containerization.
"""

import uvicorn
import argparse
import sys
import os
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.api.config import get_config
from src.api.main import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_development_server(host: str = "127.0.0.1", 
                          port: int = 8000, 
                          reload: bool = True,
                          log_level: str = "info"):
    """
    Run development server with hot reloading
    
    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable hot reloading
        log_level: Logging level
    """
    logger.info(f"Starting development server on {host}:{port}")
    logger.info("Hot reloading: enabled" if reload else "Hot reloading: disabled")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
        reload_dirs=[str(project_root / "src")]
    )


def run_production_server(host: str = "0.0.0.0", 
                         port: int = 8000,
                         workers: int = None,
                         log_level: str = "info",
                         access_log: bool = True):
    """
    Run production server with multiple workers
    
    Args:
        host: Host to bind to
        port: Port to listen on
        workers: Number of worker processes
        log_level: Logging level
        access_log: Enable access logging
    """
    config = get_config("production")
    
    # Use config defaults if not specified
    workers = workers or config.workers
    
    logger.info(f"Starting production server on {host}:{port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Log level: {log_level}")
    
    # Production server configuration
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=access_log,
        reload=False,
        # Production optimizations
        loop="uvloop",
        http="httptools",
        # SSL configuration (if certificates available)
        ssl_keyfile=os.getenv("SSL_KEYFILE"),
        ssl_certfile=os.getenv("SSL_CERTFILE"),
    )


def run_gunicorn_server(host: str = "0.0.0.0",
                       port: int = 8000,
                       workers: int = None,
                       worker_class: str = "uvicorn.workers.UvicornWorker"):
    """
    Run server using Gunicorn (recommended for production)
    
    Args:
        host: Host to bind to
        port: Port to listen on  
        workers: Number of worker processes
        worker_class: Gunicorn worker class
    """
    try:
        import gunicorn.app.wsgiapp as wsgi
    except ImportError:
        logger.error("Gunicorn not installed. Install with: pip install gunicorn")
        return
    
    config = get_config("production")
    workers = workers or config.workers
    
    logger.info(f"Starting Gunicorn server on {host}:{port}")
    logger.info(f"Workers: {workers}, Worker class: {worker_class}")
    
    # Gunicorn configuration
    gunicorn_config = {
        'bind': f'{host}:{port}',
        'workers': workers,
        'worker_class': worker_class,
        'worker_connections': 1000,
        'max_requests': 1000,
        'max_requests_jitter': 100,
        'timeout': 30,
        'keepalive': 5,
        'preload_app': True,
        'accesslog': '-',
        'errorlog': '-',
        'loglevel': 'info'
    }
    
    # Create Gunicorn application
    class StandaloneApplication(wsgi.BaseApplication):
        def __init__(self, app, options=None):
            self.options = options or {}
            self.application = app
            super().__init__()
        
        def load_config(self):
            for key, value in self.options.items():
                self.cfg.set(key.lower(), value)
        
        def load(self):
            return self.application
    
    StandaloneApplication(app, gunicorn_config).run()


def check_environment():
    """
    Check if environment is properly configured
    
    Returns:
        bool: True if environment is ready
    """
    logger.info("Checking environment configuration...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8+ required")
        return False
    
    # Check required environment variables
    required_vars = ['ENVIRONMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Using default configuration")
    
    # Check configuration
    try:
        config = get_config()
        logger.info(f"Configuration loaded: {config.__class__.__name__}")
        logger.info(f"Debug mode: {config.debug_mode}")
        logger.info(f"Available models: {config.available_models}")
        
        # Validate configuration
        from src.api.config import validate_config
        warnings = validate_config(config)
        
        if warnings:
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        return False


def setup_logging(log_level: str = "info", log_file: Optional[str] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")


def print_startup_info():
    """
    Print startup information and API endpoints
    """
    config = get_config()
    
    print("\n" + "="*60)
    print("   CLAIMS RISK CLASSIFICATION API")
    print("="*60)
    print(f"Version: 1.0.0")
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Debug Mode: {config.debug_mode}")
    print(f"Available Models: {', '.join(config.available_models)}")
    print(f"Max Batch Size: {config.max_batch_size}")
    print(f"Monitoring: {'Enabled' if config.enable_monitoring else 'Disabled'}")
    print("\nAPI Endpoints:")
    print("  POST /predict           - Single claim prediction")
    print("  POST /predict/batch     - Batch claims prediction")
    print("  GET  /models           - List available models")
    print("  GET  /health           - Health check")
    print("  GET  /health/detailed  - Detailed health check")
    print("  GET  /monitoring/drift - Data drift status")
    print("  GET  /monitoring/performance - Performance metrics")
    print("\nDocumentation:")
    print(f"  Swagger UI: http://localhost:{config.port}/docs")
    print(f"  ReDoc:      http://localhost:{config.port}/redoc")
    print("="*60 + "\n")


def main():
    """
    Main entry point for the server
    """
    parser = argparse.ArgumentParser(
        description="Claims Risk Classification API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dev                    # Development server with hot reload
  %(prog)s --prod --workers 4       # Production server with 4 workers  
  %(prog)s --gunicorn               # Production server with Gunicorn
  %(prog)s --check                  # Check environment only
        """
    )
    
    # Server modes
    parser.add_argument(
        "--dev", action="store_true",
        help="Run in development mode with hot reload"
    )
    parser.add_argument(
        "--prod", action="store_true",
        help="Run in production mode"
    )
    parser.add_argument(
        "--gunicorn", action="store_true",
        help="Run with Gunicorn (recommended for production)"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check environment and exit"
    )
    
    # Server configuration
    parser.add_argument(
        "--host", default=None,
        help="Host to bind to (default: config-based)"
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Port to listen on (default: config-based)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes (production/gunicorn only)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level", choices=['debug', 'info', 'warning', 'error'], 
        default='info', help="Logging level"
    )
    parser.add_argument(
        "--log-file", help="Log file path (optional)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    
    # Check environment
    if not check_environment():
        logger.error("Environment check failed")
        sys.exit(1)
    
    if args.check:
        logger.info("Environment check passed")
        return
    
    # Get configuration
    config = get_config()
    host = args.host or config.host
    port = args.port or config.port
    
    # Print startup info
    print_startup_info()
    
    # Run appropriate server
    try:
        if args.dev or (not args.prod and not args.gunicorn):
            run_development_server(
                host=host, port=port, 
                log_level=args.log_level
            )
        elif args.gunicorn:
            run_gunicorn_server(
                host=host, port=port,
                workers=args.workers
            )
        else:  # production mode
            run_production_server(
                host=host, port=port,
                workers=args.workers,
                log_level=args.log_level
            )
    
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
