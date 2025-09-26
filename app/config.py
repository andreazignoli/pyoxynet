"""
PyOxynet Configuration Module
Handles environment-specific configurations for scientific API deployment

IMPORTANT: Requires Python 3.10.x for TensorFlow Lite compatibility
Dependencies: pyoxynet[tflite]==0.1.8, tflite-runtime, numpy==1.26.4
"""
import os
import sys
from typing import Optional

# Validate Python version for TensorFlow Lite compatibility
if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
    raise RuntimeError("PyOxynet requires Python 3.10.x for TensorFlow Lite compatibility")


class Config:
    """Base configuration class for PyOxynet scientific API"""
    
    def __init__(self):
        # Security Configuration
        self.SECRET_KEY: str = os.environ.get('SECRET_KEY') or os.urandom(32).hex()
        
        # Flask Configuration
        self.DEBUG: bool = False
        self.TESTING: bool = False
        
        # File Upload Configuration
        self.UPLOAD_FOLDER: str = os.path.join('staticFiles', 'uploads')
        self.MAX_CONTENT_LENGTH: int = 16 * 1024 * 1024  # 16MB max file size
        self.ALLOWED_EXTENSIONS: set = {'csv', 'txt', 'json'}
        
        # CPET Scientific Configuration
        self.MAX_BATCH_SIZE: int = 100  # Maximum files per batch
        self.MAX_BATCH_MEMORY_GB: float = 2.0  # Memory limit per batch
        self.PROCESSING_TIMEOUT_MINUTES: int = 10  # Processing timeout
        
        # TensorFlow Lite Configuration
        self.TFLITE_MODEL_PATH: str = os.path.join('tf_lite_models', 'tfl_model.tflite')
        
        # Medical Data Privacy Configuration
        self.AUTO_CLEANUP_MINUTES: int = 60  # Auto-cleanup uploaded files after 60 minutes
        self.NO_PERSISTENT_STORAGE: bool = True  # Never persist medical data
        
        # API Configuration
        self.API_VERSION: str = "1.0.0"
        self.API_TITLE: str = "PyOxynet Scientific CPET Analysis API"
        self.API_DESCRIPTION: str = "Production API for cardiopulmonary exercise test analysis"
        
        # Logging Configuration
        self.LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
        self.LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class DevelopmentConfig(Config):
    """Development environment configuration"""
    
    def __init__(self):
        super().__init__()
        self.DEBUG = True
        self.LOG_LEVEL = 'DEBUG'
        self.AUTO_CLEANUP_MINUTES = 10  # Faster cleanup in development


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    TESTING = False
    
    def __init__(self):
        super().__init__()
        # Production should always have SECRET_KEY set via environment
        self.SECRET_KEY = os.environ.get('SECRET_KEY')
        
        if not self.SECRET_KEY:
            raise ValueError("SECRET_KEY environment variable must be set in production")


class TestingConfig(Config):
    """Testing environment configuration"""
    
    def __init__(self):
        super().__init__()
        self.TESTING = True
        self.SECRET_KEY = 'testing-secret-key-not-for-production'
        self.MAX_BATCH_SIZE = 10  # Smaller batches for testing
        self.AUTO_CLEANUP_MINUTES = 1  # Fast cleanup for tests
        
        # Add missing attributes for security manager compatibility
        self.UPLOAD_FOLDER = 'test_uploads'
        self.DEBUG = False
        self.WTF_CSRF_ENABLED = False
        self.LOG_LEVEL = 'DEBUG'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env: Optional[str] = None) -> Config:
    """Get configuration based on environment"""
    env = env or os.environ.get('FLASK_ENV', 'default')
    return config.get(env, config['default'])