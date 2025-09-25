"""
PyOxynet Logging Configuration
Scientific API logging with medical data privacy compliance
"""
import logging
import logging.handlers
import os
from datetime import datetime
from typing import Dict, Any
import json


class PyOxynetFormatter(logging.Formatter):
    """Custom formatter for PyOxynet scientific API with privacy considerations"""
    
    def format(self, record: logging.LogRecord) -> str:
        # Add scientific context to log records
        if hasattr(record, 'cpet_session_id'):
            record.session = record.cpet_session_id
        if hasattr(record, 'processing_phase'):
            record.phase = record.processing_phase
            
        # Sanitize medical data from logs
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            record.msg = self._sanitize_message(record.msg)
            
        return super().format(record)
    
    def _sanitize_message(self, message: str) -> str:
        """Remove potential medical data from log messages"""
        # Remove common CPET data patterns that might contain personal info
        import re
        
        # Remove potential patient identifiers or personal data
        patterns = [
            r'\b[A-Z]{2,}\s*\d{2,}\b',  # Patient codes like "ABC123"
            r'\b\d{2,}/\d{2,}/\d{4}\b',  # Dates
            r'\b\d{3}-\d{2}-\d{4}\b',   # SSN-like patterns
        ]
        
        sanitized = message
        for pattern in patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized


class CPETProcessingLogger:
    """Specialized logger for CPET processing operations"""
    
    def __init__(self, name: str, config):
        self.logger = logging.getLogger(name)
        self.config = config
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with PyOxynet-specific configuration"""
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL.upper()))
        
        # Console handler for development
        if self.config.DEBUG:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(PyOxynetFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(session)s] - %(message)s'
            ))
            self.logger.addHandler(console_handler)
        
        # File handler for production
        if not self.config.DEBUG:
            file_handler = logging.handlers.RotatingFileHandler(
                'pyoxynet_api.log',
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(PyOxynetFormatter(self.config.LOG_FORMAT))
            self.logger.addHandler(file_handler)
    
    def log_cpet_processing_start(self, session_id: str, file_count: int, batch_size_mb: float):
        """Log start of CPET batch processing"""
        self.logger.info(
            f"Starting CPET batch processing",
            extra={
                'cpet_session_id': session_id,
                'processing_phase': 'batch_start',
                'file_count': file_count,
                'batch_size_mb': batch_size_mb
            }
        )
    
    def log_cpet_processing_complete(self, session_id: str, processing_time: float, success_count: int, error_count: int):
        """Log completion of CPET batch processing"""
        self.logger.info(
            f"CPET batch processing complete - {success_count} succeeded, {error_count} failed",
            extra={
                'cpet_session_id': session_id,
                'processing_phase': 'batch_complete',
                'processing_time_seconds': processing_time,
                'success_count': success_count,
                'error_count': error_count
            }
        )
    
    def log_model_inference(self, session_id: str, model_name: str, inference_time: float):
        """Log TensorFlow Lite model inference"""
        self.logger.debug(
            f"Model inference completed: {model_name}",
            extra={
                'cpet_session_id': session_id,
                'processing_phase': 'model_inference',
                'model_name': model_name,
                'inference_time_ms': inference_time * 1000
            }
        )
    
    def log_file_cleanup(self, session_id: str, files_cleaned: int):
        """Log medical data file cleanup"""
        self.logger.info(
            f"Medical data cleanup completed - {files_cleaned} files removed",
            extra={
                'cpet_session_id': session_id,
                'processing_phase': 'data_cleanup',
                'files_cleaned': files_cleaned
            }
        )
    
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security-related events"""
        self.logger.warning(
            f"Security event: {event_type}",
            extra={
                'processing_phase': 'security',
                'event_type': event_type,
                'event_details': json.dumps(details, default=str)
            }
        )
    
    def log_scientific_validation_error(self, session_id: str, validation_type: str, error_details: str):
        """Log scientific validation errors"""
        self.logger.error(
            f"Scientific validation failed: {validation_type}",
            extra={
                'cpet_session_id': session_id,
                'processing_phase': 'scientific_validation',
                'validation_type': validation_type,
                'error_details': error_details
            }
        )


def setup_logging(config) -> CPETProcessingLogger:
    """Setup PyOxynet logging infrastructure"""
    # Ensure log directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Create main API logger
    api_logger = CPETProcessingLogger('pyoxynet.api', config)
    
    return api_logger