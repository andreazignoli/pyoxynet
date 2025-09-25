"""
PyOxynet Secure Flask Application
Phase 1: Foundation & Security - Modernized scientific CPET analysis API
"""
import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import flask
from flask import Flask, request, render_template, jsonify, session
from flasgger import Swagger, swag_from
import numpy as np
import pandas as pd

# Import our new security infrastructure
from config import get_config
from logging_config import setup_logging
from security import create_security_manager

# Suppress scientific computing warnings
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='divide by zero encountered')


class PyOxynetAPI:
    """Secure PyOxynet API Application"""
    
    def __init__(self, environment: Optional[str] = None):
        self.app = Flask(__name__)
        self.config = get_config(environment)()
        self.app.config.from_object(self.config)
        
        # Initialize secure logging
        self.logger = setup_logging(self.config)
        
        # Initialize security manager
        self.security = create_security_manager(self.config, self.logger)
        
        # Initialize API documentation
        self.swagger = Swagger(self.app, template={
            "swagger": "2.0",
            "info": {
                "title": self.config.API_TITLE,
                "description": self.config.API_DESCRIPTION,
                "version": self.config.API_VERSION,
                "termsOfService": "Medical data is processed temporarily and never stored permanently",
                "contact": {
                    "name": "PyOxynet Scientific API",
                    "url": "https://github.com/andreazignoli/pyoxynet"
                }
            },
            "securityDefinitions": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        })
        
        # Register routes
        self._register_routes()
        
        # Register error handlers
        self._register_error_handlers()
        
        self.logger.logger.info("PyOxynet secure API initialized", extra={
            'processing_phase': 'application_start',
            'api_version': self.config.API_VERSION,
            'environment': environment or 'default'
        })
    
    def _register_routes(self):
        """Register secure API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint for deployment monitoring"""
            return jsonify({
                'status': 'healthy',
                'api_version': self.config.API_VERSION,
                'timestamp': datetime.utcnow().isoformat(),
                'environment': 'production' if not self.config.DEBUG else 'development'
            })
        
        @self.app.route('/api/v1/cpet/analyze', methods=['POST'])
        @swag_from({
            'tags': ['CPET Analysis'],
            'summary': 'Analyze single CPET file',
            'description': 'Upload and analyze a single cardiopulmonary exercise test file with scientific validation',
            'consumes': ['multipart/form-data'],
            'parameters': [{
                'name': 'file',
                'in': 'formData',
                'type': 'file',
                'required': True,
                'description': 'CPET data file (CSV format with VO2, VCO2, VE columns)'
            }],
            'responses': {
                200: {
                    'description': 'Analysis completed successfully',
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'success': {'type': 'boolean'},
                            'job_id': {'type': 'string'},
                            'status': {'type': 'string'},
                            'results': {'type': 'object'},
                            'processing_time_seconds': {'type': 'number'},
                            'metadata': {'type': 'object'}
                        }
                    }
                },
                400: {'description': 'Invalid file or data format'},
                413: {'description': 'File too large'},
                500: {'description': 'Processing error'}
            },
            'security': [{'ApiKeyAuth': []}]
        })
        def analyze_single_cpet():
            """Secure single CPET file analysis endpoint"""
            session_id = str(uuid.uuid4())
            start_time = datetime.utcnow()
            
            try:
                self.logger.log_cpet_processing_start(session_id, 1, 0)
                
                # Validate request
                if 'file' not in request.files:
                    return jsonify({
                        'success': False,
                        'error': 'No file provided',
                        'job_id': session_id
                    }), 400
                
                uploaded_file = request.files['file']
                
                # Security validation
                validation_result = self.security['file_validator'].validate_file(uploaded_file)
                
                if not validation_result['valid']:
                    self.logger.log_security_event('file_validation_failed', {
                        'session_id': session_id,
                        'errors': validation_result['errors']
                    })
                    return jsonify({
                        'success': False,
                        'error': 'File validation failed',
                        'details': validation_result['errors'],
                        'job_id': session_id
                    }), 400
                
                # Save file temporarily for processing
                file_path, filename = self.security['data_manager'].save_temp_file(uploaded_file, session_id)
                
                # Process CPET data (simplified for Phase 1)
                processing_result = self._process_cpet_file_secure(file_path, session_id, validation_result)
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Log completion
                self.logger.log_cpet_processing_complete(
                    session_id, 
                    processing_time, 
                    1 if processing_result['success'] else 0,
                    0 if processing_result['success'] else 1
                )
                
                # Immediate cleanup (medical data privacy)
                self.security['data_manager'].cleanup_session_files(session_id)
                
                response = {
                    'success': processing_result['success'],
                    'job_id': session_id,
                    'status': 'completed' if processing_result['success'] else 'failed',
                    'results': processing_result.get('results', {}),
                    'processing_time_seconds': processing_time,
                    'metadata': {
                        'api_version': self.config.API_VERSION,
                        'file_info': validation_result['file_info'],
                        'cpet_info': validation_result['cpet_info']
                    }
                }
                
                if not processing_result['success']:
                    response['error'] = processing_result.get('error', 'Processing failed')
                    return jsonify(response), 500
                
                return jsonify(response)
                
            except Exception as e:
                # Error handling with cleanup
                self.security['data_manager'].cleanup_session_files(session_id)
                self.logger.log_security_event('processing_error', {
                    'session_id': session_id,
                    'error': str(e)
                })
                
                return jsonify({
                    'success': False,
                    'error': 'Internal processing error',
                    'job_id': session_id
                }), 500
        
        @self.app.route('/', methods=['GET'])
        def homepage():
            """Secure homepage with API information"""
            return render_template('homepage.html', 
                                 api_version=self.config.API_VERSION,
                                 environment='Development' if self.config.DEBUG else 'Production')
    
    def _register_error_handlers(self):
        """Register secure error handlers"""
        
        @self.app.errorhandler(413)
        def file_too_large(error):
            """Handle file too large errors"""
            self.logger.log_security_event('file_too_large', {
                'max_size_mb': self.config.MAX_CONTENT_LENGTH / 1024 / 1024
            })
            return jsonify({
                'success': False,
                'error': 'File too large',
                'max_size_mb': self.config.MAX_CONTENT_LENGTH / 1024 / 1024
            }), 413
        
        @self.app.errorhandler(404)
        def not_found(error):
            """Handle not found errors"""
            return jsonify({
                'success': False,
                'error': 'Endpoint not found',
                'api_version': self.config.API_VERSION
            }), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            """Handle internal errors with privacy protection"""
            self.logger.log_security_event('internal_error', {
                'error_type': type(error).__name__
            })
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'api_version': self.config.API_VERSION
            }), 500
    
    def _process_cpet_file_secure(self, file_path: str, session_id: str, validation_result: Dict) -> Dict[str, Any]:
        """
        Secure CPET file processing with scientific validation
        Simplified for Phase 1 - will be enhanced in Phase 2
        """
        try:
            # Load and validate data
            df = pd.read_csv(file_path)
            
            # Basic scientific analysis (Phase 1 implementation)
            analysis_results = {
                'file_info': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'duration_estimated_minutes': self._estimate_duration(df)
                },
                'basic_stats': {
                    'vo2_mean': float(df['VO2'].mean()) if 'VO2' in df.columns else None,
                    'vco2_mean': float(df['VCO2'].mean()) if 'VCO2' in df.columns else None,
                    've_mean': float(df['VE'].mean()) if 'VE' in df.columns else None,
                },
                'data_quality': validation_result['cpet_info'].get('data_quality_score', 0)
            }
            
            # Log successful processing
            self.logger.logger.info(f"CPET file processed successfully", extra={
                'cpet_session_id': session_id,
                'processing_phase': 'file_processing',
                'rows_processed': len(df),
                'data_quality_score': analysis_results['data_quality']
            })
            
            return {
                'success': True,
                'results': analysis_results
            }
            
        except Exception as e:
            self.logger.log_scientific_validation_error(
                session_id, 
                'file_processing', 
                str(e)
            )
            return {
                'success': False,
                'error': f'Processing error: {str(e)}'
            }
    
    def _estimate_duration(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate test duration from data"""
        if 'Time' in df.columns:
            time_col = df['Time'].dropna()
            if len(time_col) > 0:
                return float((time_col.max() - time_col.min()) / 60.0)
        return None


def create_app(environment: Optional[str] = None) -> Flask:
    """Application factory for PyOxynet secure API"""
    pyoxynet_api = PyOxynetAPI(environment)
    return pyoxynet_api.app


# Application entry point
if __name__ == '__main__':
    # Get environment from environment variable
    env = os.environ.get('FLASK_ENV', 'development')
    
    # Create secure application
    app = create_app(env)
    
    # Run with appropriate settings
    if env == 'development':
        app.run(
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 9098)),
            debug=True
        )
    else:
        # Production should use gunicorn
        print("Production mode - use gunicorn to run the application")
        print("Example: gunicorn -w 4 -b 0.0.0.0:9098 app_secure:create_app()")