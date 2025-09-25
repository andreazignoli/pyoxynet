"""
CPET Analysis API Routes
RESTful endpoints for CPET analysis functionality
"""
from flask import Blueprint, request, jsonify, current_app
import pandas as pd
import logging
from typing import Dict, Any
import io
import json

from ...core.services.cpet_service import CPETAnalysisService


cpet_bp = Blueprint('cpet_analysis', __name__, url_prefix='/api/v1/cpet')

# Initialize service (will be done properly with dependency injection later)
cpet_service = None


def init_cpet_routes(app):
    """Initialize CPET routes with application context"""
    global cpet_service
    cpet_service = CPETAnalysisService(logger=app.logger)
    app.register_blueprint(cpet_bp)


@cpet_bp.route('/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze a single CPET file
    
    Expects multipart/form-data with 'file' field
    Returns comprehensive analysis results
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'code': 'NO_FILE'
            }), 400
        
        file_storage = request.files['file']
        if file_storage.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        # Get analysis options
        options = {}
        if 'options' in request.form:
            try:
                options = json.loads(request.form['options'])
            except json.JSONDecodeError:
                current_app.logger.warning("Invalid options JSON provided")
        
        # Get or create session ID
        session_id = request.form.get('session_id', 'api_session')
        
        # Run analysis
        results = cpet_service.analyze_file(file_storage, session_id, options)
        
        if not results['success']:
            return jsonify(results), 422
        
        # Return successful results
        response_data = {
            'success': True,
            'data': {
                'session_id': results['session_id'],
                'filename': results['filename'],
                'timestamp': results['timestamp'],
                'ml_analysis': results['ml_analysis'],
                'analysis_report': results['analysis_report'],
                'processing_info': results['processing_info']
            },
            'visualizations': {
                k: v for k, v in results['visualizations'].items() 
                if k != 'plot_config'  # Exclude plot config from API response
            },
            'metadata': {
                'api_version': '1.0.0',
                'processing_time_ms': None  # Could add timing later
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        current_app.logger.error(f"CPET file analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during analysis',
            'code': 'ANALYSIS_ERROR'
        }), 500


@cpet_bp.route('/analyze-data', methods=['POST'])
def analyze_json_data():
    """
    Analyze CPET data provided as JSON
    
    Expects JSON with CPET data array
    Returns analysis results
    """
    try:
        # Validate JSON request
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json',
                'code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data_json = request.get_json()
        
        # Validate required fields
        if 'data' not in data_json:
            return jsonify({
                'success': False,
                'error': 'Missing required field: data',
                'code': 'MISSING_DATA'
            }), 400
        
        # Convert to DataFrame
        try:
            data = pd.DataFrame(data_json['data'])
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Invalid data format: {str(e)}',
                'code': 'INVALID_DATA_FORMAT'
            }), 400
        
        # Get analysis options
        options = data_json.get('options', {})
        
        # Run analysis
        results = cpet_service.analyze_data(data, options)
        
        if not results['success']:
            return jsonify(results), 422
        
        # Return results
        response_data = {
            'success': True,
            'data': {
                'timestamp': results['timestamp'],
                'ml_analysis': results['ml_analysis'],
                'analysis_report': results['analysis_report'],
                'processing_info': results['processing_info']
            },
            'visualizations': {
                k: v for k, v in results['visualizations'].items() 
                if k != 'plot_config'
            },
            'metadata': {
                'api_version': '1.0.0',
                'data_points_processed': len(data)
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        current_app.logger.error(f"CPET data analysis failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during analysis',
            'code': 'ANALYSIS_ERROR'
        }), 500


@cpet_bp.route('/validate', methods=['POST'])
def validate_file():
    """
    Validate CPET file without full analysis
    
    Returns validation results and data preview
    """
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'code': 'NO_FILE'
            }), 400
        
        file_storage = request.files['file']
        if file_storage.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'code': 'EMPTY_FILENAME'
            }), 400
        
        session_id = request.form.get('session_id', 'validation_session')
        
        # Process file for validation
        data_result = cpet_service.data_service.process_file(file_storage, session_id)
        
        if not data_result['success']:
            return jsonify({
                'success': False,
                'error': 'File processing failed',
                'errors': data_result['errors'],
                'code': 'PROCESSING_ERROR'
            }), 422
        
        # Validate data
        validation_result = cpet_service.data_service.validate_cpet_data(data_result['data'])
        
        # Create preview (first 10 rows)
        preview_data = data_result['data'].head(10).to_dict('records')
        
        response_data = {
            'success': True,
            'validation': validation_result,
            'metadata': data_result['metadata'],
            'preview': {
                'rows': preview_data,
                'total_rows': len(data_result['data']),
                'columns': list(data_result['data'].columns)
            }
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        current_app.logger.error(f"File validation failed: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during validation',
            'code': 'VALIDATION_ERROR'
        }), 500


@cpet_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for CPET analysis service
    """
    try:
        status = cpet_service.get_service_status()
        
        # Determine overall health
        ml_status = status['ml_service']['status']
        overall_healthy = ml_status == 'ready'
        
        response_data = {
            'healthy': overall_healthy,
            'status': 'healthy' if overall_healthy else 'degraded',
            'services': status,
            'timestamp': status['last_check'],
            'version': '1.0.0'
        }
        
        status_code = 200 if overall_healthy else 503
        return jsonify(response_data), status_code
        
    except Exception as e:
        current_app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'healthy': False,
            'status': 'error',
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 500


@cpet_bp.route('/model-info', methods=['GET'])
def get_model_info():
    """
    Get information about the loaded ML model
    """
    try:
        model_info = cpet_service.ml_service.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'timestamp': pd.Timestamp.now().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Model info request failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'code': 'MODEL_INFO_ERROR'
        }), 500


# Error handlers for this blueprint
@cpet_bp.errorhandler(413)
def file_too_large(e):
    """Handle file upload size limit exceeded"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'code': 'FILE_TOO_LARGE',
        'max_size': '10MB'
    }), 413


@cpet_bp.errorhandler(415)
def unsupported_media_type(e):
    """Handle unsupported file type"""
    return jsonify({
        'success': False,
        'error': 'Unsupported file type',
        'code': 'UNSUPPORTED_FILE_TYPE',
        'supported_types': ['csv', 'xlsx', 'xls', 'txt']
    }), 415