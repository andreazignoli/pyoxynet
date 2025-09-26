"""
PyOxynet CPET Analysis API v1
Modern RESTful API with OpenAPI documentation
"""
import time
from flask import request, current_app
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
import pandas as pd
from typing import Dict, Any

from ...core.services.cpet_service import CPETAnalysisService
from ...core.utils.api_response import APIResponse, APIResponseTimer


# Create namespace instance for v1 API
from flask_restx import Namespace

api = Namespace(
    'cpet',
    path='/api/v1',
    description="""
    **Professional CPET Analysis API for Exercise Physiologists and Researchers**
    
    PyOxynet provides scientifically accurate cardiopulmonary exercise test (CPET) analysis 
    using machine learning models trained on exercise physiology data. Our API enables 
    researchers worldwide to integrate advanced CPET analysis into their workflows.
    
    ## Key Features
    - **Multi-format Support**: CSV, Excel, TXT files from major metabolimeters
    - **ML-Powered Analysis**: Exercise domain classification and threshold detection  
    - **Scientific Visualizations**: 15+ interactive plots for comprehensive analysis
    - **Medical Data Privacy**: HIPAA-compliant data handling with automatic cleanup
    - **Research-Grade Accuracy**: Validated against exercise physiology literature
    
    ## Analysis Capabilities
    - Exercise domain classification (Moderate/Heavy/Severe intensity)
    - Ventilatory threshold detection (VT1, VT2)
    - Metabolic analysis and substrate utilization
    - Interactive visualizations and comprehensive reporting
    - Data quality assessment and validation
    
    ## Getting Started
    1. Upload your CPET data file or send JSON data
    2. Receive comprehensive analysis results with visualizations
    3. Download reports and integrate findings into your research
    
    For detailed examples and SDKs, visit our documentation.
    """
)

# Service instance (will be injected)
cpet_service: CPETAnalysisService = None


def init_analysis_api(service: CPETAnalysisService):
    """Initialize API with service dependency"""
    global cpet_service
    cpet_service = service


# API Models for documentation
analysis_options = api.model('AnalysisOptions', {
    'include_nine_panel': fields.Boolean(
        default=True,
        description='Include comprehensive 9-panel visualization'
    ),
    'include_metabolic_analysis': fields.Boolean(
        default=True,
        description='Include metabolic analysis plots'
    ),
    'threshold_detection_method': fields.String(
        default='ml',
        enum=['ml', 'traditional', 'hybrid'],
        description='Method for threshold detection'
    ),
    'domain_classification': fields.Boolean(
        default=True,
        description='Perform exercise domain classification'
    ),
    'confidence_threshold': fields.Float(
        default=0.7,
        min=0.0,
        max=1.0,
        description='Minimum confidence for predictions'
    )
})

cpet_data_point = api.model('CPETDataPoint', {
    'TIME': fields.Float(description='Time in seconds'),
    'VO2': fields.Float(required=True, description='Oxygen uptake (ml/min)', min=0),
    'VCO2': fields.Float(required=True, description='CO2 output (ml/min)', min=0),
    'VE': fields.Float(required=True, description='Minute ventilation (L/min)', min=0),
    'HR': fields.Float(description='Heart rate (bpm)', min=0, max=300),
    'RER': fields.Float(description='Respiratory exchange ratio', min=0.5, max=2.0),
    'LOAD': fields.Float(description='Workload (watts)', min=0)
})

json_analysis_request = api.model('JSONAnalysisRequest', {
    'data': fields.List(
        fields.Nested(cpet_data_point),
        required=True,
        description='Array of CPET data points (minimum 10 points)'
    ),
    'options': fields.Nested(analysis_options, description='Analysis configuration'),
    'metadata': fields.Raw(description='Additional metadata (subject ID, protocol, etc.)')
})

domain_probabilities = api.model('DomainProbabilities', {
    'Moderate': fields.Float(description='Moderate intensity domain probability'),
    'Heavy': fields.Float(description='Heavy intensity domain probability'),
    'Severe': fields.Float(description='Severe intensity domain probability')
})

ventilatory_thresholds = api.model('VentilatoryThresholds', {
    'VT1': fields.Float(description='First ventilatory threshold (ml/min)'),
    'VT2': fields.Float(description='Second ventilatory threshold (ml/min)')
})

ml_analysis = api.model('MLAnalysis', {
    'domain_probabilities': fields.Nested(domain_probabilities, required=True),
    'dominant_domain': fields.String(required=True, description='Most likely domain'),
    'confidence': fields.Float(required=True, description='Overall confidence score'),
    'ventilatory_thresholds': fields.Nested(ventilatory_thresholds, required=True)
})

analysis_summary = api.model('AnalysisSummary', {
    'vo2_max': fields.Float(description='Maximum VO2 (ml/min)'),
    'dominant_domain': fields.String(description='Primary exercise domain'),
    'confidence': fields.Float(description='Analysis confidence'),
    'test_duration_minutes': fields.Float(description='Total test duration'),
    'data_quality': fields.String(description='Overall data quality rating')
})

standard_response = api.model('StandardResponse', {
    'success': fields.Boolean(required=True, description='Operation success status'),
    'status': fields.String(required=True, description='Response status'),
    'data': fields.Raw(description='Response data payload'),
    'metadata': fields.Raw(description='Response metadata'),
    'errors': fields.Raw(description='Error information if applicable')
})

error_response = api.model('ErrorResponse', {
    'success': fields.Boolean(default=False, description='Always false for errors'),
    'status': fields.String(default='error', description='Always "error"'),
    'data': fields.Raw(default=None, description='Always null for errors'),
    'metadata': fields.Raw(description='Response metadata'),
    'error': fields.Raw(required=True, description='Error details')
})


# File upload parser
file_upload_parser = reqparse.RequestParser()
file_upload_parser.add_argument(
    'file',
    type=FileStorage,
    location='files',
    required=True,
    help='CPET data file (CSV, Excel, or TXT format)'
)
file_upload_parser.add_argument(
    'options',
    type=str,
    location='form',
    help='JSON string with analysis options'
)
file_upload_parser.add_argument(
    'session_id',
    type=str,
    location='form',
    help='Optional session identifier for tracking'
)

# Validation parser
validation_parser = reqparse.RequestParser()
validation_parser.add_argument(
    'file',
    type=FileStorage,
    location='files',
    required=True,
    help='CPET data file to validate'
)
validation_parser.add_argument(
    'preview_rows',
    type=int,
    location='form',
    default=10,
    help='Number of rows to include in preview'
)


@api.route('/analyze/file')
class FileAnalysisResource(Resource):
    """File-based CPET Analysis"""
    
    @api.doc('analyze_file')
    @api.expect(file_upload_parser, validate=True)
    @api.response(200, 'Analysis completed successfully', standard_response)
    @api.response(400, 'Bad request', error_response)
    @api.response(422, 'Validation failed', error_response)
    @api.response(413, 'File too large', error_response)
    @api.response(415, 'Unsupported file type', error_response)
    @api.response(500, 'Internal server error', error_response)
    def post(self):
        """
        Analyze CPET data from uploaded file
        
        Upload a CPET data file (CSV, Excel, or TXT) for comprehensive analysis.
        Supports data from major metabolimeters with automatic format detection.
        
        **Supported File Formats:**
        - CSV (comma, semicolon, or tab separated)
        - Excel (.xlsx, .xls)
        - Text files (.txt)
        
        **Required CPET Variables:**
        - VO2 (oxygen uptake in ml/min)
        - VCO2 (carbon dioxide output in ml/min)  
        - VE (minute ventilation in L/min)
        
        **Optional Variables:**
        - HR (heart rate in bpm)
        - RER (respiratory exchange ratio)
        - Load/Power (workload in watts)
        - Time (seconds from start)
        
        **Returns:**
        Comprehensive analysis including ML-powered domain classification,
        threshold detection, scientific visualizations, and detailed reporting.
        """
        with APIResponseTimer() as timer:
            try:
                args = file_upload_parser.parse_args()
                file_storage = args['file']
                
                # Validate file presence
                if not file_storage or file_storage.filename == '':
                    return APIResponse.error(
                        message="No file provided or file is empty",
                        error_code="NO_FILE"
                    )
                
                # Parse analysis options
                options = {}
                if args.get('options'):
                    import json
                    try:
                        options = json.loads(args['options'])
                    except json.JSONDecodeError:
                        return APIResponse.error(
                            message="Invalid options JSON format",
                            error_code="VALIDATION_ERROR"
                        )
                
                session_id = args.get('session_id', f'api_{int(time.time())}')
                
                # Perform analysis
                results = cpet_service.analyze_file(file_storage, session_id, options)
                
                if not results['success']:
                    return APIResponse.error(
                        message="File analysis failed",
                        errors=results.get('errors', []),
                        error_code="ANALYSIS_ERROR",
                        status_code=422
                    )
                
                # Format response data
                response_data = {
                    'session_id': results['session_id'],
                    'filename': results['filename'],
                    'analysis': {
                        'ml_results': results['ml_analysis'],
                        'report': results['analysis_report'],
                        'processing_info': results['processing_info']
                    },
                    'visualizations': {
                        k: v for k, v in results['visualizations'].items()
                        if k != 'plot_config' and v.get('success', False)
                    }
                }
                
                return APIResponse.success(
                    data=response_data,
                    message="CPET analysis completed successfully",
                    metadata={
                        'data_points': results['processing_info']['total_data_points'],
                        'analysis_quality': results['processing_info']['data_quality']
                    },
                    processing_time=timer.elapsed_time
                )
                
            except Exception as e:
                current_app.logger.error(f"File analysis failed: {e}")
                return APIResponse.internal_error(
                    message="Analysis processing failed"
                )


@api.route('/analyze/data')
class DataAnalysisResource(Resource):
    """JSON Data CPET Analysis"""
    
    @api.doc('analyze_data')
    @api.expect(json_analysis_request, validate=True)
    @api.response(200, 'Analysis completed successfully', standard_response)
    @api.response(400, 'Bad request', error_response)
    @api.response(422, 'Validation failed', error_response)
    @api.response(500, 'Internal server error', error_response)
    def post(self):
        """
        Analyze CPET data provided as JSON
        
        Send CPET data directly as JSON for analysis. Ideal for integration
        with existing systems or programmatic analysis workflows.
        
        **Request Format:**
        ```json
        {
          "data": [
            {"TIME": 0, "VO2": 500, "VCO2": 400, "VE": 15, "HR": 60},
            {"TIME": 30, "VO2": 800, "VCO2": 720, "VE": 25, "HR": 80},
            ...
          ],
          "options": {
            "include_nine_panel": true,
            "threshold_detection_method": "ml",
            "confidence_threshold": 0.7
          }
        }
        ```
        
        **Data Requirements:**
        - Minimum 10 data points for reliable analysis
        - Required: VO2, VCO2, VE values
        - Optional: TIME, HR, RER, LOAD values
        
        **Returns:**
        Same comprehensive analysis as file upload, including domain
        classification, threshold detection, and visualizations.
        """
        with APIResponseTimer() as timer:
            try:
                if not request.is_json:
                    return APIResponse.error(
                        message="Content-Type must be application/json",
                        error_code="VALIDATION_ERROR"
                    )
                
                request_data = request.get_json()
                
                # Validate request structure
                if 'data' not in request_data:
                    return APIResponse.error(
                        message="Missing required field: data",
                        error_code="VALIDATION_ERROR"
                    )
                
                # Convert to DataFrame
                try:
                    data = pd.DataFrame(request_data['data'])
                    if len(data) < 10:
                        return APIResponse.error(
                            message="Minimum 10 data points required for analysis",
                            error_code="INSUFFICIENT_DATA",
                            status_code=422
                        )
                except Exception as e:
                    return APIResponse.error(
                        message=f"Invalid data format: {str(e)}",
                        error_code="INVALID_CPET_DATA",
                        status_code=422
                    )
                
                options = request_data.get('options', {})
                
                # Perform analysis
                results = cpet_service.analyze_data(data, options)
                
                if not results['success']:
                    return APIResponse.error(
                        message="Data analysis failed",
                        errors=results.get('errors', []),
                        error_code="ANALYSIS_ERROR",
                        status_code=422
                    )
                
                # Format response data
                response_data = {
                    'analysis': {
                        'ml_results': results['ml_analysis'],
                        'report': results['analysis_report'],
                        'processing_info': results['processing_info']
                    },
                    'visualizations': {
                        k: v for k, v in results['visualizations'].items()
                        if k != 'plot_config' and v.get('success', False)
                    }
                }
                
                return APIResponse.success(
                    data=response_data,
                    message="CPET data analysis completed successfully",
                    metadata={
                        'data_points': len(data),
                        'analysis_quality': results['processing_info']['data_quality']
                    },
                    processing_time=timer.elapsed_time
                )
                
            except Exception as e:
                current_app.logger.error(f"Data analysis failed: {e}")
                return APIResponse.internal_error(
                    message="Analysis processing failed"
                )


@api.route('/validate')
class DataValidationResource(Resource):
    """CPET Data Validation"""
    
    @api.doc('validate_file')
    @api.expect(validation_parser, validate=True)
    @api.response(200, 'Validation completed', standard_response)
    @api.response(400, 'Bad request', error_response)
    @api.response(422, 'Validation failed', error_response)
    @api.response(500, 'Internal server error', error_response)
    def post(self):
        """
        Validate CPET data file without full analysis
        
        Quick validation of CPET data files to check format, required columns,
        and data quality before performing full analysis. Useful for pre-flight
        checks and data quality assessment.
        
        **Validation Checks:**
        - File format and readability
        - Required CPET columns presence
        - Data type validation
        - Range validation for physiological values
        - Missing data assessment
        - Data quality scoring
        
        **Returns:**
        - Validation results (pass/fail with details)
        - Data preview (first N rows)
        - Column mapping and metadata
        - Data quality assessment
        - Recommendations for improvement
        """
        with APIResponseTimer() as timer:
            try:
                args = validation_parser.parse_args()
                file_storage = args['file']
                preview_rows = args.get('preview_rows', 10)
                
                if not file_storage or file_storage.filename == '':
                    return APIResponse.error(
                        message="No file provided or file is empty",
                        error_code="NO_FILE"
                    )
                
                session_id = f'validation_{int(time.time())}'
                
                # Process file
                data_result = cpet_service.data_service.process_file(
                    file_storage, session_id
                )
                
                if not data_result['success']:
                    return APIResponse.error(
                        message="File processing failed",
                        errors=data_result['errors'],
                        error_code="FILE_PROCESSING_ERROR",
                        status_code=422
                    )
                
                # Validate data
                validation_result = cpet_service.data_service.validate_cpet_data(
                    data_result['data']
                )
                
                # Create data preview
                preview_data = data_result['data'].head(preview_rows).to_dict('records')
                
                response_data = {
                    'validation': {
                        'valid': validation_result['valid'],
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings'],
                        'recommendations': validation_result['recommendations']
                    },
                    'metadata': data_result['metadata'],
                    'preview': {
                        'rows': preview_data,
                        'total_rows': len(data_result['data']),
                        'columns': list(data_result['data'].columns),
                        'preview_count': len(preview_data)
                    }
                }
                
                status_message = (
                    "Data validation passed" if validation_result['valid']
                    else "Data validation failed"
                )
                
                return APIResponse.success(
                    data=response_data,
                    message=status_message,
                    metadata={
                        'filename': file_storage.filename,
                        'file_size_bytes': len(file_storage.read() or b''),
                        'validation_status': 'passed' if validation_result['valid'] else 'failed'
                    },
                    processing_time=timer.elapsed_time
                )
                
            except Exception as e:
                current_app.logger.error(f"Validation failed: {e}")
                return APIResponse.internal_error(
                    message="Validation processing failed"
                )


@api.route('/health')
class HealthCheckResource(Resource):
    """API Health Check"""
    
    @api.doc('health_check')
    @api.response(200, 'Service healthy', standard_response)
    @api.response(503, 'Service degraded', standard_response)
    def get(self):
        """
        API health check endpoint
        
        Check the health and availability of the PyOxynet CPET analysis service.
        Monitors all critical service components including ML models, data processing,
        and visualization services.
        
        **Health Status Levels:**
        - **healthy**: All services operational
        - **degraded**: Some services have issues but core functionality available
        - **unhealthy**: Critical services unavailable
        
        **Monitored Components:**
        - Data processing service
        - ML inference service (TensorFlow Lite models)
        - Visualization service
        - File validation service
        
        **Returns:**
        Detailed health status for each service component with timestamps
        and service availability information.
        """
        try:
            status = cpet_service.get_service_status()
            
            # Determine overall health
            ml_status = status['ml_service']['status']
            overall_healthy = ml_status in ['ready', 'degraded']
            
            health_level = 'healthy' if ml_status == 'ready' else 'degraded'
            status_code = 200 if overall_healthy else 503
            
            response_data = {
                'status': health_level,
                'services': status,
                'uptime_check': True,
                'api_version': "1.0.0"
            }
            
            message = (
                "All services operational" if health_level == 'healthy'
                else "Some services degraded but operational"
            )
            
            # Return raw dict for Flask-RESTX serialization
            return {
                'success': True,
                'status': 'ok',
                'data': response_data,
                'metadata': {
                    'api_version': '1.0.0',
                    'timestamp': pd.Timestamp.now().isoformat()
                },
                'message': message,
                'errors': None
            }, status_code
            
        except Exception as e:
            current_app.logger.error(f"Health check failed: {e}")
            return APIResponse.error(
                message="Health check failed",
                error_code="SERVICE_UNAVAILABLE",
                status_code=503
            )


@api.route('/models/info')
class ModelInfoResource(Resource):
    """ML Model Information"""
    
    @api.doc('model_info')
    @api.response(200, 'Model information retrieved', standard_response)
    @api.response(500, 'Model information unavailable', error_response)
    def get(self):
        """
        Get ML model information and capabilities
        
        Retrieve detailed information about the loaded TensorFlow Lite models
        used for CPET analysis, including model architecture, input/output specs,
        and analysis capabilities.
        
        **Model Information:**
        - Model file path and version
        - Input/output tensor specifications
        - Supported analysis types
        - Model training metadata
        - Performance characteristics
        
        **Use Cases:**
        - Verify model compatibility
        - Understand analysis capabilities
        - Debug integration issues
        - Model version tracking
        """
        try:
            model_info = cpet_service.ml_service.get_model_info()
            
            response_data = {
                'model': model_info,
                'capabilities': {
                    'domain_classification': True,
                    'threshold_detection': True,
                    'supported_variables': ['VO2', 'VCO2', 'VE', 'HR', 'RER'],
                    'min_data_points': 10,
                    'max_data_points': 10000
                },
                'analysis_types': [
                    'exercise_domain_classification',
                    'ventilatory_threshold_detection',
                    'data_quality_assessment'
                ]
            }
            
            return APIResponse.success(
                data=response_data,
                message="Model information retrieved successfully"
            )
            
        except Exception as e:
            current_app.logger.error(f"Model info request failed: {e}")
            return APIResponse.internal_error(
                message="Failed to retrieve model information"
            )


# Error handlers for this API
@api.errorhandler
def handle_validation_error(error):
    """Handle request validation errors"""
    return APIResponse.validation_error(
        validation_errors={"request": str(error)},
        message="Request validation failed"
    )


@api.errorhandler(FileNotFoundError)
def handle_file_not_found(error):
    """Handle file not found errors"""
    return APIResponse.error(
        message="Required file or resource not found",
        error_code="FILE_NOT_FOUND",
        status_code=404
    )


@api.errorhandler(ValueError)
def handle_value_error(error):
    """Handle value errors from data processing"""
    return APIResponse.error(
        message=f"Data processing error: {str(error)}",
        error_code="DATA_PROCESSING_ERROR",
        status_code=422
    )