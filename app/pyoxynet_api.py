#!/usr/bin/env python3
"""
PyOxynet CPET Analysis API - Unified Documentation & Implementation

Professional CPET Analysis API with comprehensive documentation and working examples.
Combines the best features from both legacy and modern implementations.

QUICK START:
1. Start the server: python pyoxynet_api.py
2. View docs: http://127.0.0.1:5000/docs/
3. API info: http://127.0.0.1:5000/info
4. Test endpoint: POST http://127.0.0.1:5000/analyze/data

WORKING EXAMPLES:
- See test_api_examples.md for complete working examples
- example_request_minimal.json - Simple 50-point test
- example_request_dataset.json - Complete dataset format
- Both files contain verified, working data formats

REQUIREMENTS:
- Minimum 40 data points for moving window analysis  
- Required fields: VO2, VCO2, VE, PetO2, PetCO2
- Units: VO2/VCO2 (ml/min), VE (L/min), PetO2/PetCO2 (mmHg)

ANALYSIS METHOD:
- 40-point sliding window with TensorFlow Lite inference
- Domain classification: Moderate/Heavy/Severe
- Threshold detection: VT1/VT2 from domain transitions
- Temporal analysis of complete CPET tests
"""
import os
import logging
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS

# Import services
from pyoxynet_api.core.services.cpet_service import CPETAnalysisService
from pyoxynet_api.core.utils.api_response import APIResponse, NumpyEncoder

def create_unified_api():
    """Create the unified PyOxynet API with comprehensive documentation"""
    app = Flask(__name__)
    app.json_encoder = NumpyEncoder
    
    # Enable CORS
    CORS(app)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create API with comprehensive documentation
    api = Api(
        app,
        title="PyOxynet CPET Analysis API",
        version="1.0.0", 
        description="Professional CPET Analysis API for Exercise Physiologists. ML-powered exercise domain classification and ventilatory threshold detection using PyOxynet's TensorFlow Lite models with 40-point moving window analysis.",
        doc="/docs/",
        contact="PyOxynet Team",
        contact_email="support@pyoxynet.com"
    )
    
    # Configure custom JSON encoder
    @api.representation('application/json')
    def custom_json(data, code, headers=None):
        """Custom JSON representation with NumPy support"""
        settings = {'indent': 4} if app.debug else {}
        import json
        dumped = json.dumps(data, cls=NumpyEncoder, **settings)
        from flask import make_response
        resp = make_response(dumped, code)
        resp.headers.extend(headers or {})
        resp.headers['Content-Type'] = 'application/json'
        return resp
    
    # Define comprehensive data models
    cpet_point = api.model('CPETDataPoint', {
        't': fields.Float(description='Time in seconds (optional)', example=-240),
        'VO2': fields.Float(required=True, description='Oxygen uptake (ml/min)', example=1680),
        'VCO2': fields.Float(required=True, description='CO2 output (ml/min)', example=1090),
        'VE': fields.Float(required=True, description='Minute ventilation (L/min)', example=31.3),
        'PetO2': fields.Float(required=True, description='End-tidal O2 pressure (mmHg)', example=76.506),
        'PetCO2': fields.Float(required=True, description='End-tidal CO2 pressure (mmHg)', example=37.0712),
        'R': fields.Float(description='Respiratory exchange ratio (calculated if not provided)', example=0.65),
        'VE/VO2': fields.Float(description='Ventilatory equivalent for oxygen (calculated if not provided)', example=18.7),
        'VE/VCO2': fields.Float(description='Ventilatory equivalent for CO2 (calculated if not provided)', example=28.8)
    })
    
    dataset = api.model('CPETDataset', {
        'id': fields.String(required=True, description='Dataset identifier', example='test_001'),
        'VO2max': fields.Float(description='Maximum oxygen uptake (ml/min)', example=4694.3448),
        'LT': fields.Float(description='Lactate threshold (ml/min)', example=3150),
        'LT_vo2max': fields.String(description='LT as percentage of VO2max', example='67%'),
        'RCP': fields.Float(description='Respiratory compensation point (ml/min)', example=3700),
        'RCP_vo2max': fields.String(description='RCP as percentage of VO2max', example='79%'),
        'data': fields.List(fields.Nested(cpet_point), required=True, description='CPET measurement points (minimum 40 required)')
    })
    
    analysis_request = api.model('AnalysisRequest', {
        'datasets': fields.List(fields.Nested(dataset), description='Array of complete CPET datasets (pyoxynet format)'),
        'data': fields.List(fields.Nested(cpet_point), 
                          description='Direct array of CPET data points (minimum 40 points required)',
                          min_items=40)
    })
    
    # Initialize services
    cpet_service = CPETAnalysisService()
    
    @api.route('/info')
    class APIInfo(Resource):
        """Complete API Information"""
        def get(self):
            """Get comprehensive API information and usage guide"""
            return {
                "name": "PyOxynet CPET Analysis API",
                "version": "1.0.0",
                "description": "ML-powered exercise domain classification and ventilatory threshold detection",
                "endpoints": {
                    "info": "GET /info - This endpoint (comprehensive API information)",
                    "health": "GET /health - API health check",
                    "analyze": "POST /analyze/data - Analyze CPET data with moving window",
                    "docs": "GET /docs/ - Interactive Swagger documentation"
                },
                "requirements": {
                    "minimum_data_points": 40,
                    "required_fields": ["VO2", "VCO2", "VE", "PetO2", "PetCO2"],
                    "optional_fields": ["t", "R", "VE/VO2", "VE/VCO2"],
                    "units": {
                        "VO2_VCO2": "ml/min (milliliters per minute)",
                        "VE": "L/min (liters per minute)", 
                        "PetO2_PetCO2": "mmHg (millimeters of mercury)",
                        "t": "seconds"
                    }
                },
                "analysis_method": {
                    "technique": "40-point sliding window with TensorFlow Lite inference",
                    "domains": ["Moderate", "Heavy", "Severe"],
                    "thresholds": ["VT1 (Moderate to Heavy)", "VT2 (Heavy to Severe)"],
                    "coverage": "Analysis covers (total_points - 39) predictions"
                },
                "examples": {
                    "test_files": ["example_request_minimal.json", "example_request_dataset.json"],
                    "documentation": "test_api_examples.md",
                    "curl_examples": [
                        "curl -X POST http://127.0.0.1:5000/analyze/data -H 'Content-Type: application/json' -d @example_request_minimal.json"
                    ]
                },
                "data_formats": {
                    "direct_data": {"data": [{"VO2": 1680, "VCO2": 1090, "VE": 31.3, "PetO2": 76.506, "PetCO2": 37.0712}]},
                    "dataset_format": {"datasets": [{"id": "test_001", "data": "array_of_40_plus_cpet_points"}]}
                }
            }

    @api.route('/health')
    class HealthCheck(Resource):
        """Health Check Endpoint"""
        def get(self):
            """Check API health status and model availability"""
            try:
                # Check if ML service is working
                model_info = cpet_service.ml_service.get_model_info()
                return {
                    "status": "healthy", 
                    "message": "PyOxynet API is running",
                    "ml_model": {
                        "loaded": model_info.get("loaded", False),
                        "type": model_info.get("type", "unknown")
                    },
                    "endpoints": ["health", "info", "analyze/data", "docs/"]
                }
            except Exception as e:
                return {
                    "status": "degraded",
                    "message": f"API running but ML service issue: {str(e)}",
                    "endpoints": ["health", "info", "docs/"]
                }

    @api.route('/analyze/data')
    class DataAnalysis(Resource):
        """CPET Data Analysis with Moving Window"""
        
        @api.expect(analysis_request, validate=True)
        @api.doc('analyze_cpet_data', 
                responses={
                    200: 'Analysis completed successfully',
                    400: 'Bad Request - Invalid data format or insufficient data points',
                    422: 'Validation Error - Missing required fields',
                    500: 'Internal Server Error - Analysis failed'
                })
        def post(self):
            """
            Analyze CPET data using PyOxynet's 40-point moving window approach
            
            Performs moving window analysis across complete CPET test to classify exercise domains and detect ventilatory thresholds.
            
            Requires minimum 40 data points. Two input formats supported:
            1. Direct data array: {"data": [array of CPET points]}
            2. Dataset format: {"datasets": [{"id": "test", "data": [array]}]}
            
            Required fields: VO2, VCO2, VE, PetO2, PetCO2
            Units: VO2/VCO2 (ml/min), VE (L/min), PetO2/PetCO2 (mmHg)
            """
            try:
                request_data = api.payload
                
                # Handle both direct data and datasets format
                if 'data' in request_data and request_data['data']:
                    data = request_data['data']
                elif 'datasets' in request_data and request_data['datasets']:
                    # Extract data from first dataset
                    dataset = request_data['datasets'][0]
                    data = dataset.get('data', [])
                else:
                    return APIResponse.error(
                        message="No data provided",
                        errors=["Either 'data' array or 'datasets' array must be provided"]
                    )
                
                # Validate minimum data requirements for PyOxynet
                if len(data) < 40:
                    return APIResponse.error(
                        message=f"Insufficient data for PyOxynet moving window analysis",
                        errors=[
                            f"Minimum 40 data points required. Provided: {len(data)} points.",
                            "PyOxynet uses a 40-point sliding window for threshold detection.",
                            "Each prediction requires 40 consecutive CPET measurements.",
                            f"With {len(data)} points, analysis coverage would be: {max(0, len(data) - 39)}/{len(data)} predictions."
                        ]
                    )
                
                # Perform comprehensive CPET analysis
                results = cpet_service.analyze_data(data, {})
                
                if results['success']:
                    return APIResponse.success(
                        data=results,
                        message=f"Moving window analysis completed successfully. Coverage: {results.get('processing_info', {}).get('moving_window_coverage', 'unknown')}"
                    )
                else:
                    return APIResponse.error(
                        message="CPET analysis failed",
                        errors=results.get('errors', ['Unknown analysis error'])
                    )
                    
            except Exception as e:
                app.logger.error(f"Analysis error: {str(e)}")
                return APIResponse.internal_error(f"Analysis error: {str(e)}")
    
    return app

if __name__ == '__main__':
    app = create_unified_api()
    print("🚀 Starting PyOxynet CPET Analysis API")
    print("📚 Documentation: http://127.0.0.1:5000/docs/")
    print("ℹ️  API Info: http://127.0.0.1:5000/info")
    print("❤️  Health Check: http://127.0.0.1:5000/health")
    print("🔬 Analysis: POST http://127.0.0.1:5000/analyze/data")
    print()
    print("📁 Working Examples:")
    print("   - example_request_minimal.json")
    print("   - example_request_dataset.json") 
    print("   - test_api_examples.md")
    app.run(debug=True, port=5000)