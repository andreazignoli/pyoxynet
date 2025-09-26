"""
PyOxynet Modern API Application
Production-ready Flask application with OpenAPI documentation
"""
import os
import logging
from flask import Flask, redirect, url_for
from flask_restx import Api
from flask_cors import CORS
from typing import Optional

from config import DevelopmentConfig, ProductionConfig, TestingConfig
from security import create_security_manager
from .core.services.cpet_service import CPETAnalysisService
from .api.v1.analysis_api import api as analysis_api_v1, init_analysis_api
from .core.utils.api_response import APIResponse


def create_api_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory for PyOxynet API
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configured Flask API application
    """
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.environ.get('FLASK_ENV', 'development')
    
    if config_name == 'production':
        app.config.from_object(ProductionConfig())
    elif config_name == 'testing':
        app.config.from_object(TestingConfig())
    else:
        app.config.from_object(DevelopmentConfig())
    
    # Initialize logging
    setup_logging(app)
    
    # Initialize security
    security_manager = create_security_manager(app.config, app.logger)
    app.security_manager = security_manager
    
    # Configure CORS for API
    CORS(app, 
         origins=['*'],  # Configure appropriately for production
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
         allow_headers=['Content-Type', 'Authorization', 'X-API-Key'])
    
    # Create main API instance
    main_api = Api(
        app,
        title="PyOxynet CPET Analysis Platform",
        version="1.0.0",
        description="""
        **Professional CPET Analysis API for Exercise Physiologists and Researchers**
        
        The PyOxynet platform provides scientifically accurate cardiopulmonary exercise test (CPET) 
        analysis using machine learning models trained on exercise physiology data. Our RESTful API 
        enables researchers worldwide to integrate advanced CPET analysis into their workflows with 
        enterprise-grade reliability and medical data privacy compliance.
        
        ## 🔬 Scientific Features
        - **ML-Powered Domain Classification**: Moderate/Heavy/Severe exercise intensity domains
        - **Threshold Detection**: Automated VT1/VT2 identification using validated algorithms
        - **Metabolic Analysis**: Substrate utilization, fat oxidation, energy expenditure
        - **Data Quality Assessment**: Comprehensive validation and quality scoring
        - **Multi-format Support**: CSV, Excel, TXT from major metabolimeters (Cortex, Cosmed, etc.)
        
        ## 📊 Visualization & Reporting
        - **15+ Scientific Plots**: Interactive Plotly visualizations
        - **Comprehensive Reports**: Detailed analysis with clinical interpretation
        - **Export Options**: JSON, CSV, PDF reports for research publications
        - **9-Panel Analysis**: Complete CPET overview in single visualization
        
        ## 🔒 Medical Data Privacy
        - **HIPAA Compliant**: Automatic data cleanup and secure handling
        - **Session Isolation**: Individual processing sessions with privacy controls
        - **No Data Retention**: Files automatically removed after processing
        - **Audit Logging**: Complete processing audit trails for compliance
        
        ## 🚀 Developer Experience
        - **RESTful Design**: Clean, predictable API following REST principles
        - **OpenAPI Documentation**: Interactive documentation with live testing
        - **SDKs Available**: Python, R, JavaScript clients for easy integration
        - **Code Examples**: Ready-to-use snippets for common workflows
        - **Comprehensive Error Handling**: Detailed error messages and codes
        
        ## 📈 Enterprise Ready
        - **High Performance**: Optimized for large-scale research workflows
        - **Reliable Processing**: Robust error handling and graceful degradation
        - **Monitoring**: Health checks and service status endpoints
        - **Scalable Architecture**: Modular design supports horizontal scaling
        
        ## Getting Started
        
        ### Quick Analysis (cURL)
        ```bash
        curl -X POST "https://api.pyoxynet.com/api/v1/analyze/file" \\
             -F "file=@your_cpet_data.csv" \\
             -F "options={\"include_nine_panel\": true}"
        ```
        
        ### Python SDK
        ```python
        import pyoxynet
        
        client = pyoxynet.Client(api_key="your_api_key")
        results = client.analyze_file("cpet_data.csv")
        
        print(f"Dominant domain: {results.ml_analysis.dominant_domain}")
        print(f"Confidence: {results.ml_analysis.confidence}")
        ```
        
        ### R Integration
        ```r
        library(pyoxynet)
        
        results <- analyze_cpet_file("data.csv")
        plot_vo2_time(results$visualizations$vo2_time_plot)
        ```
        
        ## Support & Documentation
        - **API Documentation**: [docs.pyoxynet.com/api](https://docs.pyoxynet.com/api)
        - **Scientific Methods**: [docs.pyoxynet.com/science](https://docs.pyoxynet.com/science)
        - **Integration Guide**: [docs.pyoxynet.com/integration](https://docs.pyoxynet.com/integration)
        - **Support**: support@pyoxynet.com
        
        Built with ❤️ for the exercise physiology research community.
        """,
        doc="/",  # Root serves documentation
        prefix="",
        contact="PyOxynet Team",
        contact_email="support@pyoxynet.com",
        license="MIT",
        license_url="https://opensource.org/licenses/MIT",
        external_docs={
            "description": "Complete Documentation",
            "url": "https://docs.pyoxynet.com"
        }
    )
    
    # Initialize services
    cpet_service = CPETAnalysisService(logger=app.logger)
    
    # Initialize API v1
    init_analysis_api(cpet_service)
    
    # Add namespaces
    main_api.add_namespace(analysis_api_v1)
    
    # Add root redirect to docs
    @app.route('/')
    def index():
        """Redirect to API documentation"""
        return redirect('/docs/')
    
    # Add API info endpoint
    @app.route('/api')
    def api_info():
        """API information and version details"""
        return APIResponse.success(
            data={
                "name": "PyOxynet CPET Analysis API",
                "version": "1.0.0", 
                "description": "Professional CPET analysis for exercise physiologists",
                "endpoints": {
                    "v1": "/api/v1/",
                    "docs": "/docs/",
                    "health": "/api/v1/health"
                },
                "features": [
                    "ML-powered exercise domain classification",
                    "Ventilatory threshold detection", 
                    "Multi-format file support",
                    "Interactive visualizations",
                    "Medical data privacy compliance"
                ],
                "supported_formats": ["CSV", "Excel", "TXT"],
                "max_file_size_mb": 10,
                "rate_limits": {
                    "free_tier": "100 requests/hour",
                    "authenticated": "1000 requests/hour"
                }
            },
            message="PyOxynet API v1.0.0 - Ready for CPET analysis"
        )[0]  # Return just the response, not tuple
    
    # Global error handlers
    register_error_handlers(app, main_api)
    
    app.logger.info(f"PyOxynet API application created with {config_name} configuration")
    
    return app


def setup_logging(app: Flask) -> None:
    """Configure application logging for API"""
    if not app.debug and not app.testing:
        # Production logging
        if app.config.get('LOG_FILE'):
            file_handler = logging.FileHandler(app.config['LOG_FILE'])
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [%(name)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('PyOxynet API startup')


def register_error_handlers(app: Flask, api: Api) -> None:
    """Register comprehensive error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return APIResponse.error(
            message="Bad request - invalid request format",
            error_code="BAD_REQUEST",
            status_code=400
        )
    
    @app.errorhandler(404)
    def not_found(error):
        return APIResponse.not_found("API endpoint")
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return APIResponse.error(
            message="HTTP method not allowed for this endpoint",
            error_code="METHOD_NOT_ALLOWED", 
            status_code=405
        )
    
    @app.errorhandler(413)
    def file_too_large(error):
        return APIResponse.error(
            message=f"File size exceeds maximum allowed ({app.config.get('MAX_CONTENT_LENGTH', 10*1024*1024)} bytes)",
            error_code="FILE_TOO_LARGE",
            details={
                "max_size_mb": (app.config.get('MAX_CONTENT_LENGTH', 10*1024*1024) / 1024 / 1024),
                "supported_formats": ["CSV", "Excel", "TXT"]
            },
            status_code=413
        )
    
    @app.errorhandler(415)
    def unsupported_media_type(error):
        return APIResponse.error(
            message="Unsupported file type or content type",
            error_code="UNSUPPORTED_MEDIA_TYPE",
            details={
                "supported_file_types": ["CSV", "Excel (.xlsx/.xls)", "TXT"],
                "supported_content_types": ["multipart/form-data", "application/json"]
            },
            status_code=415
        )
    
    @app.errorhandler(422)
    def unprocessable_entity(error):
        return APIResponse.error(
            message="Request validation failed",
            error_code="UNPROCESSABLE_ENTITY",
            status_code=422
        )
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        return APIResponse.rate_limited(
            retry_after=3600  # 1 hour
        )
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Internal server error: {error}')
        return APIResponse.internal_error(
            message="An unexpected error occurred during processing"
        )
    
    @app.errorhandler(503)
    def service_unavailable(error):
        return APIResponse.error(
            message="Service temporarily unavailable",
            error_code="SERVICE_UNAVAILABLE",
            details={
                "retry_after": "Please try again in a few minutes",
                "status_endpoint": "/api/v1/health"
            },
            status_code=503
        )


if __name__ == '__main__':
    # Development server
    app = create_api_app('development')
    
    print("\n" + "="*80)
    print("🚀 PyOxynet CPET Analysis API v1.0.0")
    print("="*80)
    print(f"📖 Interactive Documentation: http://localhost:5000/")
    print(f"🔬 Analysis Endpoint: http://localhost:5000/api/v1/analyze/file")
    print(f"❤️  Health Check: http://localhost:5000/api/v1/health")
    print(f"📊 API Info: http://localhost:5000/api")
    print("="*80 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=True
    )