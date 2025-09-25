"""
Modern PyOxynet Flask Application
Modular architecture with service layers and proper separation of concerns
"""
import os
import logging
from flask import Flask
from flask_cors import CORS
from typing import Optional

from config import DevelopmentConfig, ProductionConfig, TestingConfig
from security import create_security_manager
from .api.routes.cpet_analysis import init_cpet_routes
from .api.routes.web_interface import init_web_routes


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory for PyOxynet Flask app
    
    Args:
        config_name: Configuration name (development, production, testing)
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
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
    
    # Configure CORS
    CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])  # Frontend dev server
    
    # Register blueprints
    init_cpet_routes(app)
    init_web_routes(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Add health check endpoint
    @app.route('/health')
    def health_check():
        """Application health check"""
        return {
            'status': 'healthy',
            'service': 'pyoxynet-api',
            'version': '2.0.0',
            'environment': config_name
        }
    
    app.logger.info(f"PyOxynet application created with {config_name} configuration")
    
    return app


def setup_logging(app: Flask) -> None:
    """Configure application logging"""
    if not app.debug and not app.testing:
        # Production logging configuration
        if app.config.get('LOG_FILE'):
            file_handler = logging.FileHandler(app.config['LOG_FILE'])
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
        
        # Set appropriate log level
        app.logger.setLevel(logging.INFO)
        app.logger.info('PyOxynet application startup')


def register_error_handlers(app: Flask) -> None:
    """Register global error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        if error.description:
            return {
                'error': 'Bad Request',
                'message': error.description,
                'code': 400
            }, 400
        return {
            'error': 'Bad Request',
            'message': 'The request could not be understood by the server',
            'code': 400
        }, 400
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        return {
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'code': 404
        }, 404
    
    @app.errorhandler(413)
    def file_too_large(error):
        """Handle 413 Request Entity Too Large errors"""
        return {
            'error': 'File Too Large',
            'message': f'File size exceeds maximum allowed size of {app.config.get("MAX_CONTENT_LENGTH", "unknown")} bytes',
            'code': 413
        }, 413
    
    @app.errorhandler(415)
    def unsupported_media_type(error):
        """Handle 415 Unsupported Media Type errors"""
        return {
            'error': 'Unsupported Media Type',
            'message': 'The uploaded file type is not supported',
            'code': 415,
            'supported_types': ['csv', 'xlsx', 'xls', 'txt']
        }, 415
    
    @app.errorhandler(422)
    def unprocessable_entity(error):
        """Handle 422 Unprocessable Entity errors"""
        return {
            'error': 'Unprocessable Entity',
            'message': 'The request was well-formed but contains semantic errors',
            'code': 422
        }, 422
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server Error"""
        app.logger.error(f'Internal server error: {error}')
        return {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'code': 500
        }, 500


if __name__ == '__main__':
    # Development server
    app = create_app('development')
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=True
    )