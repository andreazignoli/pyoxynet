"""
Web Interface Routes
Traditional web interface for CPET analysis with HTML templates
"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, session, current_app, jsonify
import pandas as pd
import logging
from typing import Dict, Any
import uuid

from ...core.services.cpet_service import CPETAnalysisService


web_bp = Blueprint('web_interface', __name__)

# Initialize service
cpet_service = None


def init_web_routes(app):
    """Initialize web routes with application context"""
    global cpet_service
    cpet_service = CPETAnalysisService(logger=app.logger)
    app.register_blueprint(web_bp)


@web_bp.route('/')
def index():
    """Homepage with file upload interface"""
    try:
        # Generate session ID if not exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        return render_template('index.html', 
                             session_id=session['session_id'])
        
    except Exception as e:
        current_app.logger.error(f"Homepage rendering failed: {e}")
        flash('Error loading homepage', 'error')
        return render_template('error.html', 
                             error_message="Unable to load homepage"), 500


@web_bp.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """File analysis interface"""
    if request.method == 'GET':
        return render_template('analyze.html')
    
    try:
        # Handle file upload
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(url_for('web_interface.analyze'))
        
        file_storage = request.files['file']
        if file_storage.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('web_interface.analyze'))
        
        # Get session ID
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id
        
        # Get analysis options from form
        options = {
            'include_nine_panel': request.form.get('include_nine_panel') == 'on',
            'generate_pdf_report': request.form.get('generate_pdf_report') == 'on'
        }
        
        # Run analysis
        results = cpet_service.analyze_file(file_storage, session_id, options)
        
        if not results['success']:
            for error in results.get('errors', []):
                flash(f'Analysis error: {error}', 'error')
            return redirect(url_for('web_interface.analyze'))
        
        # Store results in session for results page
        session['last_analysis_results'] = {
            'session_id': results['session_id'],
            'filename': results['filename'],
            'timestamp': results['timestamp']
        }
        
        # Redirect to results page
        return redirect(url_for('web_interface.results'))
        
    except Exception as e:
        current_app.logger.error(f"File analysis failed: {e}")
        flash('Analysis failed due to internal error', 'error')
        return redirect(url_for('web_interface.analyze'))


@web_bp.route('/results')
def results():
    """Display analysis results"""
    try:
        # Check if we have results to display
        if 'last_analysis_results' not in session:
            flash('No analysis results found', 'warning')
            return redirect(url_for('web_interface.analyze'))
        
        result_info = session['last_analysis_results']
        
        # In a full implementation, we would retrieve results from database
        # For now, we'll show a results template with the basic info
        return render_template('results.html', 
                             result_info=result_info)
        
    except Exception as e:
        current_app.logger.error(f"Results display failed: {e}")
        flash('Error displaying results', 'error')
        return redirect(url_for('web_interface.analyze'))


@web_bp.route('/results-data')
def results_data():
    """AJAX endpoint to get analysis results data"""
    try:
        # This would typically retrieve from database or cache
        # For now, return placeholder structure
        if 'last_analysis_results' not in session:
            return jsonify({'error': 'No results found'}), 404
        
        # Placeholder response structure
        placeholder_data = {
            'ml_analysis': {
                'domain_probabilities': {
                    'Moderate': 0.6,
                    'Heavy': 0.3,
                    'Severe': 0.1
                },
                'dominant_domain': 'Moderate',
                'confidence': 0.85,
                'ventilatory_thresholds': {
                    'VT1': 1200,
                    'VT2': 1800
                }
            },
            'analysis_report': {
                'summary': {
                    'vo2_max': 2500,
                    'dominant_domain': 'Moderate',
                    'confidence': 0.85,
                    'data_quality': 'Good'
                },
                'interpretation': {
                    'primary_finding': 'Exercise primarily in moderate intensity domain.',
                    'clinical_significance': 'Moderate aerobic capacity indicated.',
                    'recommendations': []
                }
            },
            'visualizations': {
                'vo2_time_plot': {'success': True},
                'vo2_vco2_plot': {'success': True},
                'domain_summary': {'success': True}
            }
        }
        
        return jsonify(placeholder_data), 200
        
    except Exception as e:
        current_app.logger.error(f"Results data request failed: {e}")
        return jsonify({'error': 'Failed to load results'}), 500


@web_bp.route('/training')
def training():
    """Training interface for CPET interpretation"""
    try:
        return render_template('training.html')
        
    except Exception as e:
        current_app.logger.error(f"Training page failed: {e}")
        flash('Error loading training interface', 'error')
        return render_template('error.html', 
                             error_message="Unable to load training interface"), 500


@web_bp.route('/api-docs')
def api_docs():
    """API documentation page"""
    try:
        return render_template('api_docs.html')
        
    except Exception as e:
        current_app.logger.error(f"API docs page failed: {e}")
        flash('Error loading API documentation', 'error')
        return render_template('error.html', 
                             error_message="Unable to load API documentation"), 500


@web_bp.route('/about')
def about():
    """About PyOxynet page"""
    try:
        return render_template('about.html')
        
    except Exception as e:
        current_app.logger.error(f"About page failed: {e}")
        flash('Error loading about page', 'error')
        return render_template('error.html', 
                             error_message="Unable to load about page"), 500


# Context processors for templates
@web_bp.app_context_processor
def inject_globals():
    """Inject global variables into all templates"""
    return {
        'app_name': 'PyOxynet',
        'app_version': '2.0.0',
        'current_year': pd.Timestamp.now().year
    }


# Error handlers for web interface
@web_bp.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors for web interface"""
    return render_template('error.html', 
                         error_message="Page not found",
                         error_code=404), 404


@web_bp.errorhandler(500)
def internal_server_error(e):
    """Handle 500 errors for web interface"""
    return render_template('error.html', 
                         error_message="Internal server error",
                         error_code=500), 500