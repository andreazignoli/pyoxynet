"""
CPET Analysis Service
High-level orchestration service for complete CPET analysis workflow
"""
import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from werkzeug.datastructures import FileStorage

from .data_service import CPETDataService
from .ml_service import CPETMLService
from .visualization_service import CPETVisualizationService


class CPETAnalysisService:
    """
    High-level CPET analysis orchestration service
    
    Coordinates:
    - Data processing and validation
    - ML inference for domain classification
    - Visualization generation
    - Results aggregation and reporting
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize sub-services
        self.data_service = CPETDataService(logger=self.logger)
        self.ml_service = CPETMLService(logger=self.logger)
        self.visualization_service = CPETVisualizationService(logger=self.logger)
        
        self.logger.info("CPET Analysis Service initialized")
    
    def analyze_file(self, file_storage: FileStorage, session_id: str, options: Optional[Dict] = None) -> Dict:
        """
        Complete CPET file analysis workflow
        
        Args:
            file_storage: Uploaded CPET file
            session_id: User session identifier
            options: Analysis options and preferences
            
        Returns:
            Complete analysis results with data, predictions, and visualizations
        """
        options = options or {}
        
        try:
            self.logger.info(f"Starting CPET analysis for file: {file_storage.filename}")
            
            # Step 1: Process and validate data
            self.logger.debug("Step 1: Processing file data")
            data_result = self.data_service.process_file(file_storage, session_id)
            
            if not data_result['success']:
                return self._create_error_result("Data processing failed", data_result['errors'])
            
            data = data_result['data']
            metadata = data_result['metadata']
            
            # Step 2: Validate data for analysis
            self.logger.debug("Step 2: Validating CPET data")
            validation_result = self.data_service.validate_cpet_data(data)
            
            if not validation_result['valid']:
                return self._create_error_result("Data validation failed", validation_result['errors'])
            
            # Step 3: Run ML analysis
            self.logger.debug("Step 3: Running ML analysis")
            ml_results = self.ml_service.predict_exercise_domains(data)
            
            # Step 4: Generate visualizations
            self.logger.debug("Step 4: Generating visualizations")
            visualizations = self._generate_visualizations(data, ml_results, options)
            
            # Step 5: Create comprehensive analysis report
            self.logger.debug("Step 5: Creating analysis report")
            analysis_report = self._create_analysis_report(data, metadata, ml_results, validation_result)
            
            # Step 6: Assemble final results
            results = {
                'success': True,
                'session_id': session_id,
                'filename': file_storage.filename,
                'timestamp': pd.Timestamp.now().isoformat(),
                'data': {
                    'processed_data': data.to_dict('records'),
                    'metadata': metadata
                },
                'ml_analysis': ml_results,
                'visualizations': visualizations,
                'analysis_report': analysis_report,
                'validation': validation_result,
                'processing_info': {
                    'total_data_points': len(data),
                    'analysis_duration': metadata.get('duration_minutes', 'Unknown'),
                    'data_quality': metadata.get('data_quality', 'Unknown')
                }
            }
            
            self.logger.info(f"CPET analysis completed successfully for {file_storage.filename}")
            return results
            
        except Exception as e:
            self.logger.error(f"CPET analysis failed: {e}")
            return self._create_error_result("Analysis failed", [str(e)])
    
    def analyze_data(self, data: List[Dict], options: Optional[Dict] = None) -> Dict:
        """
        Analyze pre-loaded CPET data (for API endpoints)
        
        Args:
            data: List of CPET data points or DataFrame with standardized columns
            options: Analysis options
            
        Returns:
            Analysis results without file processing
        """
        options = options or {}
        
        try:
            self.logger.info("Starting CPET data analysis")
            
            # Convert list of dicts to DataFrame if needed
            if isinstance(data, list):
                if len(data) == 0:
                    return self._create_error_result("No data provided", ["Empty data array"])
                
                # Check minimum data requirement for pyoxynet
                if len(data) < 40:
                    return self._create_error_result(
                        f"Insufficient data: {len(data)} points", 
                        [f"Minimum 40 data points required for pyoxynet analysis. Provided: {len(data)}"]
                    )
                
                data = pd.DataFrame(data)
            
            # Validate data
            validation_result = self.data_service.validate_cpet_data(data)
            if not validation_result['valid']:
                return self._create_error_result("Data validation failed", validation_result['errors'])
            
            # Run ML analysis with moving window
            ml_results = self.ml_service.predict_exercise_domains(data)
            
            # Generate visualizations
            visualizations = self._generate_visualizations(data, ml_results, options)
            
            # Create analysis report
            metadata = self.data_service._generate_metadata(data, "direct_input", "generic")
            analysis_report = self._create_analysis_report(data, metadata, ml_results, validation_result)
            
            results = {
                'success': True,
                'timestamp': pd.Timestamp.now().isoformat(),
                'ml_analysis': ml_results,
                'visualizations': visualizations,
                'analysis_report': analysis_report,
                'validation': validation_result,
                'processing_info': {
                    'total_data_points': len(data),
                    'data_quality': metadata.get('data_quality', 'Unknown'),
                    'moving_window_coverage': ml_results.get('moving_window_info', {}).get('analysis_coverage', 'Unknown')
                }
            }
            
            self.logger.info("CPET data analysis completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"CPET data analysis failed: {e}")
            return self._create_error_result("Data analysis failed", [str(e)])
    
    def _generate_visualizations(self, data: pd.DataFrame, ml_results: Dict, options: Dict) -> Dict:
        """Generate all requested visualizations"""
        visualizations = {}
        
        try:
            # Core CPET plots (always generate)
            core_plots = [
                ('vo2_time_plot', 'create_vo2_time_plot'),
                ('vo2_vco2_plot', 'create_vo2_vco2_plot'),
                ('ventilatory_equivalents', 'create_ventilatory_equivalents_plot'),
                ('domain_summary', 'create_domain_summary_plot')
            ]
            
            for plot_name, method_name in core_plots:
                try:
                    if plot_name == 'domain_summary':
                        # Domain summary only needs ML results
                        result = getattr(self.visualization_service, method_name)(ml_results)
                    else:
                        # Other plots need data and optionally ML results
                        result = getattr(self.visualization_service, method_name)(data, ml_results)
                    
                    visualizations[plot_name] = result
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate {plot_name}: {e}")
                    visualizations[plot_name] = {'success': False, 'error': str(e)}
            
            # Optional comprehensive view
            if options.get('include_nine_panel', True):
                try:
                    nine_panel = self.visualization_service.create_nine_panel_plot(data, ml_results)
                    visualizations['nine_panel_plot'] = nine_panel
                except Exception as e:
                    self.logger.warning(f"Failed to generate nine-panel plot: {e}")
                    visualizations['nine_panel_plot'] = {'success': False, 'error': str(e)}
            
            # Add plot configuration
            visualizations['plot_config'] = self.visualization_service.get_plot_config()
            
            return visualizations
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")
            return {'error': str(e)}
    
    def _create_analysis_report(self, data: pd.DataFrame, metadata: Dict, ml_results: Dict, validation: Dict) -> Dict:
        """Create comprehensive analysis report"""
        try:
            # Extract key metrics
            vo2_max = float(data['VO2'].max()) if 'VO2' in data.columns else None
            vo2_at_vt1 = ml_results.get('ventilatory_thresholds', {}).get('VT1')
            vo2_at_vt2 = ml_results.get('ventilatory_thresholds', {}).get('VT2')
            
            # Calculate percentages of VO2max
            vt1_percent = (vo2_at_vt1 / vo2_max * 100) if vo2_at_vt1 and vo2_max else None
            vt2_percent = (vo2_at_vt2 / vo2_max * 100) if vo2_at_vt2 and vo2_max else None
            
            # Dominant domain analysis
            dominant_domain = ml_results.get('dominant_domain', 'Unknown')
            domain_confidence = ml_results.get('confidence', 0)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                dominant_domain, domain_confidence, vo2_max, vt1_percent, vt2_percent
            )
            
            report = {
                'summary': {
                    'vo2_max': vo2_max,
                    'vo2_max_units': 'ml/min',
                    'dominant_domain': dominant_domain,
                    'confidence': domain_confidence,
                    'test_duration_minutes': metadata.get('duration_minutes'),
                    'data_quality': metadata.get('data_quality', 'Unknown')
                },
                'thresholds': {
                    'vt1': {
                        'vo2_absolute': vo2_at_vt1,
                        'vo2_percent_max': vt1_percent,
                        'detected': vo2_at_vt1 is not None
                    },
                    'vt2': {
                        'vo2_absolute': vo2_at_vt2,
                        'vo2_percent_max': vt2_percent,
                        'detected': vo2_at_vt2 is not None
                    }
                },
                'domain_analysis': ml_results.get('domain_probabilities', {}),
                'interpretation': interpretation,
                'data_quality_assessment': {
                    'overall_quality': metadata.get('data_quality', 'Unknown'),
                    'total_data_points': len(data),
                    'missing_data_percentage': metadata.get('missing_data_percentage', 0),
                    'validation_warnings': validation.get('warnings', []),
                    'recommendations': validation.get('recommendations', [])
                },
                'technical_details': {
                    'metabolimeter_type': metadata.get('metabolimeter_type', 'Unknown'),
                    'sampling_rate_hz': metadata.get('sampling_rate'),
                    'analysis_timestamp': pd.Timestamp.now().isoformat(),
                    'ml_model_info': self.ml_service.get_model_info()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Analysis report creation failed: {e}")
            return {'error': str(e)}
    
    def _generate_interpretation(self, dominant_domain: str, confidence: float, vo2_max: Optional[float], 
                               vt1_percent: Optional[float], vt2_percent: Optional[float]) -> Dict:
        """Generate human-readable interpretation of results"""
        interpretation = {
            'primary_finding': '',
            'threshold_assessment': '',
            'clinical_significance': '',
            'recommendations': []
        }
        
        try:
            # Primary finding based on dominant domain
            if dominant_domain == 'Moderate':
                interpretation['primary_finding'] = f"Exercise primarily in moderate intensity domain (confidence: {confidence:.1%}). Indicates good aerobic fitness with efficient fat oxidation."
            elif dominant_domain == 'Heavy':
                interpretation['primary_finding'] = f"Exercise primarily in heavy intensity domain (confidence: {confidence:.1%}). Indicates transition zone between aerobic and anaerobic metabolism."
            elif dominant_domain == 'Severe':
                interpretation['primary_finding'] = f"Exercise primarily in severe intensity domain (confidence: {confidence:.1%}). Indicates high anaerobic contribution and metabolic stress."
            
            # Threshold assessment
            if vt1_percent and vt2_percent:
                interpretation['threshold_assessment'] = f"VT1 detected at {vt1_percent:.0f}% of VO2max, VT2 at {vt2_percent:.0f}% of VO2max. Normal threshold progression observed."
            elif vt1_percent:
                interpretation['threshold_assessment'] = f"VT1 detected at {vt1_percent:.0f}% of VO2max. VT2 not clearly identified."
            else:
                interpretation['threshold_assessment'] = "Ventilatory thresholds not clearly identifiable in this test."
            
            # Clinical significance
            if vo2_max:
                if vo2_max > 3000:
                    interpretation['clinical_significance'] = "High aerobic capacity indicated by VO2max values."
                elif vo2_max > 2000:
                    interpretation['clinical_significance'] = "Moderate aerobic capacity indicated by VO2max values."
                else:
                    interpretation['clinical_significance'] = "Lower aerobic capacity - may benefit from structured training."
            
            # Recommendations
            if confidence < 0.7:
                interpretation['recommendations'].append("Low confidence in domain classification - consider longer test duration or different protocol.")
            
            if dominant_domain == 'Moderate' and vt1_percent and vt1_percent > 80:
                interpretation['recommendations'].append("High VT1 suggests excellent aerobic fitness. Consider higher intensity training zones.")
            
            if dominant_domain == 'Severe':
                interpretation['recommendations'].append("High anaerobic contribution suggests need for improved aerobic base training.")
            
            return interpretation
            
        except Exception as e:
            self.logger.warning(f"Interpretation generation failed: {e}")
            return {'error': 'Could not generate interpretation'}
    
    def _create_error_result(self, message: str, errors: List[str]) -> Dict:
        """Create standardized error result"""
        return {
            'success': False,
            'error_message': message,
            'errors': errors,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    def get_service_status(self) -> Dict:
        """Get status of all sub-services"""
        return {
            'data_service': 'ready',
            'ml_service': {
                'status': 'ready' if self.ml_service.interpreter else 'error',
                'model_info': self.ml_service.get_model_info()
            },
            'visualization_service': 'ready',
            'last_check': pd.Timestamp.now().isoformat()
        }