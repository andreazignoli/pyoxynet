"""
Unit tests for the new modular service architecture
"""
import pytest
import pandas as pd
import numpy as np
import io
from werkzeug.datastructures import FileStorage

from pyoxynet_api.core.services.data_service import CPETDataService
from pyoxynet_api.core.services.ml_service import CPETMLService
from pyoxynet_api.core.services.visualization_service import CPETVisualizationService
from pyoxynet_api.core.services.cpet_service import CPETAnalysisService


class TestCPETDataService:
    """Test the data processing service"""
    
    def test_process_csv_file(self, test_logger, sample_cpet_data):
        """Test CSV file processing"""
        service = CPETDataService(logger=test_logger)
        
        # Create CSV file
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='test.csv',
            content_type='text/csv'
        )
        
        result = service.process_file(file_storage, 'test_session')
        
        assert result['success'] is True
        assert len(result['errors']) == 0
        assert result['data'] is not None
        assert len(result['data']) > 0
    
    def test_column_standardization(self, test_logger):
        """Test column name standardization"""
        service = CPETDataService(logger=test_logger)
        
        # Create data with non-standard column names
        data = pd.DataFrame({
            't': [1, 2, 3, 4, 5],
            'O2': [500, 600, 700, 800, 900],
            'CO2': [400, 480, 560, 640, 720],
            'Ventilation': [15, 18, 21, 24, 27]
        })
        
        standardized = service._standardize_columns(data, 'generic')
        
        assert 'TIME' in standardized.columns
        assert 'VO2' in standardized.columns
        assert 'VCO2' in standardized.columns
        assert 'VE' in standardized.columns
    
    def test_data_validation(self, test_logger, sample_cpet_data):
        """Test CPET data validation"""
        service = CPETDataService(logger=test_logger)
        
        # Test valid data
        result = service.validate_cpet_data(sample_cpet_data)
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # Test invalid data (missing required columns)
        invalid_data = sample_cpet_data.drop(columns=['VO2'])
        result = service.validate_cpet_data(invalid_data)
        assert result['valid'] is False
        assert len(result['errors']) > 0


class TestCPETMLService:
    """Test the ML inference service"""
    
    def test_model_loading(self, test_logger):
        """Test TensorFlow Lite model loading"""
        service = CPETMLService(logger=test_logger)
        
        if service.interpreter is not None:
            # Model loaded successfully
            assert service.input_details is not None
            assert service.output_details is not None
        else:
            # Model not available in test environment - this is expected
            assert service.model_path is None
    
    def test_data_preprocessing(self, test_logger, sample_cpet_data):
        """Test data preprocessing for ML inference"""
        try:
            service = CPETMLService(logger=test_logger)
            processed = service.preprocess_data(sample_cpet_data)
            
            assert processed is not None
            assert isinstance(processed, np.ndarray)
            assert len(processed.shape) == 3  # (samples, timesteps, features)
            
        except FileNotFoundError:
            pytest.skip("TensorFlow Lite model not available in test environment")
    
    def test_exercise_domain_prediction(self, test_logger, sample_cpet_data):
        """Test exercise domain prediction"""
        try:
            service = CPETMLService(logger=test_logger)
            results = service.predict_exercise_domains(sample_cpet_data)
            
            assert 'domain_probabilities' in results
            assert 'dominant_domain' in results
            assert 'confidence' in results
            
            # Check domain probabilities sum to approximately 1
            probs = results['domain_probabilities']
            total_prob = sum(probs.values())
            assert 0.9 <= total_prob <= 1.1
            
        except FileNotFoundError:
            pytest.skip("TensorFlow Lite model not available in test environment")


class TestCPETVisualizationService:
    """Test the visualization service"""
    
    def test_vo2_time_plot_creation(self, test_logger, sample_cpet_data):
        """Test VO2 vs time plot creation"""
        service = CPETVisualizationService(logger=test_logger)
        
        result = service.create_vo2_time_plot(sample_cpet_data)
        
        assert result['success'] is True
        assert result['figure'] is not None
        assert result['error'] is None
    
    def test_vo2_vco2_plot_creation(self, test_logger, sample_cpet_data):
        """Test VO2 vs VCO2 plot creation"""
        service = CPETVisualizationService(logger=test_logger)
        
        result = service.create_vo2_vco2_plot(sample_cpet_data)
        
        assert result['success'] is True
        assert result['figure'] is not None
        assert result['error'] is None
    
    def test_domain_summary_plot(self, test_logger):
        """Test domain summary plot creation"""
        service = CPETVisualizationService(logger=test_logger)
        
        # Mock ML results
        ml_results = {
            'domain_probabilities': {
                'Moderate': 0.6,
                'Heavy': 0.3,
                'Severe': 0.1
            },
            'confidence': 0.85
        }
        
        result = service.create_domain_summary_plot(ml_results)
        
        assert result['success'] is True
        assert result['figure'] is not None
        assert result['error'] is None
    
    def test_nine_panel_plot_creation(self, test_logger, sample_cpet_data):
        """Test comprehensive 9-panel plot creation"""
        service = CPETVisualizationService(logger=test_logger)
        
        result = service.create_nine_panel_plot(sample_cpet_data)
        
        assert result['success'] is True
        assert result['figure'] is not None
        assert result['error'] is None


class TestCPETAnalysisService:
    """Test the high-level analysis orchestration service"""
    
    def test_service_initialization(self, test_logger):
        """Test analysis service initialization"""
        service = CPETAnalysisService(logger=test_logger)
        
        assert service.data_service is not None
        assert service.visualization_service is not None
        # ML service might not be available in test environment
    
    def test_data_analysis_workflow(self, test_logger, sample_cpet_data):
        """Test complete data analysis workflow"""
        service = CPETAnalysisService(logger=test_logger)
        
        try:
            result = service.analyze_data(sample_cpet_data)
            
            assert result['success'] is True
            assert 'visualizations' in result
            assert 'analysis_report' in result
            assert 'validation' in result
            
        except Exception as e:
            if "TensorFlow Lite model not found" in str(e):
                pytest.skip("TensorFlow Lite model not available in test environment")
            else:
                raise
    
    def test_file_analysis_workflow(self, test_logger, sample_cpet_data):
        """Test complete file analysis workflow"""
        service = CPETAnalysisService(logger=test_logger)
        
        # Create file storage
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='test.csv',
            content_type='text/csv'
        )
        
        try:
            result = service.analyze_file(file_storage, 'test_session')
            
            assert result['success'] is True
            assert 'data' in result
            assert 'visualizations' in result
            assert 'analysis_report' in result
            
        except Exception as e:
            if "TensorFlow Lite model not found" in str(e):
                pytest.skip("TensorFlow Lite model not available in test environment")
            else:
                raise
    
    def test_service_status_check(self, test_logger):
        """Test service status reporting"""
        service = CPETAnalysisService(logger=test_logger)
        
        status = service.get_service_status()
        
        assert 'data_service' in status
        assert 'ml_service' in status
        assert 'visualization_service' in status
        assert 'last_check' in status


class TestServiceIntegration:
    """Integration tests for service interactions"""
    
    def test_data_flow_between_services(self, test_logger, sample_cpet_data):
        """Test data flow from data service to ML service"""
        data_service = CPETDataService(logger=test_logger)
        
        # Create file and process it
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='integration_test.csv',
            content_type='text/csv'
        )
        
        # Process with data service
        data_result = data_service.process_file(file_storage, 'integration_session')
        
        assert data_result['success'] is True
        
        # Validate data can be used by ML service
        validation_result = data_service.validate_cpet_data(data_result['data'])
        assert validation_result['valid'] is True
        
        # Test visualization service can handle the data
        viz_service = CPETVisualizationService(logger=test_logger)
        plot_result = viz_service.create_vo2_time_plot(data_result['data'])
        assert plot_result['success'] is True
    
    def test_error_propagation(self, test_logger):
        """Test error handling across services"""
        service = CPETAnalysisService(logger=test_logger)
        
        # Create invalid file
        invalid_file = io.BytesIO(b"not a csv file")
        file_storage = FileStorage(
            stream=invalid_file,
            filename='invalid.csv',
            content_type='text/csv'
        )
        
        result = service.analyze_file(file_storage, 'error_test_session')
        
        assert result['success'] is False
        assert 'errors' in result or 'error_message' in result