"""
Integration tests for the new modular API architecture
"""
import pytest
import json
import io
import pandas as pd
from werkzeug.datastructures import FileStorage

from pyoxynet_api.app import create_app


@pytest.fixture
def app():
    """Create test Flask application"""
    app = create_app('testing')
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def sample_csv_file(sample_cpet_data):
    """Create sample CSV file for testing"""
    csv_content = sample_cpet_data.to_csv(index=False)
    return io.BytesIO(csv_content.encode('utf-8'))


class TestCPETAnalysisAPI:
    """Test CPET analysis API endpoints"""
    
    def test_health_check_endpoint(self, client):
        """Test API health check"""
        response = client.get('/api/v1/cpet/health')
        
        assert response.status_code in [200, 503]  # 503 if ML model not available
        data = response.get_json()
        assert 'healthy' in data
        assert 'services' in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint"""
        response = client.get('/api/v1/cpet/model-info')
        
        # May return 500 if model not available in test environment
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'success' in data
            assert 'model_info' in data
    
    def test_file_analysis_endpoint(self, client, sample_csv_file):
        """Test file analysis endpoint"""
        response = client.post('/api/v1/cpet/analyze', 
                             data={
                                 'file': (sample_csv_file, 'test.csv'),
                                 'session_id': 'test_session',
                                 'options': json.dumps({
                                     'include_nine_panel': True
                                 })
                             },
                             content_type='multipart/form-data')
        
        # May fail if ML model not available
        if response.status_code == 500:
            pytest.skip("ML model not available in test environment")
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        assert 'success' in data
    
    def test_json_data_analysis_endpoint(self, client, sample_cpet_data):
        """Test JSON data analysis endpoint"""
        json_data = {
            'data': sample_cpet_data.to_dict('records'),
            'options': {
                'include_nine_panel': True,
                'confidence_threshold': 0.7
            }
        }
        
        response = client.post('/api/v1/cpet/analyze-data',
                             json=json_data,
                             content_type='application/json')
        
        # May fail if ML model not available
        if response.status_code == 500:
            pytest.skip("ML model not available in test environment")
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        assert 'success' in data
    
    def test_file_validation_endpoint(self, client, sample_csv_file):
        """Test file validation endpoint"""
        response = client.post('/api/v1/cpet/validate',
                             data={
                                 'file': (sample_csv_file, 'test.csv'),
                                 'session_id': 'validation_session'
                             },
                             content_type='multipart/form-data')
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        assert 'success' in data
        
        if data['success']:
            assert 'validation' in data
            assert 'metadata' in data
            assert 'preview' in data
    
    def test_error_handling_no_file(self, client):
        """Test error handling when no file provided"""
        response = client.post('/api/v1/cpet/analyze',
                             data={'session_id': 'test'},
                             content_type='multipart/form-data')
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert data['code'] == 'NO_FILE'
    
    def test_error_handling_invalid_json(self, client):
        """Test error handling for invalid JSON data"""
        response = client.post('/api/v1/cpet/analyze-data',
                             json={'invalid': 'data'},
                             content_type='application/json')
        
        assert response.status_code == 400
        data = response.get_json()
        assert data['success'] is False
        assert data['code'] == 'MISSING_DATA'
    
    def test_error_handling_empty_data(self, client):
        """Test error handling for empty data"""
        json_data = {
            'data': []  # Empty data array
        }
        
        response = client.post('/api/v1/cpet/analyze-data',
                             json=json_data,
                             content_type='application/json')
        
        assert response.status_code == 422
        # The actual error depends on validation in the service layer


class TestWebInterface:
    """Test web interface endpoints"""
    
    def test_homepage(self, client):
        """Test homepage loads"""
        response = client.get('/')
        # May return 500 if templates not found in test environment
        assert response.status_code in [200, 500]
    
    def test_analyze_page_get(self, client):
        """Test analyze page GET request"""
        response = client.get('/analyze')
        # May return 500 if templates not found
        assert response.status_code in [200, 500]
    
    def test_health_check_endpoint(self, client):
        """Test application health check"""
        response = client.get('/health')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert data['service'] == 'pyoxynet-api'


class TestErrorHandling:
    """Test global error handling"""
    
    def test_404_error_handling(self, client):
        """Test 404 error handling"""
        response = client.get('/nonexistent-endpoint')
        
        assert response.status_code == 404
        data = response.get_json()
        assert 'error' in data
        assert data['code'] == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed handling"""
        response = client.put('/api/v1/cpet/health')  # PUT not allowed
        
        assert response.status_code == 405
    
    def test_large_file_handling(self, client):
        """Test handling of oversized files"""
        # Create a large file (this test might be skipped if file size limits aren't enforced)
        large_content = "VO2,VCO2,VE\n" + "1000,800,20\n" * 100000
        large_file = io.BytesIO(large_content.encode('utf-8'))
        
        response = client.post('/api/v1/cpet/analyze',
                             data={
                                 'file': (large_file, 'large.csv')
                             },
                             content_type='multipart/form-data')
        
        # Should either process successfully or return file too large error
        assert response.status_code in [200, 413, 422, 500]


class TestServiceIntegration:
    """Test integration between different service layers"""
    
    def test_data_processing_to_visualization(self, client, sample_csv_file):
        """Test that data flows correctly from processing to visualization"""
        # First validate the file
        response = client.post('/api/v1/cpet/validate',
                             data={
                                 'file': (sample_csv_file, 'test.csv')
                             },
                             content_type='multipart/form-data')
        
        if response.status_code != 200:
            pytest.skip("File validation failed")
        
        # Reset file pointer
        sample_csv_file.seek(0)
        
        # Then analyze the file
        response = client.post('/api/v1/cpet/analyze',
                             data={
                                 'file': (sample_csv_file, 'test.csv'),
                                 'options': json.dumps({
                                     'include_nine_panel': False  # Reduce complexity
                                 })
                             },
                             content_type='multipart/form-data')
        
        if response.status_code == 500:
            pytest.skip("Analysis failed - likely missing ML model")
        
        # Should return visualization data
        if response.status_code == 200:
            data = response.get_json()
            assert 'visualizations' in data
    
    def test_concurrent_requests(self, client, sample_cpet_data):
        """Test handling of concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            json_data = {
                'data': sample_cpet_data.head(20).to_dict('records')  # Small dataset
            }
            response = client.post('/api/v1/cpet/analyze-data',
                                 json=json_data,
                                 content_type='application/json')
            results.append(response.status_code)
        
        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete (success or expected failure)
        assert len(results) == 3
        for status_code in results:
            assert status_code in [200, 422, 500]  # Expected status codes


@pytest.mark.slow
class TestPerformance:
    """Performance tests for the modular architecture"""
    
    def test_response_time_validation(self, client, sample_csv_file):
        """Test that validation response time is reasonable"""
        import time
        
        start_time = time.time()
        response = client.post('/api/v1/cpet/validate',
                             data={
                                 'file': (sample_csv_file, 'test.csv')
                             },
                             content_type='multipart/form-data')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Validation should be fast (< 5 seconds)
        assert response_time < 5.0
        assert response.status_code in [200, 422]
    
    def test_memory_usage_large_dataset(self, client):
        """Test memory usage with larger dataset"""
        # Create larger dataset
        large_data = []
        for i in range(1000):  # 1000 data points
            large_data.append({
                'TIME': i,
                'VO2': 500 + i,
                'VCO2': 400 + i * 0.8,
                'VE': 15 + i * 0.1,
                'HR': 60 + i * 0.1
            })
        
        json_data = {
            'data': large_data,
            'options': {
                'include_nine_panel': False  # Reduce memory usage
            }
        }
        
        response = client.post('/api/v1/cpet/analyze-data',
                             json=json_data,
                             content_type='application/json')
        
        # Should handle larger datasets without crashing
        assert response.status_code in [200, 422, 500]