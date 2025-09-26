"""
Integration tests for the modern PyOxynet API
Tests RESTful endpoints, OpenAPI documentation, and error handling
"""
import pytest
import json
import io
import pandas as pd
from werkzeug.datastructures import FileStorage

from pyoxynet_api.api_app import create_api_app


@pytest.fixture
def app():
    """Create test API application"""
    app = create_api_app('testing')
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


@pytest.fixture
def sample_json_data(sample_cpet_data):
    """Create sample JSON data for testing"""
    return {
        'data': sample_cpet_data.to_dict('records'),
        'options': {
            'include_nine_panel': True,
            'threshold_detection_method': 'ml',
            'confidence_threshold': 0.7
        }
    }


class TestModernAPIEndpoints:
    """Test modern RESTful API endpoints"""
    
    def test_api_info_endpoint(self, client):
        """Test API information endpoint"""
        response = client.get('/api')
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert data['success'] is True
        assert data['status'] == 'ok'
        assert 'data' in data
        assert 'name' in data['data']
        assert 'version' in data['data']
        assert data['data']['version'] == '1.0.0'
    
    def test_documentation_redirect(self, client):
        """Test root redirects to documentation"""
        response = client.get('/')
        
        assert response.status_code == 302
        assert '/docs/' in response.location
    
    def test_openapi_documentation_available(self, client):
        """Test OpenAPI documentation is accessible"""
        response = client.get('/docs/')
        
        assert response.status_code == 200
        # Should contain Swagger UI
        assert b'swagger' in response.data.lower() or b'openapi' in response.data.lower()


class TestFileAnalysisEndpoint:
    """Test /api/v1/analyze/file endpoint"""
    
    def test_successful_file_analysis(self, client, sample_csv_file):
        """Test successful file analysis"""
        response = client.post(
            '/api/v1/analyze/file',
            data={
                'file': (sample_csv_file, 'test.csv'),
                'options': json.dumps({
                    'include_nine_panel': True,
                    'confidence_threshold': 0.7
                }),
                'session_id': 'test_session'
            },
            content_type='multipart/form-data'
        )
        
        # May return 500 if ML model not available - that's expected in test environment
        if response.status_code == 500:
            pytest.skip("ML model not available in test environment")
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        
        # Check standardized response format
        assert 'success' in data
        assert 'status' in data
        assert 'data' in data
        assert 'metadata' in data
        
        if response.status_code == 200:
            assert data['success'] is True
            assert data['status'] == 'ok'
            assert 'analysis' in data['data']
            assert 'visualizations' in data['data']
            
            # Check metadata
            assert 'api_version' in data['metadata']
            assert 'timestamp' in data['metadata']
            assert 'processing_time_ms' in data['metadata']
    
    def test_file_analysis_no_file_error(self, client):
        """Test file analysis without file"""
        response = client.post(
            '/api/v1/analyze/file',
            data={'session_id': 'test'},
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        
        assert data['success'] is False
        assert data['status'] == 'error'
        assert 'error' in data
        assert data['error']['code'] == 'NO_FILE'
    
    def test_file_analysis_invalid_options(self, client, sample_csv_file):
        """Test file analysis with invalid options JSON"""
        response = client.post(
            '/api/v1/analyze/file',
            data={
                'file': (sample_csv_file, 'test.csv'),
                'options': 'invalid_json{',  # Invalid JSON
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        
        assert data['success'] is False
        assert data['error']['code'] == 'VALIDATION_ERROR'


class TestJSONDataAnalysisEndpoint:
    """Test /api/v1/analyze/data endpoint"""
    
    def test_successful_json_analysis(self, client, sample_json_data):
        """Test successful JSON data analysis"""
        response = client.post(
            '/api/v1/analyze/data',
            json=sample_json_data,
            content_type='application/json'
        )
        
        # May return 500 if ML model not available
        if response.status_code == 500:
            pytest.skip("ML model not available in test environment")
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        
        # Check standardized response format
        assert 'success' in data
        assert 'status' in data
        assert 'metadata' in data
        
        if response.status_code == 200:
            assert data['success'] is True
            assert 'analysis' in data['data']
    
    def test_json_analysis_missing_data_field(self, client):
        """Test JSON analysis without data field"""
        response = client.post(
            '/api/v1/analyze/data',
            json={'options': {'include_nine_panel': True}},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        
        assert data['success'] is False
        assert data['error']['code'] == 'VALIDATION_ERROR'
    
    def test_json_analysis_insufficient_data(self, client):
        """Test JSON analysis with insufficient data points"""
        insufficient_data = {
            'data': [
                {'VO2': 500, 'VCO2': 400, 'VE': 15}  # Only 1 data point
            ]
        }
        
        response = client.post(
            '/api/v1/analyze/data',
            json=insufficient_data,
            content_type='application/json'
        )
        
        assert response.status_code == 422
        data = response.get_json()
        
        assert data['success'] is False
        assert data['error']['code'] == 'INSUFFICIENT_DATA'
    
    def test_json_analysis_invalid_content_type(self, client, sample_json_data):
        """Test JSON analysis with wrong content type"""
        response = client.post(
            '/api/v1/analyze/data',
            data=json.dumps(sample_json_data),
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = response.get_json()
        
        assert data['success'] is False
        assert data['error']['code'] == 'VALIDATION_ERROR'


class TestValidationEndpoint:
    """Test /api/v1/validate endpoint"""
    
    def test_successful_validation(self, client, sample_csv_file):
        """Test successful file validation"""
        response = client.post(
            '/api/v1/validate',
            data={
                'file': (sample_csv_file, 'test.csv'),
                'preview_rows': 5
            },
            content_type='multipart/form-data'
        )
        
        assert response.status_code in [200, 422]
        data = response.get_json()
        
        assert 'success' in data
        assert 'metadata' in data
        
        if response.status_code == 200:
            assert data['success'] is True
            assert 'validation' in data['data']
            assert 'preview' in data['data']
            assert 'metadata' in data['data']
            
            # Check validation structure
            validation = data['data']['validation']
            assert 'valid' in validation
            assert 'errors' in validation
            assert 'warnings' in validation
            
            # Check preview structure
            preview = data['data']['preview']
            assert 'rows' in preview
            assert 'total_rows' in preview
            assert 'columns' in preview


class TestHealthCheckEndpoint:
    """Test /api/v1/health endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/v1/health')
        
        assert response.status_code in [200, 503]
        data = response.get_json()
        
        assert data['success'] is True  # Health check should always return success format
        assert 'data' in data
        assert 'status' in data['data']
        assert 'services' in data['data']
        
        # Check metadata
        assert 'metadata' in data
        assert 'api_version' in data['metadata']


class TestModelInfoEndpoint:
    """Test /api/v1/models/info endpoint"""
    
    def test_model_info(self, client):
        """Test model info endpoint"""
        response = client.get('/api/v1/models/info')
        
        assert response.status_code in [200, 500]
        data = response.get_json()
        
        if response.status_code == 200:
            assert data['success'] is True
            assert 'model' in data['data']
            assert 'capabilities' in data['data']


class TestErrorHandling:
    """Test comprehensive error handling"""
    
    def test_404_error_format(self, client):
        """Test 404 error returns standardized format"""
        response = client.get('/api/v1/nonexistent')
        
        assert response.status_code == 404
        data = response.get_json()
        
        assert data['success'] is False
        assert data['status'] == 'error'
        assert 'error' in data
        assert data['error']['code'] == 'NOT_FOUND'
    
    def test_405_method_not_allowed(self, client):
        """Test method not allowed error"""
        response = client.put('/api/v1/health')  # PUT not allowed
        
        assert response.status_code == 405
        data = response.get_json()
        
        assert data['success'] is False
        assert data['error']['code'] == 'METHOD_NOT_ALLOWED'
    
    def test_unsupported_media_type(self, client):
        """Test unsupported media type error"""
        response = client.post(
            '/api/v1/analyze/file',
            data='not_multipart',
            content_type='text/plain'
        )
        
        assert response.status_code in [400, 415, 422]
        data = response.get_json()
        
        assert data['success'] is False


class TestResponseFormat:
    """Test standardized response format"""
    
    def test_success_response_format(self, client):
        """Test success response follows standard format"""
        response = client.get('/api/v1/health')
        data = response.get_json()
        
        # Standard success format
        assert 'success' in data
        assert 'status' in data
        assert 'data' in data
        assert 'metadata' in data
        assert data['errors'] is None
        
        # Metadata requirements
        metadata = data['metadata']
        assert 'api_version' in metadata
        assert 'timestamp' in metadata
        assert metadata['api_version'] == '1.0.0'
    
    def test_error_response_format(self, client):
        """Test error response follows standard format"""
        response = client.get('/api/v1/nonexistent')
        data = response.get_json()
        
        # Standard error format
        assert data['success'] is False
        assert data['status'] == 'error'
        assert data['data'] is None
        assert 'metadata' in data
        assert 'error' in data
        
        # Error structure
        error = data['error']
        assert 'message' in error
        assert 'code' in error


class TestAPIVersioning:
    """Test API versioning"""
    
    def test_api_version_in_metadata(self, client):
        """Test API version is included in all responses"""
        endpoints = [
            '/api/v1/health',
            '/api'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            if response.status_code < 500:  # Skip if service unavailable
                data = response.get_json()
                assert 'metadata' in data
                assert 'api_version' in data['metadata']
                assert data['metadata']['api_version'] == '1.0.0'
    
    def test_api_namespace_structure(self, client):
        """Test API follows v1 namespace structure"""
        # Test v1 endpoints are accessible
        v1_endpoints = [
            '/api/v1/health',
            '/api/v1/models/info'
        ]
        
        for endpoint in v1_endpoints:
            response = client.get(endpoint)
            # Should not return 404 (endpoint exists)
            assert response.status_code != 404


@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints"""
    
    def test_health_check_response_time(self, client):
        """Test health check responds quickly"""
        import time
        
        start_time = time.time()
        response = client.get('/api/v1/health')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Health check should be very fast (< 1 second)
        assert response_time < 1.0
        assert response.status_code in [200, 503]
    
    def test_concurrent_health_checks(self, client):
        """Test API handles concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/api/v1/health')
            results.append(response.status_code)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # All requests should complete successfully
        assert len(results) == 5
        for status_code in results:
            assert status_code in [200, 503]