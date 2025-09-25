"""
PyOxynet Test Configuration
Test fixtures and configuration for scientific API testing
"""
import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock
import sys

# Add app directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import TestingConfig
from logging_config import setup_logging
from security import create_security_manager


@pytest.fixture(scope="session")
def app_config():
    """Test configuration fixture"""
    return TestingConfig()


@pytest.fixture(scope="session")
def test_logger(app_config):
    """Test logger fixture"""
    return setup_logging(app_config)


@pytest.fixture(scope="session")
def security_manager(app_config, test_logger):
    """Security manager fixture for testing"""
    return create_security_manager(app_config, test_logger)


@pytest.fixture
def temp_upload_dir():
    """Temporary upload directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_cpet_data():
    """Sample CPET data for testing scientific functions"""
    data = {
        'Time': range(0, 600, 10),  # 10-minute test, 10-second intervals
        'VO2': [200 + i * 2 + (i % 10) * 5 for i in range(60)],  # Realistic VO2 progression
        'VCO2': [150 + i * 1.8 + (i % 10) * 4 for i in range(60)],  # Realistic VCO2 progression
        'VE': [10 + i * 0.5 + (i % 10) * 2 for i in range(60)],  # Realistic VE progression
        'HR': [60 + i * 2 + (i % 5) * 3 for i in range(60)],  # Heart rate progression
    }
    return pd.DataFrame(data)


@pytest.fixture
def invalid_cpet_data():
    """Invalid CPET data for testing validation"""
    data = {
        'Time': range(0, 100, 10),
        'VO2': [-100, 0, 50, None, 'invalid'],  # Invalid values
        'VCO2': [100, -50, None, 200, 150],  # Invalid values
    }
    return pd.DataFrame(data)


@pytest.fixture
def minimal_cpet_data():
    """Minimal valid CPET data"""
    data = {
        'VO2': [200, 220, 240, 260, 280],
        'VCO2': [150, 165, 180, 195, 210],
        'VE': [10, 12, 14, 16, 18],
    }
    return pd.DataFrame(data)


@pytest.fixture
def cpet_csv_file(sample_cpet_data, tmp_path):
    """Create a temporary CSV file with CPET data"""
    csv_file = tmp_path / "test_cpet.csv"
    sample_cpet_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def invalid_csv_file(invalid_cpet_data, tmp_path):
    """Create a temporary CSV file with invalid CPET data"""
    csv_file = tmp_path / "invalid_cpet.csv"
    invalid_cpet_data.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def mock_tflite_interpreter():
    """Mock TensorFlow Lite interpreter for testing"""
    mock_interpreter = Mock()
    mock_interpreter.get_input_details.return_value = [{'index': 0, 'shape': [1, 40, 7]}]
    mock_interpreter.get_output_details.return_value = [{'index': 0}]
    mock_interpreter.invoke.return_value = None
    mock_interpreter.get_tensor.return_value = [[0.3, 0.5, 0.2]]  # Mock domain probabilities
    return mock_interpreter


@pytest.fixture
def session_id():
    """Generate a test session ID"""
    import uuid
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test"""
    yield
    # Cleanup logic could be added here if needed


class TestDataGenerator:
    """Utility class for generating test data"""
    
    @staticmethod
    def generate_cpet_sequence(duration_minutes: int = 10, interval_seconds: int = 10):
        """Generate realistic CPET test sequence"""
        points = duration_minutes * 60 // interval_seconds
        
        # Simulate incremental exercise test
        data = {
            'Time': list(range(0, duration_minutes * 60, interval_seconds)),
            'VO2': [200 + i * 3 + (i % 15) * 2 for i in range(points)],
            'VCO2': [150 + i * 2.5 + (i % 12) * 1.5 for i in range(points)],
            'VE': [10 + i * 0.8 + (i % 8) * 1.2 for i in range(points)],
            'HR': [60 + i * 1.5 + (i % 20) * 2 for i in range(points)],
            'RF': [15 + i * 0.3 + (i % 10) * 0.5 for i in range(points)],
        }
        
        return pd.DataFrame(data)


# Make TestDataGenerator available at module level
@pytest.fixture
def data_generator():
    """Test data generator fixture"""
    return TestDataGenerator()