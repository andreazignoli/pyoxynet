"""
Unit tests for PyOxynet security module
Testing file validation and medical data privacy compliance
"""
import pytest
import os
import tempfile
import pandas as pd
from unittest.mock import Mock, patch
from werkzeug.datastructures import FileStorage
import io

from security import CPETFileValidator, MedicalDataManager


class TestCPETFileValidator:
    """Test CPET file validation functionality"""
    
    def test_validate_valid_cpet_file(self, test_logger, sample_cpet_data, tmp_path):
        """Test validation of valid CPET file"""
        validator = CPETFileValidator(test_logger)
        
        # Create CSV content
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        # Create FileStorage object
        file_storage = FileStorage(
            stream=csv_file,
            filename='test_cpet.csv',
            content_type='text/csv'
        )
        
        result = validator.validate_file(file_storage)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
        assert 'VO2' in result['cpet_info']['available_columns']
        assert 'VCO2' in result['cpet_info']['available_columns']
        assert 'VE' in result['cpet_info']['available_columns']
        assert result['cpet_info']['row_count'] == len(sample_cpet_data)
    
    def test_validate_file_missing_required_columns(self, test_logger, tmp_path):
        """Test validation failure when required CPET columns are missing"""
        validator = CPETFileValidator(test_logger)
        
        # Create data missing required columns
        incomplete_data = pd.DataFrame({
            'Time': [1, 2, 3, 4, 5],
            'HR': [60, 65, 70, 75, 80]  # Missing VO2, VCO2, VE
        })
        
        csv_content = incomplete_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='incomplete.csv',
            content_type='text/csv'
        )
        
        result = validator.validate_file(file_storage)
        
        assert result['valid'] is False
        assert any('Missing required CPET columns' in error for error in result['errors'])
    
    def test_validate_file_security_checks(self, test_logger):
        """Test file security validation"""
        validator = CPETFileValidator(test_logger)
        
        # Test empty file
        empty_file = FileStorage(
            stream=io.BytesIO(b''),
            filename='empty.csv',
            content_type='text/csv'
        )
        result = validator.validate_file(empty_file)
        assert result['valid'] is False
        assert any('Empty file' in error for error in result['errors'])
        
        # Test file without filename
        no_filename = FileStorage(stream=io.BytesIO(b'test'), filename='')
        result = validator.validate_file(no_filename)
        assert result['valid'] is False
    
    def test_validate_file_size_limit(self, test_logger):
        """Test file size validation"""
        validator = CPETFileValidator(test_logger)
        
        # Create oversized file content
        large_content = 'VO2,VCO2,VE\n' + '100,80,10\n' * 200000  # Large CSV
        large_file = io.BytesIO(large_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=large_file,
            filename='large.csv',
            content_type='text/csv'
        )
        
        result = validator.validate_file(file_storage)
        assert result['valid'] is False
        assert any('File too large' in error for error in result['errors'])
    
    def test_validate_invalid_file_extension(self, test_logger):
        """Test rejection of invalid file extensions"""
        validator = CPETFileValidator(test_logger)
        
        exe_file = FileStorage(
            stream=io.BytesIO(b'VO2,VCO2,VE\n100,80,10\n'),
            filename='malicious.exe',
            content_type='application/octet-stream'
        )
        
        result = validator.validate_file(exe_file)
        assert result['valid'] is False
        assert any('File type not allowed' in error for error in result['errors'])
    
    def test_data_quality_validation(self, test_logger, invalid_cpet_data):
        """Test CPET data quality validation"""
        validator = CPETFileValidator(test_logger)
        
        csv_content = invalid_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='invalid_data.csv',
            content_type='text/csv'
        )
        
        result = validator.validate_file(file_storage)
        
        # Should fail due to data quality issues
        assert result['valid'] is False or len(result['warnings']) > 0


class TestMedicalDataManager:
    """Test medical data privacy compliance manager"""
    
    def test_file_registration_and_cleanup(self, test_logger, temp_upload_dir, session_id):
        """Test file registration and automatic cleanup"""
        manager = MedicalDataManager(temp_upload_dir, 1, test_logger)  # 1 minute cleanup
        
        # Create test file
        test_file_path = os.path.join(temp_upload_dir, f"{session_id}_test.csv")
        with open(test_file_path, 'w') as f:
            f.write('VO2,VCO2,VE\n100,80,10\n')
        
        # Register file
        manager.register_file(test_file_path, session_id, {'test': 'metadata'})
        
        assert test_file_path in manager.file_registry
        assert manager.file_registry[test_file_path]['session_id'] == session_id
        
        # Test manual session cleanup
        cleaned_count = manager.cleanup_session_files(session_id)
        
        assert cleaned_count == 1
        assert not os.path.exists(test_file_path)
        assert test_file_path not in manager.file_registry
    
    def test_save_temp_file(self, test_logger, temp_upload_dir, session_id, sample_cpet_data):
        """Test secure temporary file saving"""
        manager = MedicalDataManager(temp_upload_dir, 60, test_logger)
        
        # Create FileStorage object
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='test_cpet.csv',
            content_type='text/csv'
        )
        
        file_path, filename = manager.save_temp_file(file_storage, session_id)
        
        assert os.path.exists(file_path)
        assert filename == 'test_cpet.csv'
        assert file_path in manager.file_registry
        
        # Cleanup
        manager.cleanup_session_files(session_id)
    
    def test_cleanup_service_start_stop(self, test_logger, temp_upload_dir):
        """Test cleanup service lifecycle"""
        manager = MedicalDataManager(temp_upload_dir, 60, test_logger)
        
        assert manager.running is True
        assert manager.cleanup_thread is not None
        assert manager.cleanup_thread.is_alive()
        
        manager.stop_cleanup_service()
        assert manager.running is False
    
    def test_expired_files_cleanup(self, test_logger, temp_upload_dir, session_id):
        """Test cleanup of expired files"""
        manager = MedicalDataManager(temp_upload_dir, 1, test_logger)  # 1 minute cleanup
        
        # Create test file
        test_file_path = os.path.join(temp_upload_dir, f"{session_id}_test.csv")
        with open(test_file_path, 'w') as f:
            f.write('VO2,VCO2,VE\n100,80,10\n')
        
        # Register file with past cleanup time
        from datetime import datetime, timedelta
        manager.register_file(test_file_path, session_id, {'test': 'metadata'})
        manager.file_registry[test_file_path]['cleanup_time'] = datetime.utcnow() - timedelta(minutes=1)
        
        # Run cleanup
        cleaned_count = manager.cleanup_expired_files()
        
        assert cleaned_count == 1
        assert not os.path.exists(test_file_path)
        assert test_file_path not in manager.file_registry


class TestSecurityIntegration:
    """Integration tests for security components"""
    
    def test_security_manager_creation(self, app_config, test_logger):
        """Test security manager factory function"""
        from security import create_security_manager
        
        security = create_security_manager(app_config, test_logger)
        
        assert 'file_validator' in security
        assert 'data_manager' in security
        assert isinstance(security['file_validator'], CPETFileValidator)
        assert isinstance(security['data_manager'], MedicalDataManager)
    
    @pytest.mark.integration
    def test_end_to_end_file_processing(self, security_manager, sample_cpet_data, session_id):
        """Test complete file processing workflow"""
        validator = security_manager['file_validator']
        data_manager = security_manager['data_manager']
        
        # Create test file
        csv_content = sample_cpet_data.to_csv(index=False)
        csv_file = io.BytesIO(csv_content.encode('utf-8'))
        
        file_storage = FileStorage(
            stream=csv_file,
            filename='integration_test.csv',
            content_type='text/csv'
        )
        
        # Validate file
        validation_result = validator.validate_file(file_storage)
        assert validation_result['valid'] is True
        
        # Save file temporarily
        file_storage.seek(0)  # Reset stream
        file_path, filename = data_manager.save_temp_file(file_storage, session_id)
        
        assert os.path.exists(file_path)
        
        # Cleanup
        cleaned_count = data_manager.cleanup_session_files(session_id)
        assert cleaned_count == 1
        assert not os.path.exists(file_path)