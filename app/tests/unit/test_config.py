"""
Unit tests for PyOxynet configuration module
Testing environment-specific configurations and security settings
"""
import pytest
import os
from unittest.mock import patch

from config import Config, DevelopmentConfig, ProductionConfig, TestingConfig, get_config


class TestBaseConfig:
    """Test base configuration class"""
    
    def test_default_config_values(self):
        """Test default configuration values"""
        config = Config()
        
        assert config.DEBUG is False
        assert config.TESTING is False
        assert config.MAX_BATCH_SIZE == 100
        assert config.MAX_BATCH_MEMORY_GB == 2.0
        assert config.NO_PERSISTENT_STORAGE is True
        assert config.API_VERSION == "1.0.0"
        assert config.ALLOWED_EXTENSIONS == {'csv', 'txt', 'json'}
    
    def test_secret_key_generation(self):
        """Test secret key generation when not provided"""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.SECRET_KEY is not None
            assert len(config.SECRET_KEY) > 0
    
    def test_secret_key_from_environment(self):
        """Test secret key loading from environment"""
        test_key = "test-secret-key-from-env"
        with patch.dict(os.environ, {'SECRET_KEY': test_key}):
            config = Config()
            assert config.SECRET_KEY == test_key
    
    def test_log_level_from_environment(self):
        """Test log level configuration from environment"""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}):
            config = Config()
            assert config.LOG_LEVEL == 'DEBUG'


class TestDevelopmentConfig:
    """Test development configuration"""
    
    def test_development_config_values(self):
        """Test development-specific configuration values"""
        config = DevelopmentConfig()
        
        assert config.DEBUG is True
        assert config.LOG_LEVEL == 'DEBUG'
        assert config.AUTO_CLEANUP_MINUTES == 10
    
    def test_development_inherits_base_config(self):
        """Test that development config inherits from base"""
        config = DevelopmentConfig()
        
        assert config.MAX_BATCH_SIZE == 100  # From base config
        assert config.NO_PERSISTENT_STORAGE is True  # From base config
        assert config.API_VERSION == "1.0.0"  # From base config


class TestProductionConfig:
    """Test production configuration"""
    
    def test_production_requires_secret_key(self):
        """Test that production config requires SECRET_KEY environment variable"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="SECRET_KEY environment variable must be set"):
                ProductionConfig()
    
    def test_production_config_with_secret_key(self):
        """Test production config when SECRET_KEY is provided"""
        test_key = "production-secret-key"
        with patch.dict(os.environ, {'SECRET_KEY': test_key}):
            config = ProductionConfig()
            
            assert config.SECRET_KEY == test_key
            assert config.DEBUG is False
            assert config.TESTING is False
    
    def test_production_security_settings(self):
        """Test production security settings"""
        with patch.dict(os.environ, {'SECRET_KEY': 'prod-key'}):
            config = ProductionConfig()
            
            assert config.NO_PERSISTENT_STORAGE is True
            assert config.AUTO_CLEANUP_MINUTES == 60


class TestTestingConfig:
    """Test testing configuration"""
    
    def test_testing_config_values(self):
        """Test testing-specific configuration values"""
        config = TestingConfig()
        
        assert config.TESTING is True
        assert config.SECRET_KEY == 'testing-secret-key-not-for-production'
        assert config.MAX_BATCH_SIZE == 10  # Smaller for testing
        assert config.AUTO_CLEANUP_MINUTES == 1  # Fast cleanup for tests
    
    def test_testing_config_safety(self):
        """Test that testing config is safe for automated testing"""
        config = TestingConfig()
        
        # Should not require environment variables
        assert config.SECRET_KEY is not None
        assert len(config.SECRET_KEY) > 0
        
        # Should have reasonable test limits
        assert config.MAX_BATCH_SIZE <= 10
        assert config.AUTO_CLEANUP_MINUTES <= 5


class TestConfigFactory:
    """Test configuration factory function"""
    
    def test_get_config_default(self):
        """Test getting default configuration"""
        with patch.dict(os.environ, {}, clear=True):
            config_class = get_config()
            assert config_class == DevelopmentConfig
    
    def test_get_config_development(self):
        """Test getting development configuration"""
        with patch.dict(os.environ, {'FLASK_ENV': 'development'}):
            config_class = get_config()
            assert config_class == DevelopmentConfig
    
    def test_get_config_production(self):
        """Test getting production configuration"""
        config_class = get_config('production')
        assert config_class == ProductionConfig
    
    def test_get_config_testing(self):
        """Test getting testing configuration"""
        config_class = get_config('testing')
        assert config_class == TestingConfig
    
    def test_get_config_invalid_environment(self):
        """Test getting config with invalid environment name"""
        config_class = get_config('invalid_env')
        assert config_class == DevelopmentConfig  # Should fall back to default
    
    def test_get_config_from_environment_variable(self):
        """Test getting config from FLASK_ENV environment variable"""
        with patch.dict(os.environ, {'FLASK_ENV': 'testing'}):
            config_class = get_config()
            assert config_class == TestingConfig


class TestConfigSecurity:
    """Test configuration security aspects"""
    
    def test_config_secret_key_uniqueness(self):
        """Test that generated secret keys are unique"""
        with patch.dict(os.environ, {}, clear=True):
            config1 = Config()
            config2 = Config()
            
            # Generated keys should be different
            assert config1.SECRET_KEY != config2.SECRET_KEY
    
    def test_config_no_hardcoded_secrets(self):
        """Test that no secrets are hardcoded in configuration"""
        config = DevelopmentConfig()
        
        # Development should still generate random keys
        assert config.SECRET_KEY != "super secret key"  # Old hardcoded key
        assert len(config.SECRET_KEY) > 20  # Should be substantial
    
    def test_medical_data_privacy_settings(self):
        """Test medical data privacy configuration settings"""
        configs = [DevelopmentConfig(), TestingConfig()]
        
        # Production needs SECRET_KEY, so test separately
        with patch.dict(os.environ, {'SECRET_KEY': 'test-key'}):
            configs.append(ProductionConfig())
        
        for config in configs:
            assert config.NO_PERSISTENT_STORAGE is True
            assert config.AUTO_CLEANUP_MINUTES > 0
            assert config.MAX_CONTENT_LENGTH > 0  # File size limits
    
    def test_cpet_processing_limits(self):
        """Test CPET processing limits for safety"""
        config = Config()
        
        assert config.MAX_BATCH_SIZE <= 1000  # Reasonable batch limit
        assert config.MAX_BATCH_MEMORY_GB <= 5.0  # Memory safety
        assert config.PROCESSING_TIMEOUT_MINUTES > 0  # Prevent infinite processing