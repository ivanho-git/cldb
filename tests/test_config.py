"""
Tests for configuration management.
"""
import pytest
import os
from cldb.config import (
    Config, DatabaseConfig, VectorStoreConfig, MLConfig, EWCConfig,
    ReplayConfig, SafetyConfig, ConfigLoader, get_config, reset_config
)


def test_default_config():
    config = Config()
    assert config.environment == "development"
    assert config.database.host == "localhost"
    assert config.database.port == 5432


def test_database_config_url():
    db_config = DatabaseConfig(host="db.example.com", port=5433, user="admin", password="secret", name="testdb")
    assert db_config.url == "postgresql://admin:secret@db.example.com:5433/testdb"


def test_ml_config_defaults():
    ml_config = MLConfig()
    assert ml_config.learning_rate == 1e-3
    assert ml_config.batch_size == 32
    assert ml_config.hidden_dims == [128, 64]


def test_config_validation():
    config = Config()
    config.ml.learning_rate = 1.5  # Invalid
    errors = config.validate()
    assert len(errors) > 0
    assert any("learning_rate" in e for e in errors)


def test_config_validation_passes():
    config = Config()
    errors = config.validate()
    assert len(errors) == 0


def test_env_override():
    os.environ["POSTGRES_HOST"] = "env-db-host"
    os.environ["ML_LR"] = "0.005"
    
    loader = ConfigLoader()
    config = loader.load()
    
    assert config.database.host == "env-db-host"
    assert config.ml.learning_rate == 0.005
    
    # Cleanup
    del os.environ["POSTGRES_HOST"]
    del os.environ["ML_LR"]


def test_get_config_caching():
    reset_config()
    
    config1 = get_config()
    config2 = get_config()
    
    assert config1 is config2  # Same instance (cached)


def test_reset_config():
    reset_config()
    config = get_config()
    assert config is not None