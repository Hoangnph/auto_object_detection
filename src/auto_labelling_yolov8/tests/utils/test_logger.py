"""Tests for the logger module."""
import pytest
from pathlib import Path
from utils.logger import setup_logger, get_logger

def test_setup_logger_without_log_dir():
    """Test logger setup without log directory."""
    setup_logger()
    logger = get_logger("test")
    logger.info("Test message")
    # No assertion needed as we're just checking it doesn't raise an exception

def test_setup_logger_with_log_dir(tmp_path):
    """Test logger setup with log directory."""
    log_dir = tmp_path / "logs"
    setup_logger(log_dir)
    
    # Check log directory was created
    assert log_dir.exists()
    assert log_dir.is_dir()
    
    # Check log files were created
    logger = get_logger("test")
    logger.info("Test info message")
    logger.error("Test error message")
    
    assert (log_dir / "all.log").exists()
    assert (log_dir / "error.log").exists()
    
    # Check content of log files
    with open(log_dir / "all.log") as f:
        content = f.read()
        assert "Test info message" in content
        assert "Test error message" in content
        
    with open(log_dir / "error.log") as f:
        content = f.read()
        assert "Test info message" not in content
        assert "Test error message" in content

def test_get_logger_with_name():
    """Test getting logger with a specific name."""
    logger = get_logger("test_name")
    assert logger._core.extra["name"] == "test_name"

def test_get_logger_without_name():
    """Test getting logger without a name."""
    logger = get_logger()
    assert logger._core.extra["name"] is None 