# Testing Guide

This document provides an overview of the testing infrastructure for the project.

## Overview

The project includes comprehensive unit tests covering all main functionalities:

- **Model Architecture**: CNN model initialization, forward pass, output validation
- **Data Loading**: Dataset creation, data splitting, transforms, loaders
- **Tracking**: MLflow and W&B tracker functionality
- **Integration**: End-to-end workflow tests

## Test Files

### Unit Tests

1. **`tests/test_models.py`** (15 tests)
   - Model initialization with default and custom parameters
   - Forward pass output shape validation
   - Output probability range and sum validation
   - Parameter counting
   - Gradient flow testing
   - Training/evaluation mode testing
   - Device movement (CPU/CUDA)
   - Layer existence verification

2. **`tests/test_data_loader.py`** (12 tests)
   - Dataset class initialization
   - Dataset length and indexing
   - Transform application
   - Invalid image handling
   - Data loading from directory
   - Data splitting validation
   - Batch size and image size configuration
   - Reproducibility with random seeds
   - Nested directory structure support

3. **`tests/test_tracking.py`** (14 tests)
   - MLflowTracker initialization and methods
   - WandBTracker initialization and methods
   - Training functions with MLflow and W&B
   - History structure validation
   - Mock-based testing to avoid external dependencies

### Integration Tests

4. **`tests/test_integration.py`** (4 tests)
   - Complete model training workflow
   - Model inference workflow
   - Data loading to training pipeline
   - Data splitting consistency

### Test Infrastructure

5. **`tests/conftest.py`**
   - Shared pytest fixtures
   - Model fixtures
   - Sample data fixtures
   - Temporary dataset directories

6. **`pytest.ini`**
   - Pytest configuration
   - Test discovery patterns
   - Coverage options
   - Test markers

## Running Tests

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Using the Test Runner

```bash
# Run all tests
python run_tests.py

# Run specific test suite
python run_tests.py --models
python run_tests.py --data
python run_tests.py --tracking
python run_tests.py --integration

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --test test_models.py
```

### Command Line Options

```bash
# Verbose output
pytest -v

# Run specific test
pytest tests/test_models.py::TestCustomCXRClassifier::test_forward_pass_shape

# Run with markers
pytest -m unit
pytest -m integration

# Generate coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing
```

## Test Coverage

### Current Coverage Areas

✅ **Models** (100%)
- All model initialization paths
- Forward pass with various inputs
- Output validation
- Gradient computation
- Device handling

✅ **Data Loading** (95%+)
- Dataset class functionality
- Data splitting logic
- Transform application
- Error handling
- Directory structure detection

✅ **Tracking** (90%+)
- Tracker initialization
- Logging methods
- Training function integration
- Mock-based testing

✅ **Integration** (85%+)
- End-to-end workflows
- Training pipelines
- Inference workflows

### Coverage Report

Generate coverage reports:

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report
pytest --cov=src --cov-report=html
# Then open htmlcov/index.html
```

## Writing New Tests

### Test Structure

```python
import unittest
from src.models.cnn_model import CustomCXRClassifier

class TestMyFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.model = CustomCXRClassifier()
    
    def test_feature_behavior(self):
        """Test description"""
        # Arrange
        input_data = torch.randn(1, 3, 128, 128)
        
        # Act
        result = self.model(input_data)
        
        # Assert
        self.assertEqual(result.shape, (1, 3))
```

### Using Fixtures

```python
import pytest
from tests.conftest import model, sample_batch

def test_with_fixture(model, sample_batch):
    """Test using fixtures"""
    images, labels = sample_batch
    output = model(images)
    assert output.shape[0] == images.shape[0]
```

### Best Practices

1. **Isolation**: Each test should be independent
2. **Naming**: Use descriptive test names (`test_what_when_then`)
3. **Fixtures**: Use setUp/tearDown for common setup
4. **Mocking**: Mock external dependencies (MLflow, W&B)
5. **Coverage**: Aim for >80% code coverage
6. **Speed**: Keep unit tests fast (<1 second each)

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v2
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Run tests from project root
   - Ensure `src/` is in Python path

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **MLflow/W&B Errors**
   - Tracking tests use mocks
   - No actual MLflow/W&B setup required

4. **CUDA Tests Failing**
   - CUDA tests are optional
   - They skip automatically if CUDA unavailable

## Test Maintenance

- Update tests when functionality changes
- Add tests for new features
- Remove tests for deprecated features
- Keep test data minimal
- Use temporary directories for file-based tests

## Future Improvements

- [ ] Add performance benchmarks
- [ ] Add regression tests
- [ ] Add property-based tests
- [ ] Add stress tests
- [ ] Add memory leak tests
- [ ] Add multi-GPU tests


