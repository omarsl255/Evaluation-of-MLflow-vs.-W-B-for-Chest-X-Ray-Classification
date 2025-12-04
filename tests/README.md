# Unit Tests

This directory contains comprehensive unit tests for all main functionalities of the project.

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── conftest.py              # Pytest fixtures and configuration
├── test_models.py           # Tests for CNN model architecture
├── test_data_loader.py      # Tests for data loading functionality
├── test_tracking.py         # Tests for MLflow and W&B tracking
├── test_integration.py      # Integration tests for end-to-end workflows
└── README.md                # This file
```

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements.txt
```

This will install `pytest` and `pytest-cov` along with other dependencies.

### Run All Tests

```bash
# Using pytest (recommended)
pytest

# Using unittest
python -m unittest discover tests

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing
```

### Run Specific Test Files

```bash
# Test models only
pytest tests/test_models.py

# Test data loader only
pytest tests/test_data_loader.py

# Test tracking only
pytest tests/test_tracking.py

# Test integration tests only
pytest tests/test_integration.py
```

### Run Specific Test Classes or Functions

```bash
# Run a specific test class
pytest tests/test_models.py::TestCustomCXRClassifier

# Run a specific test function
pytest tests/test_models.py::TestCustomCXRClassifier::test_forward_pass_shape
```

### Run Tests with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Coverage

### Generate Coverage Report

```bash
# Terminal report
pytest --cov=src --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=src --cov-report=html
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux
```

### Coverage Targets

- **Models**: Test model initialization, forward pass, output shapes, gradients
- **Data Loading**: Test dataset creation, data splitting, transforms, loaders
- **Tracking**: Test MLflow and W&B tracker initialization, logging, training functions
- **Integration**: Test end-to-end workflows

## Test Categories

### Unit Tests

- **test_models.py**: Tests for `CustomCXRClassifier` model
  - Model initialization
  - Forward pass
  - Output validation
  - Gradient flow
  - Device movement

- **test_data_loader.py**: Tests for data loading
  - Dataset class
  - Data splitting
  - Transform application
  - Loader creation

- **test_tracking.py**: Tests for tracking functionality
  - MLflowTracker class
  - WandBTracker class
  - Training functions

### Integration Tests

- **test_integration.py**: End-to-end workflow tests
  - Complete training pipeline
  - Model inference
  - Data loading to training

## Writing New Tests

### Example Test Structure

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
        input_data = ...
        
        # Act
        result = self.model(input_data)
        
        # Assert
        self.assertEqual(result.shape, expected_shape)
```

### Using Fixtures (Pytest)

```python
import pytest
from tests.conftest import model, sample_batch

def test_with_fixture(model, sample_batch):
    """Test using fixtures"""
    images, labels = sample_batch
    output = model(images)
    assert output.shape[0] == images.shape[0]
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=src --cov-report=xml
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running tests from the project root:

```bash
# From project root
pytest tests/
```

### Missing Dependencies

Install all dependencies:

```bash
pip install -r requirements.txt
```

### MLflow/W&B Errors in Tests

Tracking tests use mocks to avoid requiring actual MLflow/W&B setup. If you see errors, check that mocks are properly configured.

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use setUp/tearDown for common setup
3. **Naming**: Use descriptive test names
4. **Coverage**: Aim for >80% code coverage
5. **Speed**: Keep unit tests fast (<1 second each)
6. **Mocking**: Mock external dependencies (MLflow, W&B)

## Test Maintenance

- Update tests when functionality changes
- Add tests for new features
- Remove tests for deprecated features
- Keep test data minimal and focused
- Use temporary directories for file-based tests


