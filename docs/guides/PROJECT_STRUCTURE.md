# Project Structure

This document describes the improved project structure for better organization and maintainability.

## Directory Structure

```
.
├── src/                          # Core source code
│   ├── __init__.py
│   ├── models/                   # Model definitions
│   │   ├── __init__.py
│   │   └── cnn_model.py          # Custom CNN architecture
│   ├── data/                     # Data handling
│   │   ├── __init__.py
│   │   └── data_loader.py        # Data loading utilities
│   ├── tracking/                 # Experiment tracking
│   │   ├── __init__.py
│   │   ├── mlflow_tracker.py     # MLflow integration
│   │   └── wandb_tracker.py      # W&B integration
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       └── visualization.py      # Visualization utilities (optional)
│
├── scripts/                      # Training and execution scripts
│   ├── train_mlflow.py           # Train with MLflow
│   ├── train_wandb.py            # Train with W&B
│   ├── compare_mlflow_wandb.py   # Compare both tools
│   ├── run_hyperparameter_tuning.py      # MLflow hyperparameter tuning
│   ├── run_wandb_hyperparameter_tuning.py # W&B hyperparameter tuning
│   └── start_mlflow_ui.py        # Start MLflow UI
│
├── tools/                        # Utility and maintenance scripts
│   ├── cleanup_duplicates.py     # Cleanup duplicate files
│   ├── improve_structure.py      # Structure improvement utilities
│   └── organize_project.py       # Project organization utilities
│
├── examples/                     # Example scripts
│   ├── example_mlflow_usage.py
│   └── example_wandb_usage.py
│
├── configs/                      # Configuration files
│   ├── mlflow/                   # MLflow configurations
│   │   ├── experiments.yaml
│   │   ├── hyperparameters.yaml
│   │   └── quick_test.yaml
│   └── wandb/                    # W&B configurations
│       ├── experiments.yaml
│       ├── hyperparameters.yaml
│       └── quick_test.yaml
│
├── docs/                         # Documentation
│   ├── MLFLOW_COMPLETE_GUIDE.md  # Complete MLflow guide (includes all topics)
│   ├── WANDB_COMPLETE_GUIDE.md   # Complete W&B guide (includes all topics)
│   ├── README.md                 # Documentation index
│   ├── BASE_CONFIG_OPTIONS.md
│   ├── STRUCTURE_IMPROVEMENTS.md
│   ├── TESTING_GUIDE.md
│   ├── DOCUMENTATION_IMPROVEMENTS.md
│   └── MIGRATION_GUIDE.md
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures and configuration
│   ├── test_models.py            # Model tests
│   ├── test_data_loader.py       # Data loader tests
│   ├── test_tracking.py          # Tracking tests
│   ├── test_integration.py       # Integration tests
│   └── README.md                 # Testing documentation
│
├── notebooks/                    # Jupyter notebooks (optional)
│   └── .gitkeep
│
├── Covid19-dataset/              # Dataset directory (gitignored if large)
│   ├── train/
│   │   ├── COVID-19/
│   │   ├── Viral Pneumonia/
│   │   └── Normal/
│   └── test/
│       ├── COVID-19/
│       ├── Viral Pneumonia/
│       └── Normal/
│
├── mlruns/                       # MLflow runs (gitignored)
├── wandb/                        # W&B runs (gitignored)
│
├── main.py                       # Main entry point (dataset download)
├── run_tests.py                  # Test runner script
├── pytest.ini                    # Pytest configuration
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup (optional)
├── .gitignore                    # Git ignore rules
├── README.md                     # Main documentation
├── QUICK_START.md                # Quick start guide
└── PROJECT_STRUCTURE.md          # This file
```

## Organization Principles

### 1. Source Code (`src/`)
- **models/**: Model architecture definitions
  - `cnn_model.py`: Custom CNN for Chest X-Ray classification
    - Architecture based on [Vinay10100/Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification)
    - Adapted for PyTorch and integrated with MLflow/W&B tracking
- **data/**: Data loading and preprocessing
  - `data_loader.py`: Dataset loading, splitting, and augmentation
- **tracking/**: Experiment tracking integrations
  - `mlflow_tracker.py`: MLflow integration for experiment tracking
  - `wandb_tracker.py`: W&B integration for experiment tracking
- **utils/**: Utility functions and helpers
  - Visualization utilities, helper functions, etc.

### 2. Scripts (`scripts/`)
- **Training Scripts**: 
  - `train_mlflow.py`: Train model with MLflow tracking
  - `train_wandb.py`: Train model with W&B tracking
- **Comparison Scripts**:
  - `compare_mlflow_wandb.py`: Compare MLflow vs W&B
- **Hyperparameter Tuning**:
  - `run_hyperparameter_tuning.py`: MLflow hyperparameter tuning
  - `run_wandb_hyperparameter_tuning.py`: W&B hyperparameter tuning
- **Utilities**:
  - `start_mlflow_ui.py`: Start MLflow UI server

### 2b. Tools (`tools/`)
- **Maintenance Scripts**:
  - `cleanup_duplicates.py`: Cleanup duplicate files
  - `improve_structure.py`: Structure improvement utilities
  - `organize_project.py`: Project organization utilities

### 3. Examples (`examples/`)
- Example usage scripts
- Demo scripts showing how to use the tracking tools
- Tutorial scripts

### 4. Configuration (`configs/`)
- **mlflow/**: MLflow experiment configurations
  - `experiments.yaml`: Specific experiments for MLflow
  - `hyperparameters.yaml`: Parameter grid for MLflow
  - `quick_test.yaml`: Quick test configuration for MLflow
- **wandb/**: W&B experiment configurations
  - `experiments.yaml`: Specific experiments for W&B
  - `hyperparameters.yaml`: Parameter grid for W&B
  - `quick_test.yaml`: Quick test configuration for W&B

### 5. Documentation (`docs/`)
- **MLflow Documentation**: `MLFLOW_COMPLETE_GUIDE.md` - Comprehensive guide covering quick start, training, hyperparameter tuning, grid search, and examples
- **W&B Documentation**: `WANDB_COMPLETE_GUIDE.md` - Comprehensive guide covering quick start, training, hyperparameter tuning, grid search, and examples
- **General Documentation**: Other documentation files (BASE_CONFIG_OPTIONS.md, TESTING_GUIDE.md, etc.)
  - README.md (documentation index)
  - BASE_CONFIG_OPTIONS.md
  - STRUCTURE_IMPROVEMENTS.md
  - TESTING_GUIDE.md
  - DOCUMENTATION_IMPROVEMENTS.md

### 6. Tests (`tests/`)
- Unit tests for models
- Unit tests for data loading
- Unit tests for tracking integrations
- Integration tests

## File Naming Conventions

### Python Files
- Use snake_case: `data_loader.py`, `mlflow_tracker.py`
- Descriptive names: `train_mlflow.py`, `compare_mlflow_wandb.py`
- Group related functionality in modules

### Configuration Files
- Use descriptive names: `experiments.yaml`, `hyperparameters.yaml`
- Organize by tool: `mlflow/`, `wandb/`
- Use YAML format for configuration files

### Documentation Files
- Use UPPER_SNAKE_CASE: `MLFLOW_COMPLETE_GUIDE.md`, `WANDB_COMPLETE_GUIDE.md`
- Descriptive names that indicate content
- All documentation files are in `docs/` directory (no subdirectories)
- Each tool has one comprehensive guide covering all topics

## Import Structure

### From Source Code
```python
from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import get_data_loaders
from src.tracking.mlflow_tracker import train_with_mlflow
from src.tracking.wandb_tracker import train_with_wandb
```

### Configuration Files
```python
config_path = "configs/mlflow/experiments.yaml"
config_path = "configs/wandb/hyperparameters.yaml"
```

## Benefits of This Structure

1. **Separation of Concerns**: Clear separation between core code, scripts, and configs
2. **Maintainability**: Easy to find and modify files
3. **Scalability**: Easy to add new features or tools
4. **Professional**: Follows Python project best practices
5. **Documentation**: Well-organized documentation
6. **Testing**: Dedicated tests directory
7. **Reusability**: Source code can be imported as a package
8. **Clarity**: Clear structure makes it easy for new contributors

## Adding New Features

### Adding a New Model
1. Create model file in `src/models/`
2. Update `src/models/__init__.py`
3. Add tests in `tests/test_models.py`

### Adding a New Tracking Tool
1. Create tracker file in `src/tracking/`
2. Update `src/tracking/__init__.py`
3. Create training script in `scripts/`
4. Add configuration files in `configs/`
5. Add documentation in `docs/`

### Adding a New Script
1. Create script in `scripts/`
2. Use imports from `src/`
3. Update documentation if needed

## Running Scripts

All scripts should be run from the project root:

```bash
# Training scripts
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# Hyperparameter tuning
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml

# Comparison
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10

# MLflow UI
python scripts/start_mlflow_ui.py
```

## Git Ignore

The following directories/files should be gitignored:
- `mlruns/`: MLflow runs
- `wandb/`: W&B runs
- `__pycache__/`: Python cache
- `*.pyc`: Compiled Python files
- `Covid19-dataset/`: Dataset (if large, or keep if small)
- `.env`: Environment variables
- `*.log`: Log files
- `tools/`: Utility scripts (optional, can be versioned)

## Next Steps

1. Ensure all files are in their correct locations
2. Update all imports to use the new structure
3. Update configuration file paths
4. Update documentation references
5. Test that everything works
6. Update README.md with new structure
