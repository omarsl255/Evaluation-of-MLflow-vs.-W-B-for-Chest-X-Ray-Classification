# Getting Started

This is a quick reference guide to help you navigate the project.

## Using the Makefile (Recommended!)

We've created a Makefile to simplify all commands. Just type `make` followed by the command:

```bash
# See all available commands
make help

# Quick W&B hyperparameter test
make wandb-quick

# Full W&B hyperparameter tuning
make wandb-tune

# Quick MLflow hyperparameter test
make mlflow-quick

# Compare MLflow vs W&B
make compare

# Train with custom parameters
make train-custom EPOCHS=50 BATCH_SIZE=64

# Start MLflow UI
make mlflow-ui
```

## Manual Commands (if not using Makefile)

### Run W&B Hyperparameter Tuning
```bash
# Quick test with default config
PYTHONPATH=. python scripts/run_wandb_hyperparameter_tuning.py --quick

# Full hyperparameter search
PYTHONPATH=. python scripts/run_wandb_hyperparameter_tuning.py

# With custom config
PYTHONPATH=. python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
```

### Run MLflow Hyperparameter Tuning
```bash
# Quick test
PYTHONPATH=. python scripts/run_hyperparameter_tuning.py --quick

# Full hyperparameter search
PYTHONPATH=. python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

### Train Models
```bash
# Train with W&B
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# Train with MLflow
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20

# Compare both
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

## Documentation

- **[README.md](README.md)** - Main project documentation
- **[Quick Start Guide](docs/guides/QUICK_START.md)** - Detailed quick start guide
- **[W&B Quick Start](docs/guides/WANDB_QUICK_START.md)** - W&B-specific guide
- **[Project Structure](docs/guides/PROJECT_STRUCTURE.md)** - Detailed structure info

## Configuration Files

- **W&B configs**: [configs/wandb/](configs/wandb/)
  - `hyperparameters.yaml` - Hyperparameter search configuration
- **MLflow configs**: [configs/mlflow/](configs/mlflow/)
  - `hyperparameters.yaml` - Hyperparameter search configuration
  - `experiments.yaml` - Specific experiments
  - `quick_test.yaml` - Quick test configuration

## Common Issues

### Command not working?
Make sure to use `PYTHONPATH=.` before your command:
```bash
PYTHONPATH=. python scripts/your_script.py
```

### Missing config file?
Check that the config file exists:
```bash
ls configs/wandb/hyperparameters.yaml
ls configs/mlflow/hyperparameters.yaml
```

### W&B not logged in?
```bash
wandb login
```

### View MLflow UI
```bash
mlflow ui
# Then visit http://localhost:5000
```
