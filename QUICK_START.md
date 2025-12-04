# Quick Start Guide

Get started with the project in minutes!

## 📋 Table of Contents

1. [Installation](#1-installation)
2. [Download Dataset](#2-download-dataset)
3. [Train with MLflow](#3-train-with-mlflow)
4. [Train with W&B](#4-train-with-wb)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Compare Both Tools](#6-compare-both-tools)
7. [Command Reference](#7-command-reference)
8. [Next Steps](#8-next-steps)

## 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification-main

# Install dependencies
pip install -r requirements.txt
```

## 2. Download Dataset

```bash
python main.py --download
```

## 3. Train with MLflow

```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

**View results:**
```bash
python -m mlflow ui
```
Open http://localhost:5000

**For detailed MLflow guide:** See [docs/MLFLOW_COMPLETE_GUIDE.md](docs/MLFLOW_COMPLETE_GUIDE.md)

## 4. Train with W&B

```bash
# Login first (Windows: use python -m wandb login)
python -m wandb login

# Then train
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

**View results:** Results are automatically uploaded to https://wandb.ai

**For detailed W&B guide:** See [docs/WANDB_COMPLETE_GUIDE.md](docs/WANDB_COMPLETE_GUIDE.md)

## 5. Hyperparameter Tuning

### MLflow Grid Search

```bash
# Quick test (1 experiment)
python scripts/run_hyperparameter_tuning.py --quick

# Run with default config (limited experiments)
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml

# Run specific number of experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 10

# Run all 162 combinations
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 162
```

**For detailed grid search guide:** See [docs/MLFLOW_COMPLETE_GUIDE.md](docs/MLFLOW_COMPLETE_GUIDE.md) (Grid Search section)

### W&B Grid Search

```bash
# Quick test (1 experiment)
python scripts/run_wandb_hyperparameter_tuning.py --quick

# Run with default config
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml

# Run specific number of experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10
```

**For detailed grid search guide:** See [docs/WANDB_COMPLETE_GUIDE.md](docs/WANDB_COMPLETE_GUIDE.md) (Grid Search section)

## 6. Compare Both Tools

```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

## 7. Command Reference

### Training Commands

| Task | MLflow | W&B |
|------|--------|-----|
| **Basic Training** | `python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20` | `python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20` |
| **View Results** | `python -m mlflow ui` (then open http://localhost:5000) | Auto-uploaded to https://wandb.ai |

### Hyperparameter Tuning Commands

| Task | MLflow | W&B |
|------|--------|-----|
| **Quick Test** | `python scripts/run_hyperparameter_tuning.py --quick` | `python scripts/run_wandb_hyperparameter_tuning.py --quick` |
| **Default Config** | `python scripts/run_hyperparameter_tuning.py` | `python scripts/run_wandb_hyperparameter_tuning.py` |
| **Specific Config** | `python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml` | `python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml` |
| **Limit Experiments** | `python scripts/run_hyperparameter_tuning.py --max-experiments 10` | `python scripts/run_wandb_hyperparameter_tuning.py --max-experiments 10` |

### Common Options

- `--dataset_path`: Path to dataset (required for training)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--config`: Path to config file (for hyperparameter tuning)
- `--quick`: Quick test mode (1 experiment)
- `--max-experiments`: Limit number of experiments
- `--shuffle`: Shuffle experiments before running

### Configuration Files

**MLflow Configs:**
- `configs/mlflow/experiments.yaml` - Specific experiments
- `configs/mlflow/hyperparameters.yaml` - Parameter grid (162 combinations)
- `configs/mlflow/quick_test.yaml` - Quick test

**W&B Configs:**
- `configs/wandb/experiments.yaml` - Specific experiments
- `configs/wandb/hyperparameters.yaml` - Parameter grid (162 combinations)
- `configs/wandb/quick_test.yaml` - Quick test

**For config options:** See [docs/BASE_CONFIG_OPTIONS.md](docs/BASE_CONFIG_OPTIONS.md)

## 8. Next Steps

### Documentation
- **Complete Guides:**
  - [docs/MLFLOW_COMPLETE_GUIDE.md](docs/MLFLOW_COMPLETE_GUIDE.md) - Complete MLflow guide (includes quick start, training, hyperparameter tuning, grid search, and examples)
  - [docs/WANDB_COMPLETE_GUIDE.md](docs/WANDB_COMPLETE_GUIDE.md) - Complete W&B guide (includes quick start, training, hyperparameter tuning, grid search, and examples)
- **Project Info:**
  - [README.md](README.md) - Main project documentation
  - [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Project structure details
  - [docs/README.md](docs/README.md) - Documentation index

### Common Issues

**❌ Script not found:**
- ✅ Make sure you're in the project root directory
- ✅ Use `scripts/` prefix: `python scripts/train_mlflow.py`

**❌ Config file not found:**
- ✅ Use full path: `configs/mlflow/experiments.yaml`
- ✅ Check file exists in `configs/mlflow/` or `configs/wandb/`

**❌ W&B login error:**
- ✅ Use `python -m wandb login` (Windows)
- ✅ Or `wandb login` if in PATH

**❌ MLflow UI not starting:**
- ✅ Use `python -m mlflow ui` (Windows)
- ✅ Or `mlflow ui` if in PATH

