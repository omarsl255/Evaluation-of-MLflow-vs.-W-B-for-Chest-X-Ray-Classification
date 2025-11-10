# Correct Commands - Quick Reference

## ✅ Correct Command Structure

All scripts are now in the `scripts/` directory, and configs are organized by tool.

## Training Commands

### MLflow
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### W&B
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

## Hyperparameter Tuning Commands

### MLflow
```bash
# Run with default config (configs/mlflow/hyperparameters.yaml)
python scripts/run_hyperparameter_tuning.py

# Run with specific experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml

# Quick test
python scripts/run_hyperparameter_tuning.py --quick
```

### W&B
```bash
# Run with default config (configs/wandb/hyperparameters.yaml)
python scripts/run_wandb_hyperparameter_tuning.py

# Run with specific experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml

# Quick test
python scripts/run_wandb_hyperparameter_tuning.py --quick
```

## Comparison Command

```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

## Configuration Files

### MLflow Configs
- `configs/mlflow/experiments.yaml` - Specific experiments
- `configs/mlflow/hyperparameters.yaml` - Parameter grid
- `configs/mlflow/quick_test.yaml` - Quick test

### W&B Configs
- `configs/wandb/experiments.yaml` - Specific experiments
- `configs/wandb/hyperparameters.yaml` - Parameter grid
- `configs/wandb/quick_test.yaml` - Quick test

## Common Mistakes to Avoid

❌ **Wrong:**
```bash
python run_hyperparameter_tuning.py --config configs/experiments.yaml
```

✅ **Correct:**
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

## Why the New Structure?

1. **Better Organization**: Scripts are separated from source code
2. **Clear Separation**: Configs organized by tool (MLflow/W&B)
3. **Easier to Find**: Everything is in logical directories
4. **Professional**: Follows Python project best practices

## Quick Fix

If you get "file not found" errors:
1. Make sure you're in the project root directory
2. Use `scripts/` prefix for all scripts
3. Use `configs/mlflow/` or `configs/wandb/` for config files

