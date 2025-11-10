# Quick Commands Reference

## Training

### MLflow
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### W&B
```bash
# Login first
wandb login

# Then train
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

## Hyperparameter Tuning

### MLflow
```bash
# Run with default config
python scripts/run_hyperparameter_tuning.py

# Run with specific experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml

# Quick test
python scripts/run_hyperparameter_tuning.py --quick
```

### W&B
```bash
# Run with default config
python scripts/run_wandb_hyperparameter_tuning.py

# Run with specific experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml

# Quick test
python scripts/run_wandb_hyperparameter_tuning.py --quick
```

## Comparison

```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

## View Results

### MLflow UI
```bash
python -m mlflow ui
```
Open http://localhost:5000

### W&B Dashboard
Results are automatically uploaded to https://wandb.ai

## Configuration Files

- MLflow: `configs/mlflow/experiments.yaml`
- W&B: `configs/wandb/experiments.yaml`

## Common Options

- `--dataset_path`: Path to dataset (required for training)
- `--epochs`: Number of epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--config`: Path to config file (for hyperparameter tuning)
- `--quick`: Quick test mode
- `--max-experiments`: Limit number of experiments
- `--shuffle`: Shuffle experiments

## Examples

### Train a single model
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

### Run hyperparameter tuning
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

### Compare both tools
```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

