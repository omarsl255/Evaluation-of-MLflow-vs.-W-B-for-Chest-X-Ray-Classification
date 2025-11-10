# Running Grid Search with MLflow

## Quick Start

### Default Grid Search (Limited to 10 experiments)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

This will:
- Generate all parameter combinations from the grid
- Limit to 10 experiments (as configured)
- Shuffle experiments before running
- Continue on error

### Run All Combinations
To run ALL combinations, edit `configs/mlflow/hyperparameters.yaml`:

```yaml
execution:
  run_all_combinations: true  # Change to true
  max_experiments: 162  # Or remove this line
  shuffle: true
  continue_on_error: true
```

Then run:
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

### Limit to Specific Number
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 20
```

### Quick Test (Single Experiment)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --quick
```

## Current Grid Configuration

The `configs/mlflow/hyperparameters.yaml` file defines:

- **learning_rate**: [0.001, 0.0001, 0.01] (3 values)
- **batch_size**: [32, 64, 16] (3 values)
- **num_epochs**: [20, 30, 50] (3 values)
- **lr_gamma**: [0.1, 0.5] (2 values)
- **lr_step_size**: [5, 7, 10] (3 values)

**Total combinations: 3 × 3 × 3 × 2 × 3 = 162 experiments**

## Modifying the Grid

Edit `configs/mlflow/hyperparameters.yaml`:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]  # Add/remove values
  batch_size: [32, 64, 16]              # Modify as needed
  num_epochs: [20, 30, 50]              # Adjust range
  lr_gamma: [0.1, 0.5]                  # Learning rate decay
  lr_step_size: [5, 7, 10]              # Step size for decay
```

## Viewing Results

After running experiments:
```bash
python -m mlflow ui
```

Open http://localhost:5000 to compare all experiments!

## Tips

1. **Start Small**: Use `--max-experiments 10` to test first
2. **Monitor Progress**: Watch terminal output for each experiment
3. **Compare Results**: Use MLflow UI to identify best parameters
4. **Refine Grid**: Based on results, narrow down parameter ranges

## Example Workflow

1. Run quick test:
   ```bash
   python scripts/run_hyperparameter_tuning.py --quick
   ```

2. Run limited grid search:
   ```bash
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 20
   ```

3. Analyze results in MLflow UI

4. Modify grid based on results

5. Run full grid search (if needed):
   ```bash
   # Edit config to set run_all_combinations: true
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
   ```

