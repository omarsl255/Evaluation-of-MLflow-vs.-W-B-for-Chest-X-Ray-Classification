# Grid Search Guide - Using `configs/mlflow/hyperparameters.yaml`

## Overview

The `configs/mlflow/hyperparameters.yaml` file defines a parameter grid that generates **162 total combinations**:

- **learning_rate**: 3 values [0.001, 0.0001, 0.01]
- **batch_size**: 3 values [32, 64, 16]
- **num_epochs**: 3 values [20, 30, 50]
- **lr_gamma**: 2 values [0.1, 0.5]
- **lr_step_size**: 3 values [5, 7, 10]

**Total: 3 × 3 × 3 × 2 × 3 = 162 experiments**

## Running Grid Search

### Option 1: Run with Default Limit (10 experiments)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

This will:
- Generate all 162 combinations
- Randomly select 10 experiments (due to `max_experiments: 10`)
- Shuffle before running
- Continue on error

### Option 2: Run More Experiments
```bash
# Run 50 experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 50

# Run 100 experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 100
```

### Option 3: Run ALL 162 Combinations

**Method A: Use command line flag**
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 162
```

**Method B: Edit config file**
Edit `configs/mlflow/hyperparameters.yaml` and remove or comment out the `max_experiments` line:

```yaml
execution:
  # max_experiments: 10  # Commented out to run all
  shuffle: true
  continue_on_error: true
```

Then run:
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

### Option 4: Quick Test (1 experiment)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --quick
```

## Modifying the Grid

Edit `configs/mlflow/hyperparameters.yaml` to change parameter ranges:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]  # Add/remove values
  batch_size: [32, 64, 16]              # Modify as needed
  num_epochs: [20, 30, 50]              # Adjust range
  lr_gamma: [0.1, 0.5]                  # Learning rate decay
  lr_step_size: [5, 7, 10]              # Step size for decay
```

**Note**: Adding more values increases the total combinations exponentially!

## Recommended Workflow

1. **Start Small**: Run quick test first
   ```bash
   python scripts/run_hyperparameter_tuning.py --quick
   ```

2. **Limited Search**: Run 10-20 experiments to get initial results
   ```bash
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 20
   ```

3. **Analyze Results**: View in MLflow UI
   ```bash
   python -m mlflow ui
   ```

4. **Refine Grid**: Based on results, narrow down parameter ranges

5. **Full Search**: If needed, run all combinations
   ```bash
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 162
   ```

## Viewing Results

After running experiments:
```bash
python -m mlflow ui
```

Open http://localhost:5000 to:
- Compare all experiments
- Identify best hyperparameters
- View training curves
- Export results

## Tips

1. **Monitor Progress**: Watch terminal output for each experiment
2. **Use Shuffle**: Helps avoid bias from running order
3. **Continue on Error**: Allows other experiments to run even if one fails
4. **Start Small**: Begin with fewer experiments to test configuration
5. **Analyze Early**: Check results after initial runs to refine grid

## Example: Running 50 Experiments

```bash
python scripts/run_hyperparameter_tuning.py \
  --config configs/mlflow/hyperparameters.yaml \
  --max-experiments 50 \
  --shuffle
```

This will:
- Generate all 162 combinations
- Randomly select 50 experiments
- Shuffle before running
- Track all results in MLflow

## Configuration File Location

- **Grid Search Config**: `configs/mlflow/hyperparameters.yaml`
- **Specific Experiments**: `configs/mlflow/experiments.yaml`
- **Quick Test**: Use `--quick` flag with hyperparameters.yaml

