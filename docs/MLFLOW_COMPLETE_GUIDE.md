# MLflow Complete Guide

This comprehensive guide covers everything you need to know about using MLflow for experiment tracking in this Chest X-Ray Classification project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training with MLflow](#training-with-mlflow)
3. [Viewing Results](#viewing-results)
4. [Comparing Experiments](#comparing-experiments)
5. [Loading Saved Models](#loading-saved-models)
6. [MLflow UI Features](#mlflow-ui-features)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Grid Search](#grid-search)
9. [Examples](#examples)
10. [Best Practices](#best-practices)
11. [Advanced Usage](#advanced-usage)
12. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python main.py --download
```

### 3. Train with MLflow
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### 4. View Results

**Option 1: Using Python module (recommended on Windows)**
```bash
python -m mlflow ui
```

**Option 2: Using helper script**
```bash
python scripts/start_mlflow_ui.py
```

**Option 3: Using MLflow CLI (Linux/Mac)**
```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

### Quick Commands Reference

```bash
# Train a model
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20

# Train with test evaluation
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --test

# View UI
python -m mlflow ui
```

---

## Training with MLflow

### Basic Training
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20
```

### Advanced Options
```bash
python scripts/train_mlflow.py \
    --dataset_path "Covid19-dataset" \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --image_size 128 \
    --experiment_name "MyExperiment" \
    --run_name "baseline_model" \
    --test
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_path` | Path to dataset directory | Required |
| `--epochs` | Number of training epochs | 20 |
| `--batch_size` | Batch size for training | 32 |
| `--learning_rate` | Learning rate | 0.001 |
| `--image_size` | Image size for resizing | 128 |
| `--device` | Device (cuda/cpu) | cuda if available |
| `--experiment_name` | MLflow experiment name | Chest-XRay-Classification-MLflow |
| `--run_name` | Name for this run | Auto-generated |
| `--test` | Evaluate on test set | False |

### What Gets Tracked

#### Metrics (Automatically Logged)
- Training Loss (per epoch)
- Training Accuracy (per epoch)
- Validation Loss (per epoch)
- Validation Accuracy (per epoch)
- Per-class Precision, Recall, F1-score
- Test metrics (if `--test` flag used)

#### Parameters (Automatically Logged)
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer settings
- Model architecture

#### Artifacts (Automatically Saved)
- Full PyTorch model
- Confusion matrix
- Test metrics (if available)

---

## Viewing Results

### Start MLflow UI

**On Windows (recommended):**
```bash
python -m mlflow ui
```

**On Linux/Mac:**
```bash
mlflow ui
```

The UI will be available at: **http://localhost:5000**

**Note**: If the `mlflow` command is not found, always use `python -m mlflow ui` instead.

### MLflow UI Features

1. **Experiments List**: View all your experiments
2. **Runs Comparison**: Compare multiple runs side-by-side
3. **Metrics Visualization**: See training curves, validation metrics
4. **Parameters**: View hyperparameters for each run
5. **Artifacts**: Download models, confusion matrices, etc.
6. **Model Registry**: Register and manage model versions

### Navigation in MLflow UI

- **Experiments**: Left sidebar shows all experiments
- **Runs**: Each training session is a "run"
- **Metrics**: Click on a run to see detailed metrics
- **Compare**: Select multiple runs and click "Compare" to compare them
- **Download**: Click on artifacts to download models or files

---

## Comparing Experiments

### Compare Multiple Runs

1. Start MLflow UI: `python -m mlflow ui`
2. Open http://localhost:5000
3. Select multiple runs (checkboxes)
4. Click "Compare" button
5. View side-by-side comparison of:
   - Parameters (hyperparameters)
   - Metrics (accuracy, loss, etc.)
   - Training curves

### Example: Training Multiple Models

```bash
# Run 1: Baseline
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --run_name "baseline"

# Run 2: Higher learning rate
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --learning_rate 0.01 --run_name "lr_0.01"

# Run 3: Larger batch size
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 64 --run_name "batch_64"
```

Then compare all three runs in MLflow UI.

---

## Loading Saved Models

### Using MLflow to Load Models

```python
import mlflow
import mlflow.pytorch
import torch

# Load a specific run's model
run_id = "your-run-id-here"
model_uri = f"runs:/{run_id}/model"
model = mlflow.pytorch.load_model(model_uri)

# Use the model for inference
model.eval()
with torch.no_grad():
    output = model(input_tensor)
```

### Find Run ID

1. Open MLflow UI: `python -m mlflow ui`
2. Click on a run
3. Copy the Run ID from the run details page
4. Or use the MLflow API:

```python
import mlflow

# Get all runs in an experiment
experiment = mlflow.get_experiment_by_name("Chest-XRay-Classification-MLflow")
runs = mlflow.search_runs(experiment.experiment_id)

# Get the best run
best_run = runs.loc[runs['metrics.val_accuracy'].idxmax()]
best_run_id = best_run['run_id']
```

---

## MLflow UI Features

### 1. Metrics Tracking
- **Training Loss**: Tracked every epoch
- **Training Accuracy**: Tracked every epoch
- **Validation Loss**: Tracked every epoch
- **Validation Accuracy**: Tracked every epoch
- **Per-class Metrics**: Precision, Recall, F1-score for each class
- **Best Validation Accuracy**: Logged at the end

### 2. Parameters Logged
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer type
- Loss function
- Model architecture

### 3. Artifacts
- **Model**: Full PyTorch model saved
- **Confusion Matrix**: Saved as numpy array
- **Test Metrics**: If `--test` flag is used

### 4. Run Information
- Run ID
- Run name (if provided)
- Start/End time
- Status
- Tags (customizable)

---

## Hyperparameter Tuning

This section explains how to use the parameter matrix system for hyperparameter tuning.

### Quick Start

#### 1. Run with Default Configuration
```bash
python scripts/run_hyperparameter_tuning.py
```

#### 2. Run with Specific Config File
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

#### 3. Run Quick Test (Fewer Experiments)
```bash
python scripts/run_hyperparameter_tuning.py --quick
```

### Configuration Files

#### 1. `configs/mlflow/hyperparameters.yaml` - Parameter Grid
Defines a grid of parameters to try. All combinations will be generated.

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]
  batch_size: [32, 64, 16]
  num_epochs: [20, 30, 50]
```

This will generate 3 × 3 × 3 = 27 experiments.

#### 2. `configs/mlflow/experiments.yaml` - Specific Experiments
Define exact experiments to run with specific parameters.

```yaml
experiments:
  - name: "baseline"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
```

### Modifying Parameters

#### Option 1: Edit Parameter Grid (Hyperparameter Search)

Edit `configs/mlflow/hyperparameters.yaml`:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.005, 0.01]  # Add/remove values
  batch_size: [16, 32, 64, 128]        # Add more batch sizes
  num_epochs: [10, 20, 30]             # Modify epochs
  lr_gamma: [0.1, 0.5]                 # Add learning rate decay
```

**Note**: This generates ALL combinations. For 4×4×3×2 = 96 experiments!

#### Option 2: Define Specific Experiments (Recommended)

Edit `configs/mlflow/experiments.yaml`:

```yaml
experiments:
  # Add your experiment here
  - name: "my_custom_experiment"
    learning_rate: 0.002
    batch_size: 48
    num_epochs: 25
    lr_gamma: 0.2
    lr_step_size: 8
```

Add as many experiments as you want!

#### Option 3: Create New Config File

1. Copy `configs/mlflow/experiments.yaml` to `configs/mlflow/my_config.yaml`
2. Modify the parameters
3. Run: `python scripts/run_hyperparameter_tuning.py --config configs/mlflow/my_config.yaml`

### Configuration Structure

#### Base Configuration
Parameters that apply to all experiments:

```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"  # "auto", "cuda", or "cpu"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
  test_after_training: true
```

#### Experiment Parameters
Parameters that vary between experiments:

- `learning_rate`: Learning rate (float)
- `batch_size`: Batch size (int)
- `num_epochs`: Number of epochs (int)
- `lr_gamma`: Learning rate decay factor (float)
- `lr_step_size`: Learning rate decay step size (int)

#### MLflow Configuration
```yaml
mlflow_config:
  experiment_name: "Hyperparameter-Tuning"
  use_run_names: true
  run_name_template: "lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}"
```

#### Execution Settings
```yaml
execution:
  max_experiments: null  # null = run all combinations, or set a number to limit
  shuffle: true          # Shuffle before running
  continue_on_error: true  # Continue if one fails
```

### Command Line Options

```bash
python scripts/run_hyperparameter_tuning.py [OPTIONS]

Options:
  --config PATH              Path to YAML config file (default: configs/mlflow/hyperparameters.yaml)
  --quick                    Use quick test configuration
  --max-experiments N        Maximum number of experiments to run
  --shuffle                  Shuffle experiments before running
```

### Workflow

1. **Plan Your Experiments**: Decide which parameters to tune
2. **Create/Edit Config File**: Use `configs/mlflow/experiments.yaml` for specific experiments or `configs/mlflow/hyperparameters.yaml` for grid search
3. **Run Experiments**: `python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml`
4. **Monitor Progress**: Watch terminal output and check MLflow UI
5. **Analyze Results**: Compare experiments in MLflow UI

### Tips

1. **Start Small**: Begin with a few experiments
2. **Use Descriptive Names**: Makes it easy to identify experiments
3. **Organize Experiments**: Create separate config files for different purposes
4. **Limit Experiments**: Use `max_experiments` to avoid running too many
5. **Shuffle Experiments**: Helps avoid bias from running order

---

## Grid Search

This section covers grid search hyperparameter tuning with MLflow using `configs/mlflow/hyperparameters.yaml`.

### Overview

The `configs/mlflow/hyperparameters.yaml` file defines a parameter grid that generates **162 total combinations**:

- **learning_rate**: 3 values [0.001, 0.0001, 0.01]
- **batch_size**: 3 values [32, 64, 16]
- **num_epochs**: 3 values [20, 30, 50]
- **lr_gamma**: 2 values [0.1, 0.5]
- **lr_step_size**: 3 values [5, 7, 10]

**Total: 3 × 3 × 3 × 2 × 3 = 162 experiments**

### Running Grid Search

#### Option 1: Run ALL 162 Combinations (Default)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

This will:
- Generate all 162 combinations
- Run ALL experiments (due to `max_experiments: null` in config)
- Shuffle before running
- Continue on error

**Note:** The default config now runs all combinations. To limit experiments, use `--max-experiments` flag or edit the config file.

#### Option 2: Limit Number of Experiments
```bash
# Run only 10 experiments (useful for testing)
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 10

# Run 50 experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 50

# Run 100 experiments
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 100
```

**To limit via config file:** Edit `configs/mlflow/hyperparameters.yaml`:

```yaml
execution:
  max_experiments: 10  # Set to a number to limit, or null to run all
  shuffle: true
  continue_on_error: true
```

#### Option 3: Quick Test (1 experiment)
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --quick
```

### Modifying the Grid

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

### Recommended Workflow

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

5. **Full Search**: If needed, run all 162 combinations (default)
   ```bash
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
   ```

### Viewing Results

After running experiments:
```bash
python -m mlflow ui
```

Open http://localhost:5000 to:
- Compare all experiments
- Identify best hyperparameters
- View training curves
- Export results

### Tips

1. **Monitor Progress**: Watch terminal output for each experiment
2. **Use Shuffle**: Helps avoid bias from running order
3. **Continue on Error**: Allows other experiments to run even if one fails
4. **Start Small**: Begin with fewer experiments to test configuration
5. **Analyze Early**: Check results after initial runs to refine grid

---

## Examples

### Example 1: Simple Experiment Configuration

**File: `configs/mlflow/experiments.yaml`**

```yaml
experiments:
  - name: "baseline"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
  
  - name: "high_lr"
    learning_rate: 0.01
    batch_size: 32
    num_epochs: 20
  
  - name: "large_batch"
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 20
```

**Run:**
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/experiments.yaml
```

**Result:** 3 experiments will run automatically, all tracked in MLflow!

### Example 2: Parameter Grid (All Combinations)

**File: `configs/mlflow/hyperparameters.yaml`**

```yaml
parameter_grid:
  learning_rate: [0.001, 0.01]
  batch_size: [32, 64]
  num_epochs: [20, 30]
```

**Run:**
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
```

**Result:** 2 × 2 × 2 = 8 experiments will run (all combinations)

### Example 3: Learning Rate Sweep

**File: `configs/mlflow/learning_rate_sweep.yaml`**

```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  test_after_training: true

mlflow_config:
  experiment_name: "Learning-Rate-Sweep"
  use_run_names: true
  run_name_template: "lr_{learning_rate}"

experiments:
  - name: "lr_0.0001"
    learning_rate: 0.0001
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.0005"
    learning_rate: 0.0005
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.001"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.005"
    learning_rate: 0.005
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
  
  - name: "lr_0.01"
    learning_rate: 0.01
    batch_size: 32
    num_epochs: 20
    lr_gamma: 0.1
    lr_step_size: 7
```

**Run:**
```bash
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/learning_rate_sweep.yaml
```

### Example 4: Batch Size Comparison

**File: `configs/mlflow/batch_size_comparison.yaml`**

```yaml
experiments:
  - name: "batch_16"
    learning_rate: 0.001
    batch_size: 16
    num_epochs: 20
  
  - name: "batch_32"
    learning_rate: 0.001
    batch_size: 32
    num_epochs: 20
  
  - name: "batch_64"
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 20
  
  - name: "batch_128"
    learning_rate: 0.001
    batch_size: 128
    num_epochs: 20
```

### Example 5: Comprehensive Grid Search

**File: `configs/mlflow/comprehensive_search.yaml`**

```yaml
parameter_grid:
  learning_rate: [0.0001, 0.001, 0.01]
  batch_size: [16, 32, 64]
  num_epochs: [20, 30]
  lr_gamma: [0.1, 0.5]

execution:
  max_experiments: 20  # Limit to 20 experiments
  shuffle: true
```

This would generate 3 × 3 × 2 × 2 = 36 combinations, but limited to 20.

---

## Best Practices

### 1. Use Descriptive Run Names
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --run_name "baseline_lr0.001_bs32"
```

### 2. Organize Experiments
```bash
# Different experiments for different purposes
python scripts/train_mlflow.py --experiment_name "Hyperparameter-Tuning" --run_name "trial_1"
python scripts/train_mlflow.py --experiment_name "Model-Architecture" --run_name "resnet50"
```

### 3. Track All Important Parameters
All hyperparameters are automatically logged. You can add custom parameters by modifying the training script.

### 4. Use Tags for Organization
You can add tags to runs programmatically:
```python
with mlflow.start_run(tags={"model_type": "CNN", "dataset": "COVID-19"}):
    # Training code
```

### 5. Regular Evaluation
Always use `--test` flag to evaluate on test set:
```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

### 6. Start Small with Grid Search
- Begin with a few experiments
- Use `--quick` flag for testing
- Gradually expand parameter space

### 7. Compare Multiple Runs
Use MLflow UI to compare all experiments easily and identify best hyperparameters.

---

## Advanced Usage

### Custom Tracking URI

By default, MLflow stores data in `./mlruns`. You can change this:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Use SQLite database
# or
mlflow.set_tracking_uri("http://localhost:5000")  # Use remote server
```

### Query Runs Programmatically

```python
import mlflow
import pandas as pd

# Search runs
runs = mlflow.search_runs(
    experiment_names=["Chest-XRay-Classification-MLflow"],
    filter_string="metrics.val_accuracy > 0.8"
)

# Get best run
best_run = runs.loc[runs['metrics.val_accuracy'].idxmax()]
print(f"Best accuracy: {best_run['metrics.val_accuracy']}")
print(f"Run ID: {best_run['run_id']}")
```

### Export Results

```python
import mlflow

# Export runs to CSV
runs = mlflow.search_runs(experiment_names=["Chest-XRay-Classification-MLflow"])
runs.to_csv("experiment_results.csv", index=False)
```

---

## Troubleshooting

### MLflow UI Not Starting

**On Windows:**
```bash
# Use Python module syntax (recommended)
python -m mlflow ui

# Or use different port
python -m mlflow ui --port 5001
```

**On Linux/Mac:**
```bash
# Check if port 5000 is in use
mlflow ui --port 5001  # Use different port
```

**If `mlflow` command not found:**
Always use `python -m mlflow ui` instead of `mlflow ui`. This works on all platforms.

### Cannot Find Runs
- Make sure you're in the project directory where `mlruns/` folder exists
- Check experiment name matches
- Verify runs were completed successfully

### Model Loading Issues
- Ensure you're using the same PyTorch version
- Check that the model architecture matches
- Verify the run_id is correct

### Too Many Experiments
- Use `--max-experiments` to limit
- Edit config to reduce parameter grid
- Use specific experiments instead of grid

### Experiments Failing
- Check dataset path is correct
- Verify parameters are valid
- Set `continue_on_error: true` to continue

### Config File Errors
- Check YAML syntax
- Verify all required parameters are present
- Use a YAML validator

---

## Example Workflow

### Complete Training and Evaluation Workflow

```bash
# 1. Train model
python scripts/train_mlflow.py \
    --dataset_path "Covid19-dataset" \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --run_name "final_model" \
    --test

# 2. Start MLflow UI
python -m mlflow ui

# 3. Open browser to http://localhost:5000
# 4. View results, compare runs, download models
```

---

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow PyTorch Integration](https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

## Quick Reference

```bash
# Train with MLflow
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20

# View UI
python -m mlflow ui

# Access UI
# Open http://localhost:5000 in browser

# Compare experiments
# Select multiple runs in UI and click "Compare"

# Load model
# Use run_id from UI: mlflow.pytorch.load_model(f"runs:/{run_id}/model")

# Grid Search (all 162 combinations)
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml

# Grid Search (limited to 10)
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 10

# Quick test
python scripts/run_hyperparameter_tuning.py --quick
```

