# Weights & Biases (W&B) Complete Guide

This comprehensive guide covers everything you need to know about using Weights & Biases for experiment tracking in this Chest X-Ray Classification project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training with W&B](#training-with-wandb)
3. [Viewing Results](#viewing-results)
4. [Comparing Experiments](#comparing-experiments)
5. [Loading Saved Models](#loading-saved-models)
6. [W&B Dashboard Features](#wandb-dashboard-features)
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

### 2. Login to W&B
```bash
# On Windows (recommended)
python -m wandb login

# On Linux/Mac
wandb login
```
Follow the instructions to create a free account and get your API key.

### 3. Download Dataset
```bash
python main.py --download
```

### 4. Train with W&B
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

### 5. View Results
Results are automatically uploaded to your W&B dashboard at https://wandb.ai

### Quick Commands Reference

```bash
# Login to W&B
python -m wandb login  # or wandb login on Linux/Mac

# Train a model
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# Train with test evaluation
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --test

# Grid Search (all 162 combinations)
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml

# Grid Search (limited to 10)
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10

# Quick test
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --quick
```

---

## Training with W&B

### Basic Training
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20
```

### Advanced Options
```bash
python scripts/train_wandb.py \
    --dataset_path "Covid19-dataset" \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.0001 \
    --image_size 128 \
    --project_name "MyProject" \
    --run_name "baseline_model" \
    --entity "my-team" \
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
| `--project_name` | W&B project name | Chest-XRay-Classification-WB |
| `--run_name` | Name for this run | Auto-generated |
| `--entity` | W&B entity/team name | None (your personal account) |
| `--test` | Evaluate on test set | False |

### What Gets Tracked

#### Metrics (Automatically Logged)
- Training Loss (per epoch)
- Training Accuracy (per epoch)
- Validation Loss (per epoch)
- Validation Accuracy (per epoch)
- Per-class Precision, Recall, F1-score
- Test metrics (if `--test` flag used)
- System metrics (CPU, GPU, memory)

#### Parameters (Automatically Logged)
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer settings
- Model architecture

#### Artifacts (Automatically Saved)
- Full PyTorch model (as artifact)
- Confusion matrix (as image)
- Test metrics (if available)

---

## Viewing Results

### W&B Dashboard
1. Go to https://wandb.ai
2. Login to your account
3. Select your project
4. View runs, metrics, and visualizations

### W&B Dashboard Features

1. **Runs Table**: View all your runs in a table
2. **Metrics Visualization**: See training curves, validation metrics
3. **Parameters**: View hyperparameters for each run
4. **System Metrics**: CPU, GPU, memory usage
5. **Media**: View confusion matrices, sample predictions
6. **Artifacts**: Download models, datasets, etc.
7. **Compare**: Compare multiple runs side-by-side
8. **Sweeps**: Organize hyperparameter sweeps

### Navigation in W&B Dashboard

- **Projects**: Left sidebar shows all projects
- **Runs**: Each training session is a "run"
- **Metrics**: Click on a run to see detailed metrics
- **Compare**: Select multiple runs and click "Compare"
- **Download**: Download models, artifacts, or export data

---

## Comparing Experiments

### Compare Multiple Runs

1. Go to https://wandb.ai
2. Select your project
3. Select multiple runs (checkboxes)
4. Click "Compare" button
5. View side-by-side comparison of:
   - Parameters (hyperparameters)
   - Metrics (accuracy, loss, etc.)
   - Training curves
   - System metrics

### Example: Training Multiple Models

```bash
# Run 1: Baseline
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --run_name "baseline"

# Run 2: Higher learning rate
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --learning_rate 0.01 --run_name "lr_0.01"

# Run 3: Larger batch size
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 64 --run_name "batch_64"
```

Then compare all three runs in W&B dashboard.

---

## Loading Saved Models

### Using W&B to Load Models

```python
import wandb
import torch
import os
from src.models.cnn_model import CustomCXRClassifier

# Initialize W&B API
api = wandb.Api()

# Get a specific run
run = api.run("your-entity/your-project/run_id")

# Download model artifact
artifact = run.use_artifact("model:latest")
artifact_dir = artifact.download()

# Load model
model = CustomCXRClassifier(in_channels=3, num_classes=3)
model.load_state_dict(torch.load(os.path.join(artifact_dir, "model.pth")))
model.eval()
```

### Find Run ID

1. Go to W&B dashboard: https://wandb.ai
2. Click on a run
3. Copy the Run ID from the run URL or details page
4. Or use the W&B API:

```python
import wandb

# Get all runs in a project
api = wandb.Api()
runs = api.runs("your-entity/your-project")

# Get the best run
best_run = max(runs, key=lambda r: r.summary.get('val/accuracy', 0))
best_run_id = best_run.id
```

---

## W&B Dashboard Features

### 1. Metrics Tracking
- **Training Loss**: Tracked every epoch
- **Training Accuracy**: Tracked every epoch
- **Validation Loss**: Tracked every epoch
- **Validation Accuracy**: Tracked every epoch
- **Per-class Metrics**: Precision, Recall, F1-score for each class
- **Best Validation Accuracy**: Logged at the end
- **System Metrics**: CPU, GPU, memory usage

### 2. Parameters Logged
- Learning rate
- Batch size
- Number of epochs
- Image size
- Optimizer type
- Loss function
- Model architecture

### 3. Artifacts
- **Model**: Full PyTorch model saved as artifact
- **Confusion Matrix**: Saved as image
- **Test Metrics**: If `--test` flag is used

### 4. Run Information
- Run ID
- Run name (if provided)
- Start/End time
- Status
- Tags (customizable)
- Notes (add notes to runs)

---

## Hyperparameter Tuning

This section explains how to use the parameter matrix system for hyperparameter tuning with Weights & Biases.

### Quick Start

#### 1. Login to W&B
```bash
# On Windows (recommended)
python -m wandb login

# On Linux/Mac
wandb login
```

#### 2. Run with Default Configuration
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
```

#### 3. Run with Specific Config File
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml
```

#### 4. Run Quick Test (Fewer Experiments)
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --quick
```

#### 5. Run Limited Number of Experiments
```bash
# Run only 10 experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10
```

### Configuration Files

#### 1. `configs/wandb/hyperparameters.yaml` - Parameter Grid
Defines a grid of parameters to try. All combinations will be generated.

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]
  batch_size: [32, 64, 16]
  num_epochs: [20, 30, 50]
```

This will generate 3 × 3 × 3 = 27 experiments.

#### 2. `configs/wandb/experiments.yaml` - Specific Experiments
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

Edit `configs/wandb/hyperparameters.yaml`:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.005, 0.01]  # Add/remove values
  batch_size: [16, 32, 64, 128]        # Add more batch sizes
  num_epochs: [10, 20, 30]             # Modify epochs
  lr_gamma: [0.1, 0.5]                 # Add learning rate decay
```

**Note**: This generates ALL combinations. For 4×4×3×2 = 96 experiments!

#### Option 2: Define Specific Experiments (Recommended)

Edit `configs/wandb/experiments.yaml`:

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

1. Copy `configs/wandb/hyperparameters.yaml` to `configs/wandb/my_config.yaml`
2. Modify the parameters
3. Run: `python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/my_config.yaml`

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

#### W&B Configuration
```yaml
wandb_config:
  project_name: "Hyperparameter-Tuning-WB"
  entity: null  # Set to your team name if using teams
  use_run_names: true
  run_name_template: "lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}"
```

#### Execution Settings
```yaml
execution:
  # GRID SEARCH: Set max_experiments to null to run ALL combinations
  # Or set a number to limit experiments (useful for testing)
  max_experiments: null  # null = run all combinations, or set a number to limit
  shuffle: true  # Shuffle experiments before running (useful for parallel execution)
  continue_on_error: true  # Continue to next experiment if one fails
```

### Command Line Options

```bash
python scripts/run_wandb_hyperparameter_tuning.py [OPTIONS]

Options:
  --config PATH              Path to YAML config file (default: configs/wandb/hyperparameters.yaml)
  --quick                    Use quick test configuration (fewer experiments)
  --max-experiments N        Maximum number of experiments to run (overrides config)
  --shuffle                  Shuffle experiments before running (overrides config)
  --entity ENTITY            W&B entity/team name (overrides config)
```

### Workflow

1. **Plan Your Experiments**: Decide which parameters to tune
2. **Create/Edit Config File**: Use `configs/wandb/experiments.yaml` for specific experiments or `configs/wandb/hyperparameters.yaml` for grid search
3. **Login to W&B**: `python -m wandb login` or `wandb login`
4. **Run Experiments**: `python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml`
5. **Monitor Progress**: Watch terminal output and check W&B dashboard
6. **Analyze Results**: Compare experiments in W&B dashboard

### Tips

1. **Start Small**: Begin with a few experiments
2. **Use Descriptive Names**: Makes it easy to identify experiments
3. **Organize Experiments**: Create separate config files for different purposes
4. **Limit Experiments**: Use `max_experiments` to avoid running too many
5. **Use Teams/Entities**: Share projects with your team using `entity` parameter

---

## Grid Search

This section covers grid search hyperparameter tuning with W&B using `configs/wandb/hyperparameters.yaml`.

### Overview

The `configs/wandb/hyperparameters.yaml` file defines a parameter grid that generates **162 total combinations**:

- **learning_rate**: 3 values [0.001, 0.0001, 0.01]
- **batch_size**: 3 values [32, 64, 16]
- **num_epochs**: 3 values [20, 30, 50]
- **lr_gamma**: 2 values [0.1, 0.5]
- **lr_step_size**: 3 values [5, 7, 10]

**Total: 3 × 3 × 3 × 2 × 3 = 162 experiments**

### Running Grid Search

#### Option 1: Run All 162 Combinations
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
```

This will:
- Generate all 162 combinations
- Run all experiments (since `max_experiments: null`)
- Shuffle before running (if enabled)
- Continue on error
- Log all results to W&B dashboard

#### Option 2: Run Limited Number of Experiments
```bash
# Run only 10 experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10

# Run 50 experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 50

# Run 100 experiments
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 100
```

#### Option 3: Quick Test (1 experiment)
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --quick
```

This uses the `quick_test` configuration which runs only 1 experiment for quick testing.

### Modifying the Grid

Edit `configs/wandb/hyperparameters.yaml` to change parameter ranges:

```yaml
parameter_grid:
  learning_rate: [0.001, 0.0001, 0.01]      # 3 values
  batch_size: [32, 64, 16]                   # 3 values
  num_epochs: [20, 30, 50]                   # 3 values
  lr_gamma: [0.1, 0.5]                       # 2 values
  lr_step_size: [5, 7, 10]                  # 3 values
```

**Note**: Adding more values increases the total combinations exponentially!

**Example**: If you add one more learning rate value:
- New total: 4 × 3 × 3 × 2 × 3 = 216 experiments

### Configuration Options

#### Limit Experiments in Config File

Edit `configs/wandb/hyperparameters.yaml`:

```yaml
execution:
  max_experiments: 10  # Limit to 10 experiments
  shuffle: true
  continue_on_error: true
```

#### Run All Combinations

Set `max_experiments: null` in config or use command line:

```yaml
execution:
  max_experiments: null  # null = run all combinations
  shuffle: true
  continue_on_error: true
```

### Recommended Workflow

1. **Start Small**: Run quick test first
   ```bash
   python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --quick
   ```

2. **Limited Search**: Run 10-20 experiments to get initial results
   ```bash
   python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 20
   ```

3. **Analyze Results**: View in W&B dashboard
   - Go to https://wandb.ai
   - Select your project: "Hyperparameter-Tuning-WB"
   - Compare runs and identify best hyperparameters

4. **Refine Grid**: Based on results, narrow down parameter ranges

5. **Full Search**: If needed, run all combinations
   ```bash
   python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
   ```

### Viewing Results

After running experiments:

1. Go to https://wandb.ai
2. Login to your account
3. Select your project: "Hyperparameter-Tuning-WB"
4. View all runs in the dashboard

#### W&B Dashboard Features

- **Runs Table**: View all experiments in a sortable table
- **Metrics Visualization**: Compare training curves, validation metrics
- **Parameters**: View hyperparameters for each run
- **Compare**: Select multiple runs and compare side-by-side
- **Filter**: Filter runs by parameters or metrics
- **Export**: Export results to CSV or JSON

#### Comparing Experiments

1. Select multiple runs (checkboxes)
2. Click "Compare" button
3. View side-by-side:
   - Parameters (hyperparameters)
   - Metrics (accuracy, loss, etc.)
   - Training curves
   - System metrics

### Tips

1. **Monitor Progress**: Watch terminal output for each experiment
2. **Use Shuffle**: Helps avoid bias from running order
3. **Continue on Error**: Allows other experiments to run even if one fails
4. **Start Small**: Begin with fewer experiments to test configuration
5. **Analyze Early**: Check results after initial runs to refine grid
6. **Use Descriptive Run Names**: The config uses templates like `lr_{learning_rate}_bs_{batch_size}_ep_{num_epochs}`

### Example: Running 50 Experiments

```bash
python scripts/run_wandb_hyperparameter_tuning.py \
  --config configs/wandb/hyperparameters.yaml \
  --max-experiments 50 \
  --shuffle
```

This will:
- Generate all 162 combinations
- Randomly select 50 experiments (due to shuffle)
- Shuffle before running
- Track all results in W&B dashboard
- Continue even if one experiment fails

---

## Examples

### Example 1: Simple Experiment Configuration

**File: `configs/wandb/experiments.yaml`**

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
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml
```

**Result:** 3 experiments will run automatically, all tracked in W&B dashboard!

### Example 2: Parameter Grid (All Combinations)

**File: `configs/wandb/hyperparameters.yaml`**

```yaml
parameter_grid:
  learning_rate: [0.001, 0.01]
  batch_size: [32, 64]
  num_epochs: [20, 30]
```

**Run:**
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
```

**Result:** 2 × 2 × 2 = 8 experiments will run (all combinations)

### Example 3: Learning Rate Sweep

**File: `configs/wandb/learning_rate_sweep.yaml`**

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

wandb_config:
  project_name: "Learning-Rate-Sweep-WB"
  entity: null
  use_run_names: true

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
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/learning_rate_sweep.yaml
```

### Example 4: Batch Size Comparison

**File: `configs/wandb/batch_size_comparison.yaml`**

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

### Example 5: Team Collaboration

**Using Teams/Entities**

```yaml
wandb_config:
  project_name: "Shared-Project-WB"
  entity: "my-team"  # Your team name
  use_run_names: true
```

**Run:**
```bash
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/experiments.yaml --entity "my-team"
```

### W&B Specific Features

#### 1. Real-time Monitoring
- View experiments in real-time as they run
- See system metrics (CPU, GPU, memory)
- Monitor training progress live

#### 2. Team Collaboration
- Share projects with your team
- Compare experiments across team members
- Collaborate on hyperparameter tuning

#### 3. Artifacts
- Download models directly from W&B
- Version control for models
- Share models with team members

#### 4. Sweeps
- Use W&B sweeps for automated hyperparameter search
- Grid search, random search, Bayesian optimization
- Advanced hyperparameter tuning

---

## Best Practices

### 1. Use Descriptive Run Names
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --run_name "baseline_lr0.001_bs32"
```

### 2. Organize Projects
```bash
# Different projects for different purposes
python scripts/train_wandb.py --project_name "Hyperparameter-Tuning" --run_name "trial_1"
python scripts/train_wandb.py --project_name "Model-Architecture" --run_name "resnet50"
```

### 3. Use Teams/Entities
```bash
# Share projects with your team
python scripts/train_wandb.py --entity "my-team" --project_name "Shared-Project"
```

### 4. Add Tags
You can add tags to runs programmatically:
```python
wandb.init(tags=["baseline", "CNN", "COVID-19"])
```

### 5. Regular Evaluation
Always use `--test` flag to evaluate on test set:
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --test
```

### 6. Start Small with Grid Search
- Begin with a few experiments
- Use `--quick` flag for testing
- Gradually expand parameter space

### 7. Compare Multiple Runs
Use W&B dashboard to compare all experiments easily and identify best hyperparameters.

---

## Advanced Usage

### Custom Tracking

```python
import wandb

# Initialize W&B
wandb.init(project="MyProject", name="custom_run")

# Log custom metrics
wandb.log({"custom_metric": value})

# Log images
wandb.log({"sample_images": [wandb.Image(img) for img in images]})

# Log tables
wandb.log({"predictions": wandb.Table(data=table_data)})
```

### Query Runs Programmatically

```python
import wandb

# Initialize API
api = wandb.Api()

# Get runs
runs = api.runs("your-entity/your-project")

# Filter runs
good_runs = [r for r in runs if r.summary.get('val/accuracy', 0) > 0.8]

# Get best run
best_run = max(runs, key=lambda r: r.summary.get('val/accuracy', 0))
print(f"Best accuracy: {best_run.summary['val/accuracy']}")
print(f"Run ID: {best_run.id}")
```

### Export Results

```python
import wandb
import pandas as pd

# Initialize API
api = wandb.Api()

# Get runs
runs = api.runs("your-entity/your-project")

# Convert to DataFrame
runs_df = pd.DataFrame([r.summary for r in runs])
runs_df.to_csv("wandb_results.csv", index=False)
```

---

## Troubleshooting

### W&B Not Logging
- Make sure you're logged in: `python -m wandb login` or `wandb login`
- Check internet connection (W&B requires internet)
- Verify API key is correct

### Cannot Find Runs
- Check project name matches
- Verify you're logged into the correct W&B account
- Check entity/team name if using teams

### Model Loading Issues
- Ensure you're using the same PyTorch version
- Check that the model architecture matches
- Verify the run_id is correct

### Offline Mode
If you're offline, W&B can run in offline mode:
```bash
# On Windows
python -m wandb offline

# On Linux/Mac
wandb offline
```
Sync later with:
```bash
# On Windows
python -m wandb sync

# On Linux/Mac
wandb sync
```

### Too Many Experiments
- Use `--max-experiments N` to limit (e.g., `--max-experiments 10`)
- Edit config to reduce parameter grid
- Use specific experiments instead of grid
- Set `max_experiments: 10` in config file

### Experiments Failing
- Check dataset path is correct
- Verify parameters are valid
- Set `continue_on_error: true` to continue

### Config File Errors
- Check YAML syntax
- Verify all required parameters are present
- Use a YAML validator

### Can't Find Runs in Dashboard
- Check project name matches: "Hyperparameter-Tuning-WB"
- Verify you're logged into correct W&B account
- Check entity/team name if using teams

---

## Example Workflow

### Complete Training and Evaluation Workflow

```bash
# 1. Login to W&B
python -m wandb login  # or wandb login on Linux/Mac

# 2. Train model
python scripts/train_wandb.py \
    --dataset_path "Covid19-dataset" \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --run_name "final_model" \
    --test

# 3. View results in W&B dashboard
# Go to https://wandb.ai and select your project
```

---

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [W&B API Reference](https://docs.wandb.ai/ref/python/api)

---

## Quick Reference

```bash
# Login to W&B
python -m wandb login  # or wandb login on Linux/Mac

# Train with W&B
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20

# Grid Search (162 combinations)
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml

# Grid Search (limited to 10)
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10

# Quick test
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --quick

# View results
# Go to https://wandb.ai

# Compare experiments
# Select multiple runs in W&B dashboard and click "Compare"

# Load model
# Use W&B API: wandb.Api().run("entity/project/run_id")
```

