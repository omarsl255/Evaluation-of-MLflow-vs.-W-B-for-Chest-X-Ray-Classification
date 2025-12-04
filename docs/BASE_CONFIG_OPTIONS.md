# Base Config Options Reference

This document lists all available options for `base_config` in the hyperparameters YAML files.

## Complete Base Config Options

### Dataset Configuration

```yaml
base_config:
  # Dataset path (required)
  dataset_path: "Covid19-dataset"  # Path to your dataset folder
  # Options: Any valid path string
  # Examples:
  #   dataset_path: "Covid19-dataset"
  #   dataset_path: "data/chest-xray"
  #   dataset_path: "C:/Users/YourName/Datasets/Covid19"
```

### Image Processing

```yaml
  # Image size for resizing (all images will be resized to this)
  image_size: 128
  # Options: Common values
  #   image_size: 64    # Smaller, faster training
  #   image_size: 128   # Default, balanced
  #   image_size: 224   # Larger, more detail (slower)
  #   image_size: 256   # High resolution (slower, more memory)
  #   image_size: 512   # Very high resolution (very slow)
```

### Device Selection

```yaml
  # Computing device
  device: "auto"  # Recommended: automatically detects GPU/CPU
  # Options:
  #   device: "auto"   # Automatically use CUDA if available, else CPU
  #   device: "cuda"   # Force GPU (will fail if no GPU available)
  #   device: "cpu"   # Force CPU (slower but always works)
  #   device: "cuda:0"  # Use specific GPU (if multiple GPUs)
  #   device: "cuda:1"  # Use second GPU
```

### Data Splitting

```yaml
  # Data split ratios (must sum to 1.0)
  train_split: 0.8   # 80% for training
  val_split: 0.1     # 10% for validation
  test_split: 0.1    # 10% for testing
  
  # Options: Any values that sum to 1.0
  # Examples:
  #   train_split: 0.7, val_split: 0.15, test_split: 0.15  # More validation/test
  #   train_split: 0.85, val_split: 0.1, test_split: 0.05  # More training data
  #   train_split: 0.9, val_split: 0.05, test_split: 0.05   # Maximum training data
```

### Random Seed

```yaml
  # Random seed for reproducibility
  random_seed: 42
  # Options: Any integer
  # Examples:
  #   random_seed: 42    # Default
  #   random_seed: 123    # Different seed
  #   random_seed: 2024  # Year-based seed
  #   random_seed: 0     # Zero seed
```

### Optimizer

```yaml
  # Optimizer type
  optimizer: "Adam"
  # Options: Supported optimizers
  #   optimizer: "Adam"        # Default, good for most cases
  #   optimizer: "SGD"         # Stochastic Gradient Descent
  #   optimizer: "AdamW"       # Adam with weight decay
  #   optimizer: "RMSprop"     # RMSprop optimizer
  #   optimizer: "Adagrad"     # Adagrad optimizer
  #   optimizer: "Adadelta"    # Adadelta optimizer
```

### Loss Function

```yaml
  # Loss function
  loss_function: "CrossEntropyLoss"
  # Options: Supported loss functions
  #   loss_function: "CrossEntropyLoss"  # Default, standard for classification
  #   loss_function: "NLLLoss"           # Negative Log Likelihood Loss
  # Note: Other loss functions may require additional configuration
```

### Learning Rate Schedule

```yaml
  # Learning rate scheduler parameters
  # These are base values - can be overridden in parameter_grid
  
  # Step size for learning rate decay (epochs between reductions)
  lr_step_size: 7
  # Options: Any positive integer
  # Examples:
  #   lr_step_size: 5    # Decay every 5 epochs (more frequent)
  #   lr_step_size: 7    # Default
  #   lr_step_size: 10   # Decay every 10 epochs (less frequent)
  #   lr_step_size: 15   # Decay every 15 epochs
  
  # Learning rate decay factor (multiply LR by this value)
  lr_gamma: 0.1
  # Options: Float between 0 and 1
  # Examples:
  #   lr_gamma: 0.1      # Default, reduce to 10% (strong decay)
  #   lr_gamma: 0.5      # Reduce to 50% (moderate decay)
  #   lr_gamma: 0.9      # Reduce to 90% (gentle decay)
  #   lr_gamma: 0.2      # Reduce to 20% (moderate-strong decay)
```

### Evaluation

```yaml
  # Whether to evaluate on test set after training
  test_after_training: true
  # Options: Boolean
  #   test_after_training: true   # Evaluate on test set (recommended)
  #   test_after_training: false   # Skip test evaluation (faster)
```

## Complete Example Configurations

### Example 1: Fast Training (Small Images, CPU)
```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 64              # Smaller images = faster
  device: "cpu"                # Force CPU
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
  lr_step_size: 5              # More frequent decay
  lr_gamma: 0.5                # Moderate decay
  test_after_training: true
```

### Example 2: High Quality Training (Large Images, GPU)
```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 224              # Larger images = more detail
  device: "cuda"               # Force GPU
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  optimizer: "AdamW"           # Adam with weight decay
  loss_function: "CrossEntropyLoss"
  lr_step_size: 10             # Less frequent decay
  lr_gamma: 0.1                # Strong decay
  test_after_training: true
```

### Example 3: Maximum Training Data
```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"
  train_split: 0.9             # 90% for training
  val_split: 0.05              # 5% for validation
  test_split: 0.05             # 5% for testing
  random_seed: 42
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
  lr_step_size: 7
  lr_gamma: 0.1
  test_after_training: true
```

### Example 4: More Validation Data
```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"
  train_split: 0.7             # 70% for training
  val_split: 0.2               # 20% for validation (more reliable validation)
  test_split: 0.1              # 10% for testing
  random_seed: 42
  optimizer: "SGD"             # SGD optimizer
  loss_function: "CrossEntropyLoss"
  lr_step_size: 7
  lr_gamma: 0.1
  test_after_training: true
```

### Example 5: Gentle Learning Rate Schedule
```yaml
base_config:
  dataset_path: "Covid19-dataset"
  image_size: 128
  device: "auto"
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_seed: 42
  optimizer: "Adam"
  loss_function: "CrossEntropyLoss"
  lr_step_size: 10             # Less frequent
  lr_gamma: 0.9                # Gentle decay (90% of previous)
  test_after_training: true
```

## Parameter Grid vs Base Config

**Base Config**: Parameters that are the same for ALL experiments
- These are fixed values applied to every experiment
- Examples: `dataset_path`, `image_size`, `device`, `optimizer`

**Parameter Grid**: Parameters that vary between experiments
- These generate different combinations
- Examples: `learning_rate`, `batch_size`, `num_epochs`, `lr_gamma`, `lr_step_size`

**Note**: If a parameter appears in both `base_config` and `parameter_grid`, the `parameter_grid` value will override the `base_config` value for that specific experiment.

## Quick Reference Table

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `dataset_path` | string | "Covid19-dataset" | Any valid path | Path to dataset folder |
| `image_size` | int | 128 | 64, 128, 224, 256, 512 | Image resize dimension |
| `device` | string | "auto" | "auto", "cuda", "cpu", "cuda:0" | Computing device |
| `train_split` | float | 0.8 | 0.0-1.0 | Training data proportion |
| `val_split` | float | 0.1 | 0.0-1.0 | Validation data proportion |
| `test_split` | float | 0.1 | 0.0-1.0 | Test data proportion |
| `random_seed` | int | 42 | Any integer | Random seed for reproducibility |
| `optimizer` | string | "Adam" | "Adam", "SGD", "AdamW", "RMSprop" | Optimizer type |
| `loss_function` | string | "CrossEntropyLoss" | "CrossEntropyLoss", "NLLLoss" | Loss function |
| `lr_step_size` | int | 7 | Any positive integer | Epochs between LR decay |
| `lr_gamma` | float | 0.1 | 0.0-1.0 | LR decay factor |
| `test_after_training` | bool | true | true, false | Evaluate on test set |

## Tips

1. **Image Size**: Larger images = better quality but slower training and more memory
2. **Device**: Use "auto" unless you have a specific reason to force CPU/GPU
3. **Splits**: Ensure train_split + val_split + test_split = 1.0
4. **Random Seed**: Use the same seed for reproducible results
5. **Optimizer**: "Adam" is usually best for most cases
6. **LR Schedule**: Smaller `lr_step_size` = more frequent decay, smaller `lr_gamma` = stronger decay


