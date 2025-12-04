# Experimental Evaluation: MLflow vs. Weights & Biases (W&B)

## Project Goal

This project provides an **experimental evaluation and comparison** of **MLflow** and **Weights & Biases (W&B)** as experiment tracking tools for machine learning workflows. The evaluation focuses on:

- **User-friendliness**: Ease of setup, usage, and navigation
- **Feature set**: Capabilities, visualization, and tooling
- **Integration**: How well each tool integrates into an existing ML workflow

## Approach

To ensure a **fair and comprehensive comparison**, this project:

1. **Uses a real ML use case**: Chest X-Ray Classification (3-class: COVID-19, Viral Pneumonia, Normal)
2. **Implements automated grid search**: Runs the same hyperparameter combinations on both tools
3. **Generates sufficient data**: Executes 162+ experiments per tool for statistical significance
4. **Maintains consistency**: Same model architecture, dataset, and hyperparameters across both tools

This enables a direct, apples-to-apples comparison of MLflow and W&B under identical experimental conditions.

## ML Use Case: Chest X-Ray Classification

**Dataset**: COVID-19 Image Dataset
- **Source**: Kaggle (pranavraikokte/covid19-image-dataset)
- **Task**: 3-Way Classification
- **Classes**: 
  - COVID-19
  - Viral Pneumonia
  - Normal

This real-world medical imaging classification task serves as the experimental use case for comparing MLflow and W&B. The complexity and practical nature of this problem make it ideal for evaluating how each tool handles a production-like ML workflow.

## Project Structure

```
.
├── src/                          # Core source code
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
│       └── __init__.py
│
├── scripts/                      # Training and execution scripts
│   ├── train_mlflow.py           # Train with MLflow
│   ├── train_wandb.py            # Train with W&B
│   ├── compare_mlflow_wandb.py   # Compare both tools
│   ├── run_hyperparameter_tuning.py      # MLflow hyperparameter tuning
│   ├── run_wandb_hyperparameter_tuning.py # W&B hyperparameter tuning
│   └── start_mlflow_ui.py        # Start MLflow UI
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
│   ├── mlflow/                   # MLflow documentation
│   ├── wandb/                    # W&B documentation
│   └── examples/                 # Example documentation
│
├── tests/                        # Unit tests
│   └── __init__.py
│
├── notebooks/                    # Jupyter notebooks (optional)
│
├── Covid19-dataset/              # Dataset directory
│   ├── train/
│   └── test/
│
├── mlruns/                       # MLflow runs (gitignored)
├── wandb/                        # W&B runs (gitignored)
│
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── PROJECT_STRUCTURE.md          # Detailed structure documentation
└── README.md                     # This file
```

**See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure documentation.**

## Model Architecture

The project uses a custom CNN architecture (`CustomCXRClassifier`) designed for Chest X-Ray classification. The model architecture is based on the implementation from [Vinay10100/Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification).
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/655d4102-d1ae-4006-8591-800dcecd7f3e" />

### Architecture Details:

- **Input**: RGB images (128x128 pixels)
- **Architecture**: 
  - Convolutional layer 1: 16 filters, 3x3 kernel, ReLU activation, followed by MaxPooling2D (2x2 pool size)
  - Convolutional layer 2: 64 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.25)
  - Convolutional layer 3: 128 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.3)
  - Convolutional layer 4: 128 filters, 3x3 kernel, ReLU activation, padding='same', followed by MaxPooling2D (2x2 pool size), Dropout (0.4)
  - Flatten layer
  - Dense layer 1: 128 neurons, ReLU activation, Dropout (0.25)
  - Dense layer 2: 64 neurons, ReLU activation
  - Output layer: 3 neurons (one for each class), softmax activation
- **Output**: 3 classes (COVID-19, Viral Pneumonia, Normal)
- **Features**: Dropout regularization, MaxPooling, Fully Connected layers

**Note**: This architecture has been adapted from the original implementation to work with PyTorch and integrated with MLflow and W&B for experiment tracking.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Evaluation-of-MLflow-vs.-W-B-for-Chest-X-Ray-Classification-main
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle API (for dataset download)

1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account → API → Create New Token
3. Download `kaggle.json` and place it in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)

### 4. Set up W&B (optional, for W&B tracking)

```bash
wandb login
```

Follow the instructions to create a free account and get your API key.

## Usage

### 1. Download Dataset

```bash
python main.py --download
```

This will download the COVID-19 Image Dataset from Kaggle to your local directory.

### 2. Train with MLflow

```bash
python scripts/train_mlflow.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 32
```

**Note**: If you install the package (`pip install -e .`), you can also use:
```bash
train-mlflow --dataset_path "Covid19-dataset" --epochs 20
```

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--experiment_name`: MLflow experiment name (default: Chest-XRay-Classification-MLflow)
- `--run_name`: MLflow run name (optional)
- `--test`: Evaluate on test set after training

**View MLflow UI:**
```bash
# On Windows (recommended)
python -m mlflow ui

# On Linux/Mac
mlflow ui
```
Then open http://localhost:5000 in your browser.

**Note**: If `mlflow` command is not found, use `python -m mlflow ui` instead.

**📖 For detailed MLflow usage instructions, see [docs/MLFLOW_COMPLETE_GUIDE.md](docs/MLFLOW_COMPLETE_GUIDE.md)**

### 2.1. Automated Grid Search for Comparison

**Key Feature**: This project uses **automated grid search** to run the **same hyperparameter combinations** on both MLflow and W&B, enabling a fair comparison.

**Grid Search Configuration:**
- **162 total combinations** across 5 hyperparameters:
  - Learning rate: 3 values [0.001, 0.0001, 0.01]
  - Batch size: 3 values [32, 64, 16]
  - Epochs: 3 values [20, 30, 50]
  - LR gamma: 2 values [0.1, 0.5]
  - LR step size: 3 values [5, 7, 10]

**Run Grid Search:**
```bash
# Run all 162 combinations (default)
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml

# Run limited number for testing
python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml --max-experiments 10

# Quick test (1 experiment)
python scripts/run_hyperparameter_tuning.py --quick
```

**Why Grid Search?**
- Ensures **identical experimental conditions** for both tools
- Generates **sufficient data** (162+ runs) for meaningful comparison
- Automates the process, eliminating manual bias
- Enables statistical analysis of tracking tool performance

**📖 See [docs/MLFLOW_COMPLETE_GUIDE.md](docs/MLFLOW_COMPLETE_GUIDE.md) for detailed instructions (Grid Search section)**

### 3. Train with W&B

**First, login to W&B:**
```bash
wandb login
```

**Then train:**
```bash
python scripts/train_wandb.py --dataset_path "Covid19-dataset" --epochs 20 --batch_size 32
```

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--project_name`: W&B project name (default: Chest-XRay-Classification-WB)
- `--run_name`: W&B run name (optional)
- `--entity`: W&B entity/team name (optional)
- `--test`: Evaluate on test set after training

**View Results:**
Results are automatically uploaded to your W&B dashboard at https://wandb.ai

**📖 For detailed W&B usage instructions, see [docs/WANDB_COMPLETE_GUIDE.md](docs/WANDB_COMPLETE_GUIDE.md)**

### 3.1. Automated Grid Search for Comparison

**Same Grid Search as MLflow**: The W&B implementation uses the **identical grid search configuration** (162 combinations) to ensure fair comparison.

**Run Grid Search:**
```bash
# Run all 162 combinations (default)
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml

# Run limited number for testing
python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml --max-experiments 10

# Quick test (1 experiment)
python scripts/run_wandb_hyperparameter_tuning.py --quick
```

**Comparison Benefits:**
- **Identical experiments** run on both tools
- **Same hyperparameters** tested across platforms
- **Consistent evaluation** of user experience and features
- **Quantitative data** for objective comparison

**📖 See [docs/WANDB_COMPLETE_GUIDE.md](docs/WANDB_COMPLETE_GUIDE.md) for detailed instructions (Grid Search section)**

### 4. Direct Comparison: MLflow vs W&B

**Automated Comparison Script:**
```bash
python scripts/compare_mlflow_wandb.py --dataset_path "Covid19-dataset" --epochs 10
```

This script runs the **same experiment** with both tracking tools side-by-side and provides a direct comparison of:
- **Training time**: Performance overhead of each tool
- **Model performance metrics**: Accuracy, loss, per-class metrics
- **Best validation accuracy**: Model quality comparison
- **Test set performance**: Generalization comparison
- **User experience**: Setup complexity, ease of use
- **Feature comparison**: Visualization, collaboration, integration

**For Comprehensive Comparison:**
After running grid search on both tools (162 experiments each), you can:
1. Compare visualization quality and clarity
2. Evaluate ease of experiment navigation
3. Assess collaboration features
4. Analyze integration complexity
5. Review feature richness and usefulness

**Options:**
- `--dataset_path`: Path to the dataset directory (required)
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--image_size`: Image size for resizing (default: 128)
- `--device`: Device to use (default: cuda if available, else cpu)
- `--mlflow_experiment`: MLflow experiment name
- `--wandb_project`: W&B project name
- `--wandb_entity`: W&B entity/team name (optional)
- `--skip_mlflow`: Skip MLflow experiment
- `--skip_wandb`: Skip W&B experiment

## Dataset Structure

The dataset should be organized as follows:

```
dataset_path/
├── COVID-19/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Viral Pneumonia/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Normal/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

The data loader automatically handles variations in folder names (case-insensitive matching).

## Features Tracked

### MLflow
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Training and validation metrics (loss, accuracy)
- Per-class metrics (precision, recall, F1-score)
- Model artifacts
- Confusion matrix
- Best model checkpoint

### W&B
- Hyperparameters (learning rate, batch size, epochs, etc.)
- Training and validation metrics (loss, accuracy)
- Per-class metrics (precision, recall, F1-score)
- Real-time metrics visualization
- Confusion matrix plots
- Model artifacts
- Gradient and parameter tracking
- Learning rate scheduling

## Experimental Comparison: MLflow vs W&B

This project enables you to **empirically evaluate** both tools through:

### Comparison Dimensions

1. **User-Friendliness**
   - Setup complexity and time
   - Ease of navigation and finding experiments
   - Learning curve and documentation quality
   - Command-line vs GUI experience

2. **Feature Set**
   - Visualization quality and customization
   - Experiment comparison capabilities
   - Model management and versioning
   - Integration with other tools
   - Real-time monitoring and alerts

3. **Integration into Workflow**
   - Code changes required
   - API simplicity and flexibility
   - Workflow disruption
   - Scalability for large teams
   - Local vs cloud deployment options

### Quick Comparison Overview

**MLflow**
- ✅ Local tracking by default (no account required)
- ✅ Simple UI: `mlflow ui`
- ✅ Good for local experiments and model registry
- ✅ Open-source and self-hostable
- ⚠️ Basic visualization compared to W&B
- ⚠️ Limited collaboration features
- ⚠️ No real-time monitoring

**W&B**
- ✅ Rich visualization and collaboration features
- ✅ Real-time monitoring and alerts
- ✅ Advanced experiment comparison tools
- ✅ Cloud-based (accessible from anywhere)
- ⚠️ Requires account (free tier available)
- ⚠️ Cloud-based (may require internet)
- ⚠️ More complex setup for self-hosting

**Note**: Run the grid search experiments on both tools to form your own empirical conclusions based on your specific use case and requirements.

## Experimental Results & Analysis

### Running the Full Comparison

1. **Execute Grid Search on Both Tools:**
   ```bash
   # MLflow: Run 162 experiments
   python scripts/run_hyperparameter_tuning.py --config configs/mlflow/hyperparameters.yaml
   
   # W&B: Run 162 experiments (same hyperparameters)
   python scripts/run_wandb_hyperparameter_tuning.py --config configs/wandb/hyperparameters.yaml
   ```

2. **Compare Results:**
   - **MLflow UI**: `python -m mlflow ui` → http://localhost:5000
   - **W&B Dashboard**: https://wandb.ai → Select your project

3. **Evaluate:**
   - **User Experience**: Which tool is easier to navigate and use?
   - **Visualization**: Which provides clearer, more useful visualizations?
   - **Features**: Which tool offers features most valuable for your workflow?
   - **Integration**: Which integrates more smoothly into your existing setup?

### What to Compare

After running experiments, evaluate:

- **Setup & Onboarding**: Time and complexity to get started
- **Experiment Management**: Ease of organizing and finding runs
- **Visualization Quality**: Clarity and usefulness of charts and graphs
- **Comparison Tools**: Ability to compare multiple experiments effectively
- **Collaboration**: Team sharing and collaboration features
- **Performance**: Overhead and impact on training time
- **Scalability**: Handling of large numbers of experiments
- **Documentation**: Quality and completeness of guides

### Expected Outcomes

This experimental evaluation will help you:
- **Make informed decisions** about which tool fits your needs
- **Understand trade-offs** between local vs cloud solutions
- **Evaluate features** in the context of your actual workflow
- **Compare user experience** through hands-on usage

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- 8GB+ RAM recommended
- Sufficient disk space for dataset and model artifacts

## Troubleshooting

### Dataset Download Issues
- Ensure Kaggle API credentials are set up correctly
- Check that `kagglehub` is installed: `pip install kagglehub`
- Verify internet connection

### W&B Login Issues
- Run `wandb login` and follow the instructions
- Ensure you have a W&B account (free tier is available)

### CUDA/GPU Issues
- Install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Memory Issues
- Reduce batch size: `--batch_size 16`
- Reduce image size: `--image_size 64`
- Use CPU if GPU memory is limited: `--device cpu`

## Research & Evaluation Purpose

This project is designed for **experimental evaluation and research purposes**. It provides:

- A **systematic approach** to comparing ML experiment tracking tools
- **Reproducible methodology** for fair tool comparison
- **Real-world use case** (medical imaging) for practical evaluation
- **Automated grid search** for comprehensive data collection
- **Objective framework** for evaluating user-friendliness, features, and integration

The goal is to help researchers, practitioners, and teams make **data-driven decisions** when choosing between MLflow and W&B for their ML workflows.

## License

This project is for educational and research purposes.

## Citation

If you use this project, please cite:
- COVID-19 Image Dataset: [Kaggle Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- Model Architecture: [Vinay10100/Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification)
- MLflow: [MLflow Documentation](https://mlflow.org/)
- Weights & Biases: [W&B Documentation](https://wandb.ai/)

## Author

Evaluation of MLflow vs. W&B for Chest X-Ray Classification

## Acknowledgments

- [Vinay10100](https://github.com/Vinay10100) for the original CNN architecture implementation in [Chest-X-Ray-Classification](https://github.com/Vinay10100/Chest-X-Ray-Classification)
- Kaggle for hosting the COVID-19 Image Dataset
- MLflow team for the excellent experiment tracking tool
- Weights & Biases team for the comprehensive MLOps platform
