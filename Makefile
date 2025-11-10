.PHONY: help install download clean test
.DEFAULT_GOAL := help

# Variables
PYTHON := python
DATASET_PATH := Covid19-dataset
EPOCHS := 20
BATCH_SIZE := 32

help: ## Show this help message
	@echo "Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make wandb-quick          # Quick W&B test"
	@echo "  make wandb-tune           # Full W&B hyperparameter tuning"
	@echo "  make mlflow-quick         # Quick MLflow test"
	@echo "  make compare              # Compare MLflow vs W&B"

# Installation & Setup
install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install package in development mode
	pip install -e .

download: ## Download dataset from Kaggle
	$(PYTHON) main.py --download

wandb-login: ## Login to Weights & Biases
	wandb login

# W&B Training
wandb-train: ## Train model with W&B (default: 20 epochs)
	PYTHONPATH=. $(PYTHON) scripts/train_wandb.py --dataset_path $(DATASET_PATH) --epochs $(EPOCHS) --batch_size $(BATCH_SIZE)

wandb-quick: ## Quick W&B hyperparameter test (1 experiment, 10 epochs)
	PYTHONPATH=. $(PYTHON) scripts/run_wandb_hyperparameter_tuning.py --quick

wandb-tune: ## Full W&B hyperparameter tuning (10 experiments)
	PYTHONPATH=. $(PYTHON) scripts/run_wandb_hyperparameter_tuning.py

wandb-tune-all: ## Run all W&B hyperparameter combinations
	PYTHONPATH=. $(PYTHON) scripts/run_wandb_hyperparameter_tuning.py --max-experiments 999

# MLflow Training
mlflow-train: ## Train model with MLflow (default: 20 epochs)
	PYTHONPATH=. $(PYTHON) scripts/train_mlflow.py --dataset_path $(DATASET_PATH) --epochs $(EPOCHS) --batch_size $(BATCH_SIZE)

mlflow-quick: ## Quick MLflow hyperparameter test (1 experiment, 10 epochs)
	PYTHONPATH=. $(PYTHON) scripts/run_hyperparameter_tuning.py --quick

mlflow-tune: ## Full MLflow hyperparameter tuning (10 experiments)
	PYTHONPATH=. $(PYTHON) scripts/run_hyperparameter_tuning.py

mlflow-tune-all: ## Run all MLflow hyperparameter combinations
	PYTHONPATH=. $(PYTHON) scripts/run_hyperparameter_tuning.py --max-experiments 999

mlflow-ui: ## Start MLflow UI (visit http://localhost:5000)
	mlflow ui

# Comparison
compare: ## Compare MLflow vs W&B (10 epochs each)
	PYTHONPATH=. $(PYTHON) scripts/compare_mlflow_wandb.py --dataset_path $(DATASET_PATH) --epochs 10

compare-full: ## Compare MLflow vs W&B (full training)
	PYTHONPATH=. $(PYTHON) scripts/compare_mlflow_wandb.py --dataset_path $(DATASET_PATH) --epochs $(EPOCHS)

# Testing & Validation
test: ## Run tests
	pytest tests/

# Cleanup
clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-runs: ## Clean up MLflow and W&B run directories
	@read -p "Are you sure you want to delete mlruns/ and wandb/? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf mlruns/ wandb/; \
		echo "Cleaned up run directories"; \
	fi

clean-all: clean clean-runs ## Clean everything (cache, runs, temp files)

# Custom training with parameters
train-custom: ## Custom W&B training (use: make train-custom EPOCHS=50 BATCH_SIZE=64)
	PYTHONPATH=. $(PYTHON) scripts/train_wandb.py \
		--dataset_path $(DATASET_PATH) \
		--epochs $(EPOCHS) \
		--batch_size $(BATCH_SIZE)

# Quick commands
quick-test: wandb-quick ## Alias for wandb-quick

full-test: wandb-tune mlflow-tune ## Run both W&B and MLflow hyperparameter tuning
