"""
Unit tests for MLflow and W&B tracking functionality
"""

import unittest
import os
import tempfile
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import Mock, patch, MagicMock

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.tracking.mlflow_tracker import MLflowTracker, train_with_mlflow
from src.tracking.wandb_tracker import WandBTracker, train_with_wandb
from src.models.cnn_model import CustomCXRClassifier


class TestMLflowTracker(unittest.TestCase):
    """Test cases for MLflowTracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment_name = "test-experiment"
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_tracker_initialization(self):
        """Test MLflowTracker initialization"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        self.assertEqual(tracker.experiment_name, self.experiment_name)
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_tracker_initialization_custom_uri(self, mock_mlflow):
        """Test MLflowTracker with custom tracking URI"""
        # Mock MLflow to avoid Windows file:// URI issues
        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "0"
        
        tracking_uri = self.temp_dir
        tracker = MLflowTracker(
            experiment_name=self.experiment_name,
            tracking_uri=tracking_uri
        )
        self.assertEqual(tracker.experiment_name, self.experiment_name)
        mock_mlflow.set_tracking_uri.assert_called_once_with(tracking_uri)
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_start_run(self, mock_mlflow):
        """Test start_run method"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        run_name = "test-run"
        tags = {"tag1": "value1"}
        
        tracker.start_run(run_name=run_name, tags=tags)
        mock_mlflow.start_run.assert_called_once_with(
            run_name=run_name, tags=tags
        )
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_log_params(self, mock_mlflow):
        """Test log_params method"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        params = {"learning_rate": 0.001, "batch_size": 32}
        
        tracker.log_params(params)
        mock_mlflow.log_params.assert_called_once_with(params)
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_log_metrics(self, mock_mlflow):
        """Test log_metrics method"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        metrics = {"loss": 0.5, "accuracy": 0.9}
        step = 10
        
        tracker.log_metrics(metrics, step=step)
        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=step)
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_log_model(self, mock_mlflow):
        """Test log_model method"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        model = CustomCXRClassifier()
        artifact_path = "test-model"
        
        tracker.log_model(model, artifact_path=artifact_path)
        mock_mlflow.pytorch.log_model.assert_called_once_with(
            model, artifact_path
        )
    
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_end_run(self, mock_mlflow):
        """Test end_run method"""
        tracker = MLflowTracker(experiment_name=self.experiment_name)
        tracker.end_run()
        mock_mlflow.end_run.assert_called_once()


class TestWandBTracker(unittest.TestCase):
    """Test cases for WandBTracker"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_name = "test-project"
        self.entity = None
        self.config = {"learning_rate": 0.001, "batch_size": 32}
    
    def test_tracker_initialization(self):
        """Test WandBTracker initialization"""
        tracker = WandBTracker(
            project_name=self.project_name,
            entity=self.entity,
            config=self.config
        )
        self.assertEqual(tracker.project_name, self.project_name)
        self.assertEqual(tracker.entity, self.entity)
        self.assertEqual(tracker.config, self.config)
    
    def test_tracker_initialization_defaults(self):
        """Test WandBTracker initialization with defaults"""
        tracker = WandBTracker()
        self.assertIsNotNone(tracker.project_name)
        self.assertIsNone(tracker.entity)
        self.assertEqual(tracker.config, {})
    
    @patch('src.tracking.wandb_tracker.wandb.init')
    def test_init_run(self, mock_wandb_init):
        """Test init method"""
        tracker = WandBTracker(
            project_name=self.project_name,
            entity=self.entity,
            config=self.config
        )
        run_name = "test-run"
        tags = ["tag1", "tag2"]
        
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        result = tracker.init(run_name=run_name, tags=tags)
        
        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        self.assertEqual(call_kwargs['project'], self.project_name)
        self.assertEqual(call_kwargs['entity'], self.entity)
        self.assertEqual(call_kwargs['name'], run_name)
        self.assertEqual(call_kwargs['tags'], tags)
    
    @patch('src.tracking.wandb_tracker.wandb')
    def test_log(self, mock_wandb):
        """Test log method"""
        tracker = WandBTracker()
        metrics = {"loss": 0.5, "accuracy": 0.9}
        step = 10
        
        tracker.log(metrics, step=step)
        mock_wandb.log.assert_called_once_with(metrics, step=step)
    
    @patch('src.tracking.wandb_tracker.wandb')
    def test_log_with_commit(self, mock_wandb):
        """Test log method with commit parameter"""
        tracker = WandBTracker()
        metrics = {"loss": 0.5}
        step = 10
        commit = True
        
        tracker.log(metrics, step=step, commit=commit)
        mock_wandb.log.assert_called_once_with(metrics, step=step, commit=commit)
    
    @patch('src.tracking.wandb_tracker.wandb')
    @patch('src.tracking.wandb_tracker.torch')
    def test_log_model(self, mock_torch, mock_wandb):
        """Test log_model method"""
        tracker = WandBTracker()
        model = CustomCXRClassifier()
        artifact_path = "test-model.pth"
        
        tracker.log_model(model, artifact_path=artifact_path)
        mock_torch.save.assert_called_once()
        mock_wandb.save.assert_called_once_with(artifact_path)
    
    @patch('src.tracking.wandb_tracker.wandb')
    def test_watch(self, mock_wandb):
        """Test watch method"""
        tracker = WandBTracker()
        model = CustomCXRClassifier()
        log_freq = 100
        
        tracker.watch(model, log_freq=log_freq)
        mock_wandb.watch.assert_called_once_with(model, log_freq=log_freq)
    
    @patch('src.tracking.wandb_tracker.wandb')
    def test_finish(self, mock_wandb):
        """Test finish method"""
        tracker = WandBTracker()
        tracker.finish()
        mock_wandb.finish.assert_called_once()


class TestTrainWithMLflow(unittest.TestCase):
    """Test cases for train_with_mlflow function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = CustomCXRClassifier()
        self.device = torch.device('cpu')
        self.num_epochs = 2
        self.class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
        
        # Create dummy data loaders with all 3 classes represented
        train_data = torch.randn(30, 3, 128, 128)
        # Ensure all 3 classes are represented
        train_labels = torch.tensor([0]*10 + [1]*10 + [2]*10)
        train_dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=4)
        
        val_data = torch.randn(15, 3, 128, 128)
        # Ensure all 3 classes are represented in validation set
        val_labels = torch.tensor([0]*5 + [1]*5 + [2]*5)
        val_dataset = TensorDataset(val_data, val_labels)
        self.val_loader = DataLoader(val_dataset, batch_size=4)
        
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 4,
            'lr_step_size': 7,
            'lr_gamma': 0.1
        }
    
    @patch('src.tracking.mlflow_tracker.MLflowTracker')
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_train_with_mlflow_basic(self, mock_mlflow, mock_tracker_class):
        """Test basic training with MLflow"""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = lambda x: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda *args: None
        
        model, history = train_with_mlflow(
            self.model,
            self.train_loader,
            self.val_loader,
            self.config,
            self.num_epochs,
            self.device,
            self.class_names,
            experiment_name="test-exp",
            run_name="test-run"
        )
        
        # Check that model is returned
        self.assertIsNotNone(model)
        # Check that history is returned
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('train_acc', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)
    
    @patch('src.tracking.mlflow_tracker.MLflowTracker')
    @patch('src.tracking.mlflow_tracker.mlflow')
    def test_train_with_mlflow_history_structure(self, mock_mlflow, mock_tracker_class):
        """Test that training history has correct structure"""
        mock_tracker = MagicMock()
        mock_tracker_class.return_value = mock_tracker
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = lambda x: mock_run
        mock_mlflow.start_run.return_value.__exit__ = lambda *args: None
        
        model, history = train_with_mlflow(
            self.model,
            self.train_loader,
            self.val_loader,
            self.config,
            self.num_epochs,
            self.device,
            self.class_names
        )
        
        # Check history structure
        self.assertEqual(len(history['train_loss']), self.num_epochs)
        self.assertEqual(len(history['train_acc']), self.num_epochs)
        self.assertEqual(len(history['val_loss']), self.num_epochs)
        self.assertEqual(len(history['val_acc']), self.num_epochs)


class TestTrainWithWandB(unittest.TestCase):
    """Test cases for train_with_wandb function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = CustomCXRClassifier()
        self.device = torch.device('cpu')
        self.num_epochs = 2
        self.class_names = ['COVID-19', 'Viral Pneumonia', 'Normal']
        
        # Create dummy data loaders with all 3 classes represented
        train_data = torch.randn(30, 3, 128, 128)
        # Ensure all 3 classes are represented
        train_labels = torch.tensor([0]*10 + [1]*10 + [2]*10)
        train_dataset = TensorDataset(train_data, train_labels)
        self.train_loader = DataLoader(train_dataset, batch_size=4)
        
        val_data = torch.randn(15, 3, 128, 128)
        # Ensure all 3 classes are represented in validation set
        val_labels = torch.tensor([0]*5 + [1]*5 + [2]*5)
        val_dataset = TensorDataset(val_data, val_labels)
        self.val_loader = DataLoader(val_dataset, batch_size=4)
        
        self.config = {
            'learning_rate': 0.001,
            'batch_size': 4,
            'lr_step_size': 7,
            'lr_gamma': 0.1
        }
    
    @patch('src.tracking.wandb_tracker.WandBTracker')
    @patch('src.tracking.wandb_tracker.wandb')
    def test_train_with_wandb_basic(self, mock_wandb, mock_tracker_class):
        """Test basic training with W&B"""
        mock_tracker = MagicMock()
        mock_run = MagicMock()
        mock_tracker.init.return_value = mock_run
        mock_tracker_class.return_value = mock_tracker
        
        model, history, run = train_with_wandb(
            self.model,
            self.train_loader,
            self.val_loader,
            self.config,
            self.num_epochs,
            self.device,
            self.class_names,
            project_name="test-project",
            run_name="test-run"
        )
        
        # Check that model is returned
        self.assertIsNotNone(model)
        # Check that history is returned
        self.assertIsNotNone(history)
        self.assertIn('train_loss', history)
        self.assertIn('train_acc', history)
        self.assertIn('val_loss', history)
        self.assertIn('val_acc', history)
    
    @patch('src.tracking.wandb_tracker.WandBTracker')
    @patch('src.tracking.wandb_tracker.wandb')
    def test_train_with_wandb_history_structure(self, mock_wandb, mock_tracker_class):
        """Test that training history has correct structure"""
        mock_tracker = MagicMock()
        mock_run = MagicMock()
        mock_tracker.init.return_value = mock_run
        mock_tracker_class.return_value = mock_tracker
        
        model, history, run = train_with_wandb(
            self.model,
            self.train_loader,
            self.val_loader,
            self.config,
            self.num_epochs,
            self.device,
            self.class_names
        )
        
        # Check history structure
        self.assertEqual(len(history['train_loss']), self.num_epochs)
        self.assertEqual(len(history['train_acc']), self.num_epochs)
        self.assertEqual(len(history['val_loss']), self.num_epochs)
        self.assertEqual(len(history['val_acc']), self.num_epochs)


if __name__ == '__main__':
    unittest.main()


