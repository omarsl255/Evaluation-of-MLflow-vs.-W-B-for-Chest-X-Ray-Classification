"""
Integration tests for end-to-end workflows
"""

import unittest
import torch
import tempfile
import shutil
import os
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.models.cnn_model import CustomCXRClassifier
from src.data.data_loader import load_dataset_from_directory


class TestModelTrainingIntegration(unittest.TestCase):
    """Integration tests for model training workflow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_test_dataset()
        self.device = torch.device('cpu')
    
    def setup_test_dataset(self):
        """Create a test dataset"""
        classes = ['COVID-19', 'Viral Pneumonia', 'Normal']
        for class_name in classes:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 20 test images per class
            for i in range(20):
                img_path = os.path.join(class_dir, f'image_{i}.jpg')
                img = Image.new('RGB', (128, 128), color=(i*12, i*12, i*12))
                img.save(img_path)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_model_training_workflow(self):
        """Test complete model training workflow"""
        # Load dataset
        train_loader, val_loader, test_loader, class_names = \
            load_dataset_from_directory(
                self.temp_dir,
                image_size=128,
                batch_size=4,
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                random_seed=42
            )
        
        # Create model
        model = CustomCXRClassifier()
        model = model.to(self.device)
        
        # Setup training
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 1 epoch
        model.train()
        for epoch in range(1):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        self.assertGreater(accuracy, 0)  # Should have some accuracy
        self.assertLessEqual(accuracy, 1)  # Should be <= 1
    
    def test_model_inference_workflow(self):
        """Test model inference workflow"""
        # Create model
        model = CustomCXRClassifier()
        model.eval()
        
        # Create dummy input
        test_input = torch.randn(1, 3, 128, 128)
        
        # Inference
        with torch.no_grad():
            output = model(test_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        # Check output
        self.assertEqual(output.shape, (1, 3))
        self.assertEqual(predicted_class.shape, (1,))
        self.assertGreaterEqual(predicted_class.item(), 0)
        self.assertLess(predicted_class.item(), 3)


class TestDataPipelineIntegration(unittest.TestCase):
    """Integration tests for data pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_test_dataset()
    
    def setup_test_dataset(self):
        """Create a test dataset"""
        classes = ['COVID-19', 'Viral Pneumonia', 'Normal']
        for class_name in classes:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 15 test images per class
            for i in range(15):
                img_path = os.path.join(class_dir, f'image_{i}.jpg')
                img = Image.new('RGB', (128, 128), color=(i*16, i*16, i*16))
                img.save(img_path)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_loading_to_training_pipeline(self):
        """Test data loading through to training pipeline"""
        # Load dataset
        train_loader, val_loader, test_loader, class_names = \
            load_dataset_from_directory(
                self.temp_dir,
                image_size=128,
                batch_size=4,
                random_seed=42
            )
        
        # Verify loaders
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Get a batch
        images, labels = next(iter(train_loader))
        
        # Verify batch structure
        self.assertEqual(images.shape[0], labels.shape[0])
        self.assertEqual(images.shape[1], 3)  # RGB channels
        self.assertEqual(images.shape[2], 128)  # Height
        self.assertEqual(images.shape[3], 128)  # Width
        self.assertTrue(torch.all(labels >= 0))
        self.assertTrue(torch.all(labels < 3))
    
    def test_data_splitting_consistency(self):
        """Test that data splits are consistent"""
        train_loader1, val_loader1, test_loader1, _ = \
            load_dataset_from_directory(
                self.temp_dir,
                random_seed=42
            )
        
        train_loader2, val_loader2, test_loader2, _ = \
            load_dataset_from_directory(
                self.temp_dir,
                random_seed=42
            )
        
        # Check consistency
        self.assertEqual(
            len(train_loader1.dataset),
            len(train_loader2.dataset)
        )
        self.assertEqual(
            len(val_loader1.dataset),
            len(val_loader2.dataset)
        )
        self.assertEqual(
            len(test_loader1.dataset),
            len(test_loader2.dataset)
        )


if __name__ == '__main__':
    unittest.main()


