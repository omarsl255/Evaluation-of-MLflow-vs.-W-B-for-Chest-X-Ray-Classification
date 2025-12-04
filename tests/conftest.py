"""
Pytest configuration and shared fixtures
"""

import pytest
import torch
import tempfile
import shutil
import os
from PIL import Image

# Set matplotlib to use non-interactive backend for testing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from src.models.cnn_model import CustomCXRClassifier


@pytest.fixture
def model():
    """Fixture for creating a model instance"""
    return CustomCXRClassifier()


@pytest.fixture
def device():
    """Fixture for device"""
    return torch.device('cpu')


@pytest.fixture
def sample_batch():
    """Fixture for creating a sample batch"""
    batch_size = 4
    channels = 3
    height = 128
    width = 128
    images = torch.randn(batch_size, channels, height, width)
    labels = torch.randint(0, 3, (batch_size,))
    return images, labels


@pytest.fixture
def temp_dataset_dir():
    """Fixture for creating a temporary dataset directory"""
    temp_dir = tempfile.mkdtemp()
    
    # Create class directories
    classes = ['COVID-19', 'Viral Pneumonia', 'Normal']
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(temp_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 10 test images per class
        for i in range(10):
            img_path = os.path.join(class_dir, f'image_{i}.jpg')
            img = Image.new('RGB', (128, 128), color=(i*25, i*25, i*25))
            img.save(img_path)
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config():
    """Fixture for training configuration"""
    return {
        'learning_rate': 0.001,
        'batch_size': 32,
        'lr_step_size': 7,
        'lr_gamma': 0.1,
        'num_epochs': 5
    }


@pytest.fixture
def class_names():
    """Fixture for class names"""
    return ['COVID-19', 'Viral Pneumonia', 'Normal']


