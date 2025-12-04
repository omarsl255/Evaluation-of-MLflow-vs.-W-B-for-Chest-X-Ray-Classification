"""
Unit tests for data loading functionality
"""

import unittest
import os
import tempfile
import shutil
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.data_loader import (
    COVID19ChestXRayDataset,
    load_dataset_from_directory
)


class TestCOVID19ChestXRayDataset(unittest.TestCase):
    """Test cases for COVID19ChestXRayDataset"""
    
    def setUp(self):
        """Set up test fixtures with temporary directory structure"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_test_dataset()
    
    def setup_test_dataset(self):
        """Create a temporary test dataset structure"""
        # Create class directories
        classes = ['COVID-19', 'Viral Pneumonia', 'Normal']
        self.image_paths = []
        self.labels = []
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 5 test images per class
            for i in range(5):
                img_path = os.path.join(class_dir, f'image_{i}.jpg')
                # Create a simple test image
                img = Image.new('RGB', (128, 128), color=(i*50, i*50, i*50))
                img.save(img_path)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test dataset initialization"""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        dataset = COVID19ChestXRayDataset(
            self.image_paths, 
            self.labels, 
            transform=transform
        )
        self.assertEqual(len(dataset), 15)  # 5 images * 3 classes
    
    def test_dataset_length(self):
        """Test dataset __len__ method"""
        dataset = COVID19ChestXRayDataset(self.image_paths, self.labels)
        self.assertEqual(len(dataset), len(self.image_paths))
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        dataset = COVID19ChestXRayDataset(
            self.image_paths, 
            self.labels, 
            transform=transform
        )
        
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertEqual(label, self.labels[0])
    
    def test_dataset_with_transform(self):
        """Test dataset with transforms"""
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        dataset = COVID19ChestXRayDataset(
            self.image_paths, 
            self.labels, 
            transform=transform
        )
        
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 128, 128))
    
    def test_dataset_without_transform(self):
        """Test dataset without transforms"""
        dataset = COVID19ChestXRayDataset(self.image_paths, self.labels)
        image, label = dataset[0]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(label, self.labels[0])
    
    def test_dataset_tensor_index(self):
        """Test dataset with tensor index"""
        dataset = COVID19ChestXRayDataset(self.image_paths, self.labels)
        tensor_idx = torch.tensor(0)
        image, label = dataset[tensor_idx]
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(label, self.labels[0])
    
    def test_dataset_invalid_image_handling(self):
        """Test dataset handles invalid images gracefully"""
        # Create a dataset with a non-existent image path
        invalid_paths = self.image_paths + ['/nonexistent/path/image.jpg']
        invalid_labels = self.labels + [0]
        
        dataset = COVID19ChestXRayDataset(invalid_paths, invalid_labels)
        # Should return a black image as fallback
        image, label = dataset[-1]
        self.assertIsInstance(image, Image.Image)


class TestLoadDatasetFromDirectory(unittest.TestCase):
    """Test cases for load_dataset_from_directory function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.setup_test_dataset_structure()
    
    def setup_test_dataset_structure(self):
        """Create a test dataset structure"""
        # Create flat structure
        classes = ['COVID-19', 'Viral Pneumonia', 'Normal']
        for class_name in classes:
            class_dir = os.path.join(self.temp_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 10 test images per class
            for i in range(10):
                img_path = os.path.join(class_dir, f'image_{i}.jpg')
                img = Image.new('RGB', (128, 128), color=(i*25, i*25, i*25))
                img.save(img_path)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_dataset_basic(self):
        """Test basic dataset loading"""
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
        
        # Check that loaders are created
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        self.assertIsInstance(test_loader, DataLoader)
        
        # Check class names
        self.assertEqual(class_names, ['COVID-19', 'Viral Pneumonia', 'Normal'])
    
    def test_load_dataset_splits(self):
        """Test dataset splitting"""
        train_loader, val_loader, test_loader, _ = \
            load_dataset_from_directory(
                self.temp_dir,
                batch_size=1,
                train_split=0.7,
                val_split=0.2,
                test_split=0.1,
                random_seed=42
            )
        
        # Count samples in each split
        train_count = len(train_loader.dataset)
        val_count = len(val_loader.dataset)
        test_count = len(test_loader.dataset)
        total = train_count + val_count + test_count
        
        # Should have 30 total images (10 per class * 3 classes)
        self.assertEqual(total, 30)
        
        # Check approximate split ratios (allowing for rounding)
        self.assertGreater(train_count, 18)  # ~70% of 30 = 21
        self.assertGreater(val_count, 4)     # ~20% of 30 = 6
        self.assertGreater(test_count, 1)    # ~10% of 30 = 3
    
    def test_load_dataset_batch_size(self):
        """Test dataset loading with different batch sizes"""
        for batch_size in [1, 4, 8, 16]:
            train_loader, _, _, _ = load_dataset_from_directory(
                self.temp_dir,
                batch_size=batch_size,
                random_seed=42
            )
            self.assertEqual(train_loader.batch_size, batch_size)
    
    def test_load_dataset_image_size(self):
        """Test dataset loading with different image sizes"""
        for image_size in [64, 128, 224]:
            train_loader, _, _, _ = load_dataset_from_directory(
                self.temp_dir,
                image_size=image_size,
                batch_size=4,
                random_seed=42
            )
            
            # Get a sample batch
            images, labels = next(iter(train_loader))
            # Check image shape: (batch, channels, height, width)
            self.assertEqual(images.shape[2], image_size)
            self.assertEqual(images.shape[3], image_size)
    
    def test_load_dataset_reproducibility(self):
        """Test that dataset loading is reproducible with same seed"""
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
        
        # Check that splits are the same
        self.assertEqual(len(train_loader1.dataset), len(train_loader2.dataset))
        self.assertEqual(len(val_loader1.dataset), len(val_loader2.dataset))
        self.assertEqual(len(test_loader1.dataset), len(test_loader2.dataset))
    
    def test_load_dataset_invalid_path(self):
        """Test dataset loading with invalid path"""
        with self.assertRaises(FileNotFoundError):
            load_dataset_from_directory('/nonexistent/path')
    
    def test_load_dataset_nested_structure(self):
        """Test dataset loading with nested structure (Covid19-dataset/train)"""
        # Create nested structure
        nested_dir = os.path.join(self.temp_dir, 'Covid19-dataset', 'train')
        os.makedirs(nested_dir, exist_ok=True)
        
        classes = ['Covid', 'Viral Pneumonia', 'Normal']
        for class_name in classes:
            class_dir = os.path.join(nested_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create 20 test images per class (enough for proper splitting)
            for i in range(20):
                img_path = os.path.join(class_dir, f'image_{i}.jpg')
                img = Image.new('RGB', (128, 128), color=(i*12, i*12, i*12))
                img.save(img_path)
        
        # Test loading from parent directory
        parent_dir = os.path.join(self.temp_dir, 'Covid19-dataset')
        train_loader, val_loader, test_loader, class_names = \
            load_dataset_from_directory(
                parent_dir,
                batch_size=4,
                random_seed=42
            )
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertEqual(class_names, ['COVID-19', 'Viral Pneumonia', 'Normal'])
    
    def test_load_dataset_data_loader_iteration(self):
        """Test that data loaders can be iterated"""
        train_loader, val_loader, test_loader, _ = \
            load_dataset_from_directory(
                self.temp_dir,
                batch_size=4,
                random_seed=42
            )
        
        # Test iteration
        batch = next(iter(train_loader))
        images, labels = batch
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)
        self.assertEqual(images.shape[0], labels.shape[0])
        self.assertLessEqual(images.shape[0], 4)  # batch_size


if __name__ == '__main__':
    unittest.main()


