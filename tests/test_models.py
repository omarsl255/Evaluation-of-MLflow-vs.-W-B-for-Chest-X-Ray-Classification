"""
Unit tests for CNN model architecture
"""

import unittest
import torch
import torch.nn as nn
from src.models.cnn_model import CustomCXRClassifier, INPUT_CHANNELS, NUM_CLASSES


class TestCustomCXRClassifier(unittest.TestCase):
    """Test cases for CustomCXRClassifier model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = CustomCXRClassifier()
        self.batch_size = 4
        self.input_channels = INPUT_CHANNELS
        self.num_classes = NUM_CLASSES
        self.image_size = 128
    
    def test_model_initialization(self):
        """Test model initialization with default parameters"""
        model = CustomCXRClassifier()
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.fc_input_features, 6272)
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters"""
        model = CustomCXRClassifier(in_channels=1, num_classes=5)
        self.assertIsInstance(model, nn.Module)
        # Test forward pass with custom input
        test_input = torch.randn(1, 1, 128, 128)
        output = model(test_input)
        self.assertEqual(output.shape, (1, 5))
    
    def test_forward_pass_shape(self):
        """Test forward pass output shape"""
        test_input = torch.randn(self.batch_size, self.input_channels, 
                                self.image_size, self.image_size)
        output = self.model(test_input)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
    
    def test_forward_pass_output_range(self):
        """Test that output values are in valid probability range [0, 1]"""
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size)
        output = self.model(test_input)
        
        # Check that all values are between 0 and 1 (softmax output)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 1))
    
    def test_forward_pass_probability_sum(self):
        """Test that output probabilities sum to 1 (softmax property)"""
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size)
        output = self.model(test_input)
        
        # Check that probabilities sum to 1 for each sample
        prob_sums = torch.sum(output, dim=1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones(1), atol=1e-5))
    
    def test_model_parameters_count(self):
        """Test that model has expected number of parameters"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                              if p.requires_grad)
        
        # Model should have approximately 1M+ parameters
        self.assertGreater(total_params, 1000000)
        self.assertEqual(total_params, trainable_params)
    
    def test_model_different_input_sizes(self):
        """Test model with different input sizes"""
        # Test with 64x64 input
        test_input_64 = torch.randn(1, self.input_channels, 64, 64)
        try:
            output_64 = self.model(test_input_64)
            # If it works, check output shape
            self.assertEqual(output_64.shape[1], self.num_classes)
        except RuntimeError:
            # Expected if input size is too small for architecture
            pass
    
    def test_model_gradient_flow(self):
        """Test that gradients can flow through the model"""
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size, 
                                requires_grad=False)
        output = self.model(test_input)
        
        # Create a dummy loss
        target = torch.randint(0, self.num_classes, (1,))
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for at least some parameters
        has_gradients = any(p.grad is not None for p in self.model.parameters())
        self.assertTrue(has_gradients)
    
    def test_model_eval_mode(self):
        """Test model in evaluation mode"""
        self.model.eval()
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size)
        
        with torch.no_grad():
            output = self.model(test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertFalse(self.model.training)
    
    def test_model_train_mode(self):
        """Test model in training mode"""
        self.model.train()
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size)
        output = self.model(test_input)
        
        self.assertEqual(output.shape, (1, self.num_classes))
        self.assertTrue(self.model.training)
    
    def test_model_device_movement(self):
        """Test moving model to different devices"""
        # Test CPU (should always work)
        model_cpu = CustomCXRClassifier()
        test_input = torch.randn(1, self.input_channels, 
                                self.image_size, self.image_size)
        output_cpu = model_cpu(test_input)
        self.assertEqual(output_cpu.shape, (1, self.num_classes))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = CustomCXRClassifier().cuda()
            test_input_cuda = test_input.cuda()
            output_cuda = model_cuda(test_input_cuda)
            self.assertEqual(output_cuda.shape, (1, self.num_classes))
    
    def test_model_layers_exist(self):
        """Test that all expected layers exist in the model"""
        # Check convolutional layers
        self.assertTrue(hasattr(self.model, 'conv1'))
        self.assertTrue(hasattr(self.model, 'conv2'))
        self.assertTrue(hasattr(self.model, 'conv3'))
        self.assertTrue(hasattr(self.model, 'conv4'))
        
        # Check pooling layers
        self.assertTrue(hasattr(self.model, 'pool1'))
        self.assertTrue(hasattr(self.model, 'pool2'))
        self.assertTrue(hasattr(self.model, 'pool3'))
        self.assertTrue(hasattr(self.model, 'pool4'))
        
        # Check dropout layers
        self.assertTrue(hasattr(self.model, 'dropout1'))
        self.assertTrue(hasattr(self.model, 'dropout2'))
        self.assertTrue(hasattr(self.model, 'dropout3'))
        self.assertTrue(hasattr(self.model, 'dropout_fc1'))
        
        # Check fully connected layers
        self.assertTrue(hasattr(self.model, 'fc1'))
        self.assertTrue(hasattr(self.model, 'fc2'))
        self.assertTrue(hasattr(self.model, 'output_layer'))


if __name__ == '__main__':
    unittest.main()


