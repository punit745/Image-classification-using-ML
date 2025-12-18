"""
Basic smoke tests for the image classification pipeline.
"""

import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_ann_model, build_cnn_model
from utils import load_cifar10_data, predict_image
from config import MODEL_CONFIG, CIFAR10_CLASSES


class TestDataLoading(unittest.TestCase):
    """Tests for data loading functionality."""
    
    def test_load_cifar10_data(self):
        """Test that CIFAR-10 data loads correctly (skipped if no network)."""
        try:
            X_train, y_train, X_test, y_test = load_cifar10_data()
            
            # Check shapes
            self.assertEqual(len(X_train.shape), 4)
            self.assertEqual(len(X_test.shape), 4)
            self.assertEqual(X_train.shape[1:], (32, 32, 3))
            self.assertEqual(X_test.shape[1:], (32, 32, 3))
            
            # Check normalization
            self.assertGreaterEqual(X_train.min(), 0.0)
            self.assertLessEqual(X_train.max(), 1.0)
            self.assertGreaterEqual(X_test.min(), 0.0)
            self.assertLessEqual(X_test.max(), 1.0)
            
            # Check labels
            self.assertEqual(len(y_train.shape), 2)
            self.assertEqual(len(y_test.shape), 2)
            self.assertGreaterEqual(y_train.min(), 0)
            self.assertLessEqual(y_train.max(), 9)
        except Exception as e:
            if "URL fetch failure" in str(e) or "No address" in str(e):
                self.skipTest("Network unavailable - skipping data loading test")
            else:
                raise


class TestModels(unittest.TestCase):
    """Tests for model building."""
    
    def test_build_ann_model(self):
        """Test that ANN model builds correctly."""
        model = build_ann_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertEqual(len(model.layers), 3)
        
        # Check input/output shapes
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        
        # Check compilation
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)
    
    def test_build_cnn_model(self):
        """Test that CNN model builds correctly."""
        model = build_cnn_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertGreater(len(model.layers), 5)
        
        # Check input/output shapes
        self.assertEqual(model.input_shape, (None, 32, 32, 3))
        self.assertEqual(model.output_shape, (None, 10))
        
        # Check compilation
        self.assertIsNotNone(model.optimizer)
        self.assertIsNotNone(model.loss)


class TestPrediction(unittest.TestCase):
    """Tests for prediction pipeline."""
    
    def test_predict_single_image(self):
        """Test prediction on a single image."""
        # Create mock data instead of loading from network
        test_image = np.random.rand(32, 32, 3).astype('float32')
        
        # Build a simple model
        model = build_ann_model()
        
        # Test prediction function
        predicted_class = predict_image(model, test_image)
        
        # Check prediction is valid class
        self.assertIsInstance(predicted_class, (int, np.integer))
        self.assertGreaterEqual(predicted_class, 0)
        self.assertLessEqual(predicted_class, 9)
    
    def test_predict_batch(self):
        """Test prediction on a batch of images."""
        # Create mock data instead of loading from network
        test_batch = np.random.rand(5, 32, 32, 3).astype('float32')
        
        # Build a simple model
        model = build_ann_model()
        
        # Test prediction on batch
        predicted_class = predict_image(model, test_batch)
        
        # Check prediction is valid
        self.assertIsInstance(predicted_class, (int, np.integer))
        self.assertGreaterEqual(predicted_class, 0)
        self.assertLessEqual(predicted_class, 9)


class TestConfig(unittest.TestCase):
    """Tests for configuration."""
    
    def test_config_values(self):
        """Test that config has required values."""
        self.assertIn('batch_size', MODEL_CONFIG)
        self.assertIn('epochs', MODEL_CONFIG)
        self.assertIn('learning_rate', MODEL_CONFIG)
        
        # Check types
        self.assertIsInstance(MODEL_CONFIG['batch_size'], int)
        self.assertIsInstance(MODEL_CONFIG['epochs'], int)
        self.assertIsInstance(MODEL_CONFIG['learning_rate'], float)
        
    def test_cifar10_classes(self):
        """Test that CIFAR-10 classes are correct."""
        self.assertEqual(len(CIFAR10_CLASSES), 10)
        self.assertIn('airplane', CIFAR10_CLASSES)
        self.assertIn('cat', CIFAR10_CLASSES)
        self.assertIn('truck', CIFAR10_CLASSES)


def run_smoke_tests():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("Running Smoke Tests for Image Classification Pipeline")
    print("="*60 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestPrediction))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_smoke_tests()
    sys.exit(0 if success else 1)
