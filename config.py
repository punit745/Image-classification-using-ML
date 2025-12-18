"""
Configuration file for the Image Classification project.
Contains all hyperparameters and settings.
"""

# Model configuration
MODEL_CONFIG = {
    # Data parameters
    'image_height': 32,
    'image_width': 32,
    'image_channels': 3,
    'num_classes': 10,
    
    # Training parameters
    'batch_size': 64,
    'epochs': 20,
    'validation_split': 0.2,
    
    # Optimizer parameters
    'learning_rate': 0.001,
    'optimizer': 'adam',
    
    # Model architecture - CNN
    'cnn_filters': [32, 64, 128, 256],
    'cnn_kernel_size': (3, 3),
    'cnn_pool_size': (2, 2),
    'dropout_rate': 0.3,
    'dense_units': 64,
    
    # Model architecture - ANN
    'ann_dense_units': 2048,
    
    # Callbacks
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
    
    # Paths
    'model_save_path': 'saved_models/',
    'checkpoint_path': 'checkpoints/',
}

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
