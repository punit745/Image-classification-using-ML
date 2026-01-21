"""
Model architectures for image classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import MODEL_CONFIG


def build_ann_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Build an Artificial Neural Network (ANN) model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(MODEL_CONFIG['ann_dense_units'], activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name='ANN_Model')
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Build a Convolutional Neural Network (CNN) model.
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential(name='CNN_Model')
    
    # Add input layer
    model.add(layers.Input(shape=input_shape))
    
    # Add convolutional blocks
    for filters in MODEL_CONFIG['cnn_filters']:
        model.add(layers.Conv2D(
            filters,
            kernel_size=MODEL_CONFIG['cnn_kernel_size'],
            padding='same',
            activation='relu'
        ))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPooling2D(MODEL_CONFIG['cnn_pool_size']))
    
    # Flatten and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(MODEL_CONFIG['dense_units'], activation='relu'))
    model.add(layers.Dropout(MODEL_CONFIG['dropout_rate']))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=MODEL_CONFIG['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_callbacks(model_name='model'):
    """
    Get callbacks for training.
    
    Args:
        model_name: Name for saved model files
    
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=MODEL_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=MODEL_CONFIG['reduce_lr_factor'],
            patience=MODEL_CONFIG['reduce_lr_patience'],
            verbose=1,
            min_lr=1e-7
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=f"{MODEL_CONFIG['checkpoint_path']}{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    return callbacks


def load_model(model_path):
    """
    Load a saved model.
    
    Args:
        model_path: Path to saved model file
    
    Returns:
        Loaded Keras model
    """
    return keras.models.load_model(model_path)
