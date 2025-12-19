"""
Utility functions for data preprocessing and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from config import CIFAR10_CLASSES


def load_cifar10_data():
    """
    Load and preprocess CIFAR-10 dataset.
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - Normalized training and test data
    """
    # Load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values to [0, 1]
    X_train = x_train.astype('float32') / 255.0
    X_test = x_test.astype('float32') / 255.0
    
    return X_train, y_train, X_test, y_test


def visualize_samples(x_data, y_data, num_samples=10, dataset_name="Training"):
    """
    Visualize sample images from the dataset.
    
    Args:
        x_data: Image data (not normalized for visualization)
        y_data: Labels
        num_samples: Number of samples to display
        dataset_name: Name of the dataset (for title)
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 6))
    
    for i in range(num_samples):
        image = x_data[i]
        
        # If image is normalized, denormalize for display
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        axes[i].imshow(image)
        axes[i].axis('off')
        
        # Get class name
        class_idx = y_data[i][0] if len(y_data[i].shape) > 0 else y_data[i]
        axes[i].set_title(CIFAR10_CLASSES[class_idx])
    
    plt.suptitle(f'{dataset_name} Samples', fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    
    Args:
        history: Keras History object from model.fit()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='lower right')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


def predict_image(model, image, return_probabilities=False):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained Keras model
        image: Image array (can be single image or batch)
        return_probabilities: If True, return full probability distribution
    
    Returns:
        Predicted class index and optionally probabilities
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Get predictions
    predictions = model.predict(image, verbose=0)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    if return_probabilities:
        return predicted_class, predictions[0]
    return predicted_class


def display_prediction(image, predicted_class, confidence=None):
    """
    Display an image with its prediction.
    
    Args:
        image: Image array
        predicted_class: Predicted class index
        confidence: Optional confidence score
    """
    plt.figure(figsize=(6, 6))
    
    # Denormalize if needed
    if image.max() <= 1.0:
        display_image = (image * 255).astype(np.uint8)
    else:
        display_image = image.astype(np.uint8)
    
    plt.imshow(display_image)
    plt.axis('off')
    
    title = f'Predicted: {CIFAR10_CLASSES[predicted_class]}'
    if confidence is not None:
        title += f' ({confidence*100:.2f}%)'
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.show()
