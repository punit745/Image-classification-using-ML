"""
Training script for image classification models.
"""

import os
import argparse
import tensorflow as tf
from models import build_ann_model, build_cnn_model, get_callbacks
from utils import load_cifar10_data, plot_training_history
from config import MODEL_CONFIG


def train_model(model_type='cnn', epochs=None, batch_size=None):
    """
    Train a model on CIFAR-10 dataset.
    
    Args:
        model_type: Type of model ('ann' or 'cnn')
        epochs: Number of training epochs (uses config default if None)
        batch_size: Batch size for training (uses config default if None)
    
    Returns:
        Trained model and training history
    """
    # Set parameters
    epochs = epochs or MODEL_CONFIG['epochs']
    batch_size = batch_size or MODEL_CONFIG['batch_size']
    
    print(f"\n{'='*50}")
    print(f"Training {model_type.upper()} Model")
    print(f"{'='*50}\n")
    
    # Load data
    print("Loading CIFAR-10 data...")
    X_train, y_train, X_test, y_test = load_cifar10_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Image shape: {X_train.shape[1:]}\n")
    
    # Build model
    print(f"Building {model_type.upper()} model...")
    if model_type.lower() == 'ann':
        model = build_ann_model()
    elif model_type.lower() == 'cnn':
        model = build_cnn_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'ann' or 'cnn'.")
    
    # Display model summary
    model.summary()
    print()
    
    # Create directories for saving
    os.makedirs(MODEL_CONFIG['model_save_path'], exist_ok=True)
    os.makedirs(MODEL_CONFIG['checkpoint_path'], exist_ok=True)
    
    # Get callbacks
    callbacks = get_callbacks(model_name=f'{model_type}_model')
    
    # Train model
    print(f"Training for {epochs} epochs with batch size {batch_size}...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Save final model
    model_path = os.path.join(MODEL_CONFIG['model_save_path'], f'{model_type}_model_final.h5')
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    return model, history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train image classification models')
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=['ann', 'cnn'],
        help='Model type to train (ann or cnn)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help=f'Number of training epochs (default: {MODEL_CONFIG["epochs"]})'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help=f'Batch size for training (default: {MODEL_CONFIG["batch_size"]})'
    )
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(history)
    
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
