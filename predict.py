"""
Inference script for making predictions with trained models.
"""

import argparse
import numpy as np
from PIL import Image
from models import load_model
from config import CIFAR10_CLASSES


def preprocess_image(image_path, target_size=(32, 32)):
    """
    Load and preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    # Load image
    img = Image.open(image_path)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize
    img = img.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(img).astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict(model_path, image_path):
    """
    Make a prediction on an image.
    
    Args:
        model_path: Path to the saved model
        image_path: Path to the image to classify
    
    Returns:
        Predicted class and confidence
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Preprocess image
    print(f"Processing image {image_path}...")
    img_array = preprocess_image(image_path)
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Predicted Class: {CIFAR10_CLASSES[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"{'='*50}\n")
    
    # Display top 3 predictions
    print("Top 3 predictions:")
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    for i, idx in enumerate(top_3_indices, 1):
        print(f"{i}. {CIFAR10_CLASSES[idx]}: {predictions[0][idx] * 100:.2f}%")
    
    return predicted_class, confidence


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make predictions with trained models')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the trained model file'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the image to classify'
    )
    
    args = parser.parse_args()
    
    # Make prediction
    predict(args.model, args.image)


if __name__ == '__main__':
    main()
