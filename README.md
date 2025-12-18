# Image Classification using Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive image classification project implementing both Convolutional Neural Networks (CNN) and Artificial Neural Networks (ANN) for the CIFAR-10 dataset. This project includes a clean, modular codebase with training scripts, inference tools, and a web-based interface using Streamlit.

## Project Overview

This project demonstrates the implementation and comparison of different neural network architectures for image classification tasks:

- **CNN Model**: Advanced architecture with multiple convolutional layers, batch normalization, and dropout for robust image feature extraction
- **ANN Model**: Fully connected neural network for baseline comparison
- **Transfer Learning**: Pre-trained models (MobileNetV2, ResNet50, EfficientNetB0) for general image classification

## Features

- ✨ Clean, modular code architecture
- 📊 Configurable hyperparameters via `config.py`
- 🎯 Training with callbacks (early stopping, learning rate reduction, checkpointing)
- 📈 Visualization of training metrics and predictions
- 🌐 Web interface using Streamlit for interactive predictions
- 🧪 Comprehensive test suite for validation
- 📦 Easy setup with `requirements.txt`

## Dataset: CIFAR-10

The CIFAR-10 dataset consists of:
- **60,000 images** (50,000 training + 10,000 test)
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Image dimensions**: 32x32 pixels with RGB channels

## Project Structure

```
Image-classification-using-ML/
├── config.py                 # Configuration and hyperparameters
├── models.py                 # Model architectures (CNN, ANN)
├── utils.py                  # Data loading, preprocessing, and visualization
├── train.py                  # Training script
├── predict.py                # Inference script
├── test_pipeline.py          # Test suite
├── Streamlit_app.py          # Web interface
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── notebooks/               # Jupyter notebooks (optional)

    
   

  
│   ├── Implementation-of-ML-model-for-image-classification.ipynb
│   └── MobileNet_TransferLearning.ipynb
├── saved_models/            # Saved trained models (created during training)
└── checkpoints/             # Training checkpoints (created during training)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/punit745/Image-classification-using-ML.git
cd Image-classification-using-ML
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.15.0
- NumPy 1.24.3
- Matplotlib 3.8.0
- Pillow 10.1.0
- Streamlit 1.28.1
- scikit-learn 1.3.2

## Usage

### 1. Training Models

Train the CNN model (recommended):
```bash
python train.py --model cnn --epochs 20 --batch-size 64
```

Train the ANN model:
```bash
python train.py --model ann --epochs 10 --batch-size 64
```

The script will:
- Automatically download CIFAR-10 dataset
- Train the model with specified parameters
- Save the best model based on validation accuracy
- Generate training history plots
- Display test accuracy

**Training Output**:
- Best model: `checkpoints/cnn_model_best.h5` or `checkpoints/ann_model_best.h5`
- Final model: `saved_models/cnn_model_final.h5` or `saved_models/ann_model_final.h5`

### 2. Making Predictions

Classify a single image:
```bash
python predict.py --model saved_models/cnn_model_final.h5 --image path/to/your/image.jpg
```

Example output:
```
==================================================
Predicted Class: cat
Confidence: 87.34%
==================================================

Top 3 predictions:
1. cat: 87.34%
2. dog: 8.21%
3. bird: 2.15%
```

### 3. Web Interface (Streamlit)

Launch the interactive web application:
```bash
streamlit run Streamlit_app.py
```

This provides a user-friendly interface to:
- Upload images for classification
- Choose between different pre-trained models:
  - MobileNetV2 (ImageNet)
  - ResNet50 (ImageNet)
  - EfficientNetB0 (ImageNet)
  - Custom CIFAR-10 model
- View predictions with confidence scores

### 4. Running Tests

Run the test suite to verify the pipeline:
```bash
python test_pipeline.py
```

This will test:
- Data loading and preprocessing
- Model building and compilation
- Prediction pipeline
- Configuration settings

## Model Architectures

### CNN Model
```
Conv2D (32 filters) → BatchNorm → MaxPool →
Conv2D (64 filters) → BatchNorm → MaxPool →
Conv2D (128 filters) → BatchNorm → MaxPool →
Conv2D (256 filters) → BatchNorm → MaxPool →
Flatten → Dense (64) → Dropout (0.3) → Dense (10, softmax)
```

**Features**:
- Batch normalization for stable training
- Dropout for regularization
- Adam optimizer with configurable learning rate
- Early stopping and learning rate reduction

### ANN Model
```
Flatten → Dense (2048, relu) → Dense (10, softmax)
```

**Features**:
- Simple fully connected architecture
- Good baseline for comparison

## Configuration

Edit `config.py` to customize hyperparameters:

```python
MODEL_CONFIG = {
    'batch_size': 64,           # Training batch size
    'epochs': 20,               # Number of epochs
    'learning_rate': 0.001,     # Initial learning rate
    'dropout_rate': 0.3,        # Dropout rate
    'cnn_filters': [32, 64, 128, 256],  # CNN filter sizes
    # ... and more
}
```

## Results

Typical performance on CIFAR-10:

| Model | Test Accuracy | Parameters | Training Time |
|-------|--------------|------------|---------------|
| ANN   | ~50-55%      | ~6.3M      | ~5 min        |
| CNN   | ~75-80%      | ~1.2M      | ~15 min       |

*Note: Results may vary based on hardware and training parameters*

## Key Improvements

This refactored version includes:

1. **Code Quality**:
   - Modular design with separate files for models, utilities, and training
   - Proper error handling and validation
   - Consistent coding style and documentation

2. **Functionality**:
   - Configurable hyperparameters
   - Training callbacks (early stopping, LR reduction, checkpointing)
   - Comprehensive testing suite
   - Better preprocessing and data handling

3. **Usability**:
   - Command-line interface for training and inference
   - Web interface for interactive predictions
   - Clear documentation and usage examples

4. **Best Practices**:
   - Version-controlled dependencies
   - Proper `.gitignore` for Python projects
   - Separation of concerns
   - Reusable components

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'tensorflow'`  
**Solution**: Install dependencies with `pip install -r requirements.txt`

**Issue**: Model file not found in Streamlit app  
**Solution**: Train a model first using `python train.py --model cnn`

**Issue**: Out of memory during training  
**Solution**: Reduce batch size in config.py or use `--batch-size 32`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Acknowledgments

This project was developed as part of an internship to explore and implement various deep learning architectures for image classification tasks.

## Contact

For questions or feedback, please open an issue on the GitHub repository.
