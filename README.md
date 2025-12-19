# Image-classification-using-ML

## Project Overview
This README documents my internship project focused on implementing an image classification model using Convolutional Neural Networks (CNNs) and Artificial Neural Networks (ANNs). The project leverages the CIFAR-10 dataset, a standard dataset for image recognition tasks, to classify images into one of ten categories.

## Objective
  The primary objective of this project was to:

    Understand the basic and advanced concepts of Machine Learning (ML) and Deep Learning (DL).
    Implement image classification models using CNN and ANN architectures.
    Explore the performance of different models on real-world datasets.

## Concepts Learned

**Machine Learning Basics**
    Supervised learning techniques.
    Data preprocessing, feature scaling, and dataset partitioning.

**Deep Learning Fundamentals**
    Understanding Neural Networks and their architectures.
    Backpropagation and optimization techniques like Adam and SGD.
    Overfitting, underfitting, and regularization methods

## Dataset: CIFAR-10
    **The CIFAR-10 dataset consists of:**

    60,000 images categorized into 10 classes, each with 6,000 images.
    Image dimensions of 32x32 pixels with RGB channels.
    Classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Tools and Libraries

    **Programming Language:** Python
    **Libraries:** TensorFlow, NumPy, Matplotlib, Pillow, Streamlit

## Implementation Highlights
    
    **CNN Architecture**
          Input Layer: Accepts 32x32 RGB images.
          Convolutional Layers: Extracts spatial features using filters.
          Pooling Layers: Reduces feature dimensionality.
          Fully Connected Layers: Maps extracted features to output categories.
          Activation Function: ReLU for hidden layers, Softmax for output.
    **ANN Architecture**
          Input Layer: Accepts flattened image data (3072 features).
          Hidden Layers: Multiple dense layers with ReLU activation.
          Output Layer: Softmax activation for classification.

## Key Features of the Project

    Successfully classified images into ten distinct categories with significant accuracy.
    Compared the performance of CNNs and ANNs, highlighting the advantages of using CNN for image data.
    Learned advanced concepts of DL such as dropout, batch normalization, and learning rate scheduling.

  
## Challenges Faced
    Balancing between model complexity and training time.
    Preventing overfitting on the dataset using techniques like dropout.
    Optimizing hyperparameters for better accuracy.

## Results
    Achieved X% accuracy using the CNN model.
    Achieved Y% accuracy using the ANN model.
    

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/punit745/Image-classification-using-ML.git
cd Image-classification-using-ML
```

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Run the Streamlit Web Application (Recommended)

The easiest way to interact with the models is through the Streamlit web interface:

```bash
streamlit run Streamlit_app.py
```

This will open a web browser with an interactive UI where you can:
- Choose from multiple pre-trained models (MobileNetV2, ResNet50, EfficientNetB0)
- Upload images for classification
- Use the custom CIFAR-10 trained model
- See predictions with confidence scores

#### Option 2: Run in Jupyter Notebook

To explore the model training and implementation details:

1. Install Jupyter if not already installed:
```bash
pip install jupyter
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open either of these notebooks:
   - `Implementation-of-ML-model-for-image-classification.ipynb` - Original CNN/ANN implementation
   - `MobileNet_TransferLearning.ipynb` - Transfer learning with MobileNet

4. Run the cells sequentially to:
   - Load and preprocess the CIFAR-10 dataset
   - Build and train CNN/ANN models
   - Evaluate model performance
   - Make predictions on test images

### Quick Start Example

If you want to use the models programmatically:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Build a Simple CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Summary of the Model
model.summary()
```
    
# References
    CIFAR-10 Dataset: CIFAR-10 Official Website
    TensorFlow Documentation: TensorFlow
    Keras Documentation: Keras
    
# Conclusion
    This internship project served as a stepping stone into the fascinating world of Machine Learning and Deep Learning. By implementing image classification models using CNNs and ANNs, I gained valuable     hands-on experience and a deeper understanding of the principles underlying artificial intelligence.

    
   

  
