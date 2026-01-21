# 🚀 Step-by-Step Guide to Run This Project Locally

This guide will help you set up and run the Image Classification project on your local machine, even if you're new to machine learning or Python projects.

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Running the Project](#running-the-project)
4. [Verification](#verification)
5. [Common Issues & Solutions](#common-issues--solutions)

---

## Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required Software

1. **Python 3.9 or higher** (Python 3.9, 3.10, or 3.11 recommended)
   - Download from: https://www.python.org/downloads/
   - During installation, make sure to check "Add Python to PATH"

2. **Git** (for cloning the repository)
   - Download from: https://git-scm.com/downloads

3. **pip** (Python package manager - usually comes with Python)

### System Requirements
- **RAM**: At least 4GB (8GB recommended for training)
- **Storage**: At least 2GB free space
- **Internet**: Required for downloading datasets and dependencies

---

## Installation Steps

### Step 1: Verify Python Installation

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and verify Python is installed:

```bash
python --version
```

or

```bash
python3 --version
```

**Expected output**: `Python 3.9.x`, `Python 3.10.x`, or `Python 3.11.x`

If you get an error, please install Python from the link above.

---

### Step 2: Clone the Repository

Navigate to the directory where you want to download the project:

```bash
cd Desktop  # Or any directory of your choice
```

Clone the repository:

```bash
git clone https://github.com/punit745/Image-classification-using-ML.git
```

Navigate into the project directory:

```bash
cd Image-classification-using-ML
```

---

### Step 3: Create a Virtual Environment (Recommended)

Creating a virtual environment keeps your project dependencies isolated:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**How to know it's activated?** Your terminal prompt should show `(venv)` at the beginning.

---

### Step 4: Install Dependencies

With your virtual environment activated, install all required packages:

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.16.1
- NumPy 1.26.4
- Matplotlib 3.8.0
- Pillow 10.3.0
- Streamlit 1.28.1
- scikit-learn 1.3.2

**Note**: This may take 5-10 minutes depending on your internet speed.

---

### Step 5: Verify Installation

Check if TensorFlow is installed correctly:

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

**Expected output**: `TensorFlow version: 2.16.1`

---

## Running the Project

Now you're ready to run the project! You have three main options:

### Option 1: Train a Model (First Time Users Start Here)

Train the CNN model on CIFAR-10 dataset:

```bash
python train.py --model cnn --epochs 20 --batch-size 64
```

**What happens:**
- Downloads CIFAR-10 dataset automatically (~170MB)
- Trains the model for 20 epochs (~15-20 minutes)
- Saves the best model in `checkpoints/` directory
- Saves the final model in `saved_models/` directory
- Displays training progress and accuracy

**For faster training (lower accuracy):**
```bash
python train.py --model cnn --epochs 5 --batch-size 64
```

**To train the simpler ANN model:**
```bash
python train.py --model ann --epochs 10 --batch-size 64
```

---

### Option 2: Make Predictions on Images

After training a model, classify any image:

```bash
python predict.py --model saved_models/cnn_model_final.h5 --image path/to/your/image.jpg
```

**Example with sample images:**
```bash
python predict.py --model saved_models/cnn_model_final.h5 --image sample_images/Image-1.jpeg
```

**Expected output:**
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

---

### Option 3: Run the Web Interface (Streamlit App)

Launch the interactive web application:

```bash
streamlit run Streamlit_app.py
```

**What happens:**
- Opens a web browser automatically
- Shows a beautiful web interface
- Allows you to upload images and get predictions
- Provides multiple pre-trained models to choose from

**Access the app at**: http://localhost:8501

**To stop the app**: Press `Ctrl+C` in the terminal

---

### Option 4: Run Tests

Verify everything is working correctly:

```bash
python test_pipeline.py
```

This tests:
- Data loading
- Model building
- Prediction pipeline
- Configuration settings

---

## Verification

### ✅ Successful Installation Checklist

- [ ] Python 3.9+ is installed and accessible
- [ ] Virtual environment is created and activated
- [ ] All dependencies are installed without errors
- [ ] TensorFlow imports successfully
- [ ] Can run `train.py` or `test_pipeline.py` without errors

### 📁 Expected Directory Structure After Training

```
Image-classification-using-ML/
├── saved_models/           # Contains trained models
│   ├── cnn_model_final.h5
│   └── ann_model_final.h5 (if trained)
├── checkpoints/            # Contains best models during training
│   ├── cnn_model_best.h5
│   └── ann_model_best.h5
├── training_history.png    # Training plots (auto-generated)
└── [other project files]
```

---

## Common Issues & Solutions

### Issue 1: `python: command not found`

**Solution**: Try using `python3` instead of `python`:
```bash
python3 --version
python3 train.py --model cnn
```

---

### Issue 2: `pip: command not found`

**Solution**: Try using `python3 -m pip` instead:
```bash
python3 -m pip install -r requirements.txt
```

---

### Issue 3: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**: 
1. Make sure your virtual environment is activated
2. Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

### Issue 4: Out of Memory Error During Training

**Solution**: Reduce the batch size:
```bash
python train.py --model cnn --batch-size 32
```

Or edit `config.py` and change `'batch_size': 64` to `'batch_size': 32`

---

### Issue 5: Streamlit Not Opening in Browser

**Solution**:
1. Check the terminal for the URL (usually http://localhost:8501)
2. Manually open that URL in your browser
3. If port 8501 is busy, Streamlit will use another port (check terminal)

---

### Issue 6: Model File Not Found in Streamlit

**Solution**: Train a model first before running the Streamlit app:
```bash
python train.py --model cnn --epochs 5
streamlit run Streamlit_app.py
```

---

### Issue 7: TensorFlow Installation Issues on Mac M1/M2

**Solution**: Use the following commands:
```bash
pip install tensorflow-macos
pip install tensorflow-metal
```

Then install other dependencies:
```bash
pip install -r requirements.txt --no-deps
pip install numpy matplotlib pillow streamlit scikit-learn
```

---

### Issue 8: Permission Denied Error

**Solution**: 
- On Mac/Linux: Use `sudo` carefully or fix directory permissions
- Better: Use virtual environment (recommended approach above)

---

## 🎯 Quick Start for Impatient Users

If you just want to see it working quickly:

```bash
# 1. Clone and navigate
git clone https://github.com/punit745/Image-classification-using-ML.git
cd Image-classification-using-ML

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch web interface (uses pre-trained ImageNet models)
streamlit run Streamlit_app.py
```

The Streamlit app will work immediately with pre-trained ImageNet models (MobileNetV2, ResNet50, EfficientNetB0). To use the custom CIFAR-10 model, you'll need to train it first.

---

## 🔄 Deactivating Virtual Environment

When you're done working on the project:

```bash
deactivate
```

To reactivate later:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

---

## 📚 What's Next?

After successfully setting up the project:

1. **Experiment with different models**: Try ANN vs CNN
2. **Adjust hyperparameters**: Edit `config.py` to customize training
3. **Test with your own images**: Use `predict.py` with your photos
4. **Explore the code**: Check out `models.py`, `train.py`, and `utils.py`
5. **Read the notebooks**: See `notebooks/` for detailed explanations

---

## 🆘 Still Having Issues?

If you encounter problems not covered here:

1. Check the main [README.md](README.md) for more details
2. Review error messages carefully - they often contain the solution
3. Open an issue on the GitHub repository with:
   - Your operating system
   - Python version
   - Complete error message
   - Steps you've already tried

---

## 🎉 Congratulations!

You've successfully set up the Image Classification project! Now you can:
- Train custom models
- Make predictions
- Use the web interface
- Experiment with different architectures

Happy coding! 🚀
