# Sample Images

This directory contains sample images that can be used for testing the prediction pipeline.

These images are from the CIFAR-10 dataset and can be used with:

```bash
# Using the prediction script
python predict.py --model saved_models/cnn_model_final.h5 --image sample_images/Image-1.jpeg

# Or through the Streamlit web interface
streamlit run Streamlit_app.py
```

## Images Included

- `Image-1.jpeg` - Sample image 1
- `Image_2.jpeg` - Sample image 2
- `Image_3.jpeg` - Sample image 3
- `Image_4.jpeg` - Sample image 4

These are low-resolution (32x32) images from the CIFAR-10 dataset classes.
