import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os


# Adding CSS styling for minimalist black & white theme
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #eaf7ff, #cce3f5);
        }
        
        /* Headings - professional black */
        h1, h2, h3, h4, h5, h6 {
            color: #0078D7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# General Image Classification with MobileNetV2
def mobilenetv2_imagenet():
    st.title("MobileNetV2 Classification")
    st.markdown("Upload an image to classify it using the MobileNetV2 model trained on ImageNet.")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            model = tf.keras.applications.MobileNetV2(weights="imagenet")
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=3
            )[0]

        st.markdown("### 📊 Classification Results")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.markdown(f"""
            <div style='background-color: #F8F8F8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #000;'>
                <strong style='font-size: 1.1rem;'>{i + 1}. {label.replace('_', ' ').title()}</strong>
                <br>
                <span style='color: #666;'>Confidence: {score * 100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)


# General Image Classification with ResNet50
def resnet50_imagenet():
    st.title("ResNet50 Classification")
    st.markdown("Upload an image to classify it using the ResNet50 model trained on ImageNet.")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            model = tf.keras.applications.ResNet50(weights="imagenet")
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.resnet50.decode_predictions(
                predictions, top=3
            )[0]

        st.markdown("### 📊 Classification Results")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.markdown(f"""
            <div style='background-color: #F8F8F8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #000;'>
                <strong style='font-size: 1.1rem;'>{i + 1}. {label.replace('_', ' ').title()}</strong>
                <br>
                <span style='color: #666;'>Confidence: {score * 100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)


# General Image Classification with EfficientNetB0
def efficientnet_imagenet():
    st.title("EfficientNetB0 Classification")
    st.markdown("Upload an image to classify it using the EfficientNetB0 model trained on ImageNet.")
    
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("Analyzing image..."):
            model = tf.keras.applications.EfficientNetB0(weights="imagenet")
            img = image.resize((224, 224))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            predictions = model.predict(img_array)
            decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(
                predictions, top=3
            )[0]

        st.markdown("### 📊 Classification Results")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.markdown(f"""
            <div style='background-color: #F8F8F8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #000;'>
                <strong style='font-size: 1.1rem;'>{i + 1}. {label.replace('_', ' ').title()}</strong>
                <br>
                <span style='color: #666;'>Confidence: {score * 100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)


# CIFAR-10 Image Classification
def cifar10_classification():
    st.title("CIFAR-10 Classification")
    st.markdown("Upload an image to classify it into one of 10 CIFAR-10 categories using our custom trained model.")
    
    uploaded_file = st.file_uploader(
        "Upload an image for CIFAR-10...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        # Check if model file exists
        model_path = "saved_models/cnn_model_final.h5"
        if not os.path.exists(model_path):
            # Fall back to old model name if it exists
            model_path = "model111.h5"
            if not os.path.exists(model_path):
                st.error("Model file not found. Please train the model first using train.py")
                return
        
        model = tf.keras.models.load_model(model_path)
        class_names = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        img = image.resize((32, 32))
        img_array = np.array(img)
        img_array = img_array.astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = np.max(predictions)

        st.write(f"Predicted Class: **{class_names[predicted_class]}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")


# Main Function for Navigation
def main():
    set_background()  # Apply the CSS styling
    
    # Page title and description
    st.title("🖼️ Image Classification Studio")
    st.markdown("""
    <p style='font-size: 1.1rem; color: #666; margin-bottom: 30px;'>
    Professional image classification using state-of-the-art deep learning models.
    Select a model and upload an image to get started.
    </p>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Model Selection")
    st.sidebar.markdown("---")
    
    choice = st.sidebar.selectbox(
        "Choose a Classification Model",
        (
            "MobileNetV2 (ImageNet)",
            "ResNet50 (ImageNet)",
            "EfficientNetB0 (ImageNet)",
            "CIFAR-10",
        ),
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    This application provides multiple pre-trained models for image classification.
    
    **ImageNet Models**: General-purpose object detection (1000 classes)
    
    **CIFAR-10 Model**: Specialized for 10 specific categories
    """)

    if choice == "MobileNetV2 (ImageNet)":
        mobilenetv2_imagenet()
    elif choice == "ResNet50 (ImageNet)":
        resnet50_imagenet()
    elif choice == "EfficientNetB0 (ImageNet)":
        efficientnet_imagenet()
    elif choice == "CIFAR-10":
        cifar10_classification()


if __name__ == "__main__":
    main()
