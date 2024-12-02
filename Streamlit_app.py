import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# Adding CSS styling for custom background and font
def set_background():
    st.markdown(
        """
        <style>
        body {
            background-color: #eaf7ff;  /* Light blue background */
            color: #333;  /* Font color */
            font-family: "Comic Sans MS", cursive;
        }
        .stApp {
            background: linear-gradient(135deg, #eaf7ff, #cce3f5); /* Gradient background */
        }
        .sidebar .sidebar-content {
            background: #3e4e68; /* Sidebar background */
            color: lightyellow;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0078D7; /* Heading color */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# General Image Classification with MobileNetV2
def mobilenetv2_imagenet():
    st.title("Object Classification with MobileNetV2")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
            predictions, top=3
        )[0]

        st.write("**Top Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. **{label}**: {score * 100:.2f}% confidence")


# General Image Classification with ResNet50
def resnet50_imagenet():
    st.title("Object Classification with ResNet50")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        model = tf.keras.applications.ResNet50(weights="imagenet")
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.resnet50.decode_predictions(
            predictions, top=3
        )[0]

        st.write("**Top Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. **{label}**: {score * 100:.2f}% confidence")


# General Image Classification with EfficientNetB0
def efficientnet_imagenet():
    st.title("Object Classification with EfficientNetB0")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        model = tf.keras.applications.EfficientNetB0(weights="imagenet")
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(
            predictions, top=3
        )[0]

        st.write("**Top Predictions:**")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}. **{label}**: {score * 100:.2f}% confidence")


# CIFAR-10 Image Classification
def cifar10_classification():
    st.title("CIFAR-10 Image Classification")
    uploaded_file = st.file_uploader(
        "Upload an image for CIFAR-10...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        if image.mode != "RGB":
            image = image.convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Classifying...")

        model = tf.keras.models.load_model("model111.h5")
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
    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox(
        "Choose a Model for Image Classification",
        (
            "MobileNetV2 (ImageNet)",
            "ResNet50 (ImageNet)",
            "EfficientNetB0 (ImageNet)",
            "CIFAR-10",
        ),
    )

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
