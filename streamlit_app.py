import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def load_model_file(model_name: str):
    """Load a model file based on its extension."""
    if model_name.endswith(".pkl"):
        return joblib.load(model_name)
    elif model_name.endswith(".h5"):
        return load_model(model_name)
    else:
        raise ValueError("Unsupported model file format")


def extract_features(img, required_size=8202) -> np.ndarray:
    """Extract SIFT features and ensure they match the required size."""
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    _, descriptors = sift.detectAndCompute(image_cv, None)

    if descriptors is not None:
        flattened_descriptors = descriptors.flatten()
        # Pad or truncate to match the required size
        if len(flattened_descriptors) >= required_size:
            return flattened_descriptors[:required_size]
        else:
            return np.pad(flattened_descriptors, (0, required_size - len(flattened_descriptors)))
    else:
        return np.zeros(required_size)  # Zero vector if no features are found


def classify_image(img: bytes, model, model_type: str) -> pd.DataFrame:
    """Classify the uploaded image based on the selected model."""
    try:
        image = Image.open(img).convert("RGB")

        if model_type in ["KNN", "ANN", "SVM"]:
            features = extract_features(image)
            probabilities = model.predict_proba([features])[0]

            probabilities = [round(prob * 100, 2) for prob in probabilities]
            prediction = np.argmax(probabilities)
        elif model_type in ["CNN with Dropout", "CNN without Dropout"]:
            image = image.resize((128, 128))  # Resize to match CNN input size
            image_array = img_to_array(image) / 255.0  # Normalize to [0, 1]
            image_array = np.expand_dims(image_array, axis=0)

            not_fractured_prob = model.predict(image_array)[0][0]
            fractured_prob = 1 - not_fractured_prob

            probabilities = [round(not_fractured_prob * 100, 2), round(fractured_prob * 100, 2)]
            prediction = 0 if not_fractured_prob >= fractured_prob else 1

        LABEL_MAPPING = {
            0: "Not Fractured",
            1: "Fractured"
        }
        class_labels = ["Not Fractured", "Fractured"]

        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability (%)": probabilities
        })
        return prediction_df.sort_values("Probability (%)", ascending=False), LABEL_MAPPING[prediction]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None


# Streamlit UI
st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

model_type = st.selectbox("Choose a model:", ["KNN", "ANN", "SVM", "CNN with Dropout", "CNN without Dropout"])

try:
    model_files = {
        "KNN": "knn_classifier.pkl",
        "ANN": "ann_sklearn.pkl",
        "SVM": "svm_classifier-2.pkl",
        "CNN with Dropout": "cnn_with_dropoutt.h5",
        "CNN without Dropout": "cnn_without_dropoutt.h5"
    }
    selected_model_file = model_files[model_type]
    model = load_model_file(selected_model_file)
except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()

if image_file:
    st.image(image_file, caption="Uploaded Image", use_column_width=True)
    pred_button = st.button("Analyze Bone Structure")

    if pred_button:
        # Perform image classification
        predictions_df, top_prediction = classify_image(image_file, model, model_type)

        if not predictions_df.empty:
            # Display top prediction
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability (%)"]:.2f}%')

            # Display all predictions
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
