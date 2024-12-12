import joblib
import pandas as pd
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import StandardScaler

# Load the scaler used during training
scaler = joblib.load("scaler.pkl")  # Ensure you save and provide this file from training

def load_model_file(model_name: str):
    if model_name.endswith(".pkl"):
        return joblib.load(model_name)
    elif model_name.endswith(".h5"):
        return load_model(model_name)
    else:
        raise ValueError("Unsupported model file format")

def extract_features(img) -> np.ndarray:
    image_cv = np.array(img)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_cv, None)

    if descriptors is not None:
        descriptors = descriptors.flatten()
        features = descriptors[:128] if len(descriptors) >= 128 else np.pad(descriptors, (0, 128 - len(descriptors)), mode='constant')
    else:
        features = np.zeros(128)
    st.write(f"Extracted Features: {features}")  # Debug log for extracted features
    return features

def normalize_features(features: np.ndarray) -> np.ndarray:
    return scaler.transform(features.reshape(1, -1))[0]

def classify_image(img: bytes, model, model_type: str) -> pd.DataFrame:
    try:
        image = Image.open(img).convert("RGB")

        if model_type in ["KNN", "ANN"]:
            features = extract_features(image)
            features = normalize_features(features)
            probabilities = model.predict_proba([features])[0]
            probabilities = [round(prob * 100, 2) for prob in probabilities]
            prediction = [np.argmax(probabilities)]

        elif model_type == "SVM":
            features = extract_features(image)
            features = normalize_features(features)
            prediction = model.predict([features])

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([features])[0]
                probabilities = [round(prob * 100, 2) for prob in probabilities]
            else:
                decision_score = model.decision_function([features])[0]
                probability_fractured = 1 / (1 + np.exp(-decision_score))
                probability_not_fractured = 1 - probability_fractured
                probabilities = [
                    round(probability_not_fractured * 100, 2),
                    round(probability_fractured * 100, 2)
                ]

        elif model_type in ["CNN with Dropout", "CNN without Dropout"]:
            image = image.resize((128, 128))
            image_array = img_to_array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            not_fractured_prob = model.predict(image_array)[0][0]
            fractured_prob = 1 - not_fractured_prob

            probabilities = [round(not_fractured_prob * 100, 2), round(fractured_prob * 100, 2)]
            prediction = [0 if not_fractured_prob >= fractured_prob else 1]

        LABEL_MAPPING = {0: "Not Fractured", 1: "Fractured"}
        class_labels = ["Not Fractured", "Fractured"]

        prediction_df = pd.DataFrame({
            "Class": class_labels,
            "Probability (%)": probabilities
        })
        return prediction_df.sort_values("Probability (%)", ascending=False), LABEL_MAPPING[prediction[0]]

    except Exception as e:
        st.error(f"An error occurred during classification: {e}")
        return pd.DataFrame(), None

st.title("Bone Structure Analysis")
st.write("Upload an X-ray or bone scan image to analyze the structure.")

image_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

model_type = st.selectbox("Choose a model:", ["KNN", "ANN", "SVM", "CNN with Dropout", "CNN without Dropout"])

try:
    model_files = {
        "KNN": "knn_classifier.pkl",
        "ANN": "ann_classifier.pkl",
        "SVM": "svm_classifier.pkl",
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
        predictions_df, top_prediction = classify_image(image_file, model, model_type)

        if not predictions_df.empty:
            st.success(f'Predicted Structure: **{top_prediction}** '
                       f'Confidence: {predictions_df.iloc[0]["Probability (%)"]:.2f}%')
            st.write("Detailed Predictions:")
            st.table(predictions_df)
        else:
            st.error("Failed to classify the image.")
