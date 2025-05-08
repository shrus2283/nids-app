import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os

# Set up page config
st.set_page_config(page_title="NIDS Prediction App", layout="wide")
st.title("Network Intrusion Detection System")
st.markdown("Upload a CSV file to predict attacks using your trained models.")

# User selections
model_type = st.selectbox("Select Model", ["KNN", "Random Forest", "CNN", "LSTM"])
classification_type = st.radio("Classification Type", ["Binary", "Multi-Class"])

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

# Helper to build filenames
def make_filename(prefix):
    return f"{prefix}_{model_type.lower().replace(' ', '_')}_{classification_type.lower().replace('-', '_')}"

# Helper to load model or trigger download
def load_or_download(path):
    if not os.path.exists(path):
        st.warning(f"{path} not found. Trying to download...")
        os.system("python download_models.py")
    if os.path.exists(path):
        st.success(f"Loaded `{path}` successfully.")
        return path
    else:
        st.error(f"File `{path}` is still missing after attempting download.")
        return None

# Start processing if file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    df_original = df.copy()

    st.subheader("Preprocessing Data")

    encoder_path = load_or_download(f"models/{make_filename('encoder')}.joblib")
    label_encoder_path = load_or_download(f"models/{make_filename('label_encoder')}.joblib")
    scaler_path = f"models/{make_filename('scaler')}.joblib" if model_type in ["KNN", "Random Forest"] else None
    if scaler_path:
        scaler_path = load_or_download(scaler_path)

    try:
        encoder = joblib.load(encoder_path)
        label_encoder = joblib.load(label_encoder_path)

        categorical_cols = ['protocol_type', 'service', 'flag']
        df[categorical_cols] = df[categorical_cols].astype(str).fillna("Unknown")
        encoded_df = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)
        df = df.select_dtypes(include=[np.number])

        if scaler_path:
            scaler = joblib.load(scaler_path)
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)

        # PCA (only for KNN)
        if model_type == "KNN":
            pca_path = load_or_download(f"models/{make_filename('pca')}.joblib")
            if pca_path:
                pca = joblib.load(pca_path)
                df = pd.DataFrame(pca.transform(df), columns=[f"pca_{i}" for i in range(pca.n_components_)])
                st.success("PCA applied.")

        X = df.astype(np.float32).to_numpy()

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # Load and use model
    model_ext = "h5" if model_type in ["CNN", "LSTM"] else "sav"
    model_file = f"models/{make_filename(model_type.lower())}.{model_ext}"
    model_path = load_or_download(model_file)

    if model_path:
        try:
            if model_ext == "h5":
                model = tf.keras.models.load_model(model_path)
                if X.ndim == 2:
                    X = X.reshape((X.shape[0], X.shape[1], 1))
                y_pred = model.predict(X)

                if classification_type == "Multi-Class":
                    y_pred_indices = np.argmax(y_pred, axis=1)
                    y_pred_labels = label_encoder.inverse_transform(y_pred_indices)
                else:
                    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
                    y_pred_labels = label_encoder.inverse_transform(y_pred_binary)
            else:
                model = joblib.load(model_path)
                y_pred = model.predict(X)
                y_pred_labels = label_encoder.inverse_transform(y_pred)

            df_original['Prediction'] = y_pred_labels
            st.subheader("Predictions")
            st.dataframe(df_original)

            # Accuracy check
            label_col = st.selectbox("If available, select the column with true labels to calculate accuracy",
                                     ["None"] + list(df_original.columns))
            if label_col != "None":
                try:
                    y_true_raw = df_original[label_col].astype(str).values
                    y_pred_raw = df_original['Prediction'].astype(str).values

                    y_true = label_encoder.transform(y_true_raw)
                    y_pred_for_acc = label_encoder.transform(y_pred_raw)

                    accuracy = accuracy_score(y_true, y_pred_for_acc)
                    st.success(f"Accuracy: {accuracy:.4f}")
                except Exception as e:
                    st.error(f"Accuracy calculation error: {e}")

            # Download predictions
            csv = df_original.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")
