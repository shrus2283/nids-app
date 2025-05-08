import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os

if not os.path.exists("models/encoder_knn_binary.joblib"):
    os.system("python download_models.py")

st.set_page_config(page_title="NIDS Prediction App", layout="wide")
st.title("Network Intrusion Detection System")
st.markdown("Upload a CSV file to predict attacks using your trained models.")

model_type = st.selectbox("Select Model", ["KNN", "Random Forest", "CNN", "LSTM"])
classification_type = st.radio("Classification Type", ["Binary", "Multi-Class"])

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    df_original = df.copy()

    st.subheader("Preprocessing Data")

    encoder_file = f"encoder_{model_type.lower()}_{classification_type.lower().replace('-', '_')}.joblib"
    label_encoder_file = f"label_encoder_{model_type.lower()}_{classification_type.lower().replace('-', '_')}.joblib"
    scaler_file = f"scaler_{model_type.lower()}_{classification_type.lower().replace('-', '_')}.joblib" if model_type in ["KNN", "Random Forest"] else None
    

    try:
        encoder = joblib.load(encoder_file)
        label_encoder = joblib.load(label_encoder_file)
        st.success("Loaded encoders successfully.")

        categorical_cols = ['protocol_type', 'service', 'flag']
        df[categorical_cols] = df[categorical_cols].astype(str).fillna("Unknown")

        encoded_df = encoder.transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

        df = df.drop(columns=categorical_cols)
        df = pd.concat([df, encoded_df], axis=1)

        df = df.select_dtypes(include=[np.number])

        if scaler_file:
            scaler = joblib.load(scaler_file)
            df = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
        if model_type in ["KNN"]:
            pca_file = f"pca_{model_type.lower()}_{classification_type.lower().replace('-', '_')}.joblib"
            if os.path.exists(pca_file):
                pca = joblib.load(pca_file)
                df = pd.DataFrame(pca.transform(df), columns=[f"pca_{i}" for i in range(pca.n_components_)])
                st.success("PCA transformation applied.")
            else:
                st.warning(f"PCA file `{pca_file}` not found. Proceeding without PCA.")


        X = df.astype(np.float32).to_numpy()

    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    model_filename = f"{model_type.lower()}_{classification_type.lower().replace('-', '_')}"
    model_path = f"models/{model_filename}.{'h5' if model_type in ['CNN', 'LSTM'] else 'sav'}"

    if not os.path.exists(model_path):
        st.error(f"Model not found at `{model_path}`.")
    else:
        st.success(f"Model `{model_filename}` loaded successfully.")

        try:
            if model_type in ["CNN", "LSTM"]:
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

                if classification_type == "Multi-Class":
                    y_pred_labels = label_encoder.inverse_transform(y_pred)
                else:
                    y_pred_labels = label_encoder.inverse_transform(y_pred)

            df_original['Prediction'] = y_pred_labels
            st.subheader("Predictions")
            st.dataframe(df_original)

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

            csv = df_original.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")
