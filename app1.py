import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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
                y_pred_labels = label_encoder.inverse_transform(y_pred)

            df_original['Prediction'] = y_pred_labels
            st.subheader("Predictions")
            st.dataframe(df_original)

            pred_counts = df_original['Prediction'].value_counts().sort_index()

            st.subheader("Predicted Class Distribution")
            st.bar_chart(pred_counts)

            st.subheader("Predicted Class Proportions")
            fig1, ax1 = plt.subplots()
            ax1.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
            ax1.axis('equal')
            st.pyplot(fig1)

            st.subheader("Prediction Counts")
            st.dataframe(pred_counts.reset_index().rename(columns={'index': 'Class', 'Prediction': 'Count'}))


            label_col = st.selectbox("If available, select the column with true labels to calculate accuracy", ["None"] + list(df_original.columns))

            if label_col != "None":
                try:
                    y_true_raw = df_original[label_col].astype(str).values
                    y_pred_raw = df_original['Prediction'].astype(str).values

                    valid_classes = set(label_encoder.classes_)
                    y_true_filtered = [y for y in y_true_raw if y in valid_classes]
                    y_pred_filtered = [y for y, y_true in zip(y_pred_raw, y_true_raw) if y_true in valid_classes]

                    if len(y_true_filtered) != len(y_pred_filtered):
                        st.warning("Some true labels were excluded because they were not in the trained label set.")

                    y_true_enc = label_encoder.transform(y_true_filtered)
                    y_pred_enc = label_encoder.transform(y_pred_filtered)

                    accuracy = accuracy_score(y_true_enc, y_pred_enc)
                    st.success(f"Accuracy: {accuracy:.4f}")

                    cm = confusion_matrix(y_true_enc, y_pred_enc)
                    fig2, ax2 = plt.subplots()
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
                    disp.plot(ax=ax2, cmap="Blues", xticks_rotation=45)
                    st.pyplot(fig2)

                except Exception as e:
                    st.error(f"Accuracy calculation error: {e}")            

            if model_type == "Random Forest":
                try:
                    importances = model.feature_importances_
                    feat_names = df.columns
                    importance_df = pd.DataFrame({'feature': feat_names, 'importance': importances})
                    importance_df = importance_df.sort_values(by='importance', ascending=False).head(10)

                    st.subheader("Top 10 Important Features")
                    st.bar_chart(importance_df.set_index('feature'))
                except Exception as e:
                    st.warning(f"Feature importance unavailable: {e}")

            csv = df_original.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions CSV", data=csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Prediction error: {e}")
