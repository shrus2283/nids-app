import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

def load_model_predictions(model_type, class_type, X_raw, X_nn):
    model_file = f"{model_type.lower()}_{class_type}.{'h5' if model_type in ['cnn', 'lstm'] else 'sav'}"
    model_path = f"models/{model_file}"

    encoder_file = f"encoder_{model_type}_{class_type}.joblib"
    label_encoder_file = f"label_encoder_{model_type}_{class_type}.joblib"
    scaler_file = f"scaler_{model_type}_{class_type}.joblib"
    pca_file = f"pca_{model_type}_{class_type}.joblib"

    encoder = joblib.load(encoder_file)
    label_encoder = joblib.load(label_encoder_file)

    X = X_raw.copy()
    cat_cols = ['protocol_type', 'service', 'flag']
    X[cat_cols] = X[cat_cols].astype(str).fillna("Unknown")

    encoded = encoder.transform(X[cat_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X = X.drop(columns=cat_cols)
    X = pd.concat([X, encoded_df], axis=1)
    X = X.select_dtypes(include=[np.number])

    if model_type in ['knn', 'random_forest']:
        scaler = joblib.load(scaler_file)
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        pca = joblib.load(pca_file)
        X_final = pca.transform(X_scaled)
        model = joblib.load(model_path)
        return model.predict_proba(X_final)

    else:
        model = tf.keras.models.load_model(model_path)
        if X_nn.ndim == 2:
            X_nn = X_nn.reshape((X_nn.shape[0], X_nn.shape[1], 1))
        probs = model.predict(X_nn)
        return probs


def build_ensemble_model(class_type, df):
    print(f"\nBuilding ensemble model for: {class_type.upper()}")

    class_tag = class_type.lower().replace('-', '_')

    label_encoder = joblib.load(f"label_encoder_knn_{class_tag.replace('class', '')}.joblib")

    X_raw = df.drop(columns=["label"])
    y_raw = df["label"].astype(str)
    y_encoded = label_encoder.transform(y_raw)

    X_nn = X_raw.select_dtypes(include=[np.number])
    X_nn = X_nn.astype(np.float32).to_numpy()

    preds_knn = load_model_predictions("knn", class_tag, X_raw, X_nn)
    preds_rf = load_model_predictions("random_forest", class_tag, X_raw, X_nn)
    preds_cnn = load_model_predictions("cnn", class_tag, X_raw, X_nn)
    preds_lstm = load_model_predictions("lstm", class_tag, X_raw, X_nn)

    X_ensemble = np.hstack([preds_knn, preds_rf, preds_cnn, preds_lstm])
    y_ensemble = y_encoded

    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(X_ensemble, y_ensemble)

    y_pred = meta_model.predict(X_ensemble)
    acc = accuracy_score(y_ensemble, y_pred)
    print(f"Accuracy of ensemble ({class_type}): {acc:.4f}")

    joblib.dump(meta_model, f"ensemble_{class_tag}.sav")
    print(f"Saved: ensemble_{class_tag}.sav")


if __name__ == "__main__":
    df = pd.read_csv("KDDTest.csv")

    build_ensemble_model("binary_class", df)
    build_ensemble_model("multi_class", df)
