import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def load_tools(model, model_type):
    base = f"models/{model}_{model_type}"
    tools = {}
    try: tools['encoder'] = joblib.load(f"{base}_encoder.joblib")
    except: pass
    try: tools['scaler'] = joblib.load(f"{base}_scaler.joblib")
    except: pass
    try: tools['pca'] = joblib.load(f"{base}_pca.joblib")
    except: pass
    try: tools['label_encoder'] = joblib.load(f"{base}_label_encoder.joblib")
    except: pass
    return tools

def preprocess(df, tools):
    df = df.copy()

    # Encode categorical features
    if 'encoder' in tools:
        for col in df.select_dtypes(include='object').columns:
            if col in tools['encoder']:
                df[col] = tools['encoder'][col].transform(df[col])
    
    # Convert to numpy array
    if isinstance(df, pd.DataFrame):
        df = df.to_numpy().astype('float32')

    # Apply scaler
    if 'scaler' in tools:
        df = tools['scaler'].transform(df)

    # Apply PCA
    if 'pca' in tools:
        df = tools['pca'].transform(df)

    return df

def predict_with_model(model_name, df, model_type='binary'):
    tools = load_tools(model_name, model_type)
    X = preprocess(df, tools)

    if model_name in ['cnn', 'lstm']:
        model = load_model(f"models/{model_name}_{model_type}_class.h5")
        X = X.reshape((X.shape[0], X.shape[1], 1))
        pred = (model.predict(X).flatten() > 0.5).astype(int)
    else:
        model = joblib.load(f"models/{model_name}_{model_type}_class.sav")
        pred = model.predict(X)

    return pred

def binary_ensemble_predict(df):
    models = ['cnn', 'lstm', 'knn', 'random_forest']
    all_preds = [predict_with_model(model, df, 'binary') for model in models]

    final_preds = np.round(np.mean(all_preds, axis=0)).astype(int)
    label_encoder = joblib.load('models/label_encoder_random_forest_binary.joblib')
    decoded_preds = label_encoder.inverse_transform(final_preds)

    return decoded_preds
