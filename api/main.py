from fastapi import FastAPI
import numpy as np
from keras.models import load_model
from api.data_preprocessing import (
    preprocess_data, create_sequences,
    TARGETS, SEQUENCE_LENGTH
)
from api.utils import (
    get_predictions_per_target_dict,
    format_predictions_json,
    get_real_values_per_target_dict
)

app = FastAPI()

print("🔧 MODE DEV : chargement local")

# Load and preprocess data
df, features = preprocess_data()

# Découpage test
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
df_test = df.iloc[train_size + val_size:]

# Séquences pour prédiction
X_test, y_test = create_sequences(df_test, TARGETS, features, SEQUENCE_LENGTH)
X_test = X_test.astype(np.float32)

# Load trained model
model = load_model("bi-lstm_model_V1.h5", compile=False)

# State API
app.state.model = model
app.state.df = df
app.state.X_test = X_test
app.state.y_test = y_test
app.state.features = features


@app.get("/")
async def root():
    return {"status": "OK"}


@app.post("/predict")
async def predict(n: int):  # tu peux utiliser n pour limiter les steps
    model = app.state.model
    df = app.state.df
    X_test = app.state.X_test
    y_test = app.state.y_test
    features = app.state.features

    pred_dict = get_predictions_per_target_dict(model, X_test, df, features, n_days=n)
    real_dict = get_real_values_per_target_dict(y_test, df, features, n_days=n)

    return {
        'pred': format_predictions_json(pred_dict),
        'real': format_predictions_json(real_dict)
    }
