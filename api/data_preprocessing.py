import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings("ignore")

TARGETS = ["thermique", "nucleaire", "eolien", "solaire", "hydraulique", "bioenergies"]
LAGS = [1, 2, 4, 8, 56]
SEQUENCE_LENGTH = 168  # 7 jours

# Prétraitement des données
def preprocess_data():
    file_name = "dataset_cleaned_3h.csv"
    df = pd.read_csv(file_name, parse_dates=["datetime"])

    # Création des LAGS
    for lag in LAGS:
        for target in TARGETS:
            df[f"{target}_lag{lag}"] = df[target].shift(lag)

    df = df.dropna()

    # Supprimer la colonne datetime si elle existe
    if "datetime" in df.columns:
        df = df.drop(columns=["datetime"])

    # Features = tout sauf les TARGETS
    features = [col for col in df.columns if col not in TARGETS]

    return df, features

# Normalisation
def normalize_data(df, features):
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    # ✅ Supprimer toute colonne datetime des features
    features = [f for f in features if not np.issubdtype(df[f].dtype, np.datetime64)]

    df[features] = scaler_X.fit_transform(df[features])
    df[TARGETS] = scaler_y.fit_transform(df[TARGETS])

    return df, scaler_X, scaler_y

# Création des séquences
def create_sequences(data, TARGETS, features, seq_length, stride=3):
    X, y = [], []
    for i in range(0, len(data) - seq_length, stride):
        X.append(data.iloc[i:i + seq_length][features].values)
        y.append(data.iloc[i + seq_length][TARGETS].values)
    return np.array(X), np.array(y)
