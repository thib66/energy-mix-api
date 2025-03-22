from fastapi import FastAPI
from keras.models import load_model
#import tensorflow as tf 
#from tensorflow import keras
from api.gcp import gcp_load_model, gcp_load_data
from data_preprocessing import (
    load_data, preprocess_data, create_sequences,
    TARGETS, SEQUENCE_LENGTH
    )
from utils import (
    get_predictions_per_target_dict, 
    format_predictions_json, 
    get_real_values_per_target_dict
)

app=FastAPI()

print("test_keras")
local_filename = gcp_load_model()
app.state.model = load_model(local_filename)
print('HAHA', type(app.state.model))


local_filename = gcp_load_data()
print(local_filename)

df = load_data(local_filename)
print(type(df))

df, features = preprocess_data(df)
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
df_test = df.iloc[train_size + val_size:]

X_test, y_test = create_sequences(df_test, TARGETS, features, SEQUENCE_LENGTH)

app.state.df = df
app.state.X_test = X_test
app.state.y_test = y_test
app.state.features = features


@app.get("/")
async def root():
    return {"status":"OK"}


@app.post("/predict")
async def predict(n:int):

    model = app.state.model
    df = app.state.df
    X_test = app.state.X_test
    y_test = app.state.y_test
    features = app.state.features


    pred_dict = get_predictions_per_target_dict(model, X_test, df, features)
    print(pred_dict)
    pred_dict = format_predictions_json(pred_dict)
    print(pred_dict)

    real_dict = get_real_values_per_target_dict(y_test, df, features)
    print(real_dict)
    real_dict = format_predictions_json(real_dict)
    print(real_dict)

    return {
        'pred' : pred_dict, 
        'real' : real_dict
    }
