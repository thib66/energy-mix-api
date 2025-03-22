from fastapi import FastAPI
from gcp import load_model

app=FastAPI()

app.state.model = load_model()

@app.get("/")
async def root():
    return {"status":"OK"}

@app.post("/predict")
async def predict(n:int):


    pred = predict(model, dataset, n)



    return {"status":"OK"}
