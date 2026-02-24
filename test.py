from fastapi import FastAPI
from joblib import load
import numpy as np

app = FastAPI()

model = load("DecisionTreeClassifier.joblib")

@app.get("/")
def home():
    return {"status": "Model API running"}

@app.post("/predict")
def predict(features: list[float]):
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr).tolist()[0]
    return {"prediction": pred}