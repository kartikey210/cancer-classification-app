from fastapi import FastAPI
import joblib
import pandas as pd

# ✅ THIS LINE IS MISSING
app = FastAPI()

# Load model
model = joblib.load("models/cancer_model.pkl")

@app.get("/")
def home():
    return {"message": "Cancer Classification API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].tolist()

        return {
            "prediction": int(prediction),
            "probability": probability
        }

    except Exception as e:
        return {"error": str(e)}
    