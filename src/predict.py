import joblib
import pandas as pd

# Load model
model = joblib.load("models/cancer_model.pkl")

def predict_sample(sample_df):
    """
    sample_df: DataFrame with same features as training data
    """
    prediction = model.predict(sample_df)
    probability = model.predict_proba(sample_df)

    return prediction[0], probability[0]


# TEST (for now)
if __name__ == "__main__":
    # Dummy input (IMPORTANT: must match feature count)
    df = pd.read_csv("data/raw/proteomics.csv")

    from preprocessing import load_and_preprocess
    df = load_and_preprocess("data/raw/proteomics.csv")

    X = df.drop(columns=["sample_id", "label"])

    sample = X.iloc[[0]]  # take one sample

    pred, prob = predict_sample(sample)

    print("Prediction:", pred)
    print("Probability:", prob)