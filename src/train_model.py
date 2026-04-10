import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

from preprocessing import load_and_preprocess
from imblearn.over_sampling import SMOTE

# Load data
df = load_and_preprocess("data/raw/proteomics.csv")

X = df.drop(columns=["sample_id", "label"])
y = df["label"]

# -------- FEATURE SELECTION --------
selector = SelectKBest(score_func=f_classif, k=100)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
X = pd.DataFrame(X_selected, columns=selected_features)

# -------- TRAIN TEST SPLIT --------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# -------- SMOTE --------
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -------- MODEL --------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# -------- EVALUATION --------
y_pred = model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------- SAVE --------
joblib.dump(model, "models/cancer_model.pkl")
joblib.dump(selected_features, "models/selected_features.pkl")

print("Model + features saved!")