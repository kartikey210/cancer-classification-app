import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Metrics
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve
)

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_and_preprocess
from utils import get_top_features, apply_pca
from image_model import predict_image

# ---------------------------------------------------
# Load model + features
# ---------------------------------------------------
model = joblib.load("models/cancer_model.pkl")
selected_features = joblib.load("models/selected_features.pkl")

# ---------------------------------------------------
# Config
# ---------------------------------------------------
st.set_page_config(page_title="Cancer Classifier", layout="wide")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
st.sidebar.title("🧬 Navigation")
option = st.sidebar.radio(
    "Go to",
    ["Home", "Upload Data", "Sample Data", "Image Prediction"]
)

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🧬 Cancer Classification System")
st.markdown("### ML-based Molecular & Image Cancer Prediction")

label_map = {
    0: "Breast Cancer (TCGA)",
    1: "Lung Cancer (CPTAC)"
}

# ===================================================
# HOME
# ===================================================
if option == "Home":

    st.write("## 📌 About Project")

    st.write("""
    This system uses machine learning to classify cancer types
    using molecular expression data and image inputs.

    Features included:
    - Molecular cancer prediction
    - PCA visualization
    - Feature importance analysis
    - ROC analysis
    - Image-based prediction
    """)

# ===================================================
# UPLOAD DATA
# ===================================================
elif option == "Upload Data":

    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            # ---------------------------------------------------
            # Load CSV
            # ---------------------------------------------------
            raw_df = pd.read_csv(uploaded_file)

            st.write("### Dataset Preview")
            st.dataframe(raw_df.head())

            if st.button("🔍 Predict"):

                df = raw_df.copy()

                # ---------------------------------------------------
                # Flexible preprocessing
                # ---------------------------------------------------

                # Remove optional metadata columns
                df = df.drop(
                    columns=[
                        "RefSeq_accession_number",
                        "gene_name"
                    ],
                    errors='ignore'
                )

                # ---------------------------------------------------
                # If molecular dataset with gene_symbol
                # ---------------------------------------------------
                if "gene_symbol" in df.columns:

                    df = df.dropna(subset=["gene_symbol"])

                    df = df.drop_duplicates(
                        subset=["gene_symbol"]
                    )

                    df = df.set_index("gene_symbol")

                    # transpose
                    df = df.T

                    df.reset_index(inplace=True)

                    df.rename(
                        columns={"index": "sample_id"},
                        inplace=True
                    )

                # ---------------------------------------------------
                # Generic tabular dataset
                # ---------------------------------------------------
                else:

                    df.reset_index(inplace=True)

                    df.rename(
                        columns={"index": "sample_id"},
                        inplace=True
                    )

                # Fill missing values
                df = df.fillna(0)

                # ---------------------------------------------------
                # Feature alignment
                # ---------------------------------------------------
                X = df.drop(
                    columns=["sample_id"],
                    errors='ignore'
                )

                # Match available features only
                matching_features = [
                    f for f in selected_features
                    if f in X.columns
                ]

                if len(matching_features) == 0:

                    st.error(
                        "❌ No matching features found between uploaded dataset and trained model."
                    )

                    st.stop()

                X = X.reindex(
                    columns=matching_features,
                    fill_value=0
                )

                # ---------------------------------------------------
                # Select first sample
                # ---------------------------------------------------
                sample = X.iloc[[0]]

                # ---------------------------------------------------
                # Prediction
                # ---------------------------------------------------
                prediction = model.predict(sample)[0]

                probability = model.predict_proba(sample)[0]

                # ---------------------------------------------------
                # Result
                # ---------------------------------------------------
                st.success(
                    f"Prediction: {label_map.get(prediction, prediction)}"
                )

                # ---------------------------------------------------
                # Confidence + Probability
                # ---------------------------------------------------
                col1, col2 = st.columns(2)

                with col1:

                    st.subheader("📊 Confidence")

                    confidence = float(max(probability))

                    st.progress(confidence)

                    st.write(f"{confidence:.2f}")

                with col2:

                    st.subheader("📈 Probability")

                    fig, ax = plt.subplots()

                    labels = [
                        "Breast",
                        "Lung"
                    ]

                    colors = [
                        "#4CAF50",
                        "#FF5733"
                    ]

                    ax.bar(
                        labels,
                        probability,
                        color=colors
                    )

                    for i, v in enumerate(probability):

                        ax.text(
                            i,
                            v + 0.01,
                            f"{v:.2f}",
                            ha='center'
                        )

                    st.pyplot(fig)

                # ---------------------------------------------------
                # Feature Importance
                # ---------------------------------------------------
                st.subheader("🔬 Top Important Features")

                try:

                    top_features = get_top_features(
                        model,
                        matching_features
                    )

                    st.dataframe(top_features.head(10))

                    fig, ax = plt.subplots(figsize=(8, 5))

                    sns.barplot(
                        x="importance",
                        y="feature",
                        data=top_features.head(10),
                        ax=ax
                    )

                    st.pyplot(fig)

                except Exception as e:

                    st.warning(
                        f"Feature importance unavailable: {e}"
                    )

                # ---------------------------------------------------
                # PCA Visualization
                # ---------------------------------------------------
                st.subheader("📊 PCA Visualization")

                try:

                    pca_df = apply_pca(X)

                    fig, ax = plt.subplots(figsize=(7, 5))

                    ax.scatter(
                        pca_df["PC1"],
                        pca_df["PC2"],
                        alpha=0.6
                    )

                    if len(pca_df) > 0:

                        ax.scatter(
                            pca_df["PC1"].iloc[0],
                            pca_df["PC2"].iloc[0],
                            color="red",
                            label="Selected Sample"
                        )

                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")

                    ax.legend()

                    st.pyplot(fig)

                except Exception as e:

                    st.warning(f"PCA failed: {e}")

        except Exception as e:

            st.error(f"Processing Error: {e}")

# ===================================================
# SAMPLE DATA / MODEL EVALUATION
# ===================================================
elif option == "Sample Data":

    st.header("🧪 Model Evaluation")

    if st.button("Run Evaluation"):

        try:

            # ---------------------------------------------------
            # Load sample evaluation dataset
            # ---------------------------------------------------
            df = load_and_preprocess(
                "data/raw/proteomics.csv"
            )

            X = df.drop(
                columns=["sample_id", "label"],
                errors='ignore'
            )

            y = df["label"]

            # ---------------------------------------------------
            # Match features
            # ---------------------------------------------------
            matching_features = [
                f for f in selected_features
                if f in X.columns
            ]

            X = X.reindex(
                columns=matching_features,
                fill_value=0
            )

            # ---------------------------------------------------
            # Predictions
            # ---------------------------------------------------
            y_pred = model.predict(X)

            y_prob = model.predict_proba(X)[:, 1]

            st.success("✅ Evaluation Complete")

            # ---------------------------------------------------
            # ROC Curve
            # ---------------------------------------------------
            st.subheader("📈 ROC Curve")

            fpr, tpr, _ = roc_curve(y, y_prob)

            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()

            ax.plot(
                fpr,
                tpr,
                label=f"AUC = {roc_auc:.2f}"
            )

            ax.plot(
                [0, 1],
                [0, 1],
                linestyle='--'
            )

            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")

            ax.legend()

            st.pyplot(fig)

            # ---------------------------------------------------
            # Confusion Matrix
            # ---------------------------------------------------
            st.subheader("🧩 Confusion Matrix")

            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots()

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=ax
            )

            st.pyplot(fig)

        except Exception as e:

            st.error(f"Evaluation Error: {e}")

# ===================================================
# IMAGE PREDICTION
# ===================================================
elif option == "Image Prediction":

    st.header("🖼 Cancer Image Prediction")

    image_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if image_file is not None:

        image = Image.open(image_file)

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Predict Image"):

            try:

                result = predict_image(image)

                st.success(f"Prediction: {result}")

            except Exception as e:

                st.error(f"Image Prediction Error: {e}")
