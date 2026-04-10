import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_and_preprocess
from utils import get_top_features, apply_pca
from image_model import predict_image

# Load model + features
model = joblib.load("models/cancer_model.pkl")
selected_features = joblib.load("models/selected_features.pkl")

# Config
st.set_page_config(page_title="Cancer Classifier", layout="wide")

# Sidebar
st.sidebar.title("🧬 Navigation")
option = st.sidebar.radio("Go to", ["Home", "Upload Data", "Sample Data", "Image Prediction"])

# Title
st.title("🧬 Cancer Classification System")
st.markdown("### ML-based Molecular & Image Cancer Prediction")

label_map = {
    0: "Breast Cancer (TCGA)",
    1: "Lung Cancer (CPTAC)"
}

# ------------------ HOME ------------------
if option == "Home":
    st.write("### 📌 About Project")
    st.write("""
    This system uses machine learning to classify cancer types 
    using molecular expression data and image inputs.
    """)

# ------------------ UPLOAD ------------------
elif option == "Upload Data":
    st.header("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("### Preview")
        st.dataframe(raw_df.head())

        if st.button("🔍 Predict"):
            try:
                df = raw_df.copy()

                # Preprocessing
                df = df.drop(columns=["RefSeq_accession_number", "gene_name"], errors='ignore')
                df = df.dropna(subset=["gene_symbol"])
                df = df.drop_duplicates(subset=["gene_symbol"])
                df = df.set_index("gene_symbol")
                df = df.T
                df.reset_index(inplace=True)
                df.rename(columns={"index": "sample_id"}, inplace=True)
                df = df.fillna(0)

                # Feature alignment
                X = df.drop(columns=["sample_id"], errors='ignore')
                X = X.reindex(columns=selected_features, fill_value=0)

                sample = X.iloc[[0]]

                # 🔥 DIRECT MODEL PREDICTION (NO API)
                prediction = model.predict(sample)[0]
                probability = model.predict_proba(sample)[0]

                st.success(f"Prediction: {label_map[prediction]}")

                # Confidence + Probability
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Confidence")
                    st.progress(float(max(probability)))
                    st.write(f"{max(probability):.2f}")

                with col2:
                    st.subheader("📈 Probability")
                    fig, ax = plt.subplots()
                    labels = ["Breast", "Lung"]
                    colors = ["#4CAF50", "#FF5733"]

                    ax.bar(labels, probability, color=colors)

                    for i, v in enumerate(probability):
                        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

                    st.pyplot(fig)

                # Feature importance
                st.subheader("🔬 Top Important Genes")

                top_features = get_top_features(model, selected_features)

                for _, row in top_features.head(3).iterrows():
                    st.markdown(f"⭐ **{row['feature']}** → {row['importance']:.3f}")

                fig, ax = plt.subplots()
                sns.barplot(
                    x="importance",
                    y="feature",
                    data=top_features,
                    ax=ax
                )
                st.pyplot(fig)

                # PCA
                st.subheader("📊 PCA Visualization")

                pca_df = apply_pca(X)

                fig, ax = plt.subplots()
                ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.6)
                ax.scatter(pca_df["PC1"].iloc[0], pca_df["PC2"].iloc[0], color="red", label="Selected Sample")
                ax.legend()
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Processing Error: {e}")

# ------------------ SAMPLE DATA ------------------
elif option == "Sample Data":
    st.header("🧪 Model Evaluation")

    if st.button("Run Evaluation"):
        try:
            df = load_and_preprocess("data/raw/proteomics.csv")

            X = df.drop(columns=["sample_id", "label"])
            y = df["label"]

            X = X.reindex(columns=selected_features, fill_value=0)

            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1]

            st.success("Evaluation Complete")

            # ROC
            st.subheader("📈 ROC Curve")

            fpr, tpr, _ = roc_curve(y, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.legend()
            st.pyplot(fig)

            # Confusion Matrix
            st.subheader("📊 Confusion Matrix")

            cm = confusion_matrix(y, y_pred)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

            # Precision-Recall
            st.subheader("📉 Precision-Recall Curve")

            precision, recall, _ = precision_recall_curve(y, y_prob)

            fig, ax = plt.subplots()
            ax.plot(recall, precision)
            st.pyplot(fig)

        except Exception as e:
            st.error(str(e))

# ------------------ IMAGE ------------------
elif option == "Image Prediction":
    st.header("🖼️ Image Classification")

    image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image)

        if st.button("Predict Image"):
            result = predict_image(image)
            st.success(result)