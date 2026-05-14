import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.decomposition import PCA

# =========================================================
# OPTIONAL UMAP IMPORT
# =========================================================

try:
    import umap
    UMAP_AVAILABLE = True

except:
    UMAP_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Cancer Type Classification",
    layout="wide"
)

# =========================================================
# LOAD MODEL FILES
# =========================================================

try:

    model = pickle.load(
        open("models/cancer_model.pkl", "rb")
    )

    top_genes = pickle.load(
        open("models/selected_features.pkl", "rb")
    )

except Exception as e:

    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# =========================================================
# TITLE
# =========================================================

st.title("🧬 Cancer Type Classification System")
st.subheader("Breast Cancer (BRCA) vs Lung Cancer (LUAD)")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Controls")

option = st.sidebar.selectbox(
    "Choose Input Method",
    ["Sample Input", "Upload CSV"]
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(data):

    pred = model.predict(data)

    prob = model.predict_proba(data)

    return pred, prob

# =========================================================
# SAMPLE INPUT
# =========================================================

if option == "Sample Input":

    st.write("Run prediction using randomly generated sample data.")

    if st.button("Run Sample Prediction"):

        sample = pd.DataFrame(
            np.random.rand(1, len(top_genes)),
            columns=top_genes
        )

        pred, prob = predict(sample)

        result = (
            "BRCA (Breast Cancer)"
            if pred[0] == 0
            else "LUAD (Lung Cancer)"
        )

        confidence = np.max(prob) * 100

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")

# =========================================================
# CSV UPLOAD
# =========================================================

elif option == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            # =================================================
            # LOAD CSV
            # =================================================

            df = pd.read_csv(uploaded_file)

            st.write("## Dataset Preview")
            st.dataframe(df.head())

            st.info(f"Dataset Shape: {df.shape}")

            # =================================================
            # CLEAN COLUMN NAMES
            # =================================================

            df.columns = df.columns.astype(str)

            df.columns = [
                col.strip()
                for col in df.columns
            ]

            # =================================================
            # REMOVE COMMON NON-FEATURE COLUMNS
            # =================================================

            remove_cols = [
                "sample_id",
                "gene_symbol",
                "gene_name",
                "id",
                "Unnamed: 0",
                "index"
            ]

            df = df.drop(
                columns=[
                    c for c in remove_cols
                    if c in df.columns
                ],
                errors="ignore"
            )

            # =================================================
            # AUTOMATIC TRANSPOSE DETECTION
            # =================================================

            if df.shape[0] > df.shape[1]:

                st.warning(
                    "Detected gene-oriented dataset. "
                    "Automatically transposing..."
                )

                first_col = df.columns[0]

                df = df.set_index(first_col).T

                df.reset_index(drop=True, inplace=True)

            # =================================================
            # FIND MATCHING FEATURES
            # =================================================

            matching_features = [
                gene
                for gene in top_genes
                if gene in df.columns
            ]

            st.success(
                f"Matching Features Found: "
                f"{len(matching_features)}"
            )

            # =================================================
            # FEATURE CHECK
            # =================================================

            if len(matching_features) < 5:

                st.error(
                    "❌ Very few matching genes found.\n\n"
                    "Please upload a compatible "
                    "molecular dataset."
                )

                st.stop()

            # =================================================
            # KEEP ONLY MATCHING FEATURES
            # =================================================

            df_model = df[matching_features].copy()

            # =================================================
            # ADD MISSING FEATURES
            # =================================================

            missing_features = list(
                set(top_genes) - set(matching_features)
            )

            for feature in missing_features:

                df_model[feature] = 0

            # =================================================
            # REORDER COLUMNS
            # =================================================

            df_model = df_model[top_genes]

            # =================================================
            # HANDLE MISSING VALUES
            # =================================================

            df_model = df_model.fillna(0)

            # =================================================
            # CONVERT TO NUMERIC
            # =================================================

            df_model = df_model.apply(
                pd.to_numeric,
                errors="coerce"
            )

            df_model = df_model.fillna(0)

            # =================================================
            # PREDICT BUTTON
            # =================================================

            if st.button("Predict"):

                pred, prob = predict(df_model)

                results = pd.DataFrame()

                results["Prediction"] = [
                    "BRCA"
                    if p == 0
                    else "LUAD"
                    for p in pred
                ]

                results["Confidence"] = np.max(
                    prob,
                    axis=1
                )

                st.write("## Prediction Results")
                st.dataframe(results)

                # =============================================
                # CLASS DISTRIBUTION
                # =============================================

                fig, ax = plt.subplots()

                results["Prediction"].value_counts().plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title("Prediction Distribution")

                st.pyplot(fig)

                # =============================================
                # PCA VISUALIZATION
                # =============================================

                st.write("## PCA Visualization")

                pca = PCA(n_components=2)

                reduced = pca.fit_transform(df_model)

                fig, ax = plt.subplots()

                ax.scatter(
                    reduced[:, 0],
                    reduced[:, 1]
                )

                ax.set_title("PCA Projection")

                st.pyplot(fig)

                # =============================================
                # UMAP VISUALIZATION
                # =============================================

                st.write("## UMAP Visualization")

                if UMAP_AVAILABLE:

                    reducer = umap.UMAP(
                        random_state=42
                    )

                    embedding = reducer.fit_transform(
                        df_model
                    )

                    fig, ax = plt.subplots()

                    ax.scatter(
                        embedding[:, 0],
                        embedding[:, 1]
                    )

                    ax.set_title("UMAP Projection")

                    st.pyplot(fig)

                else:

                    st.warning(
                        "UMAP is not installed."
                    )

        except Exception as e:

            st.error(f"❌ Processing Error: {e}")

# =========================================================
# DEMO CONFUSION MATRIX
# =========================================================

st.write("---")
st.write("## Model Evaluation Demonstration")

if st.button("Show Demo Confusion Matrix"):

    y_true = np.random.randint(0, 2, 100)

    y_pred = np.random.randint(0, 2, 100)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

# =========================================================
# DEMO ROC CURVE
# =========================================================

if st.button("Show Demo ROC Curve"):

    y_true = np.random.randint(0, 2, 100)

    y_scores = np.random.rand(100)

    fpr, tpr, _ = roc_curve(y_true, y_scores)

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
        linestyle="--"
    )

    ax.set_title("ROC Curve")

    ax.legend()

    st.pyplot(fig)

# =========================================================
# DEMO CLASSIFICATION REPORT
# =========================================================

if st.button("Show Demo Classification Report"):

    y_true = np.random.randint(0, 2, 100)

    y_pred = np.random.randint(0, 2, 100)

    report = classification_report(
        y_true,
        y_pred
    )

    st.text(report)
