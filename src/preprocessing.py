import pandas as pd
import numpy as np

def load_and_preprocess(path):
    df = pd.read_csv(path)

    # Drop unnecessary columns
    df = df.drop(columns=["RefSeq_accession_number", "gene_name"], errors='ignore')

    # Remove rows with missing gene_symbol
    df = df.dropna(subset=["gene_symbol"])

    # Remove duplicate gene names
    df = df.drop_duplicates(subset=["gene_symbol"])

    # Set gene_symbol as index
    df = df.set_index("gene_symbol")

    # Transpose
    df = df.T

    # Reset index
    df.reset_index(inplace=True)
    df.rename(columns={"index": "sample_id"}, inplace=True)

    # Fill missing values
    df = df.fillna(0)

    # -------- CREATE LABELS --------
    def assign_label(sample):
        if "TCGA" in sample:
            return 0   # BRCA (example)
        elif "CPTAC" in sample:
            return 1   # LUAD (example)
        else:
            return -1  # unknown

    df["label"] = df["sample_id"].apply(assign_label)

    # Remove unknown labels
    df = df[df["label"] != -1]

    return df


if __name__ == "__main__":
    data_path = "data/raw/proteomics.csv"

    df = load_and_preprocess(data_path)

    print("Final shape:", df.shape)
    print("\nLabel distribution:\n", df["label"].value_counts())
    print("\nSample data:\n", df.head())