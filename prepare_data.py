"""
prepare_data.py

Loads the Breast Cancer Wisconsin (Diagnostic) dataset -- a real, publicly
available clinical dataset -- and saves it as a CSV for analysis.

About the data (real, not simulated):
    569 samples, each describing a digitized image of a fine needle
    aspirate (FNA) of a breast mass. 30 numeric features are computed from
    the image (mean, standard error, and "worst"/largest value of 10 cell
    nucleus characteristics: radius, texture, perimeter, area, smoothness,
    compactness, concavity, concave points, symmetry, fractal dimension).
    The target is the diagnosis: malignant or benign.

Source: UCI Machine Learning Repository / Wisconsin Diagnostic Breast
Cancer (WDBC) dataset, originally from Dr. William H. Wolberg,
University of Wisconsin Hospitals. Distributed with scikit-learn
(`sklearn.datasets.load_breast_cancer`), so no download is required.

Reference: W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear
feature extraction for breast tumor diagnosis." IS&T/SPIE 1993
International Symposium on Electronic Imaging.

Run:
    python prepare_data.py

Output:
    data/breast_cancer_diagnostic.csv
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
import os


def prepare_dataset() -> pd.DataFrame:
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()

    # target: 0 = malignant, 1 = benign in sklearn's encoding.
    # Re-map to an explicit, readable label column, keep original 0/1 too.
    df["diagnosis"] = df["target"].map({0: "malignant", 1: "benign"})

    # Tidy column names (spaces -> underscores) for easier use in code
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = prepare_dataset()
    df.to_csv("data/breast_cancer_diagnostic.csv", index=False)
    print(f"Saved {len(df)} rows, {df.shape[1]} columns -> data/breast_cancer_diagnostic.csv")
    print(df["diagnosis"].value_counts())

