# Breast Cancer Diagnostic Data Analysis

Statistical analysis and a classification model on the **Breast Cancer
Wisconsin (Diagnostic)** dataset — 569 real patient samples, each describing
30 numeric features computed from a digitized image of a fine needle
aspirate (FNA) of a breast mass (cell nucleus radius, texture, perimeter,
smoothness, concavity, symmetry, etc.), labeled malignant or benign.

This is real, publicly available clinical data — not simulated. It's
distributed with scikit-learn (`sklearn.datasets.load_breast_cancer`),
originally from Dr. William H. Wolberg, University of Wisconsin Hospitals.

> W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear feature extraction
> for breast tumor diagnosis." *IS&T/SPIE 1993 International Symposium on
> Electronic Imaging.*

## What's in here

| File | Purpose |
|---|---|
| `prepare_data.py` | Loads the dataset and exports it to CSV |
| `analyze.py` | Runs the full analysis end-to-end and saves all figures |
| `notebooks/analysis.ipynb` | The same analysis, notebook form, with narrative alongside the code |
| `data/breast_cancer_diagnostic.csv` | The exported dataset |
| `figures/` | Output plots |
| `results.md` | Plain-text summary of the analysis output |

## Method

1. **Exploratory data analysis** — class balance, feature distributions by
   diagnosis, correlation structure among features.
2. **Hypothesis testing** — Welch's t-test on each of the 10 mean features,
   comparing malignant vs. benign tumors.
3. **Logistic regression classifier** — trained on all 30 features with a
   stratified 75/25 train/test split and standardized inputs.
4. **Model evaluation** — accuracy, precision/recall/F1, confusion matrix,
   ROC curve/AUC, and feature importance via standardized coefficients.

## Key findings

- **9 of 10** mean cell-nucleus features differ significantly (p < .05)
  between malignant and benign tumors. Only `mean_fractal_dimension` shows
  no significant difference.
- Malignancy is most strongly associated with larger, more irregular cell
  nuclei — `mean_concave_points` shows the largest group difference.
- The logistic regression classifier achieves **~96–97% accuracy** and
  **ROC AUC ≈ 0.996** on held-out test data.
- The largest standardized coefficients (`worst_texture`, `radius_error`,
  `worst_symmetry`, `mean_concave_points`) tell the same story as the
  hypothesis tests: irregularity and size variability are the dominant
  signals for malignancy in this dataset.

![Feature distributions](figures/feature_distributions.png)
![Correlation heatmap](figures/correlation_heatmap.png)
![Confusion matrix](figures/confusion_matrix.png)
![ROC curve](figures/roc_curve.png)
![Feature importance](figures/feature_importance.png)

Full output (t-test table, classification report, feature importance table)
is in [`results.md`](results.md).

## Running it

```bash
git clone https://github.com/<your-username>/breast-cancer-diagnostic-analysis.git
cd breast-cancer-diagnostic-analysis
pip install -r requirements.txt

python prepare_data.py   # exports data/breast_cancer_diagnostic.csv
python analyze.py        # runs the full analysis, saves figures/ and results.md

# or, to explore interactively:
jupyter notebook notebooks/analysis.ipynb
```

## Limitations

- This is a well-studied benchmark dataset in the ML literature, which is
  part of why performance is so high; real-world diagnostic accuracy
  depends heavily on imaging quality, population, and clinical context.
  This project is a statistics/ML exercise, not a diagnostic tool.
- Several features are highly correlated (radius, perimeter, and area all
  measure tumor size), so individual regression coefficients should be
  interpreted as part of a correlated group rather than fully independent
  effects.
