# Breast Cancer Diagnostic Data Analysis

Statistical analysis and a classification model on the **Breast Cancer
Wisconsin (Diagnostic)** dataset — 569 real patient samples, each describing
30 numeric features computed from a digitized image of a fine needle
aspirate (FNA) of a breast mass, labeled malignant or benign.

This is real, publicly available clinical data — not simulated. It's
distributed with scikit-learn (`sklearn.datasets.load_breast_cancer`),
originally from Dr. William H. Wolberg, University of Wisconsin Hospitals.

> W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear feature extraction
> for breast tumor diagnosis." *IS&T/SPIE 1993 International Symposium on
> Electronic Imaging.*

**Note on origin:** this project started from a public template exploring
the same dataset with hypothesis testing and logistic regression — a
common exercise, since WDBC is one of the standard benchmark datasets in
ML coursework. What I added on top: 5-fold cross-validation (to check the
single-split results were stable), permutation-based feature importance
(to check the coefficient rankings against a method that isn't distorted
by collinear features), a decision-threshold analysis, and the discussion
in `discussion.md` connecting the results to clinical decision-making and
patient experience.

## What's in here

| File | Purpose |
|---|---|
| `prepare_data.py` | Loads the dataset and exports it to CSV |
| `analyze.py` | Runs the full analysis end-to-end and saves all figures |
| `results.md` | Plain-text summary of the analysis output |
| `discussion.md` | Reflection connecting the results to med-tech / psychology |
| `figures/` | Output plots |

## Method

1. **Exploratory data analysis** — class balance, feature distributions by
   diagnosis, correlation structure among features.
2. **Hypothesis testing** — Welch's t-test on each of the 10 mean features,
   comparing malignant vs. benign tumors.
3. **Logistic regression classifier** — trained on all 30 features with a
   stratified 75/25 train/test split and standardized inputs, cross-checked
   with 5-fold stratified cross-validation.
4. **Feature importance** — standardized coefficients *and* permutation
   importance, compared against each other since several features are
   collinear.
5. **Decision-threshold analysis** — how precision and recall for the
   malignant class trade off as the classification threshold moves, since
   false negatives and false positives are not equally costly in a
   diagnostic setting.

## Key findings

- **9 of 10** mean cell-nucleus features differ significantly (p < .05)
  between malignant and benign tumors. Only `mean_fractal_dimension` shows
  no significant difference.
- The logistic regression classifier achieves **~96–97% accuracy** and
  **ROC AUC ≈ 0.996** on a held-out split, confirmed by 5-fold
  cross-validation (accuracy 0.974 ± 0.017).
- Feature importance is broadly consistent across two different methods
  (coefficients and permutation importance) for about two-thirds of the
  top features — the rest disagree, most likely due to correlated
  features splitting credit between them.
- At the default threshold (0.5), the model misses about 7.5% of
  malignant cases. Lowering the threshold to ~0.06 catches nearly all of
  them, at the cost of more false positives — the kind of trade-off a
  real deployment would need clinical input to resolve.

Full output is in [`results.md`](results.md). For discussion of what these
results mean beyond the statistics, see [`discussion.md`](discussion.md).

## Running it

```
pip install -r requirements.txt

python prepare_data.py   # exports data/breast_cancer_diagnostic.csv
python analyze.py        # runs the full analysis, saves figures/ and results.md
```

## Limitations

- This is a well-studied benchmark dataset in the ML literature, which is
  part of why performance is so high; real-world diagnostic accuracy
  depends heavily on imaging quality, population, and clinical context.
  This project is a statistics/ML exercise, not a diagnostic tool.
- Several features are highly correlated (radius, perimeter, and area all
  measure tumor size), so individual regression coefficients should be
  interpreted as part of a correlated group rather than fully independent
  effects — this is part of why permutation importance is included as a
  cross-check.
- The model's calibration (whether a predicted 0.8 probability really
  corresponds to an 80% real-world malignancy rate) was not evaluated
  here — only its ability to rank cases correctly (AUC). This would
  matter before showing predicted probabilities to a clinician.
- External validation on data from a different hospital/population would
  be needed before any of these findings could inform real triage
  decisions.
