# Analysis results

Dataset: Breast Cancer Wisconsin (Diagnostic), 569 real patient samples.

## 1. Exploratory data analysis

Total samples: 569
Class balance: 357 benign, 212 malignant (37.3% malignant)

Several size-related features (radius, perimeter, area) are highly correlated with each other, as expected geometrically.

## 2. Hypothesis testing: malignant vs. benign

Welch's t-test (unequal variances) comparing malignant vs. benign tumors on each mean feature:

| feature                |   malignant_mean |   benign_mean |   t_stat | p_value   |
|:-----------------------|-----------------:|--------------:|---------:|:----------|
| mean_concave_points    |            0.088 |         0.026 |   24.845 | <0.0001   |
| mean_perimeter         |          115.365 |        78.075 |   22.935 | <0.0001   |
| mean_radius            |           17.463 |        12.147 |   22.209 | <0.0001   |
| mean_concavity         |            0.161 |         0.046 |   20.332 | <0.0001   |
| mean_area              |          978.376 |       462.79  |   19.641 | <0.0001   |
| mean_compactness       |            0.145 |         0.08  |   15.818 | <0.0001   |
| mean_texture           |           21.605 |        17.915 |   11.022 | <0.0001   |
| mean_smoothness        |            0.103 |         0.092 |    9.297 | <0.0001   |
| mean_symmetry          |            0.193 |         0.174 |    8.112 | <0.0001   |
| mean_fractal_dimension |            0.063 |         0.063 |   -0.297 | 0.7667    |

9 of 10 features show a statistically significant difference (p < .05) between malignant and benign tumors.

## 3. Logistic regression classifier

Train/test split: 426 train / 143 test (stratified by diagnosis, test_size=0.25)

**Accuracy:** 0.965   **Precision:** 0.980   **Recall:** 0.925   **F1:** 0.951   **ROC AUC:** 0.996

```
              precision    recall  f1-score   support

      benign       0.96      0.99      0.97        90
   malignant       0.98      0.92      0.95        53

    accuracy                           0.97       143
   macro avg       0.97      0.96      0.96       143
weighted avg       0.97      0.97      0.96       143

```

## 4. Feature importance

Top predictors, ranked by absolute standardized logistic regression coefficient:

|                     |   coefficient |
|:--------------------|--------------:|
| worst_texture       |         1.366 |
| radius_error        |         1.267 |
| worst_symmetry      |         1.036 |
| mean_concave_points |         0.963 |
| compactness_error   |        -0.927 |
| worst_area          |         0.889 |
| worst_radius        |         0.887 |
| area_error          |         0.878 |
| mean_concavity      |         0.874 |
| worst_concavity     |         0.868 |
| perimeter_error     |         0.832 |
| worst_perimeter     |         0.739 |
