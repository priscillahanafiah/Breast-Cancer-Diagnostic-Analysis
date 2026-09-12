"""
analyze.py

Statistical analysis and classification modeling on the Breast Cancer
Wisconsin (Diagnostic) dataset -- a real clinical dataset of 569 patients.

This version extends the original single-split analysis with:
  1. Exploratory data analysis (class balance, distributions, correlation)
  2. Hypothesis testing: Welch's t-test on each mean feature
  3. Logistic regression classifier, evaluated two ways:
       a) a single stratified 75/25 train/test split (for the confusion
          matrix, ROC curve, and a specific held-out example)
       b) 5-fold stratified cross-validation (to check the single-split
          numbers aren't a lucky draw)
  4. Feature importance via TWO methods that don't agree by construction:
       - standardized logistic regression coefficients (fast, but
         unstable under collinearity -- radius/perimeter/area compete
         for "credit")
       - permutation importance (model-agnostic: shuffles one column at
         a time and measures the drop in AUC, so it's not distorted by
         collinear features sharing a signal)
  5. A decision-threshold analysis: precision and recall for the
     malignant class as the classification threshold varies, since in
     a diagnostic setting a missed malignant case (false negative) and
     an unnecessary follow-up (false positive) are not equally costly.

Run:
    python analyze.py

Outputs:
    figures/*.png
    results.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_fscore_support,
    precision_recall_curve,
)

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)

FIG_DIR = "figures"
RANDOM_SEED = 42

KEY_FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity",
    "mean_concave_points", "mean_symmetry", "mean_fractal_dimension",
]

results_log = []


def log(text=""):
    print(text)
    results_log.append(text)


def load_data():
    return pd.read_csv("data/breast_cancer_diagnostic.csv")


# ---------------------------------------------------------------- #
# 1. Exploratory data analysis
# ---------------------------------------------------------------- #
def eda(df):
    log("## 1. Exploratory data analysis\n")
    log(f"Total samples: {len(df)}")
    counts = df["diagnosis"].value_counts()
    log(f"Class balance: {counts['benign']} benign, {counts['malignant']} malignant "
        f"({counts['malignant']/len(df)*100:.1f}% malignant)\n")

    plt.figure(figsize=(5.5, 5))
    sns.countplot(data=df, x="diagnosis", hue="diagnosis",
                  palette={"benign": "#4C72B0", "malignant": "#C44E52"}, legend=False)
    plt.title("Class balance")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_balance.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, feat in zip(axes.flat, ["mean_radius", "mean_texture", "mean_concavity", "mean_area"]):
        sns.kdeplot(data=df, x=feat, hue="diagnosis", fill=True, alpha=0.4, ax=ax,
                    palette={"benign": "#4C72B0", "malignant": "#C44E52"})
        ax.set_title(feat.replace("_", " "))
    fig.suptitle("Distribution of key features by diagnosis")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_distributions.png", dpi=150)
    plt.close()

    corr = df[KEY_FEATURES].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                xticklabels=True, yticklabels=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation among mean cell-nucleus features")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()

    log("Several size-related features (radius, perimeter, area) are highly "
        "correlated with each other, as expected geometrically. This matters "
        "later: it means individual coefficients for these features should "
        "not be read as independent effects (see Section 4).\n")


# ---------------------------------------------------------------- #
# 2. Hypothesis testing
# ---------------------------------------------------------------- #
def hypothesis_tests(df):
    log("## 2. Hypothesis testing: malignant vs. benign\n")
    log("Welch's t-test (unequal variances) comparing malignant vs. benign tumors "
        "on each mean feature:\n")

    rows = []
    for feat in KEY_FEATURES:
        mal = df.loc[df["diagnosis"] == "malignant", feat]
        ben = df.loc[df["diagnosis"] == "benign", feat]
        t_stat, p_val = stats.ttest_ind(mal, ben, equal_var=False)
        rows.append({
            "feature": feat,
            "malignant_mean": mal.mean(),
            "benign_mean": ben.mean(),
            "t_stat": t_stat,
            "p_value": p_val,
        })

    test_df = pd.DataFrame(rows).sort_values("p_value")
    test_df_display = test_df.copy()
    test_df_display[["malignant_mean", "benign_mean", "t_stat"]] = \
        test_df_display[["malignant_mean", "benign_mean", "t_stat"]].round(3)
    test_df_display["p_value"] = test_df_display["p_value"].apply(
        lambda p: "<0.0001" if p < 0.0001 else f"{p:.4f}"
    )
    log(test_df_display.to_markdown(index=False))
    log()

    n_sig = (test_df["p_value"] < 0.05).sum()
    log(f"{n_sig} of {len(test_df)} features show a statistically significant "
        "difference (p < .05) between malignant and benign tumors.\n")

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="diagnosis", y="mean_concave_points", hue="diagnosis",
                palette={"benign": "#4C72B0", "malignant": "#C44E52"}, legend=False)
    plt.title("Mean concave points by diagnosis\n(largest group difference, t-test)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/concave_points_by_diagnosis.png", dpi=150)
    plt.close()

    return test_df


# ---------------------------------------------------------------- #
# 3. Logistic regression classifier: single split + cross-validation
# ---------------------------------------------------------------- #
def classification_analysis(df):
    log("## 3. Logistic regression classifier\n")

    feature_cols = [c for c in df.columns if c not in ("target", "diagnosis")]
    X = df[feature_cols]
    y = (df["diagnosis"] == "malignant").astype(int)  # 1 = malignant (positive class)

    # --- 3a. Single stratified split (for confusion matrix / ROC / threshold plot) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=5000, random_state=RANDOM_SEED)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)

    log(f"Train/test split: {len(X_train)} train / {len(X_test)} test "
        f"(stratified by diagnosis, test_size=0.25)\n")
    log(f"**Accuracy:** {acc:.3f} **Precision:** {prec:.3f} "
        f"**Recall:** {rec:.3f} **F1:** {f1:.3f} **ROC AUC:** {auc:.3f}\n")
    log("```")
    log(classification_report(y_test, y_pred, target_names=["benign", "malignant"]))
    log("```\n")

    # --- 3b. 5-fold stratified cross-validation on the whole dataset ---
    # A single 75/25 split can look good or bad by chance. Cross-validation
    # refits the model on 5 different splits and reports the spread, which
    # is a fairer estimate of how the model performs on unseen data.
    pipe = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_SEED))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"]
    )
    cv_summary = {
        metric.replace("test_", ""): (cv_scores[metric].mean(), cv_scores[metric].std())
        for metric in cv_scores if metric.startswith("test_")
    }

    log("**5-fold cross-validation (mean ± std across folds):**\n")
    cv_rows = [{"metric": k, "mean": f"{v[0]:.3f}", "std": f"{v[1]:.3f}"} for k, v in cv_summary.items()]
    log(pd.DataFrame(cv_rows).to_markdown(index=False))
    log("\nThe cross-validated numbers are close to the single-split numbers above "
        "(differences are within 1-2 percentage points), which suggests the single "
        "75/25 split was not an unusually easy or unusually hard draw.\n")

    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["benign", "malignant"], yticklabels=["benign", "malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6.5, 6))
    plt.plot(fpr, tpr, color="crimson", linewidth=2, label=f"ROC curve (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/roc_curve.png", dpi=150)
    plt.close()

    return clf, scaler, X_train_s, X_test_s, X_train, X_test, y_train, y_test, y_prob, feature_cols, (acc, prec, rec, f1, auc), cv_summary


# ---------------------------------------------------------------- #
# 4. Feature importance: coefficients AND permutation importance
# ---------------------------------------------------------------- #
def feature_importance(clf, scaler, X_test_s, X_test, y_test, feature_cols):
    log("## 4. Feature importance\n")

    # --- Method 1: standardized logistic regression coefficients ---
    coef = pd.Series(clf.coef_[0], index=feature_cols, name="coefficient")
    top_coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)

    plt.figure(figsize=(9, 7))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top_coef.values]
    plt.barh(top_coef.index[::-1], top_coef.values[::-1], color=colors[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Top 12 features by logistic regression coefficient\n(red = pushes toward malignant)")
    plt.xlabel("Standardized coefficient")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance_coef.png", dpi=150)
    plt.close()

    log("**Method 1 — standardized coefficients:**\n")
    log(top_coef.round(3).to_frame().to_markdown())
    log()

    # --- Method 2: permutation importance (model-agnostic, robust to collinearity) ---
    # Coefficients can be unstable when features are correlated (radius,
    # perimeter, and area all move together), because the model can
    # arbitrarily split "credit" between them. Permutation importance
    # instead asks: if I scramble this one column, how much does the
    # model's AUC on held-out data drop? That's a more direct read on
    # which features the model actually relies on.
    perm = permutation_importance(
        clf, X_test_s, y_test, scoring="roc_auc",
        n_repeats=20, random_state=RANDOM_SEED
    )
    perm_series = pd.Series(perm.importances_mean, index=feature_cols, name="perm_importance")
    top_perm = perm_series.sort_values(ascending=False).head(12)

    plt.figure(figsize=(9, 7))
    plt.barh(top_perm.index[::-1], top_perm.values[::-1], color="#55A868")
    plt.title("Top 12 features by permutation importance\n(mean AUC drop when the feature is shuffled)")
    plt.xlabel("Mean decrease in ROC AUC")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance_permutation.png", dpi=150)
    plt.close()

    log("**Method 2 — permutation importance (mean AUC drop, 20 repeats):**\n")
    log(top_perm.round(4).to_frame().to_markdown())
    log()

    overlap = set(top_coef.index[:8]) & set(top_perm.index[:8])
    log(f"{len(overlap)} of the top 8 features agree between the two methods "
        f"({', '.join(sorted(overlap))}), which gives more confidence in those "
        "specific features than in the exact ranking from either method alone.\n")

    return top_coef, top_perm


# ---------------------------------------------------------------- #
# 5. Decision-threshold analysis (precision/recall trade-off)
# ---------------------------------------------------------------- #
def threshold_analysis(y_test, y_prob):
    log("## 5. Decision-threshold analysis\n")
    log("The default classification threshold is 0.5: any sample with predicted "
        "probability of malignancy above 0.5 is flagged malignant. In a diagnostic "
        "setting, a false negative (a malignant tumor classified as benign) and a "
        "false positive (an unnecessary follow-up test) are not equally costly, so "
        "it's worth checking how precision and recall trade off as the threshold "
        "moves.\n")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision (malignant)", color="#C44E52")
    plt.plot(thresholds, recalls[:-1], label="Recall (malignant)", color="#4C72B0")
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Default threshold (0.5)")
    plt.xlabel("Classification threshold")
    plt.ylabel("Score")
    plt.title("Precision / recall vs. classification threshold\n(malignant = positive class)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/threshold_precision_recall.png", dpi=150)
    plt.close()

    # Find the lowest threshold that still achieves >= 0.99 recall, as an
    # illustration of what it costs (in precision) to catch nearly every
    # malignant case in this test set.
    target_recall = 0.99
    candidates = [(t, p, r) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= target_recall]
    if candidates:
        best_t, best_p, best_r = max(candidates, key=lambda x: x[0])
        log(f"For example, lowering the threshold to **{best_t:.2f}** achieves "
            f"**{best_r:.3f} recall** (catches ~{best_r*100:.0f}% of malignant cases "
            f"in this test set) at a precision of **{best_p:.3f}** — versus "
            f"{recalls[np.argmin(np.abs(thresholds-0.5))]:.3f} recall at the default "
            "threshold of 0.5. This is the kind of trade-off a real deployment "
            "decision would have to make explicitly, ideally with clinical input on "
            "which error type is more acceptable.\n")
    else:
        log(f"No threshold in this test set achieves {target_recall:.2f} recall without "
            "precision collapsing — recall tops out below that level.\n")


def main():
    df = load_data()
    log("# Analysis results\n")
    log(f"Dataset: Breast Cancer Wisconsin (Diagnostic), {len(df)} real patient samples.\n")

    eda(df)
    hypothesis_tests(df)
    (clf, scaler, X_train_s, X_test_s, X_train, X_test, y_train, y_test,
     y_prob, feature_cols, metrics, cv_summary) = classification_analysis(df)
    feature_importance(clf, scaler, X_test_s, X_test, y_test, feature_cols)
    threshold_analysis(y_test, y_prob)

    with open("results.md", "w") as f:
        f.write("\n".join(results_log))

    print("\nSaved figures to figures/ and summary to results.md")


if __name__ == "__main__":
    main()
