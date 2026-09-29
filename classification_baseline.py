"""
Task 4 — Baseline Classification Pipeline (Telco Churn)

Usage:
    python classification_baseline.py "WA_Fn-UseC_-Telco-Customer-Churn(1).csv"

Optional:
    python classification_baseline.py input.csv MODEL_COMPARISON.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def make_pipeline(class_weight, numeric_features, categorical_features):
    return Pipeline(
        steps=[
            (
                "preprocess",
                build_preprocessor(numeric_features, categorical_features),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "cm": cm,
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare baseline Logistic Regression churn classifiers."
    )
    parser.add_argument("input_csv", help="Path to the Telco churn CSV.")
    parser.add_argument(
        "report_file",
        nargs="?",
        default="MODEL_COMPARISON.md",
        help="Markdown report to create.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    report_path = Path(args.report_file)

    df = pd.read_csv(input_path)

    required = {"customerID", "TotalCharges", "Churn"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    original_shape = df.shape

    # Fix TotalCharges: blank strings represent missing numeric values.
    total_charges_raw = df["TotalCharges"].copy()
    blank_total_charges = total_charges_raw.astype(str).str.strip().eq("").sum()
    df["TotalCharges"] = pd.to_numeric(
        total_charges_raw.astype(str).str.strip().replace("", np.nan),
        errors="coerce",
    )

    total_charges_missing_after_conversion = int(df["TotalCharges"].isna().sum())

    # Binary target: No=0, Yes=1.
    if not set(df["Churn"].dropna().unique()).issubset({"No", "Yes"}):
        raise ValueError("Unexpected values found in Churn. Expected only 'No' and 'Yes'.")

    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    # customerID is a pure identifier and must not be used as a predictor.
    X = df.drop(columns=["Churn", "customerID"]).copy()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numeric_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    # Mandatory: stratified split before fitting any preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Task 2 majority-class baseline: always predict the training majority class (No Churn).
    majority_class = int(y_train.mode()[0])
    baseline_predictions = np.full(len(y_test), majority_class)
    baseline_probabilities = np.full(len(y_test), y_train.mean())

    baseline_cm = confusion_matrix(
        y_test, baseline_predictions, labels=[0, 1]
    )
    baseline_metrics = {
        "accuracy": accuracy_score(y_test, baseline_predictions),
        "precision": precision_score(
            y_test, baseline_predictions, zero_division=0
        ),
        "recall": recall_score(y_test, baseline_predictions, zero_division=0),
        "f1": f1_score(y_test, baseline_predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, baseline_probabilities),
        "cm": baseline_cm,
    }

    # Default Logistic Regression.
    default_model = make_pipeline(
        None, numeric_features, categorical_features
    )
    default_results = evaluate(
        default_model, X_train, X_test, y_train, y_test
    )

    # Imbalance-aware Logistic Regression.
    balanced_model = make_pipeline(
        "balanced", numeric_features, categorical_features
    )
    balanced_results = evaluate(
        balanced_model, X_train, X_test, y_train, y_test
    )

    # One comparison table.
    rows = [
        {
            "Model": "Task 2 Majority-Class Baseline",
            "Accuracy": baseline_metrics["accuracy"],
            "Precision": baseline_metrics["precision"],
            "Recall": baseline_metrics["recall"],
            "F1": baseline_metrics["f1"],
            "ROC-AUC": baseline_metrics["roc_auc"],
        },
        {
            "Model": "Logistic Regression (default)",
            "Accuracy": default_results["accuracy"],
            "Precision": default_results["precision"],
            "Recall": default_results["recall"],
            "F1": default_results["f1"],
            "ROC-AUC": default_results["roc_auc"],
        },
        {
            "Model": "Logistic Regression (class_weight='balanced')",
            "Accuracy": balanced_results["accuracy"],
            "Precision": balanced_results["precision"],
            "Recall": balanced_results["recall"],
            "F1": balanced_results["f1"],
            "ROC-AUC": balanced_results["roc_auc"],
        },
    ]
    results_df = pd.DataFrame(rows)

    def pct(value):
        return f"{value:.4f}"

    def cm_table(cm):
        tn, fp, fn, tp = cm.ravel()
        return (
            "| Actual \\ Predicted | No Churn | Churn |\n"
            "|---|---:|---:|\n"
            f"| No Churn | {tn} | {fp} |\n"
            f"| Churn | {fn} | {tp} |\n"
        )

    # Business framing: missing a churner (FN) is assumed more costly than a false alarm.
    default_recall = default_results["recall"]
    balanced_recall = balanced_results["recall"]
    default_precision = default_results["precision"]
    balanced_precision = balanced_results["precision"]

    if balanced_recall > default_recall:
        recall_statement = (
            "The balanced model identifies more actual churners (higher recall), "
            "which directly addresses the stated higher cost of missed churners."
        )
    else:
        recall_statement = (
            "The balanced model does not increase churn recall on this split, so "
            "the class-weighting trade-off should be interpreted from all metrics."
        )

    report = f"""# Task 4 — Baseline Classification Pipeline (Telco Churn)

## Objective

Build a leakage-safe Logistic Regression pipeline for binary churn prediction, explicitly address class imbalance, and compare both Logistic Regression versions against the Task 2 majority-class baseline.

## Dataset and target

- Input: `{input_path.name}`
- Original shape: {original_shape[0]:,} rows × {original_shape[1]:,} columns
- Target: `Churn` (`No` = 0, `Yes` = 1)
- Positive class: churn (`Yes`)
- `customerID` excluded because it is an identifier, not a predictive feature.
- `TotalCharges` converted from string to numeric; blank strings were treated as missing.
- Blank `TotalCharges` values found: {blank_total_charges}
- Missing `TotalCharges` values after conversion: {total_charges_missing_after_conversion}

## Stratified split

- Train/test split: 80/20
- Random state: {RANDOM_STATE}
- Training rows: {len(X_train):,}
- Test rows: {len(X_test):,}
- Stratification: `stratify=y`
- Training churn rate: {y_train.mean():.4f}
- Test churn rate: {y_test.mean():.4f}

The stratified split keeps the churn/non-churn class proportions approximately consistent between train and test.

## Leakage prevention and preprocessing

The split occurs **before** preprocessing is fitted.

The preprocessing is contained inside a single scikit-learn `Pipeline`/`ColumnTransformer`:

- Numeric predictors → median imputation → `StandardScaler`
- Categorical predictors → most-frequent imputation → one-hot encoding with `handle_unknown="ignore"`
- `customerID` is removed before model fitting.
- `TotalCharges` is converted to numeric before the split because this is a deterministic data-type correction, not a statistic learned from the data.
- The imputer, scaler, and encoder are fitted only on the training partition and then applied unchanged to the test partition.

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
"""
    for _, row in results_df.iterrows():
        report += (
            f"| {row['Model']} | {pct(row['Accuracy'])} | "
            f"{pct(row['Precision'])} | {pct(row['Recall'])} | "
            f"{pct(row['F1'])} | {pct(row['ROC-AUC'])} |\n"
        )

    report += f"""
## Confusion matrices

The matrices use **rows = actual class** and **columns = predicted class**.

### Task 2 majority-class baseline

This baseline always predicts the training majority class (`No Churn`).

{cm_table(baseline_metrics['cm'])}

### Logistic Regression — default class weighting

{cm_table(default_results['cm'])}

### Logistic Regression — `class_weight='balanced'`

{cm_table(balanced_results['cm'])}

## Classification reports

### Logistic Regression — default

```text
{default_results['classification_report']}
```

### Logistic Regression — balanced

```text
{balanced_results['classification_report']}
```

## Business interpretation

The majority-class baseline is useful as a sanity check, but it does not identify churners: its churn recall is **{baseline_metrics['recall']:.4f}** and its churn F1 is **{baseline_metrics['f1']:.4f}**.

The default Logistic Regression achieves churn precision **{default_precision:.4f}**, churn recall **{default_recall:.4f}**, F1 **{default_results['f1']:.4f}**, and ROC-AUC **{default_results['roc_auc']:.4f}**.

The balanced Logistic Regression achieves churn precision **{balanced_precision:.4f}**, churn recall **{balanced_recall:.4f}**, F1 **{balanced_results['f1']:.4f}**, and ROC-AUC **{balanced_results['roc_auc']:.4f}**.

{recall_statement}

Because the task specifies that **missing a churner is likely more costly than a false alarm**, the deployment discussion should place particular emphasis on churn recall and the corresponding number of false negatives. The balanced model is specifically designed to give the minority class more weight during training, which commonly shifts the precision/recall trade-off toward detecting more positives. The confusion matrices make that trade-off explicit rather than relying on accuracy alone.

ROC-AUC is also useful for assessing ranking quality across thresholds, but it should not be treated as the only decision metric under class imbalance. A production deployment would ideally select a probability threshold using the actual business costs of retention offers versus missed churn.

## Metric meanings

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** among customers predicted to churn, the proportion who actually churn.
- **Recall:** among actual churners, the proportion detected by the model.
- **F1:** harmonic mean of precision and recall; useful when both matter.
- **ROC-AUC:** ability to rank positive cases above negative cases across classification thresholds.

## Conclusion

This task demonstrates stratified validation, leakage-safe preprocessing, Logistic Regression, explicit class-imbalance handling, and business-aware evaluation. The comparison against the majority-class baseline shows whether the model provides useful churn detection beyond simply predicting the dominant class.
"""

    report_path.write_text(report, encoding="utf-8")

    print(f"Input shape:   {df.shape}")
    print(f"Train shape:   {X_train.shape}")
    print(f"Test shape:    {X_test.shape}")
    print(f"TotalCharges blanks converted to NaN: {blank_total_charges}")
    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print("\nConfusion matrices [rows=actual, columns=predicted]:")
    print("\nTask 2 Majority-Class Baseline:\n", baseline_metrics["cm"])
    print("\nLogistic Regression (default):\n", default_results["cm"])
    print("\nLogistic Regression (balanced):\n", balanced_results["cm"])
    print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
