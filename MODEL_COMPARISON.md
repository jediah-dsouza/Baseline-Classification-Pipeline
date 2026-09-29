# Task 4 — Baseline Classification Pipeline (Telco Churn)

## Objective

Build a leakage-safe Logistic Regression pipeline for binary churn prediction, explicitly address class imbalance, and compare both Logistic Regression versions against the Task 2 majority-class baseline.

## Dataset and target

- Input: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Original shape: 7,043 rows × 21 columns
- Target: `Churn` (`No` = 0, `Yes` = 1)
- Positive class: churn (`Yes`)
- `customerID` excluded because it is an identifier, not a predictive feature.
- `TotalCharges` converted from string to numeric; blank strings were treated as missing.
- Blank `TotalCharges` values found: 11
- Missing `TotalCharges` values after conversion: 11

## Stratified split

- Train/test split: 80/20
- Random state: 42
- Training rows: 5,634
- Test rows: 1,409
- Stratification: `stratify=y`
- Training churn rate: 0.2654
- Test churn rate: 0.2654

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
| Task 2 Majority-Class Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| Logistic Regression (default) | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8419 |
| Logistic Regression (class_weight='balanced') | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8413 |

## Confusion matrices

The matrices use **rows = actual class** and **columns = predicted class**.

### Task 2 majority-class baseline

This baseline always predicts the training majority class (`No Churn`).

| Actual \ Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 1035 | 0 |
| Churn | 374 | 0 |


### Logistic Regression — default class weighting

| Actual \ Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 926 | 109 |
| Churn | 165 | 209 |


### Logistic Regression — `class_weight='balanced'`

| Actual \ Predicted | No Churn | Churn |
|---|---:|---:|
| No Churn | 747 | 288 |
| Churn | 81 | 293 |


## Classification reports

### Logistic Regression — default

```text
              precision    recall  f1-score   support

    No Churn       0.85      0.89      0.87      1035
       Churn       0.66      0.56      0.60       374

    accuracy                           0.81      1409
   macro avg       0.75      0.73      0.74      1409
weighted avg       0.80      0.81      0.80      1409

```

### Logistic Regression — balanced

```text
              precision    recall  f1-score   support

    No Churn       0.90      0.72      0.80      1035
       Churn       0.50      0.78      0.61       374

    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409

```
## Reproduce

```bash
python regression_baseline.py ames_cleaned.csv
```
## Business interpretation

The majority-class baseline is useful as a sanity check, but it does not identify churners: its churn recall is **0.0000** and its churn F1 is **0.0000**.

The default Logistic Regression achieves churn precision **0.6572**, churn recall **0.5588**, F1 **0.6040**, and ROC-AUC **0.8419**.

The balanced Logistic Regression achieves churn precision **0.5043**, churn recall **0.7834**, F1 **0.6136**, and ROC-AUC **0.8413**.

The balanced model identifies more actual churners (higher recall), which directly addresses the stated higher cost of missed churners.

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
