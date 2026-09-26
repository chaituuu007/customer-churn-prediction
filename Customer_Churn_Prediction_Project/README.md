# Customer Churn Prediction Using Logistic Regression

## Objective
Build a machine-learning classification pipeline to predict customer churn probability and assign customer retention-risk levels.

## Dataset
This project uses the supplied `customer_churn_sample.csv` dataset.

- Records: 15
- Columns: 11
- Target: `Churn`
- Model: Logistic Regression
- Split: 80% training / 20% testing
- Encoding: One-Hot Encoding
- Scaling: MinMaxScaler

## Pipeline
1. Load and inspect data
2. Remove duplicate Customer IDs
3. Encode target (`Yes` = 1, `No` = 0)
4. Separate numeric and categorical features
5. Impute missing values
6. Apply MinMax scaling to numeric features
7. Apply One-Hot Encoding to categorical features
8. Split into 80/20 train-test sets
9. Train Logistic Regression
10. Evaluate Precision, Recall, F1-Score and ROC-AUC
11. Export churn probabilities and risk levels

## Important limitation
The supplied dataset contains only 15 rows. Therefore, the evaluation metrics can vary substantially depending on the train/test split and should not be interpreted as production-level model performance. A larger dataset should be used for reliable model validation.

## Outputs
- `outputs/churn_predictions.csv`
- `outputs/model_metrics.csv`
- `outputs/roc_curve.png`
- `outputs/confusion_matrix.png`

## How to run
```bash
pip install -r requirements.txt
python src/churn_model.py
```
