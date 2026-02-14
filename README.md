# Customer Churn Prediction Model

## Project Overview
This project implements a machine learning solution to predict customer churn (customers likely to leave a service). It analyzes customer data (demographics, services used, billing info) to identify at-risk users, enabling proactive retention strategies.

## Key Achievements
- **High Recall for Churners:** Successfully identifies **~76%** of customers who are actually leaving.
- **Addressed Overfitting:** Reduced the gap between training and testing accuracy from **~19%** to **~7%**, creating a robust, generalizable model.
- **Handled Imbalanced Data:** Implemented advanced techniques to ensure the model doesn't just bias towards the majority "Stay" class.

## Technical Implementation
### 1. Algorithm: XGBoost Classifier
Chosen over Random Forest for better performance on tabular data and built-in regularization.

### 2. Data Preprocessing
- **Cleaning:** Handled missing values in `TotalCharges`.
- **Encoding:** Converted categorical text (e.g., "Yes", "Male") into numbers using `LabelEncoder`.
- **Splitting:** Used `StratifiedKFold` to ensure train/test sets have the same proportion of churners.

### 3. Solving Class Imbalance
- **Problem:** Dataset had 3x more "Stayers" than "Churners", causing models to ignore churners.
- **Solution:** Used `scale_pos_weight` in XGBoost (~2.77x penalty). This forces the model to treat every missed churner as highly important without deleting valuable data.

### 4. Hyperparameter Tuning
Used `GridSearchCV` to test 36 combinations of settings.
- **Optimized Parameters:**
  - `max_depth`: 5 (Limited tree depth to prevent memorization/overfitting).
  - `n_estimators`: 100 (Number of boosting rounds).
  - `learning_rate`: 0.1 (Step size for weight updates).
  - `subsample` & `colsample_bytree`: 0.8 (Used 80% of data/features per tree to add randomness).

## Files in Repository
- `churn_prediction_xgboost.py`: The main script. Loads data, trains model, tunes hyperparameters, evaluates performance, and saves the model.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset.
- `customer_churn_model_xgboost.pkl`: The saved, trained model (ready for deployment).
- `encoders.pkl`: Saved encoding rules (required to process new data).
- `churn_env/`: (Directory) Local Python virtual environment containing dependencies.

## How to Run

1. **Activate Environment:**
   ```bash
   source churn_env/bin/activate
   ```

2. **Run Script:**
   ```bash
   python churn_prediction_xgboost.py
   ```

3. **Output:**
   The script will print accuracy metrics, the confusion matrix, and the classification report to the terminal.

## Metrics Explained
- **Precision (0.53):** When predicting "Churn", correct 53% of the time. (Higher false alarms, but necessary for safety).
- **Recall (0.75):** Correctly identifies 75% of ALL actual churners. (This is the critical metric for retention).
- **Accuracy (0.76):** Overall correct predictions across both classes.
