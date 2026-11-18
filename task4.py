
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyRegressor

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ==========================================================
# 1. PROJECT PATHS AND DATASET
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

DATA_PATH = BASE_DIR / "data" / "loan_default_customer_dataset.csv"

if not DATA_PATH.exists():
    DATA_PATH = BASE_DIR / "loan_default_customer_dataset.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Put loan_default_customer_dataset.csv "
        "inside the data folder or project folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET ==========")
print("Dataset shape:", df.shape)
print(df.head())


# ==========================================================
# 2. CLEAN TARGET COLUMN
# ==========================================================

TARGET = "loan_default"

if TARGET not in df.columns:
    raise ValueError("The dataset does not contain loan_default.")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("loan_default must contain only 0 and 1.")

if df[TARGET].nunique() != 2:
    raise ValueError("Both classes (0 and 1) must be present.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

print("\n========== CLASS DISTRIBUTION ==========")
print(y.value_counts())
print("\nClass proportions:")
print(y.value_counts(normalize=True).round(3))


# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Date decomposition
        if "application_date" in X.columns:
            date = pd.to_datetime(
                X["application_date"],
                errors="coerce"
            )

            X["application_year"] = date.dt.year
            X["application_month"] = date.dt.month
            X["application_day"] = date.dt.day

            X = X.drop(columns=["application_date"])

        # Income / loan amount interaction
        if {"annual_income", "loan_amount"}.issubset(X.columns):
            loan = pd.to_numeric(
                X["loan_amount"], errors="coerce"
            ).replace(0, np.nan)

            income = pd.to_numeric(
                X["annual_income"], errors="coerce"
            )

            X["income_loan_ratio"] = income / loan

        # Log transformation
        if "annual_income" in X.columns:
            income = pd.to_numeric(
                X["annual_income"], errors="coerce"
            )

            X["log_annual_income"] = np.log1p(
                income.clip(lower=0)
            )

        return X


# Inspect the columns after feature engineering.
preview = FeatureEngineer().fit_transform(X)

ordinal_columns = []
if "education_level" in preview.columns:
    ordinal_columns = ["education_level"]

numeric_columns = preview.select_dtypes(
    include=["number"]
).columns.tolist()

numeric_columns = [
    col for col in numeric_columns
    if col not in ordinal_columns
]

categorical_columns = preview.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

categorical_columns = [
    col for col in categorical_columns
    if col not in ordinal_columns
]

print("\nNumeric columns:", numeric_columns)
print("Categorical columns:", categorical_columns)
print("Ordinal columns:", ordinal_columns)


# ==========================================================
# 4. PREPROCESSING PIPELINE
# ==========================================================

def create_preprocessor():

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        transformers.append((
            "numeric",
            numeric_pipeline,
            numeric_columns
        ))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        transformers.append((
            "categorical",
            categorical_pipeline,
            categorical_columns
        ))

    if ordinal_columns:
        ordinal_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(
                categories=[[
                    "High School",
                    "Diploma",
                    "Bachelor",
                    "Master",
                    "Doctorate"
                ]],
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ))
        ])

        transformers.append((
            "ordinal",
            ordinal_pipeline,
            ordinal_columns
        ))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("columns", column_transformer)
    ])


def create_model_pipeline(classifier):
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier)
    ])


# ==========================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ==========================================================

X_train_all, X_test, y_train_all, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all,
    y_train_all,
    test_size=0.25,
    random_state=42,
    stratify=y_train_all
)

print("\n========== DATA SPLIT ==========")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# ==========================================================
# 6. LOGISTIC REGRESSION
# ==========================================================

print("\nTraining Logistic Regression...")

logistic_model = create_model_pipeline(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
)

logistic_model.fit(X_train, y_train)


# ==========================================================
# 7. DECISION TREE
# ==========================================================

print("\nTraining Decision Tree...")

decision_tree_model = create_model_pipeline(
    DecisionTreeClassifier(
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
)

decision_tree_model.fit(X_train, y_train)


# ==========================================================
# 8. RANDOM FOREST - GRID SEARCH CV
# ==========================================================

print("\n========== RANDOM FOREST GRID SEARCH ==========")

rf_pipeline = create_model_pipeline(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )
)

rf_parameters = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5]
}

rf_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_parameters,
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=1
)

rf_search.fit(X_train, y_train)

best_rf = rf_search.best_estimator_

print("\nRandom Forest best parameters:")
print(rf_search.best_params_)

print(
    "Random Forest best CV ROC-AUC:",
    round(rf_search.best_score_, 4)
)


# ==========================================================
# 9. XGBOOST - RANDOMIZED SEARCH CV
# ==========================================================

print("\n========== XGBOOST RANDOMIZED SEARCH ==========")

xgb_pipeline = create_model_pipeline(
    XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1
    )
)

xgb_parameters = {
    "classifier__n_estimators": [100, 200, 300, 400],
    "classifier__max_depth": [3, 4, 5, 6],
    "classifier__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameters,
    n_iter=15,
    scoring="roc_auc",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

xgb_search.fit(X_train, y_train)

best_xgb = xgb_search.best_estimator_

print("\nXGBoost best parameters:")
print(xgb_search.best_params_)

print(
    "XGBoost best CV ROC-AUC:",
    round(xgb_search.best_score_, 4)
)


# ==========================================================
# 10. VALIDATION METRICS FOR ALL CLASSIFIERS
# ==========================================================

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": best_rf,
    "XGBoost": best_xgb
}

validation_results = []
validation_probabilities = {}

print("\n========== CLASSIFICATION METRICS ==========")

for name, model in models.items():

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    validation_probabilities[name] = probabilities

    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(
        y_val, predictions, zero_division=0
    )
    recall = recall_score(
        y_val, predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, predictions, zero_division=0
    )
    auc = roc_auc_score(y_val, probabilities)

    matrix = confusion_matrix(y_val, predictions)

    validation_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC-AUC": auc
    })

    print(f"\n{name}")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))
    print("Confusion Matrix:")
    print(matrix)

validation_df = pd.DataFrame(validation_results)

validation_df.to_csv(
    REPORT_DIR / "task4_validation_results.csv",
    index=False
)

print("\nValidation comparison:")
print(validation_df.round(4).to_string(index=False))


# ==========================================================
# 11. ROC CURVES FOR THREE MODELS
# ==========================================================

plt.figure(figsize=(9, 7))

roc_model_names = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

for name in roc_model_names:

    probabilities = validation_probabilities[name]

    fpr, tpr, _ = roc_curve(y_val, probabilities)
    auc = roc_auc_score(y_val, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Validation Data")
plt.legend()
plt.grid(True)

plt.savefig(
    REPORT_DIR / "task4_roc_curves.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nROC curve saved in reports folder.")


# ==========================================================
# 12. SELECT BEST CLASSIFICATION MODEL
# ==========================================================

best_row = validation_df.loc[
    validation_df["ROC-AUC"].idxmax()
]

best_model_name = best_row["Model"]
best_model = models[best_model_name]

print("\n========== BEST CLASSIFICATION MODEL ==========")
print("Best model:", best_model_name)
print("Validation ROC-AUC:", round(best_row["ROC-AUC"], 4))
print("Validation F1:", round(best_row["F1"], 4))


# ==========================================================
# 13. DECISION THRESHOLD - MINIMUM 75% RECALL
# ==========================================================

best_val_probabilities = validation_probabilities[
    best_model_name
]

threshold_candidates = np.unique(
    np.concatenate((
        np.arange(0.01, 1.00, 0.01),
        best_val_probabilities
    ))
)

threshold_results = []

for threshold in threshold_candidates:

    threshold_predictions = (
        best_val_probabilities >= threshold
    ).astype(int)

    threshold_recall = recall_score(
        y_val,
        threshold_predictions,
        zero_division=0
    )

    threshold_precision = precision_score(
        y_val,
        threshold_predictions,
        zero_division=0
    )

    if threshold_recall >= 0.75:
        threshold_results.append({
            "Threshold": threshold,
            "Recall": threshold_recall,
            "Precision": threshold_precision
        })

if threshold_results:

    threshold_df = pd.DataFrame(threshold_results)

    # Highest precision while recall remains >= 75%.
    selected_row = threshold_df.loc[
        threshold_df["Precision"].idxmax()
    ]

    selected_threshold = float(selected_row["Threshold"])

    print("\n========== SELECTED THRESHOLD ==========")
    print("Threshold:", round(selected_threshold, 4))
    print("Recall:", round(selected_row["Recall"], 4))
    print("Precision:", round(selected_row["Precision"], 4))

    threshold_df.to_csv(
        REPORT_DIR / "task4_threshold_results.csv",
        index=False
    )

else:

    selected_threshold = 0.50

    print(
        "\nNo tested threshold achieved 75% recall "
        "on the validation set."
    )
    print("Fallback threshold:", selected_threshold)


# ==========================================================
# 14. REGRESSION METRICS VS MEAN BASELINE
# ==========================================================

print("\n========== REGRESSION EVALUATION ==========")

regression_target = "loan_amount"

regression_features = [
    "age",
    "annual_income",
    "credit_score",
    "dependents",
    "loan_term_months",
    "existing_debt",
    "employment_years",
    "previous_defaults"
]

if regression_target not in df.columns:
    raise ValueError("loan_amount column is missing for regression.")

available_regression_features = [
    col for col in regression_features
    if col in df.columns
]

if not available_regression_features:
    raise ValueError("No regression input features were found.")

regression_data = df[
    available_regression_features + [regression_target]
].copy()

regression_data[regression_target] = pd.to_numeric(
    regression_data[regression_target],
    errors="coerce"
)

regression_data = regression_data.dropna(
    subset=[regression_target]
)

X_reg = regression_data[available_regression_features]
y_reg = regression_data[regression_target]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

regression_results = []
fitted_regression_models = {}

for name, regressor in regression_models.items():

    regression_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", regressor)
    ])

    regression_pipeline.fit(X_reg_train, y_reg_train)

    predictions = regression_pipeline.predict(X_reg_val)

    mae = mean_absolute_error(y_reg_val, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_reg_val, predictions)
    )

    r2 = r2_score(y_reg_val, predictions)

    fitted_regression_models[name] = regression_pipeline

    regression_results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# Mean baseline
baseline_model = DummyRegressor(strategy="mean")

baseline_model.fit(X_reg_train, y_reg_train)

baseline_predictions = baseline_model.predict(X_reg_val)

baseline_mae = mean_absolute_error(
    y_reg_val, baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(y_reg_val, baseline_predictions)
)

baseline_r2 = r2_score(
    y_reg_val, baseline_predictions
)

regression_results.append({
    "Model": "Mean Baseline",
    "MAE": baseline_mae,
    "RMSE": baseline_rmse,
    "R2": baseline_r2
})

regression_results_df = pd.DataFrame(regression_results)

print("\nRegression comparison:")
print(regression_results_df.round(4).to_string(index=False))

regression_results_df.to_csv(
    REPORT_DIR / "task4_regression_results.csv",
    index=False
)

best_regression_row = regression_results_df[
    regression_results_df["Model"] != "Mean Baseline"
].sort_values("R2", ascending=False).iloc[0]

print("\nBest regression model:", best_regression_row["Model"])
print("MAE :", round(best_regression_row["MAE"], 4))
print("RMSE:", round(best_regression_row["RMSE"], 4))
print("R2  :", round(best_regression_row["R2"], 4))

print("\nMean baseline:")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R2  :", round(baseline_r2, 4))


# ==========================================================
# 15. FINAL TEST SET EVALUATION - ONLY ONCE
# ==========================================================

print("\n========== FINAL TEST EVALUATION ==========")

# Refit selected model using training + validation data.
# Test data has not been used for tuning or model selection.
final_model = best_model
final_model.fit(X_train_all, y_train_all)

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

test_accuracy = accuracy_score(y_test, test_predictions)

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

test_auc = roc_auc_score(y_test, test_probabilities)

test_matrix = confusion_matrix(y_test, test_predictions)

print("Best model:", best_model_name)
print("Decision threshold:", round(selected_threshold, 4))
print("Test Accuracy :", round(test_accuracy, 4))
print("Test Precision:", round(test_precision, 4))
print("Test Recall   :", round(test_recall, 4))
print("Test F1       :", round(test_f1, 4))
print("Test ROC-AUC  :", round(test_auc, 4))

print("\nTest Confusion Matrix:")
print(test_matrix)

print("\nTest Classification Report:")
print(classification_report(
    y_test,
    test_predictions,
    zero_division=0
))


# ==========================================================
# 16. SAVE FINAL MODEL AND RESULTS
# ==========================================================

joblib.dump(
    {
        "model": final_model,
        "threshold": selected_threshold,
        "model_name": best_model_name
    },
    BASE_DIR / "task4_best_model.pkl"
)

final_results = pd.DataFrame([{
    "Best Model": best_model_name,
    "Threshold": selected_threshold,
    "Accuracy": test_accuracy,
    "Precision": test_precision,
    "Recall": test_recall,
    "F1": test_f1,
    "ROC-AUC": test_auc
}])

final_results.to_csv(
    REPORT_DIR / "task4_final_test_results.csv",
    index=False
)

print("\n========== TASK 4 COMPLETED SUCCESSFULLY ==========")
print("Saved model: task4_best_model.pkl")
print("Saved reports in:", REPORT_DIR)
