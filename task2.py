
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = Path("loan_default_customer_dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH.resolve()}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["loan_default", "applicant_id"],
    errors="ignore"
)

y = df["loan_default"]


# ==========================================
# 3. FEATURE ENGINEERING TRANSFORMER
# ==========================================

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Convert application date
        date = pd.to_datetime(
            X["application_date"],
            errors="coerce"
        )

        # Feature 1: Date decomposition
        X["application_year"] = date.dt.year
        X["application_month"] = date.dt.month
        X["application_day"] = date.dt.day

        # Feature 2: Income / loan amount interaction
        X["income_loan_ratio"] = (
            X["annual_income"] /
            X["loan_amount"].replace(0, np.nan)
        )

        # Feature 3: Log transformation
        X["log_annual_income"] = np.log1p(
            X["annual_income"].clip(lower=0)
        )

        # Remove original date column
        X = X.drop(columns=["application_date"])

        return X


# ==========================================
# 4. DEFINE COLUMN GROUPS
# ==========================================

numeric_features = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "dependents",
    "loan_term_months",
    "existing_debt",
    "employment_years",
    "previous_defaults",
    "application_year",
    "application_month",
    "application_day",
    "income_loan_ratio",
    "log_annual_income"
]

categorical_features = [
    "employment_type",
    "marital_status",
    "property_area",
    "loan_purpose"
]

ordinal_features = ["education_level"]

education_order = [
    "High School",
    "Diploma",
    "Bachelor",
    "Master",
    "Doctorate"
]


# ==========================================
# 5. NUMERIC PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ==========================================
# 6. CATEGORICAL PREPROCESSING
# ==========================================

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# ==========================================
# 7. ORDINAL PREPROCESSING
# ==========================================

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[education_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])


# ==========================================
# 8. COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("ordinal", ordinal_pipeline, ordinal_features)
    ],
    remainder="drop"
)


# ==========================================
# 9. COMPLETE MACHINE LEARNING PIPELINE
# ==========================================

model_pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])


# ==========================================
# 10. CROSS VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n================================")
print("CROSS VALIDATION RESULTS")
print("================================")

print("Individual Scores:", cv_scores)
print("Pipeline CV accuracy:", round(cv_scores.mean(), 4))


# ==========================================
# 11. TRY KNN IMPUTER ON NUMERIC COLUMNS
# ==========================================

knn_preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", Pipeline(steps=[
            ("imputer", KNNImputer(n_neighbors=5)),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ],
    remainder="drop"
)

knn_pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessing", knn_preprocessor),
    ("classifier", LogisticRegression(max_iter=2000))
])

knn_scores = cross_val_score(
    knn_pipeline,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n================================")
print("KNN IMPUTER RESULTS")
print("================================")

print("KNN Individual Scores:", knn_scores)
print("KNN CV Accuracy:", round(knn_scores.mean(), 4))


# ==========================================
# 12. FIT FINAL PIPELINE
# ==========================================

model_pipeline.fit(X, y)

print("\nFinal Pipeline Trained Successfully!")


# ==========================================
# 13. SAVE PIPELINE USING JOBLIB
# ==========================================

MODEL_PATH = "loan_preprocessing_pipeline.pkl"

joblib.dump(model_pipeline, MODEL_PATH)

print("Pipeline Saved Successfully!")


# ==========================================
# 14. LOAD SAVED PIPELINE
# ==========================================

loaded_pipeline = joblib.load(MODEL_PATH)

print("Saved Pipeline Loaded Successfully!")


# ==========================================
# 15. PREDICT FOR 3 NEW RAW ROWS
# ==========================================

new_customers = pd.DataFrame([

    {
        "age": 28,
        "annual_income": 650000,
        "loan_amount": 200000,
        "credit_score": 750,
        "dependents": 1,
        "loan_term_months": 36,
        "existing_debt": 30000,
        "employment_years": 5,
        "previous_defaults": 0,
        "employment_type": "Salaried",
        "marital_status": "Single",
        "property_area": "Urban",
        "loan_purpose": "Personal",
        "education_level": "Bachelor",
        "application_date": "2026-10-15"
    },

    {
        "age": 45,
        "annual_income": 350000,
        "loan_amount": 500000,
        "credit_score": 580,
        "dependents": 3,
        "loan_term_months": 60,
        "existing_debt": 150000,
        "employment_years": 12,
        "previous_defaults": 2,
        "employment_type": "Self-employed",
        "marital_status": "Married",
        "property_area": "Rural",
        "loan_purpose": "Business",
        "education_level": "Diploma",
        "application_date": "2026-10-20"
    },

    {
        "age": 32,
        "annual_income": 900000,
        "loan_amount": 300000,
        "credit_score": 800,
        "dependents": 0,
        "loan_term_months": 24,
        "existing_debt": 20000,
        "employment_years": 8,
        "previous_defaults": 0,
        "employment_type": "Salaried",
        "marital_status": "Married",
        "property_area": "Urban",
        "loan_purpose": "Home",
        "education_level": "Master",
        "application_date": "2026-11-01"
    }

])


# Make predictions using raw data
predictions = loaded_pipeline.predict(new_customers)

print("\n================================")
print("PREDICTIONS FOR 3 NEW CUSTOMERS")
print("================================")

print("Predictions:", predictions)

print("\nPrediction Labels:")

for i, prediction in enumerate(predictions, start=1):

    label = (
        "Loan Default"
        if prediction == 1
        else "No Loan Default"
    )

    print(f"Customer {i}: {label}")


print("\n================================")
print("TASK 2 COMPLETED SUCCESSFULLY!")
print("================================")
