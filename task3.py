
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, silhouette_score

from xgboost import XGBClassifier


# ==========================================
# 1. CREATE FOLDERS
# ==========================================

Path("reports").mkdir(exist_ok=True)

print("\nTASK 3 STARTED")


# ==========================================
# 2. LOAD DATASET
# ==========================================

DATA_PATH = Path("data/loan_default_customer_dataset.csv")

if not DATA_PATH.exists():
    DATA_PATH = Path("loan_default_customer_dataset.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found. Place loan_default_customer_dataset.csv "
        "inside the data folder or project folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDATASET LOADED SUCCESSFULLY")
print("Dataset Shape:", df.shape)


# ==========================================
# 3. HANDLE MISSING TARGET VALUES
# ==========================================

print("\nMissing values in target before cleaning:",
      df["loan_default"].isna().sum())

# Remove rows with missing target
df = df.dropna(subset=["loan_default"]).copy()

# Ensure target is numeric
df["loan_default"] = pd.to_numeric(
    df["loan_default"],
    errors="coerce"
)

df = df.dropna(subset=["loan_default"]).copy()

# Ensure target is binary
valid_targets = {0, 1}

if not set(df["loan_default"].unique()).issubset(valid_targets):
    raise ValueError(
        "loan_default must contain only 0 and 1. "
        "Check the target column values."
    )

df["loan_default"] = df["loan_default"].astype(int)

print("Missing target values after cleaning:",
      df["loan_default"].isna().sum())

print("Target distribution:")
print(df["loan_default"].value_counts())

if df["loan_default"].nunique() != 2:
    raise ValueError(
        "Both target classes (0 and 1) must be present."
    )


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        date = pd.to_datetime(
            X["application_date"],
            errors="coerce"
        )

        # Date decomposition
        X["application_year"] = date.dt.year
        X["application_month"] = date.dt.month
        X["application_day"] = date.dt.day

        # Income / loan interaction
        X["income_loan_ratio"] = (
            X["annual_income"] /
            X["loan_amount"].replace(0, np.nan)
        )

        # Log transformation
        X["log_annual_income"] = np.log1p(
            X["annual_income"].clip(lower=0)
        )

        X = X.drop(
            columns=["application_date"]
        )

        return X


# ==========================================
# 5. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["loan_default", "applicant_id"],
    errors="ignore"
)

y = df["loan_default"]

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
# 6. PREPROCESSING PIPELINE
# ==========================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(
        categories=[education_order],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    ))
])

column_transformer = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("ordinal", ordinal_pipeline, ordinal_features)
    ],
    remainder="drop"
)

preprocessing = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("columns", column_transformer)
])


# ==========================================
# 7. TRAIN AND VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING ROWS:", len(X_train))
print("VALIDATION ROWS:", len(X_val))


# ==========================================
# 8. TRAIN FOUR CLASSIFIERS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
trained_models = {}

for name, classifier in models.items():

    print("\nTraining:", name)

    model = Pipeline(steps=[
        ("preprocessing", clone(preprocessing)),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    results[name] = accuracy
    trained_models[name] = model

    print("Validation Accuracy:", round(accuracy, 4))

    if name == "Random Forest":

        oob = model.named_steps["classifier"].oob_score_

        print("Random Forest OOB Score:", round(oob, 4))


# ==========================================
# 9. XGBOOST WITH EARLY STOPPING
# ==========================================

print("\nTraining XGBoost with Early Stopping...")

# Fit preprocessing only on training data
xgb_preprocessor = clone(preprocessing)

X_train_xgb = xgb_preprocessor.fit_transform(
    X_train,
    y_train
)

X_val_xgb = xgb_preprocessor.transform(X_val)

xgb_classifier = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)

# Train XGBoost on transformed training data
xgb_classifier.fit(
    X_train_xgb,
    y_train,
    eval_set=[(X_val_xgb, y_val)],
    verbose=False
)

# Create complete pipeline for future predictions
xgb_model = Pipeline(steps=[
    ("preprocessing", xgb_preprocessor),
    ("classifier", xgb_classifier)
])

xgb_predictions = xgb_model.predict(X_val)

xgb_accuracy = accuracy_score(
    y_val,
    xgb_predictions
)

results["XGBoost"] = xgb_accuracy
trained_models["XGBoost"] = xgb_model

print("XGBoost Validation Accuracy:", round(xgb_accuracy, 4))
print("XGBoost Best Iteration:", xgb_classifier.best_iteration)


# ==========================================
# 10. CLASSIFICATION RESULTS TABLE
# ==========================================

print("\n========================================")
print("CLASSIFICATION RESULTS")
print("========================================")

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "Validation Score"]
)

print(results_df.to_string(index=False))

results_df.to_csv(
    "classification_results.csv",
    index=False
)


# ==========================================
# 11. LOGISTIC REGRESSION PROBABILITIES
# ==========================================

print("\n========================================")
print("LOGISTIC REGRESSION PROBABILITIES")
print("========================================")

logistic_model = trained_models["Logistic Regression"]

first_five = X_val.iloc[:5]

probabilities = logistic_model.predict_proba(first_five)

predictions = logistic_model.predict(first_five)

probability_df = pd.DataFrame({
    "Actual": y_val.iloc[:5].values,
    "Predicted": predictions,
    "Probability_No_Default": probabilities[:, 0],
    "Probability_Default": probabilities[:, 1]
})

print(probability_df.to_string(index=False))


# ==========================================
# 12. RANDOM FOREST TOP 5 FEATURES
# ==========================================

print("\n========================================")
print("RANDOM FOREST TOP 5 FEATURES")
print("========================================")

forest_model = trained_models["Random Forest"]

forest_classifier = forest_model.named_steps["classifier"]

forest_names = (
    forest_model.named_steps["preprocessing"]
    .named_steps["columns"]
    .get_feature_names_out()
)

forest_importance = pd.DataFrame({
    "Feature": forest_names,
    "Importance": forest_classifier.feature_importances_
})

forest_importance = forest_importance.sort_values(
    by="Importance",
    ascending=False
)

print(forest_importance.head(5).to_string(index=False))


# ==========================================
# 13. XGBOOST TOP 5 FEATURES
# ==========================================

print("\n========================================")
print("XGBOOST TOP 5 FEATURES")
print("========================================")

xgb_names = (
    xgb_model.named_steps["preprocessing"]
    .named_steps["columns"]
    .get_feature_names_out()
)

xgb_importance = pd.DataFrame({
    "Feature": xgb_names,
    "Importance": xgb_classifier.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Importance",
    ascending=False
)

print(xgb_importance.head(5).to_string(index=False))


# ==========================================
# 14. REGRESSION MODELS
# ==========================================

print("\n========================================")
print("LINEAR REGRESSION, RIDGE AND LASSO")
print("========================================")

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

# Remove rows with missing loan amount
regression_df = df.dropna(subset=["loan_amount"]).copy()

X_reg = regression_df[regression_features]
y_reg = regression_df["loan_amount"]

X_reg_train, X_reg_val, y_reg_train, y_reg_val = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

regression_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_reg_train_scaled = regression_preprocessor.fit_transform(
    X_reg_train
)

X_reg_val_scaled = regression_preprocessor.transform(
    X_reg_val
)

regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000)
}

coefficient_results = {}

for name, reg_model in regression_models.items():

    reg_model.fit(
        X_reg_train_scaled,
        y_reg_train
    )

    score = reg_model.score(
        X_reg_val_scaled,
        y_reg_val
    )

    coefficient_results[name] = reg_model.coef_

    print("\nModel:", name)
    print("Validation R2 Score:", round(score, 4))
    print("Intercept:", round(reg_model.intercept_, 2))
    print("Coefficients:", np.round(reg_model.coef_, 3))

    if name == "Lasso":

        zero_count = np.sum(
            np.isclose(reg_model.coef_, 0)
        )

        print("Lasso Zero Coefficients:", zero_count)


# ==========================================
# 15. REGRESSION COEFFICIENT COMPARISON
# ==========================================

coefficient_df = pd.DataFrame(
    coefficient_results,
    index=regression_features
)

print("\nREGRESSION COEFFICIENT COMPARISON")

print(coefficient_df.round(3))

coefficient_df.to_csv(
    "regression_coefficients.csv"
)


# ==========================================
# 16. K-MEANS CLUSTERING
# ==========================================

print("\n========================================")
print("K-MEANS CLUSTERING")
print("========================================")

cluster_features = [
    "annual_income",
    "loan_amount",
    "credit_score"
]

cluster_data = df[cluster_features].copy()

cluster_imputer = SimpleImputer(
    strategy="median"
)

cluster_scaler = StandardScaler()

cluster_data = cluster_imputer.fit_transform(
    cluster_data
)

cluster_data = cluster_scaler.fit_transform(
    cluster_data
)

inertias = []
silhouette_scores = []

k_values = list(range(2, 9))

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(cluster_data)

    inertia = kmeans.inertia_

    silhouette = silhouette_score(
        cluster_data,
        labels
    )

    inertias.append(inertia)
    silhouette_scores.append(silhouette)

    print(
        f"K={k} | "
        f"Inertia={inertia:.2f} | "
        f"Silhouette Score={silhouette:.4f}"
    )


# ==========================================
# 17. ELBOW CURVE
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    inertias,
    marker="o"
)

plt.title("K-Means Elbow Curve")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid(True)

plt.savefig(
    "reports/elbow_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nElbow Curve Saved Successfully!")


# ==========================================
# 18. SILHOUETTE SCORE PLOT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    silhouette_scores,
    marker="o"
)

plt.title("K-Means Silhouette Scores")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(k_values)
plt.grid(True)

plt.savefig(
    "reports/silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Silhouette Plot Saved Successfully!")


# ==========================================
# 19. CUSTOMER SEGMENT ANALYSIS
# ==========================================

best_k = k_values[
    np.argmax(silhouette_scores)
]

print("\nBest K based on silhouette score:", best_k)

best_kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["customer_segment"] = best_kmeans.fit_predict(
    cluster_data
)

segment_summary = df.groupby(
    "customer_segment"
)[cluster_features].mean().round(2)

print("\nCUSTOMER SEGMENT ANALYSIS")
print(segment_summary)

for segment, row in segment_summary.iterrows():

    print(
        f"\nSegment {segment}: "
        f"Average Income = {row['annual_income']:.2f}, "
        f"Average Loan = {row['loan_amount']:.2f}, "
        f"Average Credit Score = {row['credit_score']:.2f}"
    )

df.to_csv(
    "customer_segments.csv",
    index=False
)


# ==========================================
# 20. SAVE TRAINED MODELS
# ==========================================

joblib.dump(
    trained_models,
    "task3_classification_models.pkl"
)

print("\nClassification Models Saved Successfully!")


# ==========================================
# 21. FINAL OUTPUT
# ==========================================

print("\n========================================")
print("ALL TASK 3 REQUIREMENTS COMPLETED!")
print("========================================")

print("\nGenerated Files:")

print("1. classification_results.csv")
print("2. regression_coefficients.csv")
print("3. reports/elbow_curve.png")
print("4. reports/silhouette_scores.png")
print("5. customer_segments.csv")
print("6. task3_classification_models.pkl")

print("\nTASK 3 COMPLETED SUCCESSFULLY!")
