
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# STEP 1: LOAD DATASET
# --------------------------------------------------

df = pd.read_csv("loan_default_customer_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# Remove ID column
X = df.drop(columns=["loan_default", "applicant_id"])
y = df["loan_default"]

# --------------------------------------------------
# STEP 2: TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

# First separate 20% test data
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Divide remaining 80% into 60% train and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

print("\nTRAIN / VALIDATION / TEST SPLIT")
print("--------------------------------")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nTest set is kept aside until final evaluation.")

# --------------------------------------------------
# STEP 3: PREPROCESSING PIPELINE
# --------------------------------------------------

numeric_columns = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# --------------------------------------------------
# STEP 4: DECISION TREE OVERFITTING
# --------------------------------------------------

depths = [1, 3, 5, 10, None]

train_accuracy = []
validation_accuracy = []

print("\nDECISION TREE OVERFITTING")
print("--------------------------------")

for depth in depths:

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    train_prediction = model.predict(X_train)
    validation_prediction = model.predict(X_val)

    train_score = accuracy_score(
        y_train, train_prediction
    )

    validation_score = accuracy_score(
        y_val, validation_prediction
    )

    train_accuracy.append(train_score)
    validation_accuracy.append(validation_score)

    print(
        f"Depth {depth}: "
        f"Train Accuracy = {train_score:.3f}, "
        f"Validation Accuracy = {validation_score:.3f}"
    )

# --------------------------------------------------
# STEP 5: PLOT OVERFITTING GRAPH
# --------------------------------------------------

depth_labels = ["1", "3", "5", "10", "None"]

plt.figure(figsize=(9, 5))

plt.plot(
    depth_labels,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    depth_labels,
    validation_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Decision Tree Maximum Depth")
plt.ylabel("Accuracy")

plt.title("Decision Tree - Overfitting Analysis")

plt.legend()
plt.grid(True)

plt.savefig("overfitting_graph.png")
plt.show()

print("\nOverfitting graph saved successfully!")

# --------------------------------------------------
# STEP 6: LOGISTIC REGRESSION - 5 FOLD CROSS VALIDATION
# --------------------------------------------------

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ))
])

cv_scores = cross_val_score(
    logistic_model,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nLOGISTIC REGRESSION - 5 FOLD CV")
print("--------------------------------")

print("Individual CV Scores:", np.round(cv_scores, 3))

print("Mean Accuracy:", round(cv_scores.mean(), 3))

print("Standard Deviation:", round(cv_scores.std(), 3))

print(
    f"CV Result: {cv_scores.mean():.3f} "
    f"+/- {cv_scores.std():.3f}"
)

# --------------------------------------------------
# STEP 7: DATA LEAKAGE DEMONSTRATION
# --------------------------------------------------

print("\nDATA LEAKAGE DEMONSTRATION")
print("--------------------------------")

# Select numerical columns
numeric_X = X.select_dtypes(include=np.number)

# Incorrect method: scaling before train-test split
bad_scaler = StandardScaler()

scaled_before_split = bad_scaler.fit_transform(
    numeric_X
)

print("Incorrect scaling: Scaler fitted before splitting.")

# Correct method: split first
X_num_train, X_num_val, y_num_train, y_num_val = train_test_split(
    numeric_X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

good_scaler = StandardScaler()

X_train_scaled = good_scaler.fit_transform(
    X_num_train
)

X_val_scaled = good_scaler.transform(
    X_num_val
)

print("Correct scaling: Scaler fitted only on training data.")
print("Training data transformed successfully.")
print("Validation data transformed successfully.")

# --------------------------------------------------
# STEP 8: SUPERVISED VS UNSUPERVISED LEARNING
# --------------------------------------------------

print("\nSUPERVISED VS UNSUPERVISED LEARNING")
print("--------------------------------")

problems = {
    "Spam Detection": "Supervised Learning",
    "Customer Segmentation": "Unsupervised Learning",
    "House Price Prediction": "Supervised Learning",
    "Anomaly Detection": "Usually Unsupervised Learning"
}

for problem, category in problems.items():
    print(problem, ":", category)

# --------------------------------------------------
# STEP 9: MACHINE LEARNING WORKFLOW
# --------------------------------------------------

print("\nMACHINE LEARNING WORKFLOW")
print("--------------------------------")

workflow = [
    "Define the problem and objective.",
    "Collect the required dataset.",
    "Perform Exploratory Data Analysis (EDA).",
    "Clean data and preprocess missing values.",
    "Split data into training, validation and testing sets.",
    "Train, validate and tune machine learning models.",
    "Evaluate the final model and deploy it."
]

for i, step in enumerate(workflow, 1):
    print(f"{i}. {step}")

print("\n================================")
print("TASK 1 COMPLETED SUCCESSFULLY!")
print("================================")
