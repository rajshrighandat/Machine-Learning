# Import required libraries

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

DATASET_FILE = "iris.csv"
TARGET_COLUMN = "species"

# If your dataset name or target column changes, update these two values only.
# Example: DATASET_FILE = "Car_Data.csv", TARGET_COLUMN = "Type"

try:
    df = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset file '{DATASET_FILE}' not found. Please update DATASET_FILE in this script.")

print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------------
# 2. Data Preprocessing
# --------------------------------------------------

# Remove unnecessary ID column
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Remove unnamed column if present
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Auto-detect target column if the dataset is different
if TARGET_COLUMN not in df.columns:
    for candidate in ['diagnosis', 'species', 'Survived', 'label', 'target', 'Class', 'Type']:
        if candidate in df.columns:
            TARGET_COLUMN = candidate
            break

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")

# Convert target variable into numerical form if it is categorical
if df[TARGET_COLUMN].dtype == object or pd.api.types.is_string_dtype(df[TARGET_COLUMN]):
    le = LabelEncoder()
    df[TARGET_COLUMN] = le.fit_transform(df[TARGET_COLUMN])
    print(f"Target labels encoded: {list(le.classes_)}")

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop(TARGET_COLUMN, axis=1)
y = df[TARGET_COLUMN]

# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. Evaluation Function
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred):

    print("\n===================================")
    print(name)
    print("===================================")

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    return accuracy, precision, recall, f1


# --------------------------------------------------
# 6. Model Selection Menu
# --------------------------------------------------

while True:
    print("\nChoose model to run:")
    print("1. Decision Tree")
    print("2. Random Forest")
    print("3. AdaBoost")
    print("4. All Models")
    print("5. Exit")

    choice = "4"
    try:
        user_choice = input("Enter your choice (1-5) or press Enter for default [4]: ").strip()
        if user_choice:
            choice = user_choice
    except KeyboardInterrupt:
        print("\nNo input received. Running all models by default.")
        choice = "4"

    # --------------------------------------------------
    # 7. Run Selected Model(s)
    # --------------------------------------------------

    if choice == '1':
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        evaluate_model("Decision Tree Classifier", y_test, pred)

    elif choice == '2':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        evaluate_model("Random Forest Classifier", y_test, pred)

    elif choice == '3':
        model = AdaBoostClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        evaluate_model("AdaBoost Classifier", y_test, pred)

    elif choice == '4':
        models = [
            ("Decision Tree Classifier", DecisionTreeClassifier(random_state=42)),
            ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("AdaBoost Classifier", AdaBoostClassifier(n_estimators=100, random_state=42))
        ]

        results = []

        for index, (name, model) in enumerate(models, start=1):
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            result = evaluate_model(name, y_test, pred)
            results.append((name, result))

            if index < len(models):
                try:
                    print("\nPress Enter to continue to the next model...")
                    input()
                except EOFError:
                    print("\nNo input received. Continuing to the next model automatically.")

        comparison = pd.DataFrame({
            'Model': [name for name, _ in results],
            'Accuracy': [r[0] for _, r in results],
            'Precision': [r[1] for _, r in results],
            'Recall': [r[2] for _, r in results],
            'F1-Score': [r[3] for _, r in results]
        })

        print("\n\nModel Comparison:")
        print(comparison)

    elif choice == '5':
        print("\nExiting the program.")
        break

    else:
        print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")

    print("\nDo you want to run another model?")
    print("1. Yes")
    print("2. No")
    try:
        next_choice = input("Enter your choice (1-2): ").strip()
        if next_choice == '1':
            continue
        elif next_choice == '2':
            print("\nExiting the program.")
            break
        else:
            print("\nInvalid choice. Exiting the program.")
            break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting the program.")
        break