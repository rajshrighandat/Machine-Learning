# Student Performance Dataset
# Step 1: Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


data = pd.read_csv("student_data.csv")

print("===== DATASET =====")
print(data.head())

print("\n===== DATASET SHAPE =====")
print(data.shape)

print("\nNumber of rows:", data.shape[0])
print("Number of columns:", data.shape[1])


print("\n===== COLUMN NAMES =====")
print(data.columns)

print("\n===== MISSING VALUES =====")
print(data.isnull().sum())

# X = input features
X = data.drop("passed", axis=1)

# y = target/output
y = data["passed"]


# Convert categorical columns into numerical values
X = pd.get_dummies(X, drop_first=True)


# Convert target:
# no  = 0
# yes = 1

y = y.map({
    "no": 0,
    "yes": 1
})


print("\n===== ENCODED FEATURES =====")
print(X.head())

print("\n===== TARGET VALUES =====")
print(y.head())




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n===== DATA SPLIT =====")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))




model = AdaBoostClassifier(
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)




model.fit(X_train, y_train)

print("\nModel training completed successfully.")




y_pred = model.predict(X_test)


print("\n===== ACTUAL VALUES =====")
print(y_test.values)

print("\n===== PREDICTED VALUES =====")
print(y_pred)


# ============================================================
# Step 11: Calculate accuracy
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== ACCURACY =====")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")




cm = confusion_matrix(y_test, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)




print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fail", "Pass"]
    )
)




importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== FEATURE IMPORTANCE =====")
print(importance.head(10))