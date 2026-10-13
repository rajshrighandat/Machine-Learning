# Import libraries
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the student dataset
df = pd.read_csv(Path(__file__).parent / "student-scores.csv")

print(df.head())
print(df.describe())

# Use subject scores to predict the student's career aspiration
features = df.filter(regex="_score$")
target = df["career_aspiration"]

# Convert career names to numbers for the classifier and confusion matrix
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target_encoded,
    test_size=0.2,
    random_state=42,
    stratify=target_encoded,
)

# Scale scores because KNN compares distances between samples
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the K-Nearest Neighbors model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict career aspirations for the test data
y_pred = knn.predict(X_test_scaled)

print("\nK-Nearest Neighbors Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)

# Display the confusion matrix
matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=range(len(label_encoder.classes_)),
)
plt.figure(figsize=(12, 9))
sns.heatmap(
    matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)
plt.xlabel("Predicted Career")
plt.ylabel("Actual Career")
plt.title("KNN Career Aspiration Prediction")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
