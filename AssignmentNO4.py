import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score
)


# Load dataset
df = pd.read_csv('loan_data.csv')


# Display dataset
print(df.head())

print(df.tail())

print("Shape:")
print(df.shape)

print("Information:")
print(df.info())

print("Description:")
print(df.describe())

print("Missing Values:")
print(df.isnull().sum())




df = pd.get_dummies(
    df,
    columns=['purpose'],
    drop_first=True
)



X = df.drop(['not.fully.paid'], axis=1)

y = df['not.fully.paid']




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




model = GaussianNB()




model.fit(X_train, y_train)


# Prediction
y_pred = model.predict(X_test)


# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))


# Precision
print("Precision:", precision_score(y_test, y_pred, zero_division=0))


# Recall
print("Recall:", recall_score(y_test, y_pred, zero_division=0))


# F1 Score
print("F1-Score:", f1_score(y_test, y_pred, zero_division=0))


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Fully Paid', 'Not Fully Paid'],
    yticklabels=['Fully Paid', 'Not Fully Paid']
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Naive Bayes")

plt.show()