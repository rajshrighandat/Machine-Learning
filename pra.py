import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset (first 100 rows)
data = pd.read_csv("ramen-ratings.csv").head(100)

# Display dataset
print(data.head())
print(data.info())
print(data.describe())

# Independent variable (X)
X = data[["Sales_Amount"]]

# Dependent variable (y)
y = data["Discount"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict values
y_pred = model.predict(X_test)

# Compare Actual vs Predicted
result = pd.DataFrame({
     y_test.values,
     y_pred
})

print(result)

# Model parameters
print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)


# Scatter Plot
plt.scatter(X_test, y_test)

# Regression Line
plt.plot(X_test, y_pred)

plt.xlabel("Sales Amount")
plt.ylabel("Discount")
plt.title("Linear Regression")
plt.legend()

plt.show() 