import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 4, 2, 5, 6])

# Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Scatter plot and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Fit Line')

# Residual lines
for xi, yi, ypi in zip(X.flatten(), y, y_pred):
    plt.vlines(x=xi, ymin=min(yi, ypi), ymax=max(yi, ypi),
               color='green', linestyle='dashed')

plt.xlabel('X (Independent)')
plt.ylabel('Y (Dependent)')
plt.title('Linear Regression with Residuals')
plt.legend()
plt.show()

# Model parameters and error metrics
print("Intercept (β0):", model.intercept_)
print("Slope (β1):", model.coef_[0])
print("MSE:", mean_squared_error(y, y_pred))
print("MAE:", mean_absolute_error(y, y_pred))

plt.show()
