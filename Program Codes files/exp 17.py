import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Two-class linearly separable data
X = np.array([
    [1, 1],
    [2, 1],
    [2, 2],
    [3, 2],
    [7, 7],
    [8, 7],
    [8, 8],
    [9, 8]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
# Create Logistic Regression model
model = LogisticRegression()
# Train model
model.fit(X, y)
# Prediction
y_pred = model.predict(X)
print("Logistic Regression")
print("-------------------")
print("Predicted Classes:", y_pred)
print("Accuracy:", model.score(X, y))
# Plot data
plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    label="Class 0"
)
plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    label="Class 1"
)
# Decision boundary
x_values = np.linspace(0, 10, 100)
w = model.coef_[0]
b = model.intercept_[0]
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Separability using Logistic Regression")
plt.legend()
plt.show()
