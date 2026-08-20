import numpy as np
import matplotlib.pyplot as plt

# Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

# Initial parameters
m = 0
c = 0

learning_rate = 0.01
epochs = 1000

n = len(X)

# Gradient Descent
for i in range(epochs):

    y_pred = m * X + c

    dm = (-2 / n) * np.sum(X * (y - y_pred))
    dc = (-2 / n) * np.sum(y - y_pred)

    m = m - learning_rate * dm
    c = c - learning_rate * dc

# Final results
print("Gradient Descent for Linear Regression")
print("---------------------------------------")
print("Slope (m):", m)
print("Intercept (c):", c)

# Prediction
y_pred = m * X + c

print("\nPredicted values:")
print(y_pred)

# Graph
plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Gradient Descent")
plt.legend()
plt.show()
