import numpy as np
import matplotlib.pyplot as plt
# Two-class data
X = np.array([
    [1, 1],
    [2, 1],
    [1, 2],
    [2, 2],
    [6, 6],
    [7, 6],
    [6, 7],
    [7, 7]
], dtype=float)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
# Initial weights and bias
weights = np.zeros(2)
bias = 0
learning_rate = 0.01
epochs = 1000
# Training
for epoch in range(epochs):
    output = np.dot(X, weights) + bias
    error = y - output
    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error)
# Prediction
output = np.dot(X, weights) + bias
predicted = (output >= 0.5).astype(int)
print("Two-Class Neural Network")
print("------------------------")
print("Weights:", weights)
print("Bias:", bias)
print("Predicted Classes:", predicted)
# Plot
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
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Two-Class Data with Linear Activation")
plt.legend()
plt.show()
