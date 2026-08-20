import numpy as np
import matplotlib.pyplot as plt

# Multi-class data
X = np.array([
    [1, 1],
    [1, 2],
    [2, 1],
    [2, 2],

    [5, 1],
    [6, 1],
    [5, 2],
    [6, 2],

    [3, 5],
    [4, 5],
    [3, 6],
    [4, 6]
], dtype=float)

# Class labels
y = np.array([
    0, 0, 0, 0,
    1, 1, 1, 1,
    2, 2, 2, 2
])

# One-hot encoding
Y = np.zeros((len(y), 3))
Y[np.arange(len(y)), y] = 1

# Initialize weights
np.random.seed(42)

weights = np.random.randn(2, 3) * 0.01
bias = np.zeros(3)

learning_rate = 0.01
epochs = 2000

# Training
for epoch in range(epochs):

    output = np.dot(X, weights) + bias

    error = Y - output

    weights += learning_rate * np.dot(X.T, error)
    bias += learning_rate * np.sum(error, axis=0)

# Prediction
output = np.dot(X, weights) + bias

predicted = np.argmax(output, axis=1)

accuracy = np.mean(predicted == y)

print("Multi-Class Neural Network")
print("--------------------------")
print("Weights:")
print(weights)

print("\nBias:")
print(bias)

print("\nActual Classes:")
print(y)

print("\nPredicted Classes:")
print(predicted)

print("\nAccuracy:", accuracy)

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

plt.scatter(
    X[y == 2, 0],
    X[y == 2, 1],
    label="Class 2"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Multi-Class Data with Linear Activation")
plt.legend()
plt.show()
