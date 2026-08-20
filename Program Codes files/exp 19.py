import numpy as np
import matplotlib.pyplot as plt

# Generate circular data
np.random.seed(42)

angles = np.linspace(0, 2 * np.pi, 100)

# Inner circle - Class 0
r1 = 1
x1 = r1 * np.cos(angles)
y1 = r1 * np.sin(angles)

# Outer circle - Class 1
r2 = 3
x2 = r2 * np.cos(angles)
y2 = r2 * np.sin(angles)

# Combine data
X = np.vstack((
    np.column_stack((x1, y1)),
    np.column_stack((x2, y2))
))

y = np.array([0] * 100 + [1] * 100)

# Linear neuron
weights = np.zeros(2)
bias = 0

learning_rate = 0.001
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

accuracy = np.mean(predicted == y)

print("Circular Data with Linear Activation")
print("------------------------------------")
print("Weights:", weights)
print("Bias:", bias)
print("Accuracy:", accuracy)

# Plot
plt.scatter(x1, y1, label="Class 0")
plt.scatter(x2, y2, label="Class 1")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Circular Data - Linear Activation")
plt.legend()
plt.axis("equal")
plt.show()
