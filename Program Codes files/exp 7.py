import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate values
x = np.linspace(-10, 10, 200)
y = sigmoid(x)

# Plot
plt.figure(figsize=(8,5))
plt.plot(x, y, color='blue', linewidth=2)
plt.title("Visualization of Logistic Regression (Sigmoid Function)")
plt.xlabel("Input (x)")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.show()
