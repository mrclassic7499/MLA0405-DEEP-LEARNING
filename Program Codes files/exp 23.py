import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)

n = 500
theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
r = 2 * theta + np.pi

X = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
y = (theta > 2 * np.pi).astype(int)

X += np.random.randn(n, 2) * 0.5

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(20,),
    activation="logistic",
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Spiral Data - Sigmoid Activation")
plt.show()
