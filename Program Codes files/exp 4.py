import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = X[:, 0] ** 2 + np.random.randn(100) * 2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

degrees = [1, 2, 15]

plt.figure(figsize=(12, 4))

for i, degree in enumerate(degrees):
    
    poly = PolynomialFeatures(degree=degree)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    train_pred = model.predict(X_train_poly)
    test_pred = model.predict(X_test_poly)
    
    train_error = mean_squared_error(y_train, train_pred)
    test_error = mean_squared_error(y_test, test_pred)
    
    print("Polynomial Degree:", degree)
    print("Training MSE:", train_error)
    print("Testing MSE:", test_error)
    print()
    
    X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    y_plot = model.predict(X_plot_poly)
    
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, label="Training Data")
    plt.scatter(X_test, y_test, label="Testing Data")
    plt.plot(X_plot, y_plot, label="Polynomial Curve")
    plt.title("Degree " + str(degree))
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

plt.tight_layout()
plt.show()
