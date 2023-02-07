import numpy as np


class PolyRegression:
    def __init__(self, degree, learning_rate=0.01, num_iterations=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None

    def fit(self, x, y):
        x = self._prepare_input_data(x)
        self.weights = np.zeros(x.shape[1])

        for _ in range(self.num_iterations):
            y_pred = x.dot(self.weights)
            error = y_pred - y
            gradient = x.T.dot(error) / len(y)
            self.weights -= self.learning_rate * gradient

    def predict(self, x):
        x = self._prepare_input_data(x)

        return x.dot(self.weights)

    def _prepare_input_data(self, x):
        x_poly = np.zeros((x.shape[0], self.degree + 1))

        for i in range(self.degree + 1):
            x_poly[:, i:i+1] = x ** i

        return x_poly
