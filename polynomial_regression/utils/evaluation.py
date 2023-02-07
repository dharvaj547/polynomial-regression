import numpy as np
import matplotlib.pyplot as plt

from polynomial_regression.models.poly_regression import PolyRegression
from polynomial_regression.utils.data_processing import load_data, normalise_data


def evaluate_model(model, x, y, x_mean, y_mean, x_std, y_std):
    y_pred = model.predict(x)
    y_pred = y_pred * y_std + y_mean
    y = y * y_std + y_mean

    # evaluate mean squared error
    mse = np.mean((y_pred - y) ** 2)

    # evaluate R^2 score
    r2 = model.score(x, y)

    return mse, r2


def main():
    x, y = load_data('data.csv')
    x, y, x_mean, x_std, y_mean, y_std = normalise_data(x, y)

    model = PolyRegression(degree=3)
    model.fit(x, y)

    mse, r2 = evaluate_model(model, x, y, x_mean, x_std, y_mean, y_std)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    x_plot = np.linspace(x.min(), x.max(), 1000)[:, np.newaxis]
    y_plot = model.predict(x_plot)
    y_plot = y_plot * y_std + y_mean

    plt.scatter(x, y)
    plt.plot(x_plot, y_plot, color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    main()

