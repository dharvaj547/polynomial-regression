import pandas as pd
import numpy as np


def load_data(filepath):
    df = pd.read_csv(filepath)
    x = df[['x']].values
    y = df[['y']].values

    return x, y


def normalise_data(x, y):
    # normalise x values
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x = (x - x_mean) / x_std

    # normalise y values
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    y = (y - y_mean) / y_std

    return x, y, x_mean, x_std, y_mean, y_std


# def main():
#     x, y = load_data('polynomial_regression/data/data.csv')
#     x, y, x_mean, x_std, y_mean, y_std = normalise_data(x, y)
#     np.savez('processed_data.npz', x=x, y=y, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
#
#
# if __name__ == '__main__':
#     main()
