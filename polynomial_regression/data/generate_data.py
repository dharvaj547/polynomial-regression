import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_data(num_points, noise_level=0.0):
    np.random.seed(0)
    x = np.linspace(0, 9, num_points).reshape(-1, 1)
    y = 1 + 2 * x + 3 * x ** 2 + 4 * x ** 3 + noise_level * np.random.randn(num_points, 1)

    return x, y


def main():
    # synthetically generate data
    x, y = generate_data(1000, noise_level=100)
    data = np.hstack((x, y))
    df = pd.DataFrame(data, columns=['x', 'y'])
    df.to_csv('data.csv', index=False)

    # plot data
    plt.scatter(x, y)
    plt.show()


if __name__ == '__main__':
    main()
