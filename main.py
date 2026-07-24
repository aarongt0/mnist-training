import numpy as np
from sklearn.datasets import fetch_openml
from pathlib import Path
from functions import sigmoid, dsigmoid

PARAMS_FNAME = 'params.npz'

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

sizes = [784, 100, 10]
n_layers = len(sizes) - 1

def init_params():
    weights = [
        np.random.randn(sizes[i+1], sizes[i]) * np.sqrt(2/(sizes[i] + sizes[i+1]))
        for i in range(n_layers)
    ]

    biases = [
        np.zeros(sizes[i+1], 1)
        for i in range(n_layers)
    ]


def save_params(weights, biases):
    np.savez(
        PARAMS_FNAME,
        **{f"W{i}": W for i, W in enumerate(weights)},
        **{f"b{i}": b for i, b in enumerate(biases)},
    )

def load_params():
    data = np.load(PARAMS_FNAME)
    return ([data[f"W{i}"] for i in range(n_layers)],
            [data[f"b{i}"] for i in range(n_layers)])

weights, biases = load_params() if Path(PARAMS_FNAME).is_file() else init_params()


