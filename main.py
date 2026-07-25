# %%

import numpy as np
from functools import partial
from sklearn.datasets import fetch_openml
from pathlib import Path
from functions import sigmoid, dsigmoid

sigmoid_vect = np.vectorize(sigmoid)
dsigmoid_vect = np.vectorize(dsigmoid)

PARAMS_FNAME = 'params.npz'
BATCH_SIZE = 100

sizes = [784, 100, 10]
n_layers = len(sizes) - 1

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test = X[:60000], X[60000:] 
y_train, y_test = y[:60000], y[60000:] 

y_train = y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]

epochs: int = round(len(X_train) / BATCH_SIZE) # Round to prevent floating-point error

# Batch-first (in rows x out columns)
def init_params():
    weights = [
        np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2/(sizes[i] + sizes[i+1]))
        for i in range(n_layers)
    ]

    biases = [
        np.zeros((1, sizes[i+1]))
        for i in range(n_layers)
    ]

    return weights, biases

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

# %%

batch_pos = 0
rng = np.random.default_rng(0)  

perm = rng.permutation(len(X_train)) # Shuffle indexes

# %%
# Training Loop
while batch_pos < len(X_train):

    idx = perm[batch_pos: batch_pos + BATCH_SIZE] # 100 random indexes from 0 - 60k

    batch_cost_sum: np.float64 = 0
    num_correct = 0

    # Feed forward
    for i in idx:
        # %%
        i = 34
        # Single image
        A = X_train[i]
        A = A[np.newaxis, :]
        for W, B in zip(weights, biases):
            Z = A @ W + B
            A = sigmoid_vect(Z)

        
        # Check if model got correct
        prediction = np.argmax(A[0])
        Y = y_train[i][0] # Actual number drawn (represented as an index of output neurons)

        if (prediction == Y): num_correct += 1
        # %%
        # Calculate Cost TODO

        mse = 0
        # NOTE: The indexes (num below) happen to be the actual value we are training, hence the lack of a dictionary
        for num, val in enumerate(A[0]):
            correctness = 0 if num != Y else 1
            se = np.square(val - correctness)
            print(f"Squaring ({val} - {correctness}) because num = {num} and Y = {Y}")
            mse += se

        mse = mse / 10
        # %%
    batch_pos += BATCH_SIZE
# %%
        


 # Assign next starting position for next epoch

