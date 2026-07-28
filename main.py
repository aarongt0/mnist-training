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
LR = 0.01
EPOCHS = 1 # NOTE: For testing

sizes = [784, 100, 10]
n_out = sizes[-1]
n_layers = len(sizes) - 1

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test = X[:60000], X[60000:] 
y_train, y_test = y[:60000], y[60000:] 


iters: int = round(len(X_train) / BATCH_SIZE) # Round to prevent floating-point error

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
for iter in range(iters):

    idx = perm[batch_pos: batch_pos + BATCH_SIZE] # 100 random indexes from 0 - 60k

    y_batch = y_train[idx]                        
    Y = np.zeros((BATCH_SIZE, n_out))
    Y[np.arange(BATCH_SIZE), y_batch] = 1.0 # Set drawn number's index to value of 1.0, else 0

    A = [
        np.empty((BATCH_SIZE, sizes[i]))
        for i in range(len(sizes))
    ]

    Z = A[1:] # Same shape

    # --- Forward Pass
    A[0] = X_train[idx] # Assign input values

    for layer in range(n_layers):
        Z[layer] = A[layer] @ weights[layer] + biases[layer]
        A[layer + 1] = sigmoid_vect(Z[layer])

    # --- Calculate accuracy
    predictions = np.argmax(A[-1], axis=1)
    truth = np.argmax(Y, axis=1)
    accuracy = np.mean(predictions == truth)

    loss_average = np.sum(np.square(A[-1] - Y)) / BATCH_SIZE

    print(f"Iteration {iter}: Loss = {loss_average}. Accuracy = {accuracy * 100:.2f}%.")

    # --- Back Propogate
    dCdA = 2 * (A[-1] - Y)

    for layer in reversed(range(n_layers)):
        dAdZ = dsigmoid_vect(Z[layer])
        dCdZ = dCdA * dAdZ # Multiply element-wise to preserve individual training images' effect on network

        # Collapse batch into usable gradients
        W_grad = (A[layer].T @ dCdZ) / BATCH_SIZE # Outer product of previous layer and dCdZ -> calculates direction for each individual weight in the layer, then averages
        B_grad = np.mean(dCdZ, axis=0, keepdims=True) # dZdB = 1, so just average terms

        # Compute next layer's dCdA before modifying weights
        dCdA = dCdZ @ weights[layer].T

        weights[layer] -= (LR * W_grad)
        biases[layer] -= (LR * B_grad)


    batch_pos += BATCH_SIZE

# %%



 # Assign next starting position for next epoch

