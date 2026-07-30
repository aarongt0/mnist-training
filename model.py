import numpy as np
from sklearn.datasets import fetch_openml
from pathlib import Path
import os

PARAMS_FNAME = 'params.npz'
BATCH_SIZE = 100

# -- Editable

sizes = [784, 100, 10]
n_out = sizes[-1]
n_layers = len(sizes) - 1

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

X = X.astype(np.float32) / 255.0
y = y.astype(np.int64)

X_train, X_test = X[:60000], X[60000:] 
y_train, y_test = y[:60000], y[60000:] 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

iters: int = round(len(X_train) / BATCH_SIZE) # Round to prevent floating-point error

# Batch-first (in rows, out columns)
def init_params():
    print("Initializing parameters...")
    weights = [
        np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2/(sizes[i] + sizes[i+1]))
        for i in range(n_layers)
    ]

    biases = [
        np.zeros((1, sizes[i+1]))
        for i in range(n_layers)
    ]
    print("Done creating params.")
    return weights, biases

def save_params(weights, biases):
    delete_params()
    np.savez(
        PARAMS_FNAME,
        **{f"W{i}": W for i, W in enumerate(weights)}, 
        **{f"b{i}": b for i, b in enumerate(biases)},
    )
    "Parameters Saved."

def delete_params():
    if Path(PARAMS_FNAME).is_file():
        print("Paramaters Deleted.")
        os.remove(PARAMS_FNAME)

def load_params():
    data = np.load(PARAMS_FNAME)
    return ([data[f"W{i}"] for i in range(n_layers)],
            [data[f"b{i}"] for i in range(n_layers)])

weights, biases = load_params() if Path(PARAMS_FNAME).is_file() else init_params()

def reset_params():
    global weights, biases
    weights, biases = init_params()
    print("Weights and Biases reset.")



# Training Loop
def run_epoch(lr: float, epochNum):

    batch_pos = 0
    rng = np.random.default_rng(epochNum) # Seed using epoch number to mix it up
    perm = rng.permutation(len(X_train))
    accuracySum = 0
    costSum = 0

    
    for iter in range(iters):

        idx = perm[batch_pos: batch_pos + BATCH_SIZE] # 100 random indexes from 0 - 60k

        y_batch = y_train[idx]                        
        Y = np.zeros((BATCH_SIZE, n_out))
        Y[np.arange(BATCH_SIZE), y_batch] = 1.0 # Set drawn number's index to value of 1.0, else 0

        A = [
            np.empty((BATCH_SIZE, sizes[i]))
            for i in range(len(sizes))
        ]

        Z = [
            np.empty((BATCH_SIZE, sizes[i + 1]))
            for i in range(n_layers)
        ]

        # --- Forward Pass
        A[0] = X_train[idx] # Assign input values

        for layer in range(n_layers):
            Z[layer] = A[layer] @ weights[layer] + biases[layer]
            A[layer + 1] = sigmoid(Z[layer])

        # --- Calculate accuracy
        predictions = np.argmax(A[-1], axis=1)
        truth = np.argmax(Y, axis=1)

        accuracy = np.mean(predictions == truth)
        cost_average = np.sum(np.square(A[-1] - Y)) / BATCH_SIZE

        accuracySum += accuracy
        costSum += cost_average

        # --- Back Propogate
        dCdA = 2 * (A[-1] - Y)

        for layer in reversed(range(n_layers)):
            dAdZ = dsigmoid(Z[layer])
            dCdZ = dCdA * dAdZ # Multiply element-wise to preserve individual training images' effect on network

            # Collapse batch into usable gradients
            W_grad = (A[layer].T @ dCdZ) / BATCH_SIZE # Outer product of previous layer and dCdZ -> calculates direction for each individual weight in the layer, then averages
            B_grad = np.mean(dCdZ, axis=0, keepdims=True) # dZdB = 1, so just average terms

            # Compute next layer's dCdA before modifying weights
            dCdA = dCdZ @ weights[layer].T

            weights[layer] -= (lr * W_grad)
            biases[layer] -= (lr * B_grad)


        batch_pos += BATCH_SIZE

    return (accuracySum / iters), (costSum / iters)

def run_test_set():
    SET_SIZE = len(X_test)

    Y = np.zeros((SET_SIZE, n_out))
    Y[np.arange(SET_SIZE), y_test] = 1.0

    A = [
        np.empty((SET_SIZE, sizes[i]))
        for i in range(len(sizes))
    ]

    Z = [
            np.empty((SET_SIZE, sizes[i + 1]))
            for i in range(n_layers)
    ]

    # --- Forward Pass
    A[0] = X_test

    for layer in range(n_layers):
        Z[layer] = A[layer] @ weights[layer] + biases[layer]
        A[layer + 1] = sigmoid(Z[layer])

    # --- Calculate accuracy
    predictions = np.argmax(A[-1], axis=1)
    truth = np.argmax(Y, axis=1)

    accuracy = np.mean(predictions == truth)
    cost_average = np.sum(np.square(A[-1] - Y)) / SET_SIZE

    return accuracy

    