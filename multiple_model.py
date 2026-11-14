import numpy as np


def predict(X, weights, bias):
    return X @ weights + bias

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):

    n = len(y_true)

    errors = y_pred - y_true

    dw = (2 / n) * (X.T @ errors)

    db = 2 * np.mean(errors)

    return dw, db

def train(X, y, learning_rate=0.1, epochs=1000, callback=None):

    n_features = X.shape[1]

    # Initialize parameters
    weights = np.zeros(n_features)
    bias = 0.0

    loss_history = []

    for epoch in range(epochs):

        # Make predictions
        y_pred = predict(X, weights, bias)

        # Calculate loss
        loss = mse_loss(y, y_pred)
        loss_history.append(loss)

        # Calculate gradients
        dw, db = compute_gradients(X, y, y_pred)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

        # Update visualization every 10 epochs
        if callback is not None and epoch % 50 == 0:
            callback(epoch, loss_history, weights.copy(), bias)

    return weights, bias, loss_history