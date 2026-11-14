import numpy as np


def predict(x, weight, bias):
    return weight * x + bias

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def compute_gradients(x, y_true, y_pred):
    errors = y_pred - y_true

    dw = 2 * np.mean(x * errors)
    db = 2 * np.mean(errors)

    return dw,db

def train(x, y, learning_rate=0.01, epochs=1000):
    weight = 0.0
    bias = 0.0

    loss_history = []

    for epoch in range(epochs):
        # Make preditctions
        y_pred = predict(x, weight, bias)

        # Calculate loss
        loss = mse_loss(y, y_pred)
        loss_history.append(loss)

        # Calculate gradients
        dw, db = compute_gradients(x, y, y_pred)

        # Update parameters
        weight -= learning_rate * dw
        bias -= learning_rate * db

    return weight, bias, loss_history


