import numpy as np

from multiple_model import (
    predict,
    mse_loss,
    compute_gradients,
    train
)


def test_predict():
    X = np.array([
        [1, 2],
        [3, 4]
    ])

    weights = np.array([2, 3])
    bias = 5

    predictions = predict(X, weights, bias)

    expected = np.array([13, 23])

    np.testing.assert_array_equal(predictions, expected)


def test_mse_loss():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])

    loss = mse_loss(actual, predicted)

    np.testing.assert_allclose(loss, 17 / 3)


def test_gradients():
    X = np.array([
        [1, 2],
        [3, 4]
    ])

    y_true = np.array([5, 10])
    y_pred = np.array([4, 8])

    dw, db = compute_gradients(X, y_true, y_pred)

    expected_dw = np.array([-7, -10])
    expected_db = -3

    np.testing.assert_allclose(dw, expected_dw)
    np.testing.assert_allclose(db, expected_db)


def test_training():
    X = np.array([
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1]
    ], dtype=float)

    y = 2 * X[:, 0] + 3 * X[:, 1] + 1

    weights, bias, losses = train(
        X,
        y,
        learning_rate=0.1,
        epochs=1000
    )

    np.testing.assert_allclose(
        weights,
        [2, 3],
        atol=0.01
    )

    np.testing.assert_allclose(
        bias,
        1,
        atol=0.01
    )

    assert losses[-1] < losses[0]