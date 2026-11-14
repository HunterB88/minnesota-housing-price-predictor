import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from multiple_model import train, predict


# Load housing dataset
df = pd.read_csv("data/minneapolis_housing.csv")

# Select input features
# Numerical features
numeric_features = [
    "ABOVEGROUNDAREA",
    "BEDROOMS",
    "BATHROOMS",
    "YEARBUILT"
]

# Categorical features
categorical_features = [
    "NEIGHBORHOOD"
]

# Select input features
X = df[numeric_features + categorical_features].copy()

# Target: actual sale prices
y = df["ADJ_SALE_PRICE"].to_numpy(dtype=float)

print(f"Loaded {len(X)} Minneapolis homes.")

# Randomly shuffle the data
rng = np.random.default_rng(42)
indices = rng.permutation(len(X))

# Shuffle rows
X = X.iloc[indices].reset_index(drop=True)
y = y[indices]

# Split dataset
split = int(len(X) * 0.8)

X_train = X.iloc[:split].copy()
X_test = X.iloc[split:].copy()

y_train = y[:split]
y_test = y[split:]

print(f"Training homes: {len(X_train)}")
print(f"Testing homes: {len(X_test)}")

# Standardize numerical features using training data only
X_mean = X_train[numeric_features].mean()
X_std = X_train[numeric_features].std(ddof=0)

# Prevent division by zero
X_std = X_std.replace(0, 1)

# Standardize numerical columns
X_train[numeric_features] = (
    X_train[numeric_features] - X_mean
) / X_std

X_test[numeric_features] = (
    X_test[numeric_features] - X_mean
) / X_std

# One-hot encode neighborhoods
X_train = pd.get_dummies(
    X_train,
    columns=categorical_features,
    drop_first=True,
    dtype=float
)

X_test = pd.get_dummies(
    X_test,
    columns=categorical_features,
    drop_first=True,
    dtype=float
)

# Ensure test data has the same columns as training data
X_test = X_test.reindex(
    columns=X_train.columns,
    fill_value=0
)

# Save feature names
feature_names = X_train.columns.tolist()

# Convert DataFrames to NumPy matrices
X_train_scaled = X_train.to_numpy(dtype=float)
X_test_scaled = X_test.to_numpy(dtype=float)

print(f"\nTotal input features: {len(feature_names)}")
print(f"Training matrix shape: {X_train_scaled.shape}")

# Standardize prices
y_mean = y_train.mean()
y_std = y_train.std()

y_train_scaled = (y_train - y_mean) / y_std

# Enable interactive plotting
plt.ion()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

loss_ax, prediction_ax, weight_ax = axes

fig.suptitle(
    "Minneapolis Housing Model — Live Training",
    fontsize=16
)

# Graph 1: Training Loss

loss_line, = loss_ax.plot(
    [],
    [],
    color="royalblue",
    linewidth=2
)

loss_ax.set_title("Training Loss")
loss_ax.set_xlabel("Epoch")
loss_ax.set_ylabel("Mean Squared Error")
loss_ax.set_xlim(0, 10000)
loss_ax.grid(True, alpha=0.3)


# Graph 2: Actual vs Predicted

# Use a sample of training homes for visualization
sample_X = X_train_scaled[:150]
sample_y = y_train[:150]

scatter = prediction_ax.scatter(
    sample_y,
    np.zeros_like(sample_y),
    alpha=0.5,
    color="darkorange"
)

price_min = min(sample_y.min(), 0)
price_max = sample_y.max() * 1.1

prediction_ax.plot(
    [price_min, price_max],
    [price_min, price_max],
    "r--",
    label="Perfect Prediction"
)

prediction_ax.set_title("Actual vs Predicted Prices")
prediction_ax.set_xlabel("Actual Price ($)")
prediction_ax.set_ylabel("Predicted Price ($)")

prediction_ax.set_xlim(price_min, price_max)
prediction_ax.set_ylim(price_min, price_max)

prediction_ax.legend()
prediction_ax.grid(True, alpha=0.3)

# Graph 3: Model Weights

feature_names = [
    "Square Feet",
    "Bedrooms",
    "Bathrooms",
    "Year Built"
]

bars = weight_ax.bar(
    feature_names,
    np.zeros(4),
    color="mediumseagreen"
)

weight_ax.set_title("Learned Weights")
weight_ax.set_ylabel("Standardized Weight")
weight_ax.set_ylim(-1.5, 1.5)

weight_ax.tick_params(axis="x", rotation=35)
weight_ax.axhline(0, color="black", linewidth=0.8)

plt.tight_layout()

def update_dashboard(epoch, losses, weights, bias):

    # Update Training Loss

    loss_line.set_data(
        range(len(losses)),
        losses
    )

    loss_ax.set_ylim(
        0,
        max(losses[0] * 1.1, 0.01)
    )

    # Update House Predictions

    # Make predictions using current weights
    predictions_scaled = predict(
        sample_X,
        weights,
        bias
    )

    # Convert predictions back to dollars
    predictions = predictions_scaled * y_std + y_mean

    # Update scatter plot
    scatter.set_offsets(
        np.column_stack((sample_y, predictions))
    )

    # Update Model Weights

    # Display the original four numerical weights
    for bar, weight in zip(bars, weights[:4]):
        bar.set_height(weight)

    # Refresh Dashboard

    fig.suptitle(
        f"Minneapolis Housing Model — Epoch {epoch}",
        fontsize=16
    )

    fig.canvas.draw_idle()

    plt.pause(0.01)

# Train model
weights, bias, losses = train(
    X_train_scaled,
    y_train_scaled,
    learning_rate=0.01,
    epochs=10000,
    callback=update_dashboard
)

print("\nTraining complete!")
print(f"Learned weights: {weights}")
print(f"Learned bias: {bias:.4f}")

# Make predictions on unseen homes
predictions_scaled = predict(
    X_test_scaled,
    weights,
    bias
)

# Convert predictions back to dollars
predictions = predictions_scaled * y_std + y_mean

# Evaluate model
mae = np.mean(np.abs(y_test - predictions))
rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(f"\nMean Absolute Error: ${mae:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}")

# Calculate R-squared
ss_res = np.sum((y_test - predictions) ** 2)
ss_total = np.sum((y_test - y_test.mean()) ** 2)

r2 = 1 - (ss_res / ss_total)

print(f"R² Score: {r2:.4f}")

from pathlib import Path

Path("results").mkdir(exist_ok=True)

fig.savefig(
    "results/training_dashboard.png",
    dpi=200,
    bbox_inches="tight"
)

plt.ioff()
plt.show()