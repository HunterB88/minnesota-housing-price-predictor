import numpy as np 
import matplotlib.pyplot as plt
from model import train, predict

# House sizes in square feet
house_sizes = np.array([1000, 1500, 2000, 2500, 3000])

# House prices in dollars
house_prices = np.array([200000, 275000, 350000, 425000, 500000])

# Stardardize our data
x_mean = np.mean(house_sizes)
x_std = np.std(house_sizes)

y_mean = np.mean(house_prices)
y_std = np.std(house_prices)

x = (house_sizes - x_mean) / x_std
y = (house_prices - y_mean) / y_std

# Train the model
weight, bias, losses = train(x,y)

# Display reults
print(f"Learned weight: {weight:.4f}")
print(f"Learned bias: {bias:.4f}")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")

# Predict the price of a new house
new_house = 2200

# Standardize using the training data statistics
new_house_scaled = (new_house - x_mean) / x_std

# Make prediction
predicted_scaled = predict(new_house_scaled, weight, bias)

# Convert preditcion back to dollars
predicted_price = predicted_scaled * y_std + y_mean

print(f"Predicted price for {new_house} sqft: ${predicted_price:,.2f}")

# Visualize training loss
plt.figure(figsize=(8, 5))

plt.plot(losses, color='blue', linewidth=2)

plt.title("Linear Regression Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")

plt.grid(True)
plt.show()