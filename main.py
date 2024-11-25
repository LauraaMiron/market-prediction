import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Simulate or Load Historical Market Data
# For simplicity, we'll simulate some data
np.random.seed(42)
days = np.arange(1, 101)  # Simulating 100 days of data
prices = 50 + 0.5 * days + np.random.normal(scale=5, size=100)  # Simulated trend with noise

# Create a DataFrame
data = pd.DataFrame({'Day': days, 'Price': prices})

# Step 2: Preprocess Data
X = data[['Day']]  # Features (e.g., time)
y = data['Price']   # Target (e.g., market price)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Visualize Predictions
plt.scatter(X, y, label='Actual Data', color='blue', alpha=0.6)
plt.plot(X, model.predict(X), label='Prediction', color='red', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Price')
plt.title('Market Price Prediction')
plt.legend()
plt.show()

# Step 5: Make Future Predictions
future_days = np.arange(101, 121).reshape(-1, 1)  # Predict for the next 20 days
future_prices = model.predict(future_days)
print("Future Prices:")
print(pd.DataFrame({'Day': future_days.flatten(), 'Predicted Price': future_prices}))
