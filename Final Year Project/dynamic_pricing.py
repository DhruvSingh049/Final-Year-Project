import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load your dataset
data = pd.read_csv('retail_price.csv')

# Feature selection (you may need to adjust this based on your dataset)
features = ['qty', 'freight_price', 'unit_price',
             'product_score', 'weekday', 'holiday', 'month', 'year',
             'volume']

# Extract features and target variable
X = data[features]
y = data['total_price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
print(f'Mean Absolute Error: {mae}')

# Visualize predicted vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Total Price')
plt.ylabel('Predicted Total Price')
plt.title('Actual vs Predicted Total Price')
plt.show()

# Deploy the model and monitor its performance in a production environment
