import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Simulate Foot Traffic Data
np.random.seed(42)  # For reproducibility
zip_codes = np.arange(10001, 10011)  # Simulated ZIP codes
population_density = np.random.randint(1000, 10000, len(zip_codes))
number_of_competitors = np.random.randint(1, 10, len(zip_codes))
average_income = np.random.randint(50000, 150000, len(zip_codes))
foot_traffic = population_density / number_of_competitors + average_income / 1000 + np.random.normal(0, 500, len(zip_codes))

data = pd.DataFrame({
    'ZIPCode': zip_codes,
    'PopulationDensity': population_density,
    'NumberOfCompetitors': number_of_competitors,
    'AverageIncome': average_income,
    'FootTraffic': foot_traffic
})

features = ['PopulationDensity', 'NumberOfCompetitors', 'AverageIncome']
X = data[features]
y = data['FootTraffic']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict foot traffic for each ZIP code
predictions = model.predict(X)

plt.figure(figsize=(10, 6))
plt.bar(data['ZIPCode'].astype(str), data['PredictedFootTraffic'], color='skyblue')
plt.title('Predicted Foot Traffic by ZIP Code')
plt.xlabel('ZIP Code')
plt.ylabel('Predicted Foot Traffic')
plt.xticks(rotation=45)
plt.show()

# Evaluate and Interpret the Model
data['PredictedFootTraffic'] = predictions
best_zip_code = data.loc[data['PredictedFootTraffic'].idxmax(), 'ZIPCode']
print(f"The best ZIP code to start a new store based on predicted foot traffic is: {best_zip_code}")

