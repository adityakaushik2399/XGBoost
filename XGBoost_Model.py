# Basic data manipulation and numerical operations
import pandas as pd
import numpy as np

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning preprocessing and utilities
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Model imports
from xgboost import XGBRegressor
from sklearn.cluster import KMeans

# Model persistence (saving and loading)
from joblib import dump, load

# Statistical testing
from scipy.stats import ttest_rel  # Import paired t-test function

# Time-related functionality (if you need to time some operations)
import time

# Step 2: Load Data
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')
train_data.head()
test_data.head()

# Handle categorical variables like gender and user_type using label encoding.
label_encoder = LabelEncoder()
for column in ['gender', 'user_type']:
    train_data[column] = label_encoder.fit_transform(train_data[column])
    test_data[column] = label_encoder.transform(test_data[column])

# List of numerical columns for scaling
numerical_columns = ['age', 'purchase_history', 'avg_session_duration', 'last_login_days',
                     'total_spend', 'product_reviews', 'adverts_targeted', 'discount_offered',
                     ]

# Apply Standard Scaling to numerical columns
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])

# Separate the features and target variables for both training and testing data.
X_train = train_data.drop(['sales', 'user_id'], axis=1) # Dropping user_id
y_train = train_data['sales']
X_test = test_data.drop(['sales', 'user_id'], axis=1) # Dropping user_id
y_test = test_data['sales']

#Let's define some models for testing
#'reg:squarederror : performing a regression task
models = {
    "XGBoost_LearningRate_0_1_MaxDepth_1": XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=1),
    "XGBoost_LearningRate_0_5_MaxDepth_1": XGBRegressor(objective='reg:squarederror', learning_rate=0.5, max_depth=1),
    "XGBoost_LearningRate_1_MaxDepth_1": XGBRegressor(objective='reg:squarederror', learning_rate=1, max_depth=1),
    "XGBoost_LearningRate_0_1_MaxDepth_2": XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=2),
    "XGBoost_LearningRate_0_5_MaxDepth_2": XGBRegressor(objective='reg:squarederror', learning_rate=0.5, max_depth=2),
    "XGBoost_LearningRate_1_MaxDepth_2": XGBRegressor(objective='reg:squarederror', learning_rate=1, max_depth=2),
    "XGBoost_LearningRate_0_1_MaxDepth_3": XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=3),
    "XGBoost_LearningRate_0_5_MaxDepth_3": XGBRegressor(objective='reg:squarederror', learning_rate=0.5, max_depth=3),
    "XGBoost_LearningRate_1_MaxDepth_3": XGBRegressor(objective='reg:squarederror', learning_rate=1, max_depth=3),
    "XGBoost_LearningRate_0_1_MaxDepth_4": XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=4),
    "XGBoost_LearningRate_0_5_MaxDepth_4": XGBRegressor(objective='reg:squarederror', learning_rate=0.5, max_depth=4),
    "XGBoost_LearningRate_1_MaxDepth_4": XGBRegressor(objective='reg:squarederror', learning_rate=1, max_depth=4)
}

# Training loop
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} - Model Trained Successfully!")
    filename = f'{name}_model.joblib'
    dump(model, filename)

# Compute R-squared for the control model
model_results = {}

# Evaluate and save performance metrics for models
for name in models.keys():
    filename = f'{name}_model.joblib'
    model = load(filename)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_results[name] = {'MSE': mse, 'R²': r2}
    print(f"{name} - Mean Squared Error: {mse}, R-squared: {r2}")

# Identifying the model with the lowest MSE and highest R²
best_model_name = min(model_results, key=lambda k: (model_results[k]['MSE'], -model_results[k]['R²']))
best_model_performance = model_results[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"- Mean Squared Error: {best_model_performance['MSE']}")
print(f"- R-squared: {best_model_performance['R²']}")


# Load control model and make predictions Performance testing
control_model = load(f'{best_model_name}_model.joblib')
control_predictions = control_model.predict(X_test)
y_pred_control = control_model.predict(X_test)
control_r2 = r2_score(y_test, y_pred_control)
control_model_name=best_model_name

# Iterate through other models and perform t-tests
for name in models.keys():
    test_model = load(f'{name}_model.joblib')
    test_predictions = test_model.predict(X_test)

    # Perform paired t-test between control and test group predictions
    t_stat, p_value = ttest_rel(control_predictions, test_predictions)

    # Print results
    print(f"A/B Testing Between {control_model_name} and {name}:")
    print(f"- T-Statistic: {t_stat}")
    print(f"- P-Value: {p_value}")
    if p_value < 0.01:
        print(f"-> Significant difference between the models (99% confidence level).\n")
    else:
        print(f"-> No significant difference between the models (99% confidence level).\n")

# Load control model for speed testing
control_model = load(f'{control_model_name}_model.joblib')
# Speed testing
model_names = []
single_pred_times = []
bulk_pred_times = []

#Speed testing for control model
start_time = time.time()
control_predictions_single = control_model.predict(X_test[0:1])  # using only the first sample
end_time = time.time()
control_single_pred_time = end_time - start_time

start_time = time.time()
control_predictions_bulk = control_model.predict(X_test[0:1000])  # using the first 1000 samples
end_time = time.time()
control_bulk_pred_time = end_time - start_time

model_names.append(control_model_name)
single_pred_times.append(control_single_pred_time)
bulk_pred_times.append(control_bulk_pred_time)

for name in models.keys():
    test_model = load(f'{name}_model.joblib')

    # Single prediction speed test
    start_time = time.time()
    test_predictions_single = test_model.predict(X_test[0:1])
    end_time = time.time()
    single_pred_time = end_time - start_time

    # Bulk prediction speed test (using 1000 samples)
    start_time = time.time()
    test_predictions_bulk = test_model.predict(X_test[0:1000])
    end_time = time.time()
    bulk_pred_time = end_time - start_time

    model_names.append(name)
    single_pred_times.append(single_pred_time)
    bulk_pred_times.append(bulk_pred_time)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(15, 12))

# Single Prediction Times
ax[0].barh(model_names, single_pred_times, color='skyblue')
ax[0].set_title('Single Prediction Speed')
ax[0].set_xlabel('Time (seconds)')
ax[0].set_ylabel('Model Names')

# Bulk Prediction Times
ax[1].barh(model_names, bulk_pred_times, color='salmon')
ax[1].set_title('Bulk Prediction Speed for 1000 Samples')
ax[1].set_xlabel('Time (seconds)')
ax[1].set_ylabel('Model Names')

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

print("Features in the dataset:")
for feature in X_train.columns:
    print(feature)

best_model = load(f'{best_model_name}_model.joblib')

feature_importance = best_model.feature_importances_

selected_features = ['adverts_targeted', 'discount_offered']
indices = [list(X_train.columns).index(feature) for feature in selected_features]
selected_importances = [feature_importance[i] for i in indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=selected_importances, y=selected_features)
plt.title('Feature Importance for Selected Features')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

print("Influential features in order:")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature} - Importance: {selected_importances[i-1]:.4f}")

# Define possible values
adverts_values = list(range(1, 11))
discount_values = list(range(1, 11))

# Create a grid for predictions
predictions = np.zeros((len(adverts_values), len(discount_values)))

# Make predictions for each combination
for i, adverts in enumerate(adverts_values):
    for j, discount in enumerate(discount_values):
        # Create a sample with all features set to their average (or median) values
        sample = X_train.mean().to_dict()
        sample['adverts_targeted'] = adverts
        sample['discount_offered'] = discount

        # Convert dictionary to DataFrame
        df_sample = pd.DataFrame([sample])

        # Make prediction
        predictions[i, j] = best_model.predict(df_sample)[0]

# Visualization for Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(predictions, xticklabels=adverts_values, yticklabels=discount_values, annot=True, cmap='YlGnBu')
plt.title('Predicted Outcomes for Different Combinations')
plt.xlabel('Adverts for Session')
plt.ylabel('Discount for Session')
plt.show()

df_predictions = pd.DataFrame({
    'Adverts': np.repeat(adverts_values, len(discount_values)),
    'Discounts': np.tile(discount_values, len(adverts_values)),
    'Predictions': predictions.ravel()
})

# Visualization for Line Graph
plt.figure(figsize=(14, 10))
sns.lineplot(x='Adverts', y='Predictions', hue='Discounts', data=df_predictions, palette='viridis', marker="o")
plt.title('Predicted Outcomes for Different Combinations')
plt.xlabel('Adverts Per Page')
plt.ylabel('Predicted Outcomes')
plt.legend(title='Discount Items Per Page')
plt.grid(True)
plt.show()

# Find the indices of the maximum predicted value
optimal_idx = np.unravel_index(predictions.argmax(), predictions.shape)
optimal_adverts = adverts_values[optimal_idx[0]]
optimal_discounts = discount_values[optimal_idx[1]]

print(f"The optimal number of adverts per page for maximizing sales is: {optimal_adverts}")
print(f"The optimal number of discounts per page for maximizing sales is: {optimal_discounts}")

sample = X_train.mean().to_dict()
sample['adverts_targeted'] = optimal_adverts
sample['discount_offered'] = optimal_discounts

# Convert dictionary to DataFrame and predict
df_sample = pd.DataFrame([sample])
predicted_sales_optimal = best_model.predict(df_sample)[0]

print(f"Predicted average sale with {optimal_adverts} adverts and {optimal_discounts} discounts per page: {predicted_sales_optimal}")
'''
Predicted average sale with 1 adverts and 2 discounts per page: 90.5713119506836
'''
df= train_data
# Calculate statistics
count = len(df)
mean_sales = df['sales'].mean()

# Print calculated values
print(f"Count: {count} (Number of instances)")
print(f"Mean (Average) Sales: Approximately ${mean_sales:.2f}")
'''
Count: 800 (Number of instances)
Mean (Average) Sales: Approximately $82.46
'''

'''
Present the potential revenue increase use projections based on historical data and model predictions.
Revenue Increase: If the model's recommendations can increase the average sales from $82.46 to $90.57.
Then over 800 instances, the potential revenue increase would be $(90.57 - 82.46) * 800 = $6,488.
This increase signifies the potential of our model's recommendations allowing us to optimize marketing and derive more value from our investments.
'''
