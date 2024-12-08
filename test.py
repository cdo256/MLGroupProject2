import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Load the data (replace with your actual dataset)
data = pd.read_excel('TrainDataset2024.xls')

# Check the first few rows of the dataset to see the structure
print(data.head())

# Remove non-numeric columns for correlation analysis
# Select only numeric columns
data_numeric = data.select_dtypes(include=[np.number])

# Assume your target column is called 'RelapseFreeSurvival (outcome)'
target = 'RelapseFreeSurvival (outcome)'

# Step 1: Select only numeric columns for feature selection
X = data_numeric.drop(columns=[target])
y = data_numeric[target]

# Step 2: Scale the features (important for Lasso, SVM, and ANN models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Fit Lasso regression model
lasso = Lasso(alpha=0.1)  # You can experiment with different values of alpha
lasso.fit(X_scaled, y)

# Step 4: Get the coefficients of the features
coefficients = lasso.coef_

# Step 5: Create a DataFrame to store feature names and their corresponding coefficients
feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})

# Step 6: Sort the features by the absolute value of their coefficients in descending order
feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
feature_importance_sorted = feature_importance.sort_values(by='Abs_Coefficient', ascending=False)

# Step 7: Select the top features (adjust as needed)
top_features = feature_importance_sorted.head(30)

# Output the top features
print("Top Features selected by Lasso:")
print(top_features[['Feature', 'Abs_Coefficient']])

# Filter the original dataset with the top selected features
X_top = X[top_features['Feature']]

# Step 8: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(alpha=0.1),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'SVM Regression': SVR(kernel='rbf', C=1.0, epsilon=0.2),
    'ANN Regression': MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42),
    'XGBoost Regression': xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    'LightGBM Regression': lgb.LGBMRegressor(objective='regression', random_state=42)
}

# Store results
results = []

# Train and evaluate each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store the results
    results.append({
        'Model': model_name,
        'Mean Squared Error': mse,
        'R2 Score': r2
    })

# Convert results to a DataFrame for easy comparison
results_df = pd.DataFrame(results)

# Display the comparison of model performances
print("\nModel Performance Comparison:")
print(results_df)

# Optionally, you can visualize the comparison using bar plots
plt.figure(figsize=(10, 6))

# Plot MSE comparison
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='Mean Squared Error', data=results_df, palette='viridis')
plt.title('Mean Squared Error Comparison')
plt.xticks(rotation=45)

# Plot R² comparison
plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2 Score', data=results_df, palette='viridis')
plt.title('R² Score Comparison')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
results_df.to_csv("test-regression-results.csv")