# Databricks notebook source
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load the Abalone dataset
# 검증 위해 test로 진행
df = spark.read.load('/mnt/path/preprocessing/').toPandas()

X = df.drop(['rings', 'index'], axis=1)
y = df['rings']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metric (Root Mean Squared Error)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the evaluation metric
print("Root Mean Squared Error:", rmse)
