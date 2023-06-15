# Databricks notebook source
# 검증 위해 test로 진행
test = spark.read.load('/mnt/path/test/').toPandas()
test.head()

# COMMAND ----------

pip install fairlearn

# COMMAND ----------

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from fairlearn.metrics import MetricFrame, selection_rate, count

# Load the Abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
df = pd.read_csv(url, names=column_names)

# Preprocess the dataset
# Assuming you want to predict if an abalone has "Rings" greater than or equal to 10
df["Target"] = (df["Rings"] >= 10).astype(int)

# Encode the "Sex" column using label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["Sex"] = label_encoder.fit_transform(df["Sex"])

# Separate the features and target variable
X = df.drop(["Rings", "Target"], axis=1)
y = df["Target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Define the sensitive feature (if any)
sensitive_feature = X_test["Sex"]  # Assuming "Sex" is the sensitive feature

# Define the prediction metric
prediction_metric = MetricFrame(metrics={"Selection Rate": selection_rate, "Count": count},
                               y_true=y_test, y_pred=model.predict(X_test),
                               sensitive_features=sensitive_feature)

# Calculate the model bias
model_bias = prediction_metric.difference(method='between_groups')
print("Model Bias:")
print(model_bias)
