# Databricks notebook source
import mlflow
logged_model = 'runs:/50aa5e86cd8d46bd9e5cb6ef51fd3c35/xgboost_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
loaded_model.predict(data)

# COMMAND ----------

data = spark.read.load('/mnt/path/test').toPandas()
data[:,-2].head()

# COMMAND ----------

data = data.iloc[:,:-2]

# COMMAND ----------

# MAGIC %fs ls dbfs:/databricks/mlflow-tracking/
