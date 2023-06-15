# Databricks notebook source
check=spark.read.load('/mnt/path/preprocessing/').toPandas()
check.head()

# COMMAND ----------

# data summarize check
dbutils.data.summarize(check)

# COMMAND ----------

import pandas as pd
import pandas_profiling

# Load the Abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
df = pd.read_csv(url, names=column_names)

# Perform data quality analysis
profile = pandas_profiling.ProfileReport(df)

# Generate the report
profile.to_file("/dbfs/tmp/data_quality_report.html")

# COMMAND ----------

dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
