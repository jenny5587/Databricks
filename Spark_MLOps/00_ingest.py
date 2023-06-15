# Databricks notebook source
# MAGIC %pip install kaggle

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME='yourname'
# MAGIC export KAGGLE_KEY='yourkey'
# MAGIC kaggle datasets download -d blastchar/telco-customer-churn --force
# MAGIC unzip telco-customer-churn.zip
# MAGIC # pip install kaggle 진행 한 이후로 작업해야함

# COMMAND ----------

# MAGIC %fs ls /mnt/

# COMMAND ----------

#export -> bash (linux)
dbutils.fs.cp("file:/databricks/driver/WA_Fn-UseC_-Telco-Customer-Churn.csv", "dbfs:/mnt/jenny_mlops/Telco-Customer-Churn.csv")

# COMMAND ----------

# MAGIC %fs ls /mnt/jenny_mlops

# COMMAND ----------

df = spark.read.csv("dbfs:/mnt/jenny_mlops/Telco-Customer-Churn.csv",header=True,inferSchema = True)
display(df)

# COMMAND ----------

# MAGIC %sql
# MAGIC create schema spark_mlops

# COMMAND ----------

df.write.format('delta').saveAsTable("spark_mlops.bronze",path='/mnt/jenny_mlops/data')

# COMMAND ----------

# MAGIC %fs ls /mnt/jenny_mlops/data
