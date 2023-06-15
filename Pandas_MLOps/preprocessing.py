# Databricks notebook source
# MAGIC %md
# MAGIC ## pre-processing
# MAGIC - 범주화, 수치화 나눠서 null 값 변환

# COMMAND ----------

df = spark.read.load('/mnt/path/delta').toPandas()
df.head()

# COMMAND ----------

df.shape

# COMMAND ----------

# 종속변수 추출
y = df.pop("rings")

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder

# 숫자 타입의 컬럼 이름들 -> Scaler를 적용
num_cols = ["length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.display()

# COMMAND ----------

df = pd.get_dummies(df)

# COMMAND ----------

df = pd.concat([df, pd.DataFrame(y)], axis=1)
# index 컬럼 추가
df['index'] = range(len(df))
df.head()

# COMMAND ----------

# spark로 만들어야 schema를 적용할 수 있음
df = spark.createDataFrame(df)

# COMMAND ----------

# DBTITLE 1,이미 feature store를 만들었으므로 주석처리 함
# # 전처리 된 테이블을 모두 feature_store에 등록
# from databricks.feature_store import FeatureStoreClient
# from databricks.feature_store import feature_table
# import uuid
# import pandas as pd
# from pyspark.sql.functions import *

# fs = FeatureStoreClient()

# ##feature store에 등록하려는 id 생성
# table_name = f"abalone" + str(uuid.uuid4())[:6]
# feature_df = df.drop(col('rings'))
# schema = feature_df.schema
# description = "feature_all"

# unique_id = uuid.uuid4()
# print(unique_id)

# ##feature store에 feature table 등록, primary_key option을 넣어주기 위해 index 생성
# fs.create_table(
#     name=table_name,
#     df=feature_df,
#     schema=feature_df.schema,
#     description = description,
#     primary_keys = ['index']
# )

# COMMAND ----------

# Delta Lake 포맷으로 저장
df.write.format('delta').mode('overwrite').save('/mnt/path/preprocessing')

# COMMAND ----------

# MAGIC %md
# MAGIC - test dataset을 위해 70,15,15로 나눠서 test data만 델타로 저장

# COMMAND ----------

from pyspark.sql.functions import col

train_size = int(0.7 * df.count())
valid_size = int(0.15 * df.count())
test_size = int(0.15 * df.count())

train = df.limit(train_size)
valid = df.limit(train_size + valid_size).subtract(train)
test = df.subtract(train).subtract(valid)

# test_s = spark.createDataFrame(test)

# test 데이터 Delta Lake 포맷으로 저장
test.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/path/test')

# COMMAND ----------

# MAGIC %fs ls /mnt/path/test

# COMMAND ----------

# dbutils.fs.rm('/mnt/path/train', True)
# dbutils.fs.rm('/mnt/path/valid', True)
# dbutils.fs.rm('/mnt/path/test', True)
