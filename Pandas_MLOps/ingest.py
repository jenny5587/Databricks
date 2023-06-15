# Databricks notebook source
import pandas as pd
#sagemaker url 뜯어보면 uci에서 데이터를 받아오는 것을 확인
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'

data = pd.read_csv(url, header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'])

# COMMAND ----------

# pandas의 데이터프레임을 먼저 로컬 파일로 저장합니다.
local_file_path = 'abalone.csv'
data.to_csv(local_file_path, index=None, header=True)

# Databricks DBFS에 파일을 업로드합니다.
dbutils.fs.put('/mnt/path/abalone.csv', open(local_file_path).read(), True)

# COMMAND ----------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# COMMAND ----------

data

# COMMAND ----------

data.describe(include="all")

# COMMAND ----------

data.info()

# COMMAND ----------

data.head()

# COMMAND ----------

data.columns = ["sex","length","diameter","height","whole_weight","shucked_weight","viscera_weight","shell_weight",'rings']

# COMMAND ----------

data['rings'] = data['rings'].astype(np.float64)

# COMMAND ----------

data.info()

# COMMAND ----------

# Spark 데이터프레임을 Delta Lake 포맷으로 변환
spark_df = spark.createDataFrame(data)

# Delta Lake 포맷으로 저장
delta_path = '/mnt/delta/path'
spark_df.write.format('delta').mode('overwrite').save('/mnt/path/delta')

# COMMAND ----------

# DBTITLE 1,delta 포맷 완
# MAGIC %fs ls /mnt/path
