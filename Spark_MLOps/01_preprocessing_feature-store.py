# Databricks notebook source
# MAGIC %run ./package

# COMMAND ----------

df = spark.read.load('/mnt/jenny_mlops/data')
display(df)

# COMMAND ----------

display(df.summary())

# COMMAND ----------

print(df.count())

# COMMAND ----------

# null check
from pyspark.sql.functions import col,isnan, when, count
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]
   ).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### object data encoding

# COMMAND ----------

integer_cols = [field for (field, dataType) in df.dtypes if ((dataType == "double") or (dataType =="int"))]
categorical_cols = [field for (field, dataType) in df.dtypes if dataType == "string"]
encode_cols = [column + "_encode" for column in categorical_cols]

# COMMAND ----------

# onehot_encoder, vec_assembler => pipeline
string_indexer = [StringIndexer(inputCol=column, outputCol=column+"_encode") for column in categorical_cols ]
vec_assembler = VectorAssembler(inputCols=integer_cols+encode_cols, outputCol="features_table")
#spark feature_store vec_assember 
pipeline = Pipeline(stages=string_indexer+[vec_assembler])
encode_df = pipeline.fit(df).transform(df)

# COMMAND ----------

display(encode_df)

# COMMAND ----------

integer_cols+encode_cols

# COMMAND ----------

# DBTITLE 1,delta preprocessing 저장
silver = encode_df.select(integer_cols+encode_cols)
silver.write.format('delta').saveAsTable("spark_mlops.silver",path='/mnt/jenny_mlops/preprocessing')

# COMMAND ----------

# silver = encode_df.select(integer_cols+encode_cols)
# display(silver)

# COMMAND ----------

# MAGIC %md
# MAGIC ### feature store 등록

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

##feature store에 등록하려는 id 생성
table_name = f"spark_mlops.feature_" + str(uuid.uuid4())[:6]
print(table_name)
feature_df=encode_df.select(integer_cols+encode_cols).drop('Churn_encode')
schema = feature_df.schema
description = "feature_store_feature_all"

##feature store에 feature table 등록
fs.create_table(
    name=table_name,
    primary_keys = ["customerID_encode"],
    df=feature_df,
    schema=feature_df.schema,
    description = description
)

# COMMAND ----------

feature_df.printSchema()
#customerID_encode = double

# COMMAND ----------

# MAGIC %md
# MAGIC - 새로운 data update

# COMMAND ----------

feature = spark.table("spark_mlops.feature_1cfab6").limit(1)
display(feature)

# COMMAND ----------

new_feature = feature.withColumn("gender_encode", when(feature.gender_encode == 1.0,0.0))\
    .withColumn("MonthlyCharges", when(feature.MonthlyCharges == 29.85, 40.0))

# COMMAND ----------

display(new_feature)

# COMMAND ----------

new_feature.printSchema()

# COMMAND ----------

fs.write_table(
  name = table_name,
  df = new_feature,
  mode = 'merge'
)

# COMMAND ----------

# MAGIC %sql
# MAGIC use database spark_mlops;
# MAGIC SHOW TABLES;

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from spark_mlops.feature_1cfab6
# MAGIC where customerID_encode == 5375.0
# MAGIC -- primary_key 건드리면 update가 되지 않고 데이터가 꼬임
