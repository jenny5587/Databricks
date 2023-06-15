# Databricks notebook source
# 검증 위해 test로 진행
test = spark.read.load('/mnt/path/test/').toPandas()
test.head()

# COMMAND ----------

X_test = test.drop(['rings','index'], axis=1)
y_test = test['rings']

# COMMAND ----------

# mlflow에서 로컬로 모델 로드
import mlflow
import mlflow.xgboost
import xgboost as xgb
import numpy as np
import mlflow.pyfunc
from sklearn.metrics import mean_squared_error

model_version = mlflow.search_runs(experiment_ids="415430439948108", order_by=["attribute.start_time DESC"], max_results=1).iloc[0].iloc[0]
latest_model_uri = f"dbfs:/databricks/mlflow-tracking/415430439948108/{model_version}/artifacts/xgboost_model"
latest_model = mlflow.xgboost.load_model(model_uri=latest_model_uri)

# 등록된 모델을 가져와서 추론
loaded_model = mlflow.pyfunc.load_model(model_uri="dbfs:/databricks/mlflow-tracking/415430439948108/4fc799435b7f426f82d2ff594ccb3592/artifacts/xgboost_model")

y_pred = loaded_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE on test set : %f" % rmse)

# COMMAND ----------

if rmse > 4:
    # get the latest version of the model
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("xgboost_model", stages=["None"])[0]

    # transition the latest version to "Production" stage
    client.transition_model_version_stage(
        name="xgboost_model",
        version=latest_version.version,
        stage="Staging")
