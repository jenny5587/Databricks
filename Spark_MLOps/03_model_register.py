# Databricks notebook source
# MAGIC %md
# MAGIC ### test_roc_auc best model choice

# COMMAND ----------

from mlflow import MlflowClient
from mlflow.entities import ViewType
experiment_id = "2305299045478060"
run = mlflow.search_runs(
    experiment_ids=experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.Test_ROC_AUC DESC","attribute.start_time DESC"],
)

# COMMAND ----------

run

# COMMAND ----------

# DBTITLE 0,모델 등록
run_id = run.iloc[0]['run_id']
model_name = "spark_mlops"
model_uri = f"runs:/{run_id}/model"
registered_model_version = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### model stage 변경

# COMMAND ----------

from mlflow import MlflowClient
import mlflow
client = mlflow.tracking.MlflowClient()
latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
latest_version

# COMMAND ----------

client.transition_model_version_stage(name = model_name,
                                      version=latest_version.version, 
                                      stage="Production")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 조건 후 model stage 변경 - 의사결정 조치 필요

# COMMAND ----------

if run['metrics.Test_Accuracy'][0]>=0.9 or run['metrics.Test_ROC_AUC'][0]>=0.8:
    print(f'model name : {model_name}')
    run[['run_id','experiment_id','metrics.Test_Accuracy','metrics.Test_ROC_AUC']].display()
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    client.transition_model_version_stage(name = model_name,version=latest_version.version, stage="Staging")
else :
    pass
