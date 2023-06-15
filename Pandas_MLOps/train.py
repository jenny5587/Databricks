# Databricks notebook source
# pre-processing 델타 포맷 데이터 로드와 inference_data_df 만들기
pre_df = spark.read.load('/mnt/path/preprocessing/')
inference_data_df = pre_df.select("index", "rings")
inference_data_df.display()

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureLookup
from databricks.feature_store import FeatureStoreClient
from sklearn.model_selection import train_test_split

fs = FeatureStoreClient()
table_name = "abalone2a2bdd"

def load_data(table_name, lookup_key):
    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]
    # FeatureLookup 클래스는 특정 테이블에서 feature 정보를 조회하기 위한 클래스입니다. table_name은 feature가 저장된 테이블 이름을, lookup_key는 feature 정보를 조회하기 위한 key 값을 나타냅니다.

    # fs.create_training_set will look up features in model_feature_lookups with matched key from inference_data_df
    training_set = fs.create_training_set(inference_data_df, model_feature_lookups, label="rings", exclude_columns="index")
    training_pd = training_set.load_df().toPandas()

    # Create train and test datasets
    X = training_pd.drop("rings", axis=1)
    y = training_pd["rings"]

    # 먼저 데이터를 85%의 훈련용 데이터와 15%의 테스트용 데이터로 분리합니다.
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # 다음으로 85% 길이의 임시 훈련 데이터를 82.35%의 훈련 데이터와 17.65%의 검증 데이터로 분리합니다.
    # 이렇게 하면 원래 데이터의 70%가 훈련 데이터, 15%가 검증 데이터로 지정됩니다.
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

    # 결과 반환
    return X_train, X_val, X_test, y_train, y_val, y_test, training_set

X_train, X_val, X_test, y_train, y_val, y_test, training_set = load_data(table_name, "index")
X_train.head()

# COMMAND ----------

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# COMMAND ----------

import pandas as pd
import xgboost as xgb
import mlflow.xgboost
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from mlflow import sklearn
import mlflow

# MLflow experiment ID를 지정합니다.
experiment_name =  "/Users/minheejung@mz.co.kr/mlflow_abalone_cp/train"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name='xgboost_test') as run:
    # Train/validation split
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)  

    # Train an XGBoost model
    xgb_params = {
        'max_depth': 5,
        'min_child_weight': 6,
        'gamma': 4,
        'subsample': 0.7,
        'objective': 'reg:linear',
        "eta": 0.2,
        "eval_metric": "rmse"
    }

    xgb_model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=50,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=10
    )

    # Make predictions on the validation set
    y_val_pred = xgb_model.predict(dval)
    mse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    # Log metrics and the XGBoost model to MLflow
    mlflow.log_param("num_boost_round", 50)
    mlflow.log_metric("mse_val", mse_val)
    print("MSE on validation set: %f" % mse_val)
    mlflow.xgboost.log_model(xgb_model, "xgboost_model")
    
    # # Register model with MLflow registry
    run_id = mlflow.active_run().info.run_id
    artifact_path ="xgboost_model"
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
    # dbfs:/databricks/mlflow-tracking/454266561554176/adc96f550c3146d2a1ec9b01694e93f7/artifacts/xgboost_model
    mlflow.register_model(model_uri, "xgboost_model")
    model_path = '/dbfs/mnt/path/model/'
    mlflow.sklearn.save_model(xgb_model,model_path)
    # End the MLflow run
    mlflow.end_run()

# COMMAND ----------

# MAGIC %fs ls /mnt/path/model

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name="xgboost_model", version=1)

model_version_details.status
