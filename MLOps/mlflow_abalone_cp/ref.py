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

# Pandas 데이터프레임을 Delta Lake 포맷으로 변환
spark_df = spark.createDataFrame(data)

# Delta Lake 포맷으로 저장
delta_path = '/mnt/delta/path'
spark_df.write.format('delta').mode('overwrite').save('/mnt/path/delta')

# COMMAND ----------

# DBTITLE 1,delta 포맷 완
# MAGIC %fs ls /mnt/delta/path/

# COMMAND ----------

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#numeric 수치화
numeric_features = list(df.columns)
numeric_features.remove("sex")
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
#categorical 범주화
categorical_features = ["sex"]
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)
# 전처리 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 전처리
X_pre = preprocessor.fit_transform(df)
# preprocess.fit_transform() 함수를 사용하면 DataFrame을 numpy 배열로 변환하면서 컬럼 이름이 사라짐
y_pre = y.to_numpy().reshape(len(y), 1)

# COMMAND ----------

print(X_pre.shape)
print(y_pre.shape)

# COMMAND ----------

X_pre

# COMMAND ----------

# DBTITLE 1,train/test/validation 분리
# 라벨 데이터 추가 및 데이터 병합
X = np.concatenate((y_pre,X_pre), axis=1)

# 데이터셋 분리
np.random.shuffle(X)
train, valid, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

# COMMAND ----------

# Pandas 데이터프레임 변환
train_df = pd.DataFrame(train)
valid_df = pd.DataFrame(valid)
test_df = pd.DataFrame(test)

# Pandas 데이터프레임을 Delta Lake 포맷으로 변환
train_df_s = spark.createDataFrame(train_df)
valid_df_s = spark.createDataFrame(valid_df)
test_df_s = spark.createDataFrame(test_df)

# COMMAND ----------

# # Delta Lake 포맷으로 저장
# train_df_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/delta/train')
# valid_df_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/delta/valid')
# test_df_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/delta/test')

# COMMAND ----------

# MAGIC %fs ls /mnt/delta/

# COMMAND ----------

# dbutils.fs.rm('/mnt/delta/train', True)
# dbutils.fs.rm('/mnt/delta/valid', True)
# dbutils.fs.rm('/mnt/delta/test', True)

# COMMAND ----------

# MAGIC %md
# MAGIC - 그게 아니면 검증이 따로 필요하면 sagemaker 코드로 돌려볼 수도 있음
# MAGIC - 하이퍼 파라미터 튜닝 최적화 코드도 돌려서 그 하이퍼 파라미터로 xgboost 진행

# COMMAND ----------

# MAGIC %md
# MAGIC ## train 

# COMMAND ----------

train = spark.read.load('/mnt/delta/train/').toPandas()
train.head()

# COMMAND ----------

import xgboost as xgb

from sklearn.model_selection import train_test_split

# Train-Validation-Test 데이터셋 분리
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# train-validation-test set 분리
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio/(1-test_ratio), random_state=1)

#X_train,X_test,X_validation,y_validation, y_train, y_test

# XGBoost 모델 학습을 위한 데이터셋 구성
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost 모델 하이퍼파라미터 설정
param = {'max_depth': 5, 'eta': 0.2, 'silent': 0, 'objective': 'reg:linear'}
param['gamma'] = 4
param['min_child_weight'] = 6
param['subsample'] = 0.7

# XGBoost 모델 학습
xgb_model = xgb.train(params=param,
                      dtrain=dtrain,
                      num_boost_round=50,  # 트리 개수
                      evals=[(dval, 'val')],  # 검증 데이터셋 지정
                      early_stopping_rounds=50,  # 조기종료 조건
                      verbose_eval=100  # 학습 과정 출력 주기
                     )

# Test 데이터셋으로 모델 평가
y_pred = xgb_model.predict(dtest)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 하이퍼파라미터 튜닝 최적화 코드

# COMMAND ----------

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

# 모델 생성
model = RandomForestClassifier()

# Grid Search
params = {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt', 'log2']}
grid_search = GridSearchCV(model, params, cv=5)
grid_search.fit(X_train, y_train)

print("Grid Search Results:")
print("Best score: ", grid_search.best_score_)
print("Best parameters: ", grid_search.best_params_)

# Random Search
params = {'n_estimators': [100, 200, 300], 'max_features': ['auto', 'sqrt', 'log2']}
random_search = RandomizedSearchCV(model, params, cv=5, n_iter=10)
random_search.fit(X_train, y_train)

print("Random Search Results:")
print("Best score: ", random_search.best_score_)
print("Best parameters: ", random_search.best_params_)


# COMMAND ----------

# DBTITLE 1,train_test_split 후 delta 저장
# train_size = int(0.7 * len(df))
# valid_size = int(0.15 * len(df))
# test_size = int(0.15 * len(df))

# train = df[:train_size]
# valid = df[train_size:train_size + valid_size]
# test = df[train_size + valid_size:]

# # Pandas 데이터프레임을 Delta Lake 포맷으로 변환
# train_s = spark.createDataFrame(train)
# valid_s = spark.createDataFrame(valid)
# test_s = spark.createDataFrame(test)

# # Delta Lake 포맷으로 저장
# train_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/path/train')
# valid_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/path/valid')
# test_s.write.format('delta').mode('overwrite').option('header', 'false').option('index', 'false').save('/mnt/path/test')

# COMMAND ----------

# DBTITLE 1,train,valid shuffle
# from sklearn.utils import shuffle

# # train-validation-test set 분리
# train_shuffled = shuffle(train, random_state=42)
# valid_shuffled = shuffle(valid, random_state=42)

# X_train = train_shuffled.drop(['rings'], axis=1)
# y_train = train_shuffled['rings']

# X_val = valid_shuffled.drop(['rings'], axis=1)
# y_val = valid_shuffled['rings']

# # Print the shapes to verify the sizes of the splits
# print("X_train shape:", X_train.shape)
# print("X_val shape:", X_val.shape)
# print("y_train shape:", y_train.shape)
# print("y_val shape:", y_val.shape)

# COMMAND ----------

# DBTITLE 1,xgboost 모델 학습
# import mlflow
# import mlflow.pyfunc
# import mlflow.xgboost
# import numpy as np
# import sklearn
# from sklearn.ensemble import RandomForestClassifier
# from mlflow.models.signature import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env
# import cloudpickle
# import time
# import xgboost as xgb
# from sklearn.metrics import mean_squared_error,mean_absolute_error

# # XGBoost 모델 학습을 위한 데이터셋 구성
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dval = xgb.DMatrix(X_val, label=y_val)
# # dtest = xgb.DMatrix(X_test, label=y_test)

# # XGBoost 모델 설정
# xgb_model = xgb.XGBRegressor(max_depth=5, eta=0.2, gamma=4, min_child_weight=6, subsample=0.7)

# # XGBoost로 학습
# xgb_model.fit(X_train, y_train)

# # validation set으로 검증
# y_pred_val = xgb_model.predict(X_val)
# rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
# print("RMSE on validation set : %f" % (rmse_val))

# # # test set으로 검증
# # y_pred_test = xgb_model.predict(X_test)
# # rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
# # print("RMSE on test set : %f" % (rmse_test))

# COMMAND ----------

# import pandas as pd
# import xgboost as xgb
# import mlflow.xgboost
# from sklearn.metrics import mean_squared_error
# import os


# # 파라미터
# xgb_params = {
# "max_depth": 5,
# "eta": 0.2,
# "gamma": 4,
# "min_child_weight": 6,
# "subsample": 0.7,
# "objective": "reg:linear",
# "eval_metric": "rmse"
# }

# # MLflow experiment ID를 지정합니다.
# experiment_name =  "/Users/minheejung@mz.co.kr/train"
# mlflow.set_experiment(experiment_name)

# with mlflow.start_run(run_name='xgboost_mlflow')as run:
#     dtrain = xgb.DMatrix(X_train, label=y_train)
#     dval = xgb.DMatrix(X_val, label=y_val)  

#     xgb_model= xgb.train(
#     params=xgb_params,
#     dtrain=dtrain,
#     num_boost_round=50,
#     evals=[(dtrain, 'train'), (dval, 'val')]
#     )

#     # validation set으로 검증
#     y_pred_val = xgb_model.predict(dval)
#     mse_val = mean_squared_error(y_val, y_pred_val)
#     mlflow.log_metric("mse", mse_val)
#     print("MSE on validation set: %f" % mse_val)

#     model_save_path = os.path.join(os.getcwd(), "models")
#     # Log the model with artifact_path argument
#     registered_model = mlflow.xgboost.log_model(xgb_model=xgb_model, artifact_path="model", registered_model_name="xgboost")
