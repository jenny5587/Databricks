# Databricks notebook source
# Basic Functions
from pyspark.sql import functions as f
from pyspark.sql import SQLContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import isnan, when, count, col, isnull, percent_rank
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, NullType, ShortType, DateType, BooleanType, BinaryType, FloatType
from pyspark.sql.functions import *
from databricks.feature_store import feature_table
import uuid

# For feature engineering
from pyspark.ml.feature import StandardScaler, VectorAssembler, VectorIndexer, StringIndexer, OneHotEncoder, ChiSqSelector, Bucketizer, Imputer
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.window import Window
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from functools import reduce
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import array, create_map, struct
import warnings
warnings.filterwarnings('ignore')
 
# For EDA/plotting & basic funcs
import pandas as pd
import numpy as np
pd.set_option("display.max_rows", 999)
pd.set_option("display.max_columns", 200)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import random
from math import floor
import itertools 
import mlflow
from pyspark.sql import SparkSession
 
# For Modeling
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, LinearSVC, RandomForestClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from sklearn.metrics import classification_report
from sparkdl.xgboost import XgboostClassifier
from sklearn.metrics import precision_recall_curve, roc_curve

# COMMAND ----------

def precision_recall(pred_df, col_name):
  """
  Input: predicted df from a fitted model 
  Output: precision, recall used in evaluation function
  """
  rdd_pred = pred_df.select([col_name, 'label']).rdd
  metrics_m = MulticlassMetrics(rdd_pred)
  precision = metrics_m.precision(1)
  recall = metrics_m.recall(label=1)
  f2 = metrics_m.fMeasure(1.0, 2.0)
  return (precision, recall, f2)
 
# Function to graph first position of the dense vector probability
# Used in threshold_tuning function
firstelement = udf(lambda item:float(item[1]),FloatType())


def threshold_tuning(valid_df):
  """
  Input: a validated df
  Output: a panda df that contains thresholds from 0-1 and associated precision/recall/f2 score
  """
  pr_results = []
  preds_new = valid_df
  preds_new = preds_new.withColumn('pred_probability', firstelement('probability'))
  thresholds = np.arange(start=0.1, stop=1.1, step=0.1)
  c = ['c1','c2','c3','c4','c5','c6','c7','c8','c9','c10']
  i=0
  for threshold in thresholds:
    preds_new = preds_new.withColumn(c[i], f.when(preds_new["pred_probability"].cast(DoubleType()) >= threshold , 1.0).otherwise(0.0).cast(DoubleType()))
    i = i+1
  for i in range(len(thresholds)-1):                       
    precision, recall, f2 = precision_recall(preds_new, c[i])
    pr_results.append((thresholds[i], precision, recall, f2))
  pr_df = pd.DataFrame(pr_results).rename(columns={0:'Threshold',1:'Precision',2:'Recall', 3:'f2-score'})
  return pr_df
 
class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        # Note this collect could be inefficient for large datasets 
        # considering there may be one probability per datapoint (at most)
        # The Scala version takes a numBins parameter, 
        # but it doesn't seem possible to pass this from Python to Java
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)
 
 
def evaluate(preds_train, preds_valid, classifier_name):
  """
  Input: predicted model for train and validation set, classifier name
  Output: PySpark DataFrame of evaluation metrics for class 0, 1 
  Confusion Matrix, PR-Curve
  """
  rdd_train_b = preds_train.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
  rdd_valid_b = preds_valid.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
  rdd_train_m = preds_train.select(['prediction', 'label']).rdd
  rdd_valid_m = preds_valid.select(['prediction', 'label']).rdd
    
  metrics_b_train = BinaryClassificationMetrics(rdd_train_b)
  metrics_m_train = MulticlassMetrics(rdd_train_m)
  metrics_b_valid = BinaryClassificationMetrics(rdd_valid_b)
  metrics_m_valid = MulticlassMetrics(rdd_valid_m)
    
  # pull metrics
  results_pddf = pd.DataFrame({
                        ' '         : ['Train', 'Test'],
                        'PR AUC'    : [metrics_b_train.areaUnderPR, metrics_b_valid.areaUnderPR],
                        'ROC AUC'   : [metrics_b_train.areaUnderROC, metrics_b_valid.areaUnderROC],
                        'F0.5 Score': [metrics_m_train.fMeasure(label=1.0, beta=0.5), metrics_m_valid.fMeasure(1.0, beta=0.5)],
                        'F2 Score'  : [metrics_m_train.fMeasure(label=1.0, beta=2.0), metrics_m_valid.fMeasure(1.0, beta=2.0)],
                        'Recall'    : [metrics_m_train.recall(label=1), metrics_m_valid.recall(label=1)],
                        'Precision' : [metrics_m_train.precision(1), metrics_m_valid.precision(1)],
                        'Accuracy'  : [metrics_m_train.accuracy, metrics_m_valid.accuracy]})
  results_pddf = results_pddf.set_index(' ')
  # print/plot results & pprint results
  pd.set_option("display.precision", 4)
  print(classifier_name)
  print(results_pddf.T)
  print('                                        Validation Plots')
  fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
   
  preds_valid_pr = threshold_tuning(preds_valid)
  # plot the Precision-Recall curve  
  sns.set(font_scale=1, style='whitegrid')
  sns.lineplot(x='Recall', y='Precision', data=preds_valid_pr, label='PR Curve', ax=axes[0,0])
  axes[0,0].set_title('Precision-Recall Curve')
  axes[0,0].legend()
   
  sns.lineplot(x='Threshold',y='Precision',data=preds_valid_pr,label='Precision',ax=axes[0,1])
  sns.lineplot(x='Threshold',y='Recall',data=preds_valid_pr,label='Recall',ax=axes[0,1])
  sns.lineplot(x='Threshold',y='f2-score',data=preds_valid_pr,label='F2',ax=axes[0,1])
  axes[0,1].vlines(0.5,0,1,color='red')  
  axes[0,1].set_xlabel('Threshold')
  axes[0,1].set_ylabel('Percentage')
  axes[0,1].set_title('Precision and Recall vs Threshold')
  axes[0,1].legend()
 
  # plot ROC AUC
  metrics_valid = CurveMetrics(rdd_valid_b)
  points_roc = metrics_valid.get_curve('roc')
  x_val = [x[0] for x in points_roc]
  y_val = [x[1] for x in points_roc]
  sns.lineplot(x_val, y_val, color='lightsteelblue',label='ROC AUC',ax= axes[1,0])
  axes[1,0].set_ylim([0.1, 1])
  axes[1,0].set_xlabel('1-Specificity')
  axes[1,0].set_ylabel('Recall')
  axes[1,0].set_title('ROC AUC curve (Validation)')
  axes[1,0].legend()
    
  # Plot confusion matrix
  cm = metrics_m_valid.confusionMatrix().toArray()
  confusion_matrix = pd.DataFrame(cm)
  sns.heatmap(confusion_matrix,annot=True,cmap='Blues',fmt=',',ax=axes[1,1])
  size = int(preds_valid.count())
  size = f'{size:,}'
  axes[1,1].set_title('Confusion Matrix - N={}'.format(size))
  axes[1,1].set_ylabel('Actual Values')
  axes[1,1].set_xlabel('Predicted Values')
  plt.show()   
  return results_pddf,fig

# COMMAND ----------

def feature_importance_plot(model,predictions):
    gain_values=model.stages[-1].get_booster().get_score(importance_type="gain")
    index = [int(i[1:]) for i in gain_values.keys()]

    save_features = {}
    for key in predictions.schema['features'].metadata['ml_attr']['attrs'].keys():
        for i in predictions.schema['features'].metadata['ml_attr']['attrs'][key]:
            save_features[i['idx']] = i['name']

    feature_imp_dict = {}
    for n, i in enumerate(index): 
        old_feat_ind = 'f' + str(i)
        feature_imp_dict[save_features[i]] = gain_values[old_feat_ind]
    
    feature_df = pd.DataFrame.from_dict(feature_imp_dict, orient = 'index').reset_index()
    feature_df.columns = ['name', 'weight']
    
    # Feature importance
    sns.set(font_scale=1, style='whitegrid')
    fig = plt.figure(figsize=(15, 12))
    ax = sns.barplot(x='weight', y='name', data=feature_df[(feature_df.weight != 0)].sort_values('weight', ascending = False).head(25), color='blue')
    ax.set_xlabel('weight')
    ax.set_title(f'Feature Gain from XGBoost ( Number of Features : {len(feature_df[feature_df.weight != 0])})')
    plt.show()
    return fig

# COMMAND ----------

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', UserWarning)
