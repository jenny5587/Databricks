# Databricks notebook source
# MAGIC %md
# MAGIC ### model train after feature_store load

# COMMAND ----------

# MAGIC %run ./package

# COMMAND ----------

df = spark.read.format('delta').load('/mnt/jenny_mlops/preprocessing').withColumnRenamed("Churn_encode",'label')

# COMMAND ----------

display(df.summary())
#null값 없음

# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3], seed = 42)
print("There are %d training examples and %d test examples." % (train.count(), test.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ### mlflow

# COMMAND ----------

cols = train.drop('customerID_encode','label').columns
#index, label rm ->feature

# COMMAND ----------

with mlflow.start_run(experiment_id ='2305299045478060', run_name='spark_mlops_xgboost', nested=True) as run:
    # SparkSession.builder.getOrCreate()
    from xgboost.spark import SparkXGBClassifier
    vec_assembler = VectorAssembler(inputCols=cols, outputCol="features", handleInvalid='keep')

    xgb_c = SparkXGBClassifier(seed=42,
                            features_Col="features",
                            label_Col="label",
                            )
    
    pipeline = Pipeline(stages=[vec_assembler,xgb_c])
    model = pipeline.fit(train)
    train_pre = model.transform(train)
    test_pre = model.transform(test)
    #metric
    results_pddf, fig = evaluate(train_pre, test_pre, "xgb")
    #importance_feature_plot
    feature_plot = feature_importance_plot(model,test_pre)

    #metric log
    for x, row in results_pddf.iterrows():
        for col in results_pddf.columns:
            name = (x + '_' + col).replace(' ', '_')
            mlflow.log_metric(name, row[col])

    #metric parameter
    param_map = model.stages[-1].extractParamMap()
    for param, value in param_map.items():
        if value is not None:
            mlflow.log_param(param.name, value)

    mlflow.spark.log_model(model, "model")
    mlflow.log_figure(fig, "plots/evaluate_plot.png")
    mlflow.log_figure(feature_plot, "plots/feature_importance.png")
