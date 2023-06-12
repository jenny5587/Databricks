# bbc article summary mlflow pipeline.test
import mlflow

mlflow.set_experiment(f"/Users/{DA.username}/MLflow experiment")

with mlflow.start_run():
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
          # "t5-small"
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
          # LLM에는 입력 시퀀스 길이에 대한 고정 제한
            "do_sample": do_sample,
          # 'True'로 설정하면 모델이 샘플링을 사용하여 텍스트를 생성하므로 보다 다양하고 창의적인 출력이 가능
        })
    # parameter log history
    results_list = [r["summary_text"] for r in results]
    # results 리스트의 각 요소에 대해 "summary_text" 필드 값을 추출하여 results_list에 저장하는 코드
    mlflow.llm.log_predictions(
        inputs=xsum_sample["document"],
        outputs=results_list,
        prompts=["" for _ in results_list],
    )
    # MLflow에는 쿼리 및 예측 추적을 위한 API mlflow.llm.log_predictions()를 사용함
    
    signature = mlflow.models.infer_signature(
        xsum_sample["document"][0],
        mlflow.transformers.generate_signature_output(
            summarizer, xsum_sample["document"][0]
        ),
    )# 모델의 입력 및 출력 스키마를 mlflow signature에 기록함
    print(f"Signature:\n{signature}\n")

    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path="summarizer",
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )
    
    ## model name define
    model_name = f"summarizer - {DA.username}"
    model_name = model_name.replace("/", "_").replace(".", "_").replace(":", "_")
    print(model_name)
    
    ## model register
    mlflow.register_model(model_uri=model_info.model_uri, name=model_name)
    ## model staging 
    from mlflow import MlflowClient
    client = MlflowClient()
    client.transition_model_version_stage(model_name, model_version, "staging")
    mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    # pyfunc 함수를 사용하여 모델을 MLflow 모델 형식으로 패키징하면, 이 모델을 다양한 환경에서 쉽게 로드하고 실행할 수 있음
    client.transition_model_version_stage(model_name, model_version, "production")
    # model이 production 되어 있는 상태면 test delta를 prediction 진행 가능
    prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}/Production",
    env_manager="local",
    # 사용할 환경 관리자를 지정
    result_type="string",
    ) #UDF의 예상 결과 유형을 지정
