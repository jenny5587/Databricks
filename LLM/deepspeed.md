### deepspeed 구성
!deepspeed {num_gpus_flag} \
    --module training.trainer \ 
    --input-model {input_model} \
    --deepspeed {deepspeed_config} \
    --epochs 2 \
    --local-output-dir {local_output_dir} \
    --dbfs-output-dir {dbfs_output_dir} \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --logging-steps 10 \
    --save-steps 200 \
    --save-total-limit 20 \
    --eval-steps 50 \
    --warmup-steps 50 \
    --test-size 15 \
    --lr 5e-6
    
    - module training.trainer : Deepspeed에서 제공하는 (훈련에 사용하는 모듈을 지정)
    - deepspeed {deepspeed_config} : DeepSpeed 구성 파일을 지정
    - per-device-train-batch-size 2: 디바이스당 훈련 배치 크기 2로 지정
    - per-device-eval-batch-size 2: 디바이스당 평가 배치 크기 2로 지정
    - lr 5e-2: 훈련에 사용할 학습률 5e-2로 지정
    
### model 학습 후 config.json을 deepspeed config로 줘서 model 추가 & 재학습 가능할듯
<img width="659" alt="image" src="https://github.com/jenny5587/databricks/assets/103649749/a94af646-1f97-4a63-b3cc-7f98aaa20062">
