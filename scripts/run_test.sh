#!/bin/bash
# Model to be EVALUATED. (this is different to judge api url and key)
MODEL_NAME="qwen2.5-vl-3b"
API_KEY="EMPTY"
BASE_URL="http://222.29.51.247:18903/v1"
# Judge model
JUDGE_MODEL_NAME="gpt-5-mini"
# Save path of evaluation results
SAVE_PATH="./results"
# Other params, such as sampling params, reasoning mode for reasoning model and COT prompting, please check the source code and modify.
python eval/evaluate.py \
  --model_name $MODEL_NAME \
  --api_key $API_KEY \
  --api_url $BASE_URL \
  --judge_model_name $JUDGE_MODEL_NAME \
  --save_path $SAVE_PATH