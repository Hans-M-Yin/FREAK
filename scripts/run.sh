#!/bin/bash
# Model to be EVALUATED. (this is different to judge api url and key)
MODEL_NAME=""
API_KEY=""
BASE_URL=""
# Judge model
JUDGE_MODEL_NAME=""
# Save path of evaluation results
SAVE_PATH=""
# Other params, such as sampling params, reasoning mode for reasoning model and COT prompting, please check the source code and modify.
python eval/evaluate.py \
  --model_name $MODEL_NAME \
  --api_key $API_KEY \
  --api_url $BASE_URL \
  --judge_model_name $JUDGE_MODEL_NAME \
  --save_path $SAVE_PATH