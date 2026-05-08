#!/bin/bash
# Example: Run QA evaluation pipeline
# Requires: vLLM server running with Qwen3.5 model

python3 evaluation/eval_qa.py \
  --qa_json data/data_with_qa_gemini/qa_World-Knowledge.json \
  --video_dir <YOUR_VIDEO_DIR>/World-Knowledge \
  --output_dir outputs/qa_eval/ \
  --base_url http://127.0.0.1:30000/v1 \
  --api_key EMPTY \
  --model default \
  --video_fps 4
