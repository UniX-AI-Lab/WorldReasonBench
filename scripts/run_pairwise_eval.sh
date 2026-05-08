#!/bin/bash
# Example: Run pairwise comparison evaluation
python3 evaluation/reward_bench/run_pairwise_eval.py \
  --pairs-json data/statistics_model_pairs_by_task_stratified_balanced_tie_v2.json \
  --judge-model qwen3.5-27b \
  --judge-base-url http://127.0.0.1:30002/v1 \
  --num-workers 2 \
  --max-parse-attempts 3 \
  --resume
