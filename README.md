# WorldReasonBench: Human-Aligned Stress Testing of Video Generators as Future World-State Predictors

A comprehensive evaluation framework for assessing world-model reasoning in video generation models. WorldReasonBench evaluates whether generated videos demonstrate genuine understanding of physical laws, causal reasoning, temporal dynamics, and world knowledge through a two-pillar approach: (1) process-aware QA-based reasoning verification and (2) multi-dimensional quality assessment.

<div align="center">

[![arXiv](https://img.shields.io/badge/Paper-000000?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2605.10434)
[![Website](https://img.shields.io/badge/Website-000000?style=for-the-badge&logo=google-chrome&logoColor=white)](https://unix-ai-lab.github.io/WorldReasonBench/)
[![GitHub](https://img.shields.io/badge/Code-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/UniX-AI-Lab/WorldReasonBench)
[![Data](https://img.shields.io/badge/Data-0040A1?style=for-the-badge&logo=huggingface&logoColor=ffffff)](https://huggingface.co/datasets/WorldReasonBench)
[![Daily Paper](https://img.shields.io/badge/Daily_Paper-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000)](https://huggingface.co/papers/2605.10434)

</div>

---

## Directory Structure

```
WorldReasonBench/
├── data/
│   ├── *_with_prompts.json              # Task metadata with video prompts (4 categories)
│   ├── data_with_qa_gemini/
│   │   └── qa_*.json                    # QA evaluation data (open-ended + binary)
│   └── statistics_model_pairs_*.json    # Human-annotated preference pairs (5,969 pairs)
├── evaluation/
│   ├── eval_qa.py                       # QA pipeline (Stage 1: VLM answer, Stage 2: LLM judge)
│   └── reward_bench/
│       ├── __init__.py                  # Package init
│       ├── utils.py                     # Shared utilities (video/image encoding, templates)
│       ├── run_pairwise_eval.py         # Pairwise comparison evaluation
│       ├── run_pointwise_eval.py        # Pointwise S(v) scoring
│       ├── run_pointwise_eval_main_table.py  # S(v) for all model videos
│       ├── compute_pairwise_accuracy.py # Compute pairwise metrics
│       ├── compute_pointwise_metrics.py # Compute pointwise metrics (Spearman rho)
│       ├── mllm_tools/
│       │   ├── __init__.py             # Model registry
│       │   └── qwen3_5_eval.py         # Qwen3.5 OpenAI-compatible wrapper
│       └── templates/
│           └── video_generation/
│               ├── viescore.txt         # Pointwise scoring template
│               └── pairwise.txt         # Pairwise comparison template
├── scripts/
│   ├── run_qa_eval.sh                   # Example: QA evaluation
│   ├── run_pointwise_eval.sh            # Example: Pointwise evaluation
│   └── run_pairwise_eval.sh             # Example: Pairwise evaluation
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch vLLM server with Qwen3.5

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-27B \
  --port 30002 \
  --tensor-parallel-size 4 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --limit-mm-per-prompt video=1,image=1
```

The `--media-io-kwargs` flag is required for FPS-based video frame sampling.

### 3. Prepare video data

Organize your generated videos in the expected directory structure (see `MODEL_VIDEO_DIRS` in `run_pointwise_eval_main_table.py`).

## Usage

### QA-Based Reasoning Verification (Pillar I)

Evaluates whether generated videos contain the expected reasoning elements through a 2-stage pipeline:
- **Stage 1**: VLM answers questions about the video
- **Stage 2**: LLM judges answer correctness

```bash
python3 evaluation/eval_qa.py \
  --qa_json data/data_with_qa_gemini/qa_World-Knowledge.json \
  --video_dir /path/to/videos/World-Knowledge \
  --output_dir outputs/qa_eval/ \
  --base_url http://127.0.0.1:30002/v1 \
  --video_fps 4 \
  --qa_mode open_ended \
  --use_mm_processor_kwargs
```

Key metrics produced:
- **AccQA**: Simple QA accuracy
- **Score_PR**: Process-aware score combining accuracy with dynamic reasoning quality
- **Delta_RG**: Gap between easy (with hints) and difficult (without hints) accuracy

### Multi-Dimensional Quality Assessment (Pillar II)

#### Pointwise S(v) Scoring

Scores each video on 3 dimensions: reasoning correctness, content fidelity, visual aesthetics.

```bash
python3 evaluation/reward_bench/run_pointwise_eval.py \
  --pairs-json data/statistics_model_pairs_by_task_stratified_balanced_tie_v2.json \
  --judge-model qwen3.5-27b \
  --judge-base-url http://127.0.0.1:30002/v1 \
  --num-workers 2 \
  --max-parse-attempts 3 \
  --resume
```

Final score: `S(v) = 0.4 * s_reasoning + 0.3 * s_content + 0.3 * s_aesthetics`

#### Pairwise Comparison

Directly compares two videos and produces A>B / B>A / A=B verdicts.

```bash
python3 evaluation/reward_bench/run_pairwise_eval.py \
  --pairs-json data/statistics_model_pairs_by_task_stratified_balanced_tie_v2.json \
  --judge-model qwen3.5-27b \
  --judge-base-url http://127.0.0.1:30002/v1 \
  --num-workers 2 \
  --resume
```

### Computing Metrics

```bash
# Pairwise accuracy (with/without ties)
python3 evaluation/reward_bench/compute_pairwise_accuracy.py outputs/pairwise_eval.jsonl

# Pointwise correlation with human ratings
python3 evaluation/reward_bench/compute_pointwise_metrics.py \
  --videos outputs/pointwise_eval.jsonl \
  --induced-pairs outputs/pointwise_eval.induced_pairs.jsonl
```

## Benchmark Categories

| Category | Description |
|----------|-------------|
| World-Knowledge | Physics, chemistry, biology, geography reasoning |
| Human-Centric | Human behavior, social dynamics, emotion |
| Logic-Reasoning | Logical deduction, mathematical reasoning |
| Information-based-reasoning | Text comprehension, data interpretation |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_BASE_URL` | Default vLLM server URL |
| `OPENAI_API_KEY` | API key (use "EMPTY" for local vLLM) |
| `QWEN3_5_VIDEO_FPS` | Override video FPS for frame sampling |
| `QWEN3_5_NO_THINKING` | Set to "1" to disable thinking chain |
| `QWEN3_5_MAX_TOKENS` | Max generation tokens (default: 16384) |

## Citation

If you find this project helpful, please consider giving us a star and citing
our paper with:

```bibtex
@misc{wu2026worldreasonbenchhumanalignedstresstesting,
      title={WorldReasonBench: Human-Aligned Stress Testing of Video Generators as Future World-State Predictors}, 
      author={Keming Wu and Yijing Cui and Wenhan Xue and Qijie Wang and Xuan Luo and Zhiyuan Feng and Zuhao Yang and Sudong Wang and Sicong Jiang and Haowei Zhu and Zihan Wang and Ping Nie and Wenhu Chen and Bin Wang},
      year={2026},
      eprint={2605.10434},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.10434}, 
}
```

## License

This project is released under the [MIT License](LICENSE).
