"""MLLM wrappers for WorldReason-Reward-Bench evaluation."""

MLLM_LIST = [
    "qwen3.5",
    "qwen3.5-27b",
]


def MLLM_Models(model_name: str):
    if model_name in {"qwen3.5", "qwen3.5-27b"}:
        from .qwen3_5_eval import Qwen3_5

        return Qwen3_5

    raise ValueError(f"Invalid model name: {model_name}, must be one of {MLLM_LIST}")


__all__ = ["MLLM_LIST", "MLLM_Models"]
