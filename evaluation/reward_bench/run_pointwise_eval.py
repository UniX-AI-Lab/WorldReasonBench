#!/usr/bin/env python3
"""Run WorldReason-Reward-Bench point-wise evaluation with the VieScore template."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # evaluation/reward_bench -> evaluation -> project root
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils import load_template  # noqa: E402

import run_pairwise_eval as pairwise_lib  # noqa: E402


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
WEIGHTED_SCORE_WEIGHTS = (0.4, 0.3, 0.3)


@dataclass(frozen=True)
class PointwiseTemplateChunks:
    before_image: str
    before_prompt: str
    before_video: str
    after_video: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-json",
        default=str(pairwise_lib.PROJECT_ROOT / "data" / "statistics_model_pairs_by_task_stratified_balanced_tie_v2.json"),
        help="Path to statistics_model_pairs_by_task.json.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3.5-27b",
        choices=pairwise_lib.MLLM_LIST,
        help="Judge model name registered in reward_bench.",
    )
    parser.add_argument(
        "--judge-model-path",
        default=None,
        help="Optional underlying model identifier or alias override.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL override.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=None,
        help="Optional API key override.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Video-level output JSONL path. Defaults to outputs/worldreason_reward_bench/pointwise_<timestamp>.jsonl",
    )
    parser.add_argument(
        "--pair-output-jsonl",
        default=None,
        help="Optional induced pairwise JSONL path. Defaults next to --output-jsonl.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter by benchmark category, e.g. 'World-Knowledge'.",
    )
    parser.add_argument(
        "--sub-category-v2",
        default=None,
        help="Filter by benchmark sub_category_v2 label.",
    )
    parser.add_argument("--task-id", default=None, help="Only evaluate one task id.")
    parser.add_argument(
        "--pair-type",
        default=None,
        choices=["gap", "tie"],
        help="Optional pair_type filter applied before deduplicating videos.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N filtered pairs.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N filtered pairs.")
    parser.add_argument(
        "--template",
        default="viescore",
        help=(
            "Name of the pointwise template under templates/video_generation/. "
            "Default 'viescore'; use 'viescore_process_outcome' for process-outcome decoupling."
        ),
    )
    parser.add_argument(
        "--prompt-field",
        default="video_prompt",
        help="Prompt field loaded from *_with_prompts.json.",
    )
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=8,
        help="Frames sampled per video when the judge expands video inputs to images.",
    )
    parser.add_argument(
        "--frame-max-side",
        type=int,
        default=768,
        help="Resize transmitted images so their longest side is at most this many pixels.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality for encoded images and sampled video frames sent to the judge.",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=3,
        help="Retry retryable HTTP failures up to this many times.",
    )
    parser.add_argument(
        "--image-cache-size",
        type=int,
        default=256,
        help="LRU cache size for encoded input images inside the judge.",
    )
    parser.add_argument(
        "--video-cache-size",
        type=int,
        default=128,
        help="LRU cache size for sampled-and-encoded videos inside the judge.",
    )
    parser.add_argument(
        "--score-aggregation",
        default="weighted",
        choices=["weighted", "mean", "sum", "reasoning", "content", "aesthetics"],
        help=(
            "Collapse the 3 VieScore dimensions into a scalar before inducing pairwise "
            "preferences. 'weighted' uses the paper formula 0.4*reasoning + "
            "0.3*content + 0.3*aesthetics."
        ),
    )
    parser.add_argument(
        "--tie-threshold",
        type=float,
        default=0.1,
        help="If |score_A - score_B| <= this value, induce an A=B tie.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip only video_ids that already have parsed scores in the output JSONL. "
            "Rows with missing parsed_scores are re-evaluated on resume."
        ),
    )
    parser.add_argument(
        "--max-parse-attempts",
        type=int,
        default=1,
        help=(
            "If point-wise score parsing fails, re-run the current video up to this many "
            "times before moving on."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build inputs and output rows without calling the judge.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print one textual progress update every N newly evaluated videos.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Run up to this many video evaluations concurrently. Use 1 to disable concurrency.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1.")
    if args.max_parse_attempts < 1:
        parser.error("--max-parse-attempts must be at least 1.")
    if args.tie_threshold < 0:
        parser.error("--tie-threshold must be non-negative.")
    return args


def default_output_jsonl() -> Path:
    out_dir = pairwise_lib.PROJECT_ROOT / "outputs" / "worldreason_reward_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"pointwise_{timestamp}.jsonl"


def default_pair_output_jsonl(video_output_jsonl: Path) -> Path:
    return video_output_jsonl.with_name(f"{video_output_jsonl.stem}.induced_pairs.jsonl")


def build_video_id(task_id: str, model_name: str) -> str:
    return f"{task_id}::{model_name}"


def dedupe_rows_by_video_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        video_id = row.get("video_id")
        if video_id:
            indexed[video_id] = normalize_loaded_video_row(row)
    return list(indexed.values())


def parse_pointwise_template(template: str) -> PointwiseTemplateChunks:
    if "<image>" not in template or "<prompt>" not in template or "<video>" not in template:
        raise ValueError("VieScore template is missing required placeholders.")

    before_image, after_image = template.split("<image>", 1)
    before_prompt, after_prompt = after_image.split("<prompt>", 1)
    before_video, after_video = after_prompt.split("<video>", 1)
    return PointwiseTemplateChunks(
        before_image=before_image,
        before_prompt=before_prompt,
        before_video=before_video,
        after_video=after_video,
    )


def build_pointwise_inputs(
    template_chunks: PointwiseTemplateChunks,
    video_item: Dict[str, Any],
    prompt_text: str,
    max_num_frames: int,
) -> List[Dict[str, Any]]:
    interleaved_inputs: List[Dict[str, Any]] = []
    if template_chunks.before_image.strip():
        interleaved_inputs.append({"type": "text", "content": template_chunks.before_image})
    interleaved_inputs.append({"type": "image", "content": video_item["input_image_path"]})

    second_text = (
        template_chunks.before_prompt
        + prompt_text
        + template_chunks.before_video
    )
    if second_text.strip():
        interleaved_inputs.append({"type": "text", "content": second_text})
    interleaved_inputs.append(
        {
            "type": "video",
            "content": video_item["video_path"],
            "max_num_frames": max_num_frames,
            "label": "The generated video",
        }
    )
    if template_chunks.after_video.strip():
        interleaved_inputs.append({"type": "text", "content": template_chunks.after_video})
    return interleaved_inputs


def _json_candidates(raw_response: str) -> List[str]:
    stripped = (raw_response or "").strip()
    if not stripped:
        return []

    candidates: List[str] = [stripped]
    candidates.extend(match.strip() for match in JSON_BLOCK_PATTERN.findall(stripped))

    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        candidates.append(stripped[brace_start : brace_end + 1].strip())

    normalized: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate.lower().startswith("json\n"):
            candidate = candidate[5:].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            normalized.append(candidate)
    return normalized


def _coerce_score_list(value: Any) -> List[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None

    scores: List[float] = []
    for entry in value[:3]:
        try:
            scores.append(float(entry))
        except (TypeError, ValueError):
            return None
    return scores


def parse_pointwise_response(raw_response: str) -> Dict[str, Any] | None:
    for candidate in _json_candidates(raw_response):
        parsed_obj: Any
        try:
            parsed_obj = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed_obj = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue

        if not isinstance(parsed_obj, dict):
            continue

        scores = _coerce_score_list(parsed_obj.get("score"))
        if scores is None:
            continue

        reasoning_value = parsed_obj.get("reasoning")
        reasoning_text = (
            reasoning_value
            if isinstance(reasoning_value, str)
            else json.dumps(reasoning_value, ensure_ascii=False)
            if reasoning_value is not None
            else ""
        )

        return {
            "parsed_json": parsed_obj,
            "reasoning": reasoning_text.strip(),
            "scores": scores,
        }
    return None


def build_video_eval_items(
    pairs: List[Dict[str, Any]],
    task_lookup: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}

    for pair in pairs:
        task_meta = task_lookup.get(pair["task_id"], {})
        pair_id = pairwise_lib.build_pair_id(pair)
        candidates = [
            {
                "video_id": build_video_id(pair["task_id"], pair["model_1"]),
                "task_id": pair["task_id"],
                "category": task_meta.get("category"),
                "sub_category": task_meta.get("sub_category"),
                "sub_category_v2": task_meta.get("sub_category_v2"),
                "model_name": pair["model_1"],
                "human_score": pair.get("score_1"),
                "input_image_path": pair.get("input_image_path"),
                "video_path": pair.get("model_1_video_path"),
            },
            {
                "video_id": build_video_id(pair["task_id"], pair["model_2"]),
                "task_id": pair["task_id"],
                "category": task_meta.get("category"),
                "sub_category": task_meta.get("sub_category"),
                "sub_category_v2": task_meta.get("sub_category_v2"),
                "model_name": pair["model_2"],
                "human_score": pair.get("score_2"),
                "input_image_path": pair.get("input_image_path"),
                "video_path": pair.get("model_2_video_path"),
            },
        ]

        for candidate in candidates:
            existing = indexed.get(candidate["video_id"])
            if existing is None:
                candidate["source_pair_count"] = 1
                indexed[candidate["video_id"]] = candidate
                continue

            if str(existing.get("video_path") or "") != str(candidate.get("video_path") or ""):
                raise ValueError(
                    f"Inconsistent video_path for {candidate['video_id']}: "
                    f"{existing.get('video_path')} vs {candidate.get('video_path')}"
                )
            if str(existing.get("input_image_path") or "") != str(
                candidate.get("input_image_path") or ""
            ):
                raise ValueError(
                    f"Inconsistent input_image_path for {candidate['video_id']}: "
                    f"{existing.get('input_image_path')} vs {candidate.get('input_image_path')}"
                )

            existing_score = existing.get("human_score")
            candidate_score = candidate.get("human_score")
            if existing_score is not None and candidate_score is not None:
                if abs(float(existing_score) - float(candidate_score)) > 1e-9:
                    raise ValueError(
                        f"Inconsistent human_score for {candidate['video_id']} while processing {pair_id}."
                    )

            existing["source_pair_count"] = int(existing.get("source_pair_count") or 0) + 1

    return sorted(indexed.values(), key=lambda item: item["video_id"])


def _derive_score_fields(scores: List[float] | None) -> Dict[str, Any]:
    if not scores:
        return {
            "parsed_scores": None,
            "reasoning_score": None,
            "content_score": None,
            "aesthetics_score": None,
            "score_sum": None,
            "score_mean": None,
            "score_weighted": None,
        }

    score_sum = float(sum(scores))
    score_weighted = float(
        sum(weight * float(score) for weight, score in zip(WEIGHTED_SCORE_WEIGHTS, scores))
    )
    return {
        "parsed_scores": scores,
        "reasoning_score": float(scores[0]),
        "content_score": float(scores[1]),
        "aesthetics_score": float(scores[2]),
        "score_sum": score_sum,
        "score_mean": score_sum / float(len(scores)),
        "score_weighted": score_weighted,
    }


def normalize_loaded_video_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    score_fields = _derive_score_fields(_coerce_score_list(normalized.get("parsed_scores")))
    normalized.update(score_fields)
    normalized["is_parsed"] = score_fields["parsed_scores"] is not None
    return normalized


def build_video_result_row(
    video_item: Dict[str, Any],
    task_meta: Dict[str, Any],
    prompt_text: str,
    raw_response: str,
    parsed_response: Dict[str, Any] | None,
    judge_model: str,
    max_num_frames: int,
    parse_attempts_used: int,
    max_parse_attempts: int,
) -> Dict[str, Any]:
    parsed_scores = parsed_response.get("scores") if parsed_response else None
    score_fields = _derive_score_fields(parsed_scores)
    judge_reasoning = (
        parsed_response.get("reasoning", "").strip()
        if parsed_response is not None
        else (raw_response or "").strip()
    )
    return {
        "video_id": video_item["video_id"],
        "judge_model": judge_model,
        "task_id": video_item["task_id"],
        "category": task_meta.get("category"),
        "sub_category": task_meta.get("sub_category"),
        "sub_category_v2": task_meta.get("sub_category_v2"),
        "prompt_text": prompt_text,
        "model_name": video_item.get("model_name"),
        "human_score": video_item.get("human_score"),
        "input_image_path": video_item.get("input_image_path"),
        "video_path": video_item.get("video_path"),
        "source_pair_count": video_item.get("source_pair_count"),
        "max_num_frames": max_num_frames,
        "parse_attempts_used": parse_attempts_used,
        "max_parse_attempts": max_parse_attempts,
        "raw_response": raw_response,
        "parsed_json": parsed_response.get("parsed_json") if parsed_response else None,
        "judge_reasoning": judge_reasoning,
        "judge_summary": pairwise_lib.first_paragraph(judge_reasoning),
        **score_fields,
        "is_parsed": score_fields["parsed_scores"] is not None,
    }


def evaluate_video(
    video_item: Dict[str, Any],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    template_chunks: PointwiseTemplateChunks,
    args: argparse.Namespace,
    judge: Any,
) -> Dict[str, Any]:
    task_meta = task_lookup.get(video_item["task_id"])
    if task_meta is None:
        raise KeyError(f"Task id {video_item['task_id']} not found in prompt source JSONs.")

    prompt_text = prompt_lookup[video_item["task_id"]]
    inputs = build_pointwise_inputs(
        template_chunks,
        video_item,
        prompt_text,
        max_num_frames=args.max_num_frames,
    )

    if args.dry_run:
        raw_response = ""
        parsed_response = None
        parse_attempts_used = 0
    else:
        raw_response = ""
        parsed_response = None
        parse_attempts_used = 0
        for attempt_idx in range(1, args.max_parse_attempts + 1):
            parse_attempts_used = attempt_idx
            raw_response = judge(inputs)
            parsed_response = parse_pointwise_response(raw_response)
            if parsed_response is not None:
                break

    return build_video_result_row(
        video_item=video_item,
        task_meta=task_meta,
        prompt_text=prompt_text,
        raw_response=raw_response,
        parsed_response=parsed_response,
        judge_model=args.judge_model,
        max_num_frames=args.max_num_frames,
        parse_attempts_used=parse_attempts_used,
        max_parse_attempts=args.max_parse_attempts,
    )


def iter_concurrent_video_evaluations(
    video_items: List[Dict[str, Any]],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    template_chunks: PointwiseTemplateChunks,
    args: argparse.Namespace,
    judge: Any,
) -> Iterable[Dict[str, Any]]:
    executor = ThreadPoolExecutor(
        max_workers=args.num_workers,
        thread_name_prefix="pointwise-eval",
    )
    in_flight: Dict[Future, str] = {}
    video_iter = iter(video_items)

    def submit_next() -> bool:
        try:
            video_item = next(video_iter)
        except StopIteration:
            return False
        future = executor.submit(
            evaluate_video,
            video_item,
            task_lookup=task_lookup,
            prompt_lookup=prompt_lookup,
            template_chunks=template_chunks,
            args=args,
            judge=judge,
        )
        in_flight[future] = str(video_item["video_id"])
        return True

    try:
        while len(in_flight) < args.num_workers and submit_next():
            pass

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                video_id = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    import traceback as _tb
                    _tb.print_exc()
                    print(f"[WARN] Skipping failed video: {video_id} -> {exc!r}")
                    row = {
                        "video_id": video_id,
                        "is_parsed": False,
                        "parsed_scores": None,
                        "_error": str(exc),
                    }
                yield row
                submit_next()
    except Exception:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def video_row_is_parsed(row: Dict[str, Any]) -> bool:
    return _coerce_score_list(row.get("parsed_scores")) is not None


def build_video_parse_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    parsed = sum(1 for row in rows if video_row_is_parsed(row))
    return {
        "total": total,
        "parsed": parsed,
        "parse_rate": (parsed / total) if total else None,
    }


def update_video_parse_stats(stats: Dict[str, Any], row: Dict[str, Any]) -> None:
    total = int(stats.get("total") or 0) + 1
    parsed = int(stats.get("parsed") or 0)
    if video_row_is_parsed(row):
        parsed += 1
    stats.update(
        {
            "total": total,
            "parsed": parsed,
            "parse_rate": (parsed / total) if total else None,
        }
    )


def summarize_video_parse_by_field(
    rows: List[Dict[str, Any]],
    field: str,
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "UNKNOWN")
        grouped.setdefault(key, []).append(row)
    return {
        key: build_video_parse_stats(group_rows)
        for key, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def select_scalar_score(video_row: Dict[str, Any], aggregation: str) -> float | None:
    if aggregation == "weighted":
        value = video_row.get("score_weighted")
    elif aggregation == "mean":
        value = video_row.get("score_mean")
    elif aggregation == "sum":
        value = video_row.get("score_sum")
    elif aggregation == "reasoning":
        value = video_row.get("reasoning_score")
    elif aggregation == "content":
        value = video_row.get("content_score")
    elif aggregation == "aesthetics":
        value = video_row.get("aesthetics_score")
    else:
        raise ValueError(f"Unsupported score aggregation: {aggregation}")

    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def describe_score_aggregation(aggregation: str) -> str:
    if aggregation == "weighted":
        return "weighted (0.4 * reasoning + 0.3 * content + 0.3 * aesthetics)"
    return aggregation


def induce_pairwise_verdict(
    score_1: float | None,
    score_2: float | None,
    tie_threshold: float,
) -> str | None:
    if score_1 is None or score_2 is None:
        return None
    if abs(score_1 - score_2) <= tie_threshold:
        return "A=B"
    return "A>B" if score_1 > score_2 else "B>A"


def is_correct_induced_prediction(
    parsed_verdict: str | None,
    expected_verdict: str,
) -> bool | None:
    if parsed_verdict is None:
        return None
    return parsed_verdict == expected_verdict


def induced_winner_model(parsed_verdict: str | None, pair: Dict[str, Any]) -> str | None:
    if parsed_verdict == "A>B":
        return pair.get("model_1")
    if parsed_verdict == "B>A":
        return pair.get("model_2")
    return None


def build_induced_pair_rows(
    pairs: List[Dict[str, Any]],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    video_rows_by_id: Dict[str, Dict[str, Any]],
    judge_model: str,
    score_aggregation: str,
    tie_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        task_meta = task_lookup.get(pair["task_id"])
        if task_meta is None:
            raise KeyError(f"Task id {pair['task_id']} not found in prompt source JSONs.")

        video_1 = video_rows_by_id.get(build_video_id(pair["task_id"], pair["model_1"]))
        video_2 = video_rows_by_id.get(build_video_id(pair["task_id"], pair["model_2"]))
        scalar_1 = select_scalar_score(video_1 or {}, score_aggregation)
        scalar_2 = select_scalar_score(video_2 or {}, score_aggregation)
        parsed_verdict = induce_pairwise_verdict(scalar_1, scalar_2, tie_threshold)
        expected_verdict = pairwise_lib.get_expected_verdict(pair)
        rows.append(
            {
                "pair_id": pairwise_lib.build_pair_id(pair),
                "judge_model": judge_model,
                "task_id": pair["task_id"],
                "category": task_meta.get("category"),
                "sub_category": task_meta.get("sub_category"),
                "sub_category_v2": task_meta.get("sub_category_v2"),
                "prompt_text": prompt_lookup[pair["task_id"]],
                "pair_type": pair.get("pair_type"),
                "model_1": pair.get("model_1"),
                "score_1": pair.get("score_1"),
                "model_2": pair.get("model_2"),
                "score_2": pair.get("score_2"),
                "score_gap": abs(float(pair.get("score_1", 0.0)) - float(pair.get("score_2", 0.0))),
                "input_image_path": pair.get("input_image_path"),
                "model_1_video_path": pair.get("model_1_video_path"),
                "model_2_video_path": pair.get("model_2_video_path"),
                "video_1_id": build_video_id(pair["task_id"], pair["model_1"]),
                "video_2_id": build_video_id(pair["task_id"], pair["model_2"]),
                "pointwise_scores_1": (video_1 or {}).get("parsed_scores"),
                "pointwise_scores_2": (video_2 or {}).get("parsed_scores"),
                "pointwise_score_mean_1": (video_1 or {}).get("score_mean"),
                "pointwise_score_mean_2": (video_2 or {}).get("score_mean"),
                "pointwise_score_sum_1": (video_1 or {}).get("score_sum"),
                "pointwise_score_sum_2": (video_2 or {}).get("score_sum"),
                "pointwise_score_weighted_1": (video_1 or {}).get("score_weighted"),
                "pointwise_score_weighted_2": (video_2 or {}).get("score_weighted"),
                "pointwise_scalar_1": scalar_1,
                "pointwise_scalar_2": scalar_2,
                "score_aggregation": score_aggregation,
                "tie_threshold": tie_threshold,
                "parsed_verdict": parsed_verdict,
                "expected_verdict": expected_verdict,
                "is_correct": is_correct_induced_prediction(parsed_verdict, expected_verdict),
                "winner_model": induced_winner_model(parsed_verdict, pair),
            }
        )
    return rows


def save_induced_pair_rows_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    rows = pairwise_lib.dedupe_rows_by_pair_id(rows)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_scalar(value: Any) -> str:
    try:
        return "n/a" if value is None else f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def save_summary(
    path: Path,
    *,
    video_rows: List[Dict[str, Any]],
    pair_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    selected_pairs: List[Dict[str, Any]],
    selected_video_items: List[Dict[str, Any]],
    induced_pair_jsonl: Path,
) -> None:
    video_rows = dedupe_rows_by_video_id(video_rows)
    pair_rows = pairwise_lib.dedupe_rows_by_pair_id(pair_rows)
    pair_verdict_counts = Counter(row.get("parsed_verdict") or "UNPARSED" for row in pair_rows)
    expected_counts = Counter(row.get("expected_verdict") or "UNKNOWN" for row in pair_rows)
    winner_counts = Counter(row.get("winner_model") or "NO_WINNER" for row in pair_rows)
    summary = {
        "selected_pairs": len(selected_pairs),
        "selected_videos": len(selected_video_items),
        "num_video_rows": len(video_rows),
        "score_aggregation": args.score_aggregation,
        "score_aggregation_description": describe_score_aggregation(args.score_aggregation),
        "tie_threshold": args.tie_threshold,
        "induced_pair_output_jsonl": str(induced_pair_jsonl),
        "video_overall": build_video_parse_stats(video_rows),
        "video_parse_by_category": summarize_video_parse_by_field(video_rows, "category"),
        "video_parse_by_sub_category": summarize_video_parse_by_field(video_rows, "sub_category"),
        "video_parse_by_sub_category_v2": summarize_video_parse_by_field(
            video_rows, "sub_category_v2"
        ),
        "video_parse_by_model": summarize_video_parse_by_field(video_rows, "model_name"),
        "induced_pairwise": {
            "num_rows": len(pair_rows),
            "overall": pairwise_lib.build_accuracy_stats(pair_rows),
            "verdict_counts": dict(pair_verdict_counts),
            "expected_verdict_counts": dict(expected_counts),
            "winner_counts": dict(winner_counts),
            "accuracy_by_category": pairwise_lib.summarize_by_field(pair_rows, "category"),
            "accuracy_by_sub_category": pairwise_lib.summarize_by_field(
                pair_rows, "sub_category"
            ),
            "accuracy_by_sub_category_v2": pairwise_lib.summarize_by_field(
                pair_rows, "sub_category_v2"
            ),
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def print_final_summary(video_rows: List[Dict[str, Any]], pair_rows: List[Dict[str, Any]]) -> None:
    video_stats = build_video_parse_stats(video_rows)
    print(
        "video_overall:"
        f" total={video_stats['total']}"
        f" parsed={video_stats['parsed']}"
        f" parse_rate={pairwise_lib.format_pct(video_stats['parse_rate'])}"
    )
    print("induced_pairwise_accuracy:")
    pairwise_lib.print_accuracy_summary(pair_rows)


def main() -> None:
    args = parse_args()
    if (
        args.num_workers > 1
        and not args.dry_run
        and args.judge_model not in {"qwen3.5", "qwen3.5-27b"}
    ):
        raise ValueError(
            "--num-workers > 1 is currently supported only for qwen3.5 / qwen3.5-27b."
        )

    output_jsonl = Path(args.output_jsonl) if args.output_jsonl else default_output_jsonl()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    induced_pair_jsonl = (
        Path(args.pair_output_jsonl)
        if args.pair_output_jsonl
        else default_pair_output_jsonl(output_jsonl)
    )
    induced_pair_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pair_records = pairwise_lib.load_json(Path(args.pairs_json))
    task_lookup = pairwise_lib.load_task_lookup()
    template_chunks = parse_pointwise_template(load_template("video_generation", args.template))
    selected_pairs = list(pairwise_lib.iter_filtered_pairs(pair_records, args, task_lookup))
    selected_video_items = build_video_eval_items(selected_pairs, task_lookup)
    prompt_lookup = pairwise_lib.build_prompt_lookup(
        task_lookup, selected_video_items, args.prompt_field
    )

    selected_video_ids = {item["video_id"] for item in selected_video_items}
    resume_rows = (
        [
            row
            for row in dedupe_rows_by_video_id(pairwise_lib.load_jsonl_rows(output_jsonl))
            if row.get("video_id") in selected_video_ids
        ]
        if args.resume
        else []
    )
    existing_rows = [
        row for row in resume_rows if row.get("video_id") and video_row_is_parsed(row)
    ]
    requeued_unparsed_rows = [
        row for row in resume_rows if row.get("video_id") and not video_row_is_parsed(row)
    ]
    completed_video_ids = {row["video_id"] for row in existing_rows if row.get("video_id")}
    judge = None if args.dry_run else pairwise_lib.instantiate_judge(args)

    rows_written: List[Dict[str, Any]] = list(existing_rows)
    live_stats = build_video_parse_stats(existing_rows)
    pending_video_items = [
        item for item in selected_video_items if item["video_id"] not in completed_video_ids
    ]

    print(f"selected_pairs={len(selected_pairs)}")
    print(f"selected_videos={len(selected_video_items)}")
    print(f"pending_videos={len(pending_video_items)}")
    print(f"output_jsonl={output_jsonl}")
    print(f"induced_pair_output_jsonl={induced_pair_jsonl}")
    print(f"num_workers={args.num_workers}")
    print(f"max_parse_attempts={args.max_parse_attempts}")
    print(f"score_aggregation={args.score_aggregation}")
    print(
        "score_aggregation_description="
        f"{describe_score_aggregation(args.score_aggregation)}"
    )
    print(f"tie_threshold={args.tie_threshold}")
    if args.resume and existing_rows:
        print(
            "resume_status:"
            f" loaded={len(resume_rows)}"
            f" completed={len(existing_rows)}"
            f" requeue_unparsed={len(requeued_unparsed_rows)}"
            f" parse_rate={pairwise_lib.format_pct(live_stats['parse_rate'])}"
        )
    elif args.resume:
        print(
            "resume_status:"
            f" loaded={len(resume_rows)}"
            f" completed={len(existing_rows)}"
            f" requeue_unparsed={len(requeued_unparsed_rows)}"
        )

    progress = pairwise_lib.tqdm(
        pending_video_items,
        total=len(selected_video_items),
        initial=len(existing_rows),
        desc=args.category or args.judge_model,
        unit="video",
    )

    if args.num_workers == 1:
        row_iterator = (
            evaluate_video(
                video_item,
                task_lookup=task_lookup,
                prompt_lookup=prompt_lookup,
                template_chunks=template_chunks,
                args=args,
                judge=judge,
            )
            for video_item in pending_video_items
        )
    else:
        row_iterator = iter_concurrent_video_evaluations(
            pending_video_items,
            task_lookup=task_lookup,
            prompt_lookup=prompt_lookup,
            template_chunks=template_chunks,
            args=args,
            judge=judge,
        )

    for idx, row in enumerate(row_iterator, start=1):
        video_id = row["video_id"]
        scalar_value = select_scalar_score(row, args.score_aggregation)
        parse_attempts_used = row.get("parse_attempts_used")
        pairwise_lib.write_jsonl_row(output_jsonl, row)
        rows_written.append(row)
        update_video_parse_stats(live_stats, row)
        progress.update(1)
        progress.set_postfix(
            parsed=f"{live_stats['parsed']}/{live_stats['total']}",
            scalar=format_scalar(scalar_value),
        )
        if idx == 1 or idx % max(1, args.log_every) == 0 or idx == len(pending_video_items):
            pairwise_lib.progress_write(
                f"[{len(existing_rows) + idx}/{len(selected_video_items)}] {video_id} "
                f"-> scalar={format_scalar(scalar_value)} "
                f"attempts={parse_attempts_used} "
                f"parse_rate={pairwise_lib.format_pct(live_stats['parse_rate'])}"
            )

    progress.close()

    final_video_rows = dedupe_rows_by_video_id(rows_written)
    video_rows_by_id = {row["video_id"]: row for row in final_video_rows if row.get("video_id")}
    induced_pair_rows = build_induced_pair_rows(
        selected_pairs,
        task_lookup=task_lookup,
        prompt_lookup=prompt_lookup,
        video_rows_by_id=video_rows_by_id,
        judge_model=args.judge_model,
        score_aggregation=args.score_aggregation,
        tie_threshold=args.tie_threshold,
    )
    save_induced_pair_rows_jsonl(induced_pair_jsonl, induced_pair_rows)

    summary_path = output_jsonl.with_suffix(".summary.json")
    save_summary(
        summary_path,
        video_rows=final_video_rows,
        pair_rows=induced_pair_rows,
        args=args,
        selected_pairs=selected_pairs,
        selected_video_items=selected_video_items,
        induced_pair_jsonl=induced_pair_jsonl,
    )
    readable_path = output_jsonl.with_suffix(".readable.md")
    print_final_summary(final_video_rows, induced_pair_rows)
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
