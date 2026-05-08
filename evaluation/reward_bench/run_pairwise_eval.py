#!/usr/bin/env python3
"""Run WorldReason-Reward-Bench pairwise evaluation on model pairs JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from tqdm import tqdm  # type: ignore
    def progress_write(message: str) -> None:
        tqdm.write(message)
except ImportError:
    class _SimpleTqdm:
        def __init__(
            self,
            iterable=None,
            *,
            total: int | None = None,
            initial: int = 0,
            desc: str | None = None,
            unit: str = "item",
        ) -> None:
            self.iterable = iterable
            self.total = total
            self.n = initial
            self.desc = desc or "progress"
            self.unit = unit
            self.postfix: Dict[str, Any] = {}
            self._print()

        def __iter__(self):
            if self.iterable is None:
                return iter(())
            return iter(self.iterable)

        def update(self, n: int = 1) -> None:
            self.n += n
            self._print()

        def set_postfix(self, **kwargs) -> None:
            self.postfix = kwargs
            self._print()

        def close(self) -> None:
            self._print(final=True)

        def _print(self, final: bool = False) -> None:
            prefix = "[done]" if final else "[progress]"
            total_str = "?" if self.total is None else str(self.total)
            postfix = ""
            if self.postfix:
                postfix = " | " + " ".join(f"{k}={v}" for k, v in self.postfix.items())
            print(f"{prefix} {self.desc}: {self.n}/{total_str} {self.unit}{postfix}")

    def tqdm(iterable=None, **kwargs):  # type: ignore
        return _SimpleTqdm(iterable, **kwargs)

    def progress_write(message: str) -> None:
        print(message)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # evaluation/reward_bench -> evaluation -> project root
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mllm_tools import MLLM_Models, MLLM_LIST  # noqa: E402
from utils import load_template  # noqa: E402


DATA_FILES = [
    PROJECT_ROOT / "data" / "Human-Centric_with_prompts.json",
    PROJECT_ROOT / "data" / "Information-based-reasoning_with_prompts.json",
    PROJECT_ROOT / "data" / "Logic-Reasoning_with_prompts.json",
    PROJECT_ROOT / "data" / "World-Knowledge_with_prompts.json",
]

VERDICT_PATTERN = re.compile(r"\[\[(A>B|B>A|A=B=Good|A=B=Bad)\]\]")


@dataclass(frozen=True)
class PairwiseTemplateChunks:
    before_image: str
    before_prompt: str
    before_left_video: str
    before_right_video: str
    after_right_video: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-json",
        default=str(PROJECT_ROOT / "data" / "statistics_model_pairs_by_task_stratified_balanced_tie_v2.json"),
        help="Path to statistics_model_pairs_by_task.json.",
    )
    parser.add_argument(
        "--judge-model",
        default="qwen3.5-27b",
        choices=MLLM_LIST,
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
        help="Output JSONL path. Defaults to outputs/worldreason_reward_bench/pairwise_<timestamp>.jsonl",
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
        help="Optional pair_type filter.",
    )
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N filtered pairs.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most N filtered pairs.")
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
        "--resume",
        action="store_true",
        help=(
            "Skip only pair_ids that already have a parsed verdict in the output JSONL. "
            "Rows with parsed_verdict=null are re-evaluated on resume."
        ),
    )
    parser.add_argument(
        "--max-parse-attempts",
        type=int,
        default=1,
        help=(
            "If verdict parsing fails, re-run the current pair up to this many "
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
        help="Print one textual progress update every N newly evaluated pairs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Run up to this many pair evaluations concurrently. Use 1 to disable concurrency.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1.")
    if args.max_parse_attempts < 1:
        parser.error("--max-parse-attempts must be at least 1.")
    return args


def default_output_jsonl() -> Path:
    out_dir = PROJECT_ROOT / "outputs" / "worldreason_reward_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"pairwise_{timestamp}.jsonl"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_lookup() -> Dict[str, Dict[str, Any]]:
    task_lookup: Dict[str, Dict[str, Any]] = {}
    for path in DATA_FILES:
        for item in load_json(path):
            task_lookup[item["id"]] = item
    return task_lookup


def get_prompt_text(task_meta: Dict[str, Any], prompt_field: str) -> str:
    for field in (
        prompt_field,
        "video_prompt",
        "video_prompt_difficult",
        "human_hint_en",
    ):
        value = (task_meta.get(field) or "").strip()
        if value:
            return value
    raise ValueError(f"No usable prompt text found for task {task_meta.get('id')}.")


def build_prompt_lookup(
    task_lookup: Dict[str, Dict[str, Any]],
    pairs: Iterable[Dict[str, Any]],
    prompt_field: str,
) -> Dict[str, str]:
    prompt_lookup: Dict[str, str] = {}
    for pair in pairs:
        task_id = pair["task_id"]
        if task_id in prompt_lookup:
            continue
        task_meta = task_lookup.get(task_id)
        if task_meta is None:
            raise KeyError(f"Task id {task_id} not found in prompt source JSONs.")
        prompt_lookup[task_id] = get_prompt_text(task_meta, prompt_field)
    return prompt_lookup


def normalize_text_label(value: Any) -> str:
    return str(value or "").strip().lower()


def iter_filtered_pairs(
    records: List[Dict[str, Any]],
    args: argparse.Namespace,
    task_lookup: Dict[str, Dict[str, Any]],
) -> Iterable[Dict[str, Any]]:
    filtered = records
    if args.category:
        filtered = [
            row
            for row in filtered
            if normalize_text_label(task_lookup.get(row.get("task_id"), {}).get("category"))
            == normalize_text_label(args.category)
        ]
    if args.sub_category_v2:
        filtered = [
            row
            for row in filtered
            if normalize_text_label(
                task_lookup.get(row.get("task_id"), {}).get("sub_category_v2")
            )
            == normalize_text_label(args.sub_category_v2)
        ]
    if args.task_id:
        filtered = [row for row in filtered if row.get("task_id") == args.task_id]
    if args.pair_type:
        filtered = [row for row in filtered if row.get("pair_type") == args.pair_type]
    if args.offset:
        filtered = filtered[args.offset :]
    if args.limit is not None:
        filtered = filtered[: args.limit]
    return filtered


def load_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_completed_pair_ids(output_jsonl: Path) -> set[str]:
    completed: set[str] = set()
    for row in load_jsonl_rows(output_jsonl):
        pair_id = row.get("pair_id")
        if pair_id:
            completed.add(pair_id)
    return completed


def build_pair_id(pair: Dict[str, Any]) -> str:
    return f"{pair['task_id']}::{pair['model_1']}::{pair['model_2']}"


def get_expected_verdict(pair: Dict[str, Any]) -> str:
    score_1 = float(pair.get("score_1", 0.0))
    score_2 = float(pair.get("score_2", 0.0))
    if pair.get("pair_type") == "tie" or abs(score_1 - score_2) < 1e-9:
        return "A=B"
    return "A>B" if score_1 > score_2 else "B>A"


def is_correct_prediction(parsed_verdict: str | None, expected_verdict: str) -> bool | None:
    if parsed_verdict is None:
        return None
    if expected_verdict == "A=B":
        return parsed_verdict in {"A=B=Good", "A=B=Bad"}
    return parsed_verdict == expected_verdict


def parse_pairwise_template(template: str) -> PairwiseTemplateChunks:
    if (
        "<image>" not in template
        or "<prompt>" not in template
        or "<left_video>" not in template
        or "<right_video>" not in template
    ):
        raise ValueError("Pairwise template is missing required placeholders.")

    before_image, after_image = template.split("<image>", 1)
    before_prompt, after_prompt = after_image.split("<prompt>", 1)
    before_left, after_left = after_prompt.split("<left_video>", 1)
    before_right, after_right = after_left.split("<right_video>", 1)
    return PairwiseTemplateChunks(
        before_image=before_image,
        before_prompt=before_prompt,
        before_left_video=before_left,
        before_right_video=before_right,
        after_right_video=after_right,
    )


def build_pairwise_inputs(
    template_chunks: PairwiseTemplateChunks,
    pair: Dict[str, Any],
    prompt_text: str,
    max_num_frames: int,
) -> List[Dict[str, Any]]:
    interleaved_inputs: List[Dict[str, Any]] = []
    if template_chunks.before_image.strip():
        interleaved_inputs.append({"type": "text", "content": template_chunks.before_image})
    interleaved_inputs.append({"type": "image", "content": pair["input_image_path"]})

    second_text = (
        template_chunks.before_prompt
        + prompt_text
        + template_chunks.before_left_video
    )
    if second_text.strip():
        interleaved_inputs.append({"type": "text", "content": second_text})
    interleaved_inputs.append(
        {
            "type": "video",
            "content": pair["model_1_video_path"],
            "max_num_frames": max_num_frames,
            "label": f"Model A ({pair['model_1']})",
        }
    )

    if template_chunks.before_right_video.strip():
        interleaved_inputs.append(
            {"type": "text", "content": template_chunks.before_right_video}
        )
    interleaved_inputs.append(
        {
            "type": "video",
            "content": pair["model_2_video_path"],
            "max_num_frames": max_num_frames,
            "label": f"Model B ({pair['model_2']})",
        }
    )

    if template_chunks.after_right_video.strip():
        interleaved_inputs.append(
            {"type": "text", "content": template_chunks.after_right_video}
        )

    return interleaved_inputs


def parse_verdict(response_text: str) -> str | None:
    match = VERDICT_PATTERN.search(response_text)
    return match.group(1) if match else None


def clean_response_text(response_text: str) -> str:
    return VERDICT_PATTERN.sub("", response_text or "").strip()


def normalize_pairwise_verdict(value: Any) -> str | None:
    """Normalize model-output verdict variants to canonical form."""
    text = str(value or "").strip()
    if text in {"A>B", "B>A", "A=B=Good", "A=B=Bad"}:
        return text
    compressed = text.replace(" ", "")
    for variant, canonical in (
        ("A>>>B", "A>B"), ("A>>B", "A>B"),
        ("B>>>A", "B>A"), ("B>>A", "B>A"),
        ("A<B", "B>A"), ("B<A", "A>B"),
        ("A==B", "A=B=Good"), ("A=B", "A=B=Good"),
    ):
        if compressed.upper() == variant:
            return canonical
    upper = text.upper()
    if "TIE" in upper or "SAME" in upper:
        return "A=B=Good"
    return None


def first_paragraph(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    return cleaned.split("\n\n", 1)[0].strip()


def verdict_to_winner(verdict: str | None, pair: Dict[str, Any]) -> str | None:
    if verdict == "A>B":
        return pair["model_1"]
    if verdict == "B>A":
        return pair["model_2"]
    return None


def build_result_row(
    pair: Dict[str, Any],
    task_meta: Dict[str, Any],
    prompt_text: str,
    raw_response: str,
    parsed_verdict: str | None,
    judge_model: str,
    max_num_frames: int,
    parse_attempts_used: int,
    max_parse_attempts: int,
) -> Dict[str, Any]:
    expected_verdict = get_expected_verdict(pair)
    is_correct = is_correct_prediction(parsed_verdict, expected_verdict)
    judge_reasoning = clean_response_text(raw_response)
    return {
        "pair_id": build_pair_id(pair),
        "judge_model": judge_model,
        "task_id": pair["task_id"],
        "category": task_meta.get("category"),
        "sub_category": task_meta.get("sub_category"),
        "sub_category_v2": task_meta.get("sub_category_v2"),
        "prompt_text": prompt_text,
        "pair_type": pair.get("pair_type"),
        "model_1": pair.get("model_1"),
        "score_1": pair.get("score_1"),
        "model_2": pair.get("model_2"),
        "score_2": pair.get("score_2"),
        "score_gap": abs(float(pair.get("score_1", 0.0)) - float(pair.get("score_2", 0.0))),
        "input_image_path": pair.get("input_image_path"),
        "model_1_video_path": pair.get("model_1_video_path"),
        "model_2_video_path": pair.get("model_2_video_path"),
        "max_num_frames": max_num_frames,
        "parse_attempts_used": parse_attempts_used,
        "max_parse_attempts": max_parse_attempts,
        "raw_response": raw_response,
        "judge_reasoning": judge_reasoning,
        "judge_summary": first_paragraph(judge_reasoning),
        "parsed_verdict": parsed_verdict,
        "expected_verdict": expected_verdict,
        "is_correct": is_correct,
        "winner_model": verdict_to_winner(parsed_verdict, pair),
    }


REFUSED_VERDICT_SENTINEL = "REFUSED"


def extract_http_error_info(exc: BaseException) -> tuple[int | None, str | None]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        status_code = getattr(response, "status_code", None) if response is not None else None
        if isinstance(status_code, int) and 400 <= status_code < 500:
            body_text: str | None = None
            try:
                body_text = response.text  # type: ignore[union-attr]
            except Exception:
                body_text = None
            return status_code, body_text
        current = current.__cause__ or current.__context__
    return None, None


def build_refusal_row_from_exception(
    *,
    pair_id: str,
    pairs: List[Dict[str, Any]],
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    judge_model: str,
    max_num_frames: int,
    max_parse_attempts: int,
    exc: BaseException,
) -> Dict[str, Any] | None:
    status_code, body_text = extract_http_error_info(exc)
    if status_code is None:
        return None
    pair = next((p for p in pairs if build_pair_id(p) == pair_id), None)
    if pair is None:
        return None
    task_meta = task_lookup.get(pair["task_id"]) or {}
    prompt_text = prompt_lookup.get(pair["task_id"], "")
    row = build_result_row(
        pair=pair,
        task_meta=task_meta,
        prompt_text=prompt_text,
        raw_response="",
        parsed_verdict=REFUSED_VERDICT_SENTINEL,
        judge_model=judge_model,
        max_num_frames=max_num_frames,
        parse_attempts_used=0,
        max_parse_attempts=max_parse_attempts,
    )
    error_message = (body_text or str(exc) or "").strip()
    error_kind = "http_4xx"
    if "ContentPolicyViolationError" in error_message or "content safety" in error_message.lower():
        error_kind = "content_policy"
    row["error_kind"] = error_kind
    row["error_status"] = status_code
    row["error_message"] = error_message[:2000]
    row["is_correct"] = None
    row["winner_model"] = None
    return row


def evaluate_pair(
    pair: Dict[str, Any],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    template_chunks: PairwiseTemplateChunks,
    args: argparse.Namespace,
    judge: Any,
) -> Dict[str, Any]:
    task_meta = task_lookup.get(pair["task_id"])
    if task_meta is None:
        raise KeyError(f"Task id {pair['task_id']} not found in prompt source JSONs.")

    prompt_text = prompt_lookup[pair["task_id"]]
    inputs = build_pairwise_inputs(
        template_chunks,
        pair,
        prompt_text,
        max_num_frames=args.max_num_frames,
    )

    if args.dry_run:
        raw_response = ""
        parsed_verdict = None
        parse_attempts_used = 0
    else:
        raw_response = ""
        parsed_verdict = None
        parse_attempts_used = 0
        for attempt_idx in range(1, args.max_parse_attempts + 1):
            parse_attempts_used = attempt_idx
            raw_response = judge(inputs)
            parsed_verdict = parse_verdict(raw_response)
            if parsed_verdict is not None:
                break

    return build_result_row(
        pair=pair,
        task_meta=task_meta,
        prompt_text=prompt_text,
        raw_response=raw_response,
        parsed_verdict=parsed_verdict,
        judge_model=args.judge_model,
        max_num_frames=args.max_num_frames,
        parse_attempts_used=parse_attempts_used,
        max_parse_attempts=args.max_parse_attempts,
    )


def iter_concurrent_evaluations(
    pairs: List[Dict[str, Any]],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    prompt_lookup: Dict[str, str],
    template_chunks: PairwiseTemplateChunks,
    args: argparse.Namespace,
    judge: Any,
) -> Iterable[Dict[str, Any]]:
    executor = ThreadPoolExecutor(
        max_workers=args.num_workers,
        thread_name_prefix="pairwise-eval",
    )
    in_flight: Dict[Future, str] = {}
    pair_iter = iter(pairs)

    def submit_next() -> bool:
        try:
            pair = next(pair_iter)
        except StopIteration:
            return False
        future = executor.submit(
            evaluate_pair,
            pair,
            task_lookup=task_lookup,
            prompt_lookup=prompt_lookup,
            template_chunks=template_chunks,
            args=args,
            judge=judge,
        )
        in_flight[future] = build_pair_id(pair)
        return True

    try:
        while len(in_flight) < args.num_workers and submit_next():
            pass

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                pair_id = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    refusal_row = build_refusal_row_from_exception(
                        pair_id=pair_id,
                        pairs=pairs,
                        task_lookup=task_lookup,
                        prompt_lookup=prompt_lookup,
                        judge_model=args.judge_model,
                        max_num_frames=args.max_num_frames,
                        max_parse_attempts=args.max_parse_attempts,
                        exc=exc,
                    )
                    if refusal_row is None:
                        raise RuntimeError(f"Pair evaluation failed for {pair_id}") from exc
                    progress_write(
                        f"[refused] {pair_id} -> {refusal_row.get('error_kind')} "
                        f"status={refusal_row.get('error_status')} "
                        f"msg={(refusal_row.get('error_message') or '')[:120]}"
                    )
                    row = refusal_row
                yield row
                submit_next()
    except Exception:
        for future in in_flight:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def instantiate_judge(args: argparse.Namespace):
    judge_cls = MLLM_Models(args.judge_model)
    if args.judge_model in {"qwen3.5", "qwen3.5-27b"}:
        return judge_cls(
            model_path=args.judge_model_path,
            base_url=args.judge_base_url,
            api_key=args.judge_api_key,
        )
    return judge_cls()


def write_jsonl_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dedupe_rows_by_pair_id(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        pair_id = row.get("pair_id")
        if pair_id:
            indexed[pair_id] = row
    return list(indexed.values())


def build_accuracy_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    scoreable_rows = [row for row in rows if row.get("parsed_verdict") != REFUSED_VERDICT_SENTINEL]
    refused = len(rows) - len(scoreable_rows)
    total = len(scoreable_rows)
    parsed_rows = [row for row in scoreable_rows if row.get("parsed_verdict")]
    parsed = len(parsed_rows)
    correct = sum(1 for row in parsed_rows if row.get("is_correct") is True)
    parse_rate = (parsed / total) if total else None
    accuracy = (correct / parsed) if parsed else None
    e2e_accuracy = (correct / total) if total else None
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "refused": refused,
        "parse_rate": parse_rate,
        "accuracy": accuracy,
        "e2e_accuracy": e2e_accuracy,
    }


def update_accuracy_stats(stats: Dict[str, Any], row: Dict[str, Any]) -> None:
    if row.get("parsed_verdict") == REFUSED_VERDICT_SENTINEL:
        stats["refused"] = int(stats.get("refused") or 0) + 1
        return
    total = int(stats.get("total") or 0) + 1
    parsed = int(stats.get("parsed") or 0)
    correct = int(stats.get("correct") or 0)
    if row.get("parsed_verdict"):
        parsed += 1
        if row.get("is_correct") is True:
            correct += 1
    stats.update(
        {
            "total": total,
            "parsed": parsed,
            "correct": correct,
            "parse_rate": (parsed / total) if total else None,
            "accuracy": (correct / parsed) if parsed else None,
            "e2e_accuracy": (correct / total) if total else None,
        }
    )


def summarize_by_field(rows: List[Dict[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(field) or "UNKNOWN")
        grouped.setdefault(key, []).append(row)
    return {
        key: build_accuracy_stats(group_rows)
        for key, group_rows in sorted(grouped.items(), key=lambda item: item[0])
    }


def format_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def compact_path(path_value: Any) -> str:
    path_str = str(path_value or "")
    return path_str


def save_readable_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    rows = dedupe_rows_by_pair_id(rows)
    overall = build_accuracy_stats(rows)
    subcat_stats = summarize_by_field(rows, "sub_category_v2")

    lines: List[str] = [
        "# Pairwise Evaluation Report",
        "",
        "## Overall",
        "",
        f"- Total pairs: `{overall['total']}`",
        f"- Parsed pairs: `{overall['parsed']}`",
        f"- Correct pairs: `{overall['correct']}`",
        f"- Parse rate: `{format_pct(overall['parse_rate'])}`",
        f"- Accuracy: `{format_pct(overall['accuracy'])}`",
        f"- End-to-end accuracy: `{format_pct(overall['e2e_accuracy'])}`",
        "",
        "## Accuracy by Sub Category V2",
        "",
        "| Sub Category V2 | Total | Parsed | Correct | Accuracy | E2E Accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name, stats in subcat_stats.items():
        lines.append(
            f"| {name} | {stats['total']} | {stats['parsed']} | {stats['correct']} | "
            f"{format_pct(stats['accuracy'])} | {format_pct(stats['e2e_accuracy'])} |"
        )

    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                "",
                f"## {idx}. `{row.get('pair_id', 'UNKNOWN')}`",
                "",
                f"- Category: `{row.get('category', 'UNKNOWN')}`",
                f"- Sub Category: `{row.get('sub_category', 'UNKNOWN')}`",
                f"- Sub Category V2: `{row.get('sub_category_v2', 'UNKNOWN')}`",
                f"- Prompt: {row.get('prompt_text', '')}",
                (
                    f"- Pair: `A={row.get('model_1')}` (`{row.get('score_1')}`) vs "
                    f"`B={row.get('model_2')}` (`{row.get('score_2')}`), "
                    f"gap=`{row.get('score_gap')}`"
                ),
                f"- Expected verdict: `{row.get('expected_verdict', 'UNKNOWN')}`",
                f"- Judge verdict: `{row.get('parsed_verdict', 'UNPARSED')}`",
                f"- Correct: `{row.get('is_correct')}`",
                f"- Input image: `{compact_path(row.get('input_image_path'))}`",
                f"- Model A video: `{compact_path(row.get('model_1_video_path'))}`",
                f"- Model B video: `{compact_path(row.get('model_2_video_path'))}`",
                "",
                "### Judge Reasoning",
                "",
                row.get("judge_reasoning")
                or clean_response_text(row.get("raw_response") or ""),
            ]
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def print_accuracy_summary(rows: List[Dict[str, Any]]) -> None:
    overall = build_accuracy_stats(rows)
    print(
        "overall:"
        f" total={overall['total']}"
        f" parsed={overall['parsed']}"
        f" correct={overall['correct']}"
        f" parse_rate={format_pct(overall['parse_rate'])}"
        f" accuracy={format_pct(overall['accuracy'])}"
        f" e2e_accuracy={format_pct(overall['e2e_accuracy'])}"
    )

    print("accuracy_by_sub_category_v2:")
    for name, stats in summarize_by_field(rows, "sub_category_v2").items():
        print(
            f"  {name}: total={stats['total']} parsed={stats['parsed']} "
            f"correct={stats['correct']} accuracy={format_pct(stats['accuracy'])} "
            f"e2e_accuracy={format_pct(stats['e2e_accuracy'])}"
        )


def save_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    rows = dedupe_rows_by_pair_id(rows)
    verdict_counts = Counter(row.get("parsed_verdict") or "UNPARSED" for row in rows)
    expected_counts = Counter(row.get("expected_verdict") or "UNKNOWN" for row in rows)
    winner_counts = Counter(row.get("winner_model") or "NO_WINNER" for row in rows)
    summary = {
        "num_rows": len(rows),
        "overall": build_accuracy_stats(rows),
        "verdict_counts": dict(verdict_counts),
        "expected_verdict_counts": dict(expected_counts),
        "winner_counts": dict(winner_counts),
        "accuracy_by_category": summarize_by_field(rows, "category"),
        "accuracy_by_sub_category": summarize_by_field(rows, "sub_category"),
        "accuracy_by_sub_category_v2": summarize_by_field(rows, "sub_category_v2"),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


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

    pair_records = load_json(Path(args.pairs_json))
    task_lookup = load_task_lookup()
    template_chunks = parse_pairwise_template(load_template("video_generation", "pairwise"))
    selected_pairs = list(iter_filtered_pairs(pair_records, args, task_lookup))
    prompt_lookup = build_prompt_lookup(task_lookup, selected_pairs, args.prompt_field)
    selected_pair_ids = {build_pair_id(pair) for pair in selected_pairs}
    resume_rows = [
        row for row in dedupe_rows_by_pair_id(load_jsonl_rows(output_jsonl))
        if row.get("pair_id") in selected_pair_ids
    ] if args.resume else []
    existing_rows = [
        row for row in resume_rows
        if row.get("pair_id") and row.get("parsed_verdict")
    ]
    requeued_unparsed_rows = [
        row for row in resume_rows
        if row.get("pair_id") and not row.get("parsed_verdict")
    ]
    completed_pair_ids = {row["pair_id"] for row in existing_rows if row.get("pair_id")}
    judge = None if args.dry_run else instantiate_judge(args)

    rows_written: List[Dict[str, Any]] = list(existing_rows)
    live_stats = build_accuracy_stats(existing_rows)
    pending_pairs = [pair for pair in selected_pairs if build_pair_id(pair) not in completed_pair_ids]
    print(f"selected_pairs={len(selected_pairs)}")
    print(f"pending_pairs={len(pending_pairs)}")
    print(f"output_jsonl={output_jsonl}")
    print(f"num_workers={args.num_workers}")
    print(f"max_parse_attempts={args.max_parse_attempts}")
    if args.resume and existing_rows:
        print(
            "resume_status:"
            f" loaded={len(resume_rows)}"
            f" completed={len(existing_rows)}"
            f" requeue_unparsed={len(requeued_unparsed_rows)}"
            f" accuracy={format_pct(live_stats['accuracy'])}"
        )
    elif args.resume:
        print(
            "resume_status:"
            f" loaded={len(resume_rows)}"
            f" completed={len(existing_rows)}"
            f" requeue_unparsed={len(requeued_unparsed_rows)}"
        )

    progress = tqdm(
        pending_pairs,
        total=len(selected_pairs),
        initial=len(existing_rows),
        desc=args.category or args.judge_model,
        unit="pair",
    )

    if args.num_workers == 1:
        row_iterator = (
            evaluate_pair(
                pair,
                task_lookup=task_lookup,
                prompt_lookup=prompt_lookup,
                template_chunks=template_chunks,
                args=args,
                judge=judge,
            )
            for pair in pending_pairs
        )
    else:
        row_iterator = iter_concurrent_evaluations(
            pending_pairs,
            task_lookup=task_lookup,
            prompt_lookup=prompt_lookup,
            template_chunks=template_chunks,
            args=args,
            judge=judge,
        )

    for idx, row in enumerate(row_iterator, start=1):
        pair_id = row["pair_id"]
        parsed_verdict = row.get("parsed_verdict")
        parse_attempts_used = row.get("parse_attempts_used")
        write_jsonl_row(output_jsonl, row)
        rows_written.append(row)
        update_accuracy_stats(live_stats, row)
        progress.update(1)
        progress.set_postfix(
            parsed=f"{live_stats['parsed']}/{live_stats['total']}",
            acc=format_pct(live_stats["accuracy"]),
            e2e=format_pct(live_stats["e2e_accuracy"]),
        )
        if idx == 1 or idx % max(1, args.log_every) == 0 or idx == len(pending_pairs):
            progress_write(
                f"[{len(existing_rows) + idx}/{len(selected_pairs)}] {pair_id} "
                f"-> {parsed_verdict or 'UNPARSED'} | "
                f"attempts={parse_attempts_used} | "
                f"acc={format_pct(live_stats['accuracy'])} "
                f"(correct={live_stats['correct']}, parsed={live_stats['parsed']})"
            )

    progress.close()

    summary_path = output_jsonl.with_suffix(".summary.json")
    final_rows = dedupe_rows_by_pair_id(rows_written)
    save_summary(summary_path, final_rows)
    readable_path = output_jsonl.with_suffix(".readable.md")
    save_readable_markdown(readable_path, final_rows)
    print_accuracy_summary(final_rows)
    print(f"summary_json={summary_path}")
    print(f"readable_markdown={readable_path}")


if __name__ == "__main__":
    main()
