#!/usr/bin/env python3
"""Run S(v) pointwise evaluation on all videos for the main table.

Unlike run_pointwise_eval.py (which works from reward-bench pairs), this script
iterates over model video directories directly and scores every video with the
viescore.txt template to produce S(v) = 0.4*s_r + 0.3*s_c + 0.3*s_a.

Video Directory Structure:
    You must pass --video-base-dir pointing to the root of your video dataset.
    The script expects videos organized as:
      <video-base-dir>/<model_name>/<category>/*.mp4           (for closed-source models)
      <video-base-dir>/<model_name>/video_prompt_difficult/<category>/*.mp4  (for open-source models)

    Configure MODEL_VIDEO_DIRS below to match your directory layout.

Usage:
    python3 run_pointwise_eval_main_table.py \\
        --video-base-dir /path/to/video/datasets \\
        --models kling seedance2.0 \\
        --judge-base-url http://127.0.0.1:30002/v1 \\
        --output-dir outputs/sv_pointwise_main/ \\
        --video-fps 4 \\
        --num-workers 2 \\
        --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # evaluation/reward_bench -> evaluation -> project root
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mllm_tools import MLLM_Models  # noqa: E402
from utils import load_template  # noqa: E402
import run_pointwise_eval as pw_lib  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Video dataset paths ─────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
CATEGORIES = ["World-Knowledge", "Human-Centric", "Logic-Reasoning", "Information-based-reasoning"]

# Model name -> video directory pattern
# {base} = --video-base-dir, {cat} = category name
MODEL_VIDEO_DIRS = {
    # Closed-source: direct category dirs
    "kling": "{base}/kling/{cat}",
    "seedance2.0": "{base}/seedance2.0/{cat}",
    "sora2_8s": "{base}/sora2_8s/{cat}",
    "sora2_12s": "{base}/sora2_12s/{cat}",
    "veo3.1-fast": "{base}/veo3.1-fast/{cat}",
    "wan2.6": "{base}/wan2.6/{cat}",
    # Open-source: video_prompt_difficult subdir
    "ltx2": "{base}/ltx2/video_prompt_difficult/{cat}",
    "ltx2_3": "{base}/ltx2_3/video_prompt_difficult/{cat}",
    "wan2_2_i2v_a14b": "{base}/wan2_2_i2v_a14b/video_prompt_difficult/{cat}",
    "univideo": "{base}/univideo/video_prompt_difficult/{cat}",
    "hunyuan_video1_5_480p_i2v": "{base}/hunyuan_video1_5_480p_i2v/video_prompt_difficult/{cat}",
    "longcat_video": "{base}/longcat_video/i2v_refine/video_prompt_difficult/{cat}",
    "Cosmos-Predict2.5-14B_post-trained": "{base}/Cosmos-Predict2.5-14B_post-trained/video_prompt_difficult/{cat}",
}


def load_task_lookup() -> Dict[str, Dict[str, Any]]:
    """Load task metadata from *_with_prompts.json files."""
    lookup = {}
    for cat in CATEGORIES:
        path = DATA_DIR / f"{cat}_with_prompts.json"
        if not path.exists():
            continue
        for item in json.loads(path.read_text()):
            tid = item["id"]
            item["category"] = cat
            # Resolve image path
            img = item.get("image", "")
            if img and not Path(img).is_absolute():
                item["input_image_path"] = str(DATA_DIR / cat / f"{tid}.png")
            else:
                item["input_image_path"] = img
            lookup[tid] = item
    return lookup


def discover_videos(model: str, video_base: Path) -> List[Dict[str, Any]]:
    """Find all videos for a model and return eval items."""
    items = []
    pattern = MODEL_VIDEO_DIRS.get(model)
    if not pattern:
        print(f"[WARN] Unknown model: {model}")
        return items

    for cat in CATEGORIES:
        video_dir = Path(pattern.format(base=video_base, cat=cat))
        if not video_dir.exists():
            continue
        for mp4 in sorted(video_dir.glob("*.mp4")):
            task_id = mp4.stem  # e.g., "1-a-10"
            items.append({
                "video_id": f"{task_id}::{model}",
                "task_id": task_id,
                "model_name": model,
                "category": cat,
                "video_path": str(mp4),
            })
    return items


def evaluate_single_video(
    video_item: Dict[str, Any],
    *,
    task_lookup: Dict[str, Dict[str, Any]],
    template_chunks: pw_lib.PointwiseTemplateChunks,
    judge: Any,
    max_num_frames: int,
    prompt_field: str,
    max_parse_attempts: int = 3,
) -> Dict[str, Any]:
    """Score one video with the viescore template."""
    task_meta = task_lookup.get(video_item["task_id"])
    if task_meta is None:
        return {
            "video_id": video_item["video_id"],
            "task_id": video_item["task_id"],
            "model_name": video_item["model_name"],
            "category": video_item["category"],
            "parsed_scores": None,
            "is_parsed": False,
            "error": "task_id not found in lookup",
        }

    prompt_text = task_meta.get(prompt_field, task_meta.get("video_prompt", ""))
    video_item_full = {
        **video_item,
        "input_image_path": task_meta.get("input_image_path", ""),
        "sub_category_v2": task_meta.get("sub_category_v2", ""),
    }

    inputs = pw_lib.build_pointwise_inputs(
        template_chunks, video_item_full, prompt_text, max_num_frames=max_num_frames
    )

    raw_response = ""
    parsed_response = None
    parse_attempts_used = 0

    for attempt in range(1, max_parse_attempts + 1):
        parse_attempts_used = attempt
        try:
            raw_response = judge(inputs)
            parsed_response = pw_lib.parse_pointwise_response(raw_response)
            if parsed_response is not None:
                break
        except Exception as exc:
            raw_response = f"ERROR: {exc!r}"
            parsed_response = None

    parsed_scores = parsed_response.get("scores") if parsed_response else None
    score_weighted = None
    if parsed_scores and len(parsed_scores) == 3:
        score_weighted = 0.4 * parsed_scores[0] + 0.3 * parsed_scores[1] + 0.3 * parsed_scores[2]

    return {
        "video_id": video_item["video_id"],
        "task_id": video_item["task_id"],
        "model_name": video_item["model_name"],
        "category": video_item.get("category") or task_meta.get("category"),
        "sub_category_v2": task_meta.get("sub_category_v2"),
        "prompt_text": prompt_text,
        "video_path": video_item["video_path"],
        "input_image_path": task_meta.get("input_image_path"),
        "parsed_scores": parsed_scores,
        "score_weighted": score_weighted,
        "parse_attempts_used": parse_attempts_used,
        "raw_response": raw_response,
        "is_parsed": parsed_scores is not None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-base-dir", required=True,
                        help="Root directory containing model video subdirectories")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate")
    parser.add_argument("--judge-model", default="qwen3.5-27b")
    parser.add_argument("--judge-model-path", default="Qwen3.5-27B")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:30002/v1")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--output-dir", default="outputs/sv_pointwise_main")
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--max-num-frames", type=int, default=8)
    parser.add_argument("--frame-max-side", type=int, default=768)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-parse-attempts", type=int, default=3)
    parser.add_argument("--prompt-field", default="video_prompt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--template", default="viescore")
    return parser.parse_args()


def main():
    args = parse_args()
    video_base = Path(args.video_base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load task metadata
    task_lookup = load_task_lookup()
    print(f"Loaded {len(task_lookup)} tasks from prompt JSONs")

    # Load template
    template_chunks = pw_lib.parse_pointwise_template(load_template("video_generation", args.template))

    # Discover all videos
    all_videos = []
    for model in args.models:
        videos = discover_videos(model, video_base)
        all_videos.extend(videos)
        print(f"  {model}: {len(videos)} videos")
    print(f"Total videos to evaluate: {len(all_videos)}")

    # Output file
    output_jsonl = output_dir / "sv_scores.jsonl"

    # Resume: load existing results
    completed_ids = set()
    if args.resume and output_jsonl.exists():
        for line in output_jsonl.open():
            row = json.loads(line)
            if row.get("is_parsed"):
                completed_ids.add(row["video_id"])
        print(f"Resuming: {len(completed_ids)} already completed")

    pending = [v for v in all_videos if v["video_id"] not in completed_ids]
    print(f"Pending: {len(pending)} videos")

    if not pending:
        print("Nothing to do!")
        return

    # Instantiate judge
    if args.video_fps:
        os.environ["QWEN3_5_VIDEO_FPS"] = str(args.video_fps)

    judge_cls = MLLM_Models(args.judge_model)
    judge = judge_cls(
        model_path=args.judge_model_path,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
    )
    print(f"Judge: {args.judge_model} at {args.judge_base_url}")

    # Run evaluation
    parsed_count = 0
    total_count = 0

    progress = tqdm(total=len(all_videos), initial=len(completed_ids), desc="S(v) eval") if tqdm else None

    def process_result(row):
        nonlocal parsed_count, total_count
        total_count += 1
        if row.get("is_parsed"):
            parsed_count += 1
        # Write to JSONL
        with output_jsonl.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if progress:
            progress.update(1)
            progress.set_postfix(
                parsed=f"{parsed_count}/{total_count}",
                sv=f"{row.get('score_weighted', 'n/a')}"
            )

    if args.num_workers <= 1:
        for video_item in pending:
            row = evaluate_single_video(
                video_item,
                task_lookup=task_lookup,
                template_chunks=template_chunks,
                judge=judge,
                max_num_frames=args.max_num_frames,
                prompt_field=args.prompt_field,
                max_parse_attempts=args.max_parse_attempts,
            )
            process_result(row)
    else:
        executor = ThreadPoolExecutor(max_workers=args.num_workers)
        in_flight: Dict[Future, str] = {}
        video_iter = iter(pending)

        def submit_next() -> bool:
            try:
                video_item = next(video_iter)
            except StopIteration:
                return False
            future = executor.submit(
                evaluate_single_video,
                video_item,
                task_lookup=task_lookup,
                template_chunks=template_chunks,
                judge=judge,
                max_num_frames=args.max_num_frames,
                prompt_field=args.prompt_field,
                max_parse_attempts=args.max_parse_attempts,
            )
            in_flight[future] = video_item["video_id"]
            return True

        # Fill initial workers
        while len(in_flight) < args.num_workers and submit_next():
            pass

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                vid = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"video_id": vid, "is_parsed": False, "error": str(exc)}
                process_result(row)
                submit_next()

        executor.shutdown(wait=False)

    if progress:
        progress.close()

    print(f"\nDone! {parsed_count}/{total_count} parsed ({parsed_count/total_count*100:.1f}%)")
    print(f"Output: {output_jsonl}")

    # Generate per-model summary
    print("\n=== Per-model S(v) summary ===")
    rows = [json.loads(l) for l in output_jsonl.open()]
    from collections import defaultdict
    model_scores = defaultdict(list)
    for r in rows:
        if r.get("score_weighted") is not None:
            model_scores[r["model_name"]].append(r["score_weighted"])

    for model in args.models:
        scores = model_scores.get(model, [])
        if scores:
            print(f"  {model:<30} S(v)={sum(scores)/len(scores):.3f}  ({len(scores)} videos)")
        else:
            print(f"  {model:<30} no scores")


if __name__ == "__main__":
    main()
