#!/usr/bin/env python3

import argparse
import base64
import csv
import json
import math
import mimetypes
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from openai import OpenAI
else:
    OpenAI = Any  # type: ignore[misc,assignment]

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"]
VALID_QTYPES = {"factual", "detail", "temporal", "reasoning"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:30000/v1")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "default")


# -----------------------------
# Basic IO / utils
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def round4(x: float) -> float:
    return round(float(x), 4)


def safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


# ---- Process-aware metric constants ----
_PHASE_ORDER = ["factual", "temporal", "detail", "reasoning"]
_DIFFICULTY_WEIGHTS = {
    "easy":   {"correct": 0.8, "incorrect": 1.5},
    "medium": {"correct": 1.0, "incorrect": 1.0},
    "hard":   {"correct": 1.5, "incorrect": 0.6},
}
_CASCADE_DECAY = 0.8
_POC_THRESHOLD = 0.3


def _compute_process_aware_metrics(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute process-aware metrics for a single video's question results.

    Returns dict with keys: weighted_score, state_score, process_satisfaction,
    fidelity_score, mechanism_score, poc_gap, poc_flag, chain_score,
    reasoning_coverage, reasoning_radar_area, diagnosis.
    """
    if not questions:
        return {
            "weighted_score": 0.0, "state_score": None, "process_satisfaction": None,
            "fidelity_score": None, "mechanism_score": None, "poc_gap": None,
            "poc_flag": "insufficient_data", "chain_score": 0.0,
            "reasoning_coverage": 0.0, "reasoning_radar_area": 0.0, "diagnosis": "no_data",
        }

    by_type: Dict[str, List[float]] = defaultdict(list)
    for q in questions:
        score = float(q.get("score", 0))
        qtype = str(q.get("question_type", "factual")).strip().lower()
        if qtype not in _PHASE_ORDER:
            qtype = "factual"
        by_type[qtype].append(score)

    # Per-type scores
    state_score = safe_mean_or_none(by_type.get("factual", []))
    process_satisfaction = safe_mean_or_none(by_type.get("temporal", []))
    fidelity_score = safe_mean_or_none(by_type.get("detail", []))
    mechanism_score = safe_mean_or_none(by_type.get("reasoning", []))

    # Difficulty-weighted score (asymmetric)
    w_num = 0.0
    w_den = 0.0
    for q in questions:
        s = float(q.get("score", 0))
        d = str(q.get("difficulty", "medium")).strip().lower()
        wm = _DIFFICULTY_WEIGHTS.get(d, _DIFFICULTY_WEIGHTS["medium"])
        w = wm["correct"] if s >= 0.5 else wm["incorrect"]
        w_num += w * s
        w_den += w
    weighted_score = w_num / w_den if w_den > 0 else 0.0

    # POC gap & flag
    outcome_parts = [v for v in [state_score, fidelity_score] if v is not None]
    process_parts = [v for v in [process_satisfaction, mechanism_score] if v is not None]
    if outcome_parts and process_parts:
        poc_gap = safe_mean(outcome_parts) - safe_mean(process_parts)
        if poc_gap > _POC_THRESHOLD:
            poc_flag = "outcome_hacking"
        elif poc_gap < -_POC_THRESHOLD:
            poc_flag = "incomplete_transition"
        else:
            poc_flag = "consistent"
    else:
        poc_gap = None
        poc_flag = "insufficient_data"

    # Cascading penalty
    phase_rank = {p: i for i, p in enumerate(_PHASE_ORDER)}
    sorted_qs = sorted(questions, key=lambda q: phase_rank.get(
        str(q.get("question_type", "factual")).strip().lower(), 0))
    cascade_factor = 1.0
    cascade_scores: List[float] = []
    for q in sorted_qs:
        s = float(q.get("score", 0))
        qt = str(q.get("question_type", "factual")).strip().lower()
        cascade_scores.append(s * cascade_factor)
        if qt in ("temporal", "reasoning") and s < 0.5:
            cascade_factor *= _CASCADE_DECAY
    chain_score = safe_mean(cascade_scores)

    # Reasoning coverage
    types_present = set(by_type.keys()) & set(_PHASE_ORDER)
    reasoning_coverage = len(types_present) / len(_PHASE_ORDER)

    # Reasoning radar area (geometric mean)
    type_vals = [safe_mean(by_type.get(t, [])) if by_type.get(t) else 0.0 for t in _PHASE_ORDER]
    if any(v > 0 for v in type_vals):
        product = 1.0
        for v in type_vals:
            product *= max(v, 1e-9)
        reasoning_radar_area = product ** (1.0 / len(type_vals))
    else:
        reasoning_radar_area = 0.0

    # Bottleneck composite score (weighted geometric mean)
    _bn_weights = [
        (state_score, 0.2),
        (process_satisfaction, 0.3),
        (fidelity_score, 0.2),
        (mechanism_score, 0.3),
    ]
    _bn_parts = [(v, w) for v, w in _bn_weights if v is not None]
    if _bn_parts and sum(w for _, w in _bn_parts) > 0:
        _bn_w_total = sum(w for _, w in _bn_parts)
        _bn_log_sum = sum(w * math.log(max(v, 1e-9)) for v, w in _bn_parts)
        bottleneck_score = math.exp(_bn_log_sum / _bn_w_total)
    else:
        bottleneck_score = 0.0

    # Diagnosis
    s = state_score if state_score is not None else 0.0
    p = process_satisfaction if process_satisfaction is not None else 0.0
    m = mechanism_score if mechanism_score is not None else 0.0
    if s >= 0.6 and p >= 0.6 and m >= 0.6:
        diagnosis = "genuine_understanding"
    elif s >= 0.6 and (p < 0.4 or m < 0.4):
        diagnosis = "static_match_dynamic_failure"
    elif s >= 0.6 and p >= 0.6 and m < 0.4:
        diagnosis = "surface_process_no_mechanism"
    elif s < 0.4 and p < 0.4 and m < 0.4:
        diagnosis = "visual_hallucination"
    else:
        diagnosis = "partial_understanding"

    return {
        "weighted_score": round4(weighted_score),
        "state_score": round4(state_score) if state_score is not None else None,
        "process_satisfaction": round4(process_satisfaction) if process_satisfaction is not None else None,
        "fidelity_score": round4(fidelity_score) if fidelity_score is not None else None,
        "mechanism_score": round4(mechanism_score) if mechanism_score is not None else None,
        "poc_gap": round4(poc_gap) if poc_gap is not None else None,
        "poc_flag": poc_flag,
        "chain_score": round4(chain_score),
        "bottleneck_score": round4(bottleneck_score),
        "reasoning_coverage": round4(reasoning_coverage),
        "reasoning_radar_area": round4(reasoning_radar_area),
        "diagnosis": diagnosis,
    }


def normalize_question_type(question_type: str) -> str:
    q = str(question_type or "").strip().lower()
    return q if q in VALID_QTYPES else "factual"


def normalize_difficulty(difficulty: str) -> str:
    d = str(difficulty or "").strip().lower()
    return d if d in VALID_DIFFICULTIES else "unknown"


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def remove_think_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^\s*Thinking Process:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(p for p in parts if p).strip()
    return str(content)


def clean_final_answer_text(text: str) -> str:
    text = normalize_message_content(text)
    text = remove_think_blocks(text)
    text = strip_code_fences(text)
    text = re.sub(r"^\s*(Final Answer|Answer|JSON)\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = clean_final_answer_text(text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start_positions = [m.start() for m in re.finditer(r"\{", text)]
    for start in start_positions:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                return obj
                        except Exception:
                            break
    return None


def extract_final_json_block(text: str, fallback_to_any_json: bool = False) -> Optional[Dict[str, Any]]:
    raw = normalize_message_content(text)
    marker = "Final JSON:"
    idx = raw.rfind(marker)
    if idx != -1:
        candidate = raw[idx + len(marker):].strip()
        parsed = extract_first_json_object(candidate)
        if parsed is not None:
            return parsed
    if fallback_to_any_json:
        return extract_first_json_object(raw)
    return None


def normalize_items(raw: Any, qa_mode: str = "qa_pairs") -> List[Dict[str, Any]]:
    """Load items and remap QA field based on qa_mode.

    qa_mode:
      - "qa_pairs":      use item["qa_pairs"] as-is (default, legacy format)
      - "open_ended":     use item["open_ended_qa"] (Gemini format)
      - "binary":         use item["binary_qa"]     (Gemini format)
    """
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in ["data", "items", "samples"]:
            if isinstance(raw.get(key), list):
                items = raw[key]
                break
        else:
            raise ValueError("Unsupported QA JSON structure.")
    else:
        raise ValueError("Unsupported QA JSON structure.")

    # Remap the QA field to "qa_pairs" so the rest of the pipeline works unchanged
    if qa_mode in ("open_ended", "binary"):
        src_key = "open_ended_qa" if qa_mode == "open_ended" else "binary_qa"
        for item in items:
            if src_key in item:
                item["qa_pairs"] = item[src_key]
    return items


def find_video_for_id(video_dir: Path, item_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTENSIONS:
        p = video_dir / f"{item_id}{ext}"
        if p.exists():
            return p
    for p in video_dir.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.stem == item_id:
            return p
    return None


class ProgressBar:
    def __init__(self, total: int, desc: str = "", width: int = 28, enabled: bool = True) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.width = max(int(width), 10)
        self.enabled = enabled and sys.stderr.isatty()
        self.current = 0
        self._finished = False
        self._tqdm = None

        if self.enabled and self.total > 0 and tqdm is not None:
            self._tqdm = tqdm(
                total=self.total,
                desc=self.desc,
                leave=False,
                dynamic_ncols=True,
                mininterval=0.2,
                bar_format=(
                    "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
                ),
                file=sys.stderr,
            )

    def update(self, step: int = 1, suffix: str = "") -> None:
        if not self.enabled or self.total <= 0:
            return
        if self._tqdm is not None:
            if suffix:
                self._tqdm.set_postfix_str(suffix, refresh=False)
            self._tqdm.update(step)
            if self._tqdm.n >= self.total:
                self.close()
            return
        self.current = min(self.total, self.current + step)
        filled = int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 * self.current / self.total
        line = f"\r{self.desc} [{bar}] {self.current}/{self.total} {percent:5.1f}%"
        if suffix:
            line += f" | {suffix}"
        print(line, end="", file=sys.stderr, flush=True)
        if self.current >= self.total:
            self.close()

    def close(self) -> None:
        if not self.enabled or self._finished:
            return
        self._finished = True
        if self._tqdm is not None:
            self._tqdm.close()
            return
        print(file=sys.stderr, flush=True)


# -----------------------------
# OpenAI-compatible API / direct-video content
# -----------------------------
def build_client(base_url: str, api_key: str, timeout: float, max_retries: int) -> OpenAI:
    from openai import OpenAI

    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def encode_video_as_data_url(video_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(video_path))
    if not mime_type:
        mime_type = "video/mp4"
    with video_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def extract_text_from_message(message: Any) -> str:
    chunks: List[str] = []

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        chunks.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
            elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                chunks.append(item["text"].strip())

    for attr_name in ("reasoning_content", "text"):
        attr_value = getattr(message, attr_name, None)
        if isinstance(attr_value, str) and attr_value.strip():
            chunks.append(attr_value.strip())
        elif isinstance(attr_value, list):
            for item in attr_value:
                text = getattr(item, "text", None)
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
                elif isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                    chunks.append(item["text"].strip())

    if not chunks and hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
        except Exception:
            dumped = {}
        for key in ("content", "reasoning_content", "text"):
            value = dumped.get(key)
            if isinstance(value, str) and value.strip():
                chunks.append(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
                        chunks.append(item["text"].strip())

    return "\n".join(chunk for chunk in chunks if chunk).strip()


def make_video_content(video_path: Path, video_fps: float) -> Dict[str, Any]:
    content: Dict[str, Any] = {
        "type": "video_url",
        "video_url": {"url": encode_video_as_data_url(video_path)},
    }
    if video_fps > 0:
        content["fps"] = video_fps
    return content


def api_generate(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, Any]],
    max_completion_tokens: int = 512,
    temperature: float = 0.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> str:
    kwargs: Dict[str, Any] = dict(
        model=model_name,
        messages=messages,
        max_tokens=max_completion_tokens,
        temperature=temperature,
    )
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)

    if not getattr(response, "choices", None):
        raise RuntimeError("Empty choices returned from OpenAI-compatible endpoint.")

    message = response.choices[0].message
    output_text = extract_text_from_message(message)
    return output_text.strip()


# -----------------------------
# Prompts
# -----------------------------
def build_single_answer_prompt(sample: Dict[str, Any], qa: Dict[str, Any], question_index: int, binary_mode: bool = False) -> str:
    payload = {
        "id": sample.get("id", ""),
        "category": sample.get("category", ""),
        "sub_category": sample.get("sub_category", ""),
        "caption": sample.get("caption", ""),
        "video_prompt": sample.get("video_prompt", ""),
        "reasoning_analysis": sample.get("reasoning_analysis", ""),
        "key_reasoning_elements": sample.get("key_reasoning_elements", []),
        "overall_evaluation_focus": sample.get("overall_evaluation_focus", []),
        "question": {
            "question_index": question_index,
            "question": qa.get("question", ""),
            "question_type": normalize_question_type(qa.get("question_type", "")),
            "difficulty": normalize_difficulty(qa.get("difficulty", "")),
        },
    }

    if binary_mode:
        return (
            "You are answering one yes/no evaluation question about a generated video.\n\n"
            "Critical rule:\n"
            "Answer ONLY using facts that are directly visible in the video frames.\n"
            "Do NOT infer, speculate, guess, or complete missing information.\n"
            "Do NOT use commonsense, task intention, or outside knowledge to fill gaps.\n\n"
            "You MUST answer with exactly 'yes' or 'no'.\n"
            "You may briefly explain your reasoning before the answer, but the last line "
            "of your response MUST be exactly:\n"
            'Final JSON:\n{"answer": "yes"}\nor\n{"answer": "no"}\n\n'
            f"Sample metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    return (
        "You are answering one evaluation question about a generated video for InfoVBench.\n\n"
        "Critical rule:\n"
        "Answer ONLY using facts that are directly visible in the video frames.\n"
        "Do NOT infer, speculate, guess, or complete missing information.\n"
        "Do NOT use commonsense, task intention, or outside knowledge to fill gaps.\n"
        "Do NOT use any human_hint or hidden authoring hint.\n"
        "You may briefly analyze first, but keep it short.\n\n"
        "If the requested fact, detail, order, or relation is not clearly visible in the video, answer exactly:\n"
        '"unclear from the video"\n\n'
        "The last part of your response MUST be exactly in this format:\n"
        'Final JSON:\n{"answer": "..."}\n\n'
        "Rules for the final JSON:\n"
        "- It must be valid JSON.\n"
        "- It must be the last part of your response.\n"
        "- Do not use markdown code fences.\n\n"
        f"Sample metadata:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def build_binary_judge_prompt(
    question: str,
    question_type: str,
    predicted_answer: str,
    ground_truth: str,
    evaluation_criteria: str,
    difficulty: str,
) -> str:
    qtype = normalize_question_type(question_type)

    qtype_rule_map = {
        "factual": "Judge whether the predicted answer correctly describes the relevant visible fact, state, object, or outcome.",
        "detail": "Judge whether the predicted answer correctly captures the specific required detail, such as color, count, text, local position, identity, or object-specific attribute.",
        "temporal": "Judge whether the predicted answer correctly captures event order, simultaneity, progression, or before/after relations.",
        "reasoning": "Judge whether the predicted answer correctly captures causality, physical rule, commonsense rule, social logic, or mechanism implied by the video and criteria."
    }

    return (
        "You are a strict evaluator for InfoVBench QA.\n"
        "You must score this single QA as binary: 1 or 0.\n"
        "You may briefly analyze first, but keep it short.\n\n"
        "Scoring rule:\n"
        "- score = 1 only if the predicted answer is sufficiently correct according to BOTH the ground truth and the evaluation criteria.\n"
        "- score = 0 if it is incorrect, incomplete on the key point, contradicts the criteria, or unsupported.\n"
        "- Be strict.\n"
        "- Do not give partial credit.\n"
        "- If the predicted answer says 'unclear from the video', give 1 only if the criteria truly cannot be verified from the provided evidence; otherwise give 0.\n\n"
        f"Question type: {qtype}\n"
        f"Type-specific judging focus: {qtype_rule_map[qtype]}\n\n"
        "The last part of your response MUST be exactly:\n"
        "Final JSON:\n"
        '{"score": 1, "reason": "...", "verdict": "correct"}\n\n'
        "Rules for the final JSON:\n"
        "- It must be valid JSON.\n"
        "- It must be the last part of your response.\n"
        "- Use score 1 or 0 only.\n"
        "- Use verdict correct or incorrect only.\n"
        "- Do not use markdown code fences.\n\n"
        f"Question: {question}\n"
        f"Predicted answer: {predicted_answer}\n"
        f"Ground truth: {ground_truth}\n"
        f"Evaluation criteria: {evaluation_criteria}\n"
        f"Difficulty: {difficulty}\n"
    )


# -----------------------------
# Stage 1: answer one question for one video
# -----------------------------
def ask_single_video_question(
    client: OpenAI,
    model_name: str,
    video_path: Path,
    sample: Dict[str, Any],
    qa: Dict[str, Any],
    question_index: int,
    max_completion_tokens: int,
    temperature: float,
    request_sleep: float,
    video_fps: float,
    binary_mode: bool = False,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, bool]:
    content: List[Dict[str, Any]] = [
        make_video_content(video_path, video_fps=video_fps),
        {"type": "text", "text": build_single_answer_prompt(sample, qa, question_index, binary_mode=binary_mode)},
    ]

    output_text = api_generate(
        client=client,
        model_name=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a precise and conservative video QA assistant. Answer only from directly visible video content. Do not infer or guess. You may briefly analyze first, but the response must end with Final JSON: followed by one valid JSON object.",
            },
            {"role": "user", "content": content},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        extra_body=extra_body,
    )

    if request_sleep > 0:
        time.sleep(request_sleep)

    raw = normalize_message_content(output_text)
    parsed = extract_final_json_block(raw, fallback_to_any_json=False)
    if parsed and isinstance(parsed.get("answer"), str):
        ans = clean_final_answer_text(parsed.get("answer", ""))
        if ans:
            return ans, raw, True

    return "unclear from the video", raw, False


# -----------------------------
# Stage 2: binary judge each QA
# -----------------------------
def judge_answer_binary_direct(
    predicted_answer: str,
    ground_truth: str,
) -> Dict[str, Any]:
    """Direct yes/no match for binary QA -- no LLM judge needed."""
    pred_lower = predicted_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()

    pred_yn = None
    if pred_lower.startswith("yes"):
        pred_yn = "yes"
    elif pred_lower.startswith("no"):
        pred_yn = "no"
    else:
        import re as _re
        match = _re.search(r'\b(yes|no)\b', pred_lower)
        if match:
            pred_yn = match.group(1)

    if pred_yn is None:
        return {
            "score": 0,
            "reason": f"Could not extract yes/no from answer: {predicted_answer[:200]}",
            "verdict": "incorrect",
            "judge_parse_ok": False,
            "raw_judge_text": "",
            "parsed_judge_json": None,
        }

    score = 1 if pred_yn == gt_lower else 0
    return {
        "score": score,
        "reason": f"predicted={pred_yn}, ground_truth={gt_lower}",
        "verdict": "correct" if score == 1 else "incorrect",
        "judge_parse_ok": True,
        "raw_judge_text": "",
        "parsed_judge_json": {"predicted_yn": pred_yn, "ground_truth_yn": gt_lower},
    }


def judge_answer_binary(
    client: OpenAI,
    model_name: str,
    question: str,
    question_type: str,
    predicted_answer: str,
    ground_truth: str,
    evaluation_criteria: str,
    difficulty: str,
    max_completion_tokens: int,
    temperature: float,
    request_sleep: float,
) -> Dict[str, Any]:
    prompt = build_binary_judge_prompt(
        question=question,
        question_type=question_type,
        predicted_answer=predicted_answer,
        ground_truth=ground_truth,
        evaluation_criteria=evaluation_criteria,
        difficulty=difficulty,
    )

    output_text = api_generate(
        client=client,
        model_name=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evaluator. You may briefly analyze first, but the response must end with Final JSON: followed by one valid JSON object.",
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    if request_sleep > 0:
        time.sleep(request_sleep)

    raw = normalize_message_content(output_text)
    parsed = extract_final_json_block(raw, fallback_to_any_json=False)

    if not parsed:
        return {
            "score": 0,
            "reason": f"Invalid judge JSON: {clean_final_answer_text(raw)[:300]}",
            "verdict": "incorrect",
            "judge_parse_ok": False,
            "raw_judge_text": raw,
            "parsed_judge_json": None,
        }

    score = int(parsed.get("score", 0))
    score = 1 if score == 1 else 0

    verdict = str(parsed.get("verdict", "incorrect")).strip().lower()
    if verdict not in {"correct", "incorrect"}:
        verdict = "correct" if score == 1 else "incorrect"

    reason = str(parsed.get("reason", "")).strip()

    return {
        "score": score,
        "reason": reason,
        "verdict": verdict,
        "judge_parse_ok": True,
        "raw_judge_text": raw,
        "parsed_judge_json": parsed,
    }


# -----------------------------
# Aggregation
# -----------------------------
def aggregate_video_result(
    item: Dict[str, Any],
    video_path: Path,
    qa_pairs: List[Dict[str, Any]],
    predicted_answers: List[str],
    judged_results: List[Dict[str, Any]],
    answer_debugs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    question_results: List[Dict[str, Any]] = []

    sample_scores: List[float] = []
    by_qtype: Dict[str, List[float]] = defaultdict(list)
    by_difficulty: Dict[str, List[float]] = defaultdict(list)

    for idx, (qa, pred, judge) in enumerate(zip(qa_pairs, predicted_answers, judged_results), start=1):
        qtype = normalize_question_type(str(qa.get("question_type", "")))
        difficulty = normalize_difficulty(str(qa.get("difficulty", "")))
        score = int(judge.get("score", 0))
        score = 1 if score == 1 else 0

        row = {
            "question_index": idx,
            "question": str(qa.get("question", "")),
            "question_type": qtype,
            "difficulty": difficulty,
            "predicted_answer": pred,
            "ground_truth": str(qa.get("ground_truth", "")),
            "evaluation_criteria": str(qa.get("evaluation_criteria", "")),
            "score": score,
            "verdict": str(judge.get("verdict", "incorrect")),
            "judge_reason": str(judge.get("reason", "")),
            "judge_parse_ok": bool(judge.get("judge_parse_ok", False)),
            "raw_judge_text": str(judge.get("raw_judge_text", "")),
            "parsed_judge_json": judge.get("parsed_judge_json", None),
        }
        if answer_debugs and idx - 1 < len(answer_debugs):
            row.update(answer_debugs[idx - 1])
        question_results.append(row)

        sample_scores.append(float(score))
        by_qtype[qtype].append(float(score))
        by_difficulty[difficulty].append(float(score))

    # Process-aware metrics
    process_metrics = _compute_process_aware_metrics(question_results)

    return {
        "id": str(item.get("id", "")),
        "video_path": str(video_path),
        "category": str(item.get("category", "")),
        "sub_category": str(item.get("sub_category", "")),
        "overall_evaluation_focus": item.get("overall_evaluation_focus", []),
        "num_questions": len(question_results),
        "sample_score": round4(safe_mean(sample_scores)),
        "type_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_qtype.items())},
        "difficulty_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_difficulty.items())},
        # Process-aware metrics
        "weighted_score": process_metrics["weighted_score"],
        "state_score": process_metrics["state_score"],
        "process_satisfaction": process_metrics["process_satisfaction"],
        "fidelity_score": process_metrics["fidelity_score"],
        "mechanism_score": process_metrics["mechanism_score"],
        "poc_gap": process_metrics["poc_gap"],
        "poc_flag": process_metrics["poc_flag"],
        "chain_score": process_metrics["chain_score"],
        "bottleneck_score": process_metrics["bottleneck_score"],
        "reasoning_coverage": process_metrics["reasoning_coverage"],
        "reasoning_radar_area": process_metrics["reasoning_radar_area"],
        "diagnosis": process_metrics["diagnosis"],
        "questions": question_results,
    }


def build_summary(per_video_results: List[Dict[str, Any]], skipped: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall_scores = [float(v["sample_score"]) for v in per_video_results]

    by_category: Dict[str, List[float]] = defaultdict(list)
    by_sub_category: Dict[str, List[float]] = defaultdict(list)
    by_qtype: Dict[str, List[float]] = defaultdict(list)
    by_difficulty: Dict[str, List[float]] = defaultdict(list)

    total_questions = 0

    # Process-aware aggregation collectors
    all_weighted: List[float] = []
    all_process_sat: List[float] = []
    all_mechanism: List[float] = []
    all_chain: List[float] = []
    all_bottleneck: List[float] = []
    all_radar: List[float] = []
    all_state: List[float] = []
    all_fidelity: List[float] = []
    all_easy: List[float] = []
    all_hard: List[float] = []
    poc_flags: List[str] = []
    diagnoses: Dict[str, int] = defaultdict(int)

    # Per-category process-aware collectors
    cat_process: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    cat_poc_flags: Dict[str, List[str]] = defaultdict(list)

    for video in per_video_results:
        score = float(video["sample_score"])
        category = str(video.get("category", ""))
        sub_category = str(video.get("sub_category", ""))

        by_category[category].append(score)
        by_sub_category[f"{category}::{sub_category}"].append(score)

        for k, v in video.get("type_scores", {}).items():
            by_qtype[k].append(float(v))
        for k, v in video.get("difficulty_scores", {}).items():
            by_difficulty[k].append(float(v))

        total_questions += int(video.get("num_questions", 0))

        if video.get("weighted_score") is not None:
            all_weighted.append(float(video["weighted_score"]))
        if video.get("process_satisfaction") is not None:
            all_process_sat.append(float(video["process_satisfaction"]))
        if video.get("mechanism_score") is not None:
            all_mechanism.append(float(video["mechanism_score"]))
        if video.get("chain_score") is not None:
            all_chain.append(float(video["chain_score"]))
        if video.get("bottleneck_score") is not None:
            all_bottleneck.append(float(video["bottleneck_score"]))
        if video.get("reasoning_radar_area") is not None:
            all_radar.append(float(video["reasoning_radar_area"]))
        if video.get("state_score") is not None:
            all_state.append(float(video["state_score"]))
        if video.get("fidelity_score") is not None:
            all_fidelity.append(float(video["fidelity_score"]))

        diff_scores = video.get("difficulty_scores", {})
        if "easy" in diff_scores:
            all_easy.append(float(diff_scores["easy"]))
        if "hard" in diff_scores:
            all_hard.append(float(diff_scores["hard"]))

        poc_flag = video.get("poc_flag", "insufficient_data")
        poc_flags.append(poc_flag)
        diagnoses[video.get("diagnosis", "no_data")] += 1

        if category:
            for mkey in ("weighted_score", "process_satisfaction", "mechanism_score",
                         "chain_score", "bottleneck_score", "state_score", "fidelity_score"):
                if video.get(mkey) is not None:
                    cat_process[category][mkey].append(float(video[mkey]))
            cat_poc_flags[category].append(poc_flag)

    n = len(per_video_results) or 1

    category_process_aware: Dict[str, Dict[str, Any]] = {}
    for cat in sorted(cat_process.keys()):
        cn = len(cat_poc_flags.get(cat, [])) or 1
        cat_pa: Dict[str, Any] = {}
        for mkey in ("weighted_score", "process_satisfaction", "mechanism_score",
                     "chain_score", "bottleneck_score", "state_score", "fidelity_score"):
            vals = cat_process[cat].get(mkey, [])
            cat_pa[mkey] = round4(safe_mean(vals)) if vals else None
        cat_flags = cat_poc_flags.get(cat, [])
        cat_pa["outcome_hacking_rate"] = round4(sum(1 for f in cat_flags if f == "outcome_hacking") / cn)
        category_process_aware[cat] = cat_pa

    return {
        "num_videos_evaluated": len(per_video_results),
        "num_videos_skipped": len(skipped),
        "num_questions_evaluated": total_questions,
        "overall_average_score": round4(safe_mean(overall_scores)),
        "category_average_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_category.items())},
        "sub_category_average_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_sub_category.items())},
        "question_type_average_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_qtype.items())},
        "difficulty_average_scores": {k: round4(safe_mean(v)) for k, v in sorted(by_difficulty.items())},
        "process_aware": {
            "weighted_score": round4(safe_mean(all_weighted)) if all_weighted else None,
            "state_score": round4(safe_mean(all_state)) if all_state else None,
            "process_satisfaction": round4(safe_mean(all_process_sat)) if all_process_sat else None,
            "fidelity_score": round4(safe_mean(all_fidelity)) if all_fidelity else None,
            "mechanism_score": round4(safe_mean(all_mechanism)) if all_mechanism else None,
            "chain_score": round4(safe_mean(all_chain)) if all_chain else None,
            "bottleneck_score": round4(safe_mean(all_bottleneck)) if all_bottleneck else None,
            "reasoning_radar_area": round4(safe_mean(all_radar)) if all_radar else None,
            "outcome_hacking_rate": round4(sum(1 for f in poc_flags if f == "outcome_hacking") / n),
            "incomplete_transition_rate": round4(sum(1 for f in poc_flags if f == "incomplete_transition") / n),
            "consistent_rate": round4(sum(1 for f in poc_flags if f == "consistent") / n),
            "difficulty_discrimination": round4(safe_mean(all_hard) - safe_mean(all_easy)) if all_hard and all_easy else None,
            "diagnosis_distribution": dict(diagnoses),
            "per_category": category_process_aware,
        },
    }


def write_progress_files(
    output_dir: Path,
    per_video_results: List[Dict[str, Any]],
    skipped: List[Dict[str, Any]],
) -> None:
    summary = build_summary(per_video_results, skipped)

    question_rows: List[Dict[str, Any]] = []
    video_rows: List[Dict[str, Any]] = []

    for video_result in per_video_results:
        video_rows.append({
            "id": video_result["id"],
            "video_path": video_result["video_path"],
            "category": video_result["category"],
            "sub_category": video_result["sub_category"],
            "num_questions": video_result["num_questions"],
            "sample_score": video_result["sample_score"],
            **{f"type_{k}": v for k, v in video_result["type_scores"].items()},
            **{f"diff_{k}": v for k, v in video_result["difficulty_scores"].items()},
        })

        for q in video_result["questions"]:
            question_rows.append({
                "id": video_result["id"],
                "video_path": video_result["video_path"],
                "category": video_result["category"],
                "sub_category": video_result["sub_category"],
                **q,
            })

    result = {
        "summary": summary,
        "per_video": per_video_results,
        "skipped": skipped,
    }

    save_json(output_dir / "full_results.json", result)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "skipped.json", skipped)
    write_csv(output_dir / "video_scores.csv", video_rows)
    write_csv(output_dir / "question_scores.csv", question_rows)


# -----------------------------
# Main evaluation
# -----------------------------
def evaluate_dataset(
    client: OpenAI,
    model_name: str,
    qa_json_path: Path,
    video_dir: Path,
    output_dir: Path,
    max_answer_tokens: int,
    max_judge_tokens: int,
    temperature: float,
    request_sleep: float,
    limit: Optional[int],
    id_list: Optional[List[str]],
    save_every: int,
    video_fps: float,
    show_progress: bool,
    qa_mode: str = "qa_pairs",
    use_mm_processor_kwargs: bool = False,
) -> Dict[str, Any]:
    items = normalize_items(load_json(qa_json_path), qa_mode=qa_mode)
    is_binary = qa_mode == "binary"

    extra_body: Optional[Dict[str, Any]] = None
    if use_mm_processor_kwargs and video_fps > 0:
        extra_body = {
            "mm_processor_kwargs": {"fps": video_fps, "do_sample_frames": True},
        }

    if id_list is not None:
        id_set = set(id_list)
        items = [item for item in items if str(item.get("id", "")).strip() in id_set]

    if limit is not None:
        items = items[:limit]

    ensure_dir(output_dir)

    per_video_results: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    video_progress = ProgressBar(len(items), desc="Videos", enabled=show_progress)

    for idx, item in enumerate(items, start=1):
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            skipped.append({"reason": "missing_id"})
            video_progress.update(suffix="missing_id")
            if idx % save_every == 0:
                write_progress_files(output_dir, per_video_results, skipped)
            continue

        video_path = find_video_for_id(video_dir, item_id)
        if video_path is None:
            skipped.append({"id": item_id, "reason": "video_not_found"})
            video_progress.update(suffix=f"{item_id} video_not_found")
            if idx % save_every == 0:
                write_progress_files(output_dir, per_video_results, skipped)
            continue

        qa_pairs = item.get("qa_pairs", [])
        if not isinstance(qa_pairs, list) or not qa_pairs:
            skipped.append({"id": item_id, "reason": "no_qa_pairs"})
            video_progress.update(suffix=f"{item_id} no_qa_pairs")
            if idx % save_every == 0:
                write_progress_files(output_dir, per_video_results, skipped)
            continue

        predicted_answers: List[str] = []
        answer_debugs: List[Dict[str, Any]] = []
        answer_error = None
        answer_progress = ProgressBar(
            len(qa_pairs),
            desc=f"Answer {item_id}",
            enabled=show_progress,
        )
        for q_idx, qa in enumerate(qa_pairs, start=1):
            try:
                pred, raw_answer_text, parse_ok = ask_single_video_question(
                    client=client,
                    model_name=model_name,
                    video_path=video_path,
                    sample=item,
                    qa=qa,
                    question_index=q_idx,
                    max_completion_tokens=max_answer_tokens,
                    temperature=temperature,
                    request_sleep=request_sleep,
                    video_fps=video_fps,
                    binary_mode=is_binary,
                    extra_body=extra_body,
                )
                predicted_answers.append(pred)
                answer_debugs.append({
                    "answer_parse_ok": parse_ok,
                    "raw_answer_text": raw_answer_text,
                })
                answer_progress.update(suffix=f"q={q_idx}")
            except Exception as e:
                answer_error = e
                answer_progress.close()
                break
        else:
            answer_progress.close()

        if answer_error is not None:
            skipped.append({"id": item_id, "reason": f"answer_error: {answer_error}"})
            video_progress.update(suffix=f"{item_id} answer_error")
            if idx % save_every == 0:
                write_progress_files(output_dir, per_video_results, skipped)
            continue

        judged_results: List[Dict[str, Any]] = []
        judge_progress = ProgressBar(
            len(predicted_answers),
            desc=f"Judge  {item_id}",
            enabled=show_progress,
        )
        for q_idx, (qa, pred) in enumerate(zip(qa_pairs, predicted_answers), start=1):
            try:
                if is_binary:
                    judged = judge_answer_binary_direct(
                        predicted_answer=str(pred),
                        ground_truth=str(qa.get("ground_truth", "")),
                    )
                else:
                    judged = judge_answer_binary(
                        client=client,
                        model_name=model_name,
                        question=str(qa.get("question", "")),
                        question_type=str(qa.get("question_type", "")),
                        predicted_answer=str(pred),
                        ground_truth=str(qa.get("ground_truth", "")),
                        evaluation_criteria=str(qa.get("evaluation_criteria", "")),
                        difficulty=str(qa.get("difficulty", "")),
                        max_completion_tokens=max_judge_tokens,
                        temperature=temperature,
                        request_sleep=request_sleep,
                    )
            except Exception as e:
                judged = {
                    "score": 0,
                    "reason": f"judge_error: {e}",
                    "verdict": "incorrect",
                    "judge_parse_ok": False,
                    "raw_judge_text": "",
                    "parsed_judge_json": None,
                }
            judged_results.append(judged)
            judge_progress.update(suffix=f"q={q_idx}")
        judge_progress.close()

        video_result = aggregate_video_result(
            item=item,
            video_path=video_path,
            qa_pairs=qa_pairs,
            predicted_answers=predicted_answers,
            judged_results=judged_results,
            answer_debugs=answer_debugs,
        )
        per_video_results.append(video_result)
        video_progress.update(suffix=item_id)

        write_progress_files(output_dir, per_video_results, skipped)

        if idx % save_every == 0:
            write_progress_files(output_dir, per_video_results, skipped)

    video_progress.close()

    summary = build_summary(per_video_results, skipped)
    result = {
        "summary": summary,
        "per_video": per_video_results,
        "skipped": skipped,
        "config": {
            "qa_json_path": str(qa_json_path),
            "video_dir": str(video_dir),
            "output_dir": str(output_dir),
            "base_url": client.base_url.__str__(),
            "model": model_name,
            "judge_model": model_name,
            "video_fps": video_fps,
            "use_mm_processor_kwargs": use_mm_processor_kwargs,
            "max_answer_tokens": max_answer_tokens,
            "max_judge_tokens": max_judge_tokens,
            "temperature": temperature,
            "request_sleep": request_sleep,
            "scoring_mode": "binary_qa",
            "input_mode": "direct_video_openai_compatible",
            "uses_human_hint": False,
            "answer_stage_rule": "single_question_per_request_allow_brief_analysis_then_parse_final_json",
            "judge_stage_rule": "allow_brief_analysis_then_parse_final_json",
            "fallback_answer": "unclear from the video",
            "save_mode": "realtime",
            "save_every": save_every,
        },
    }

    save_json(output_dir / "full_results.json", result)
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "skipped.json", skipped)

    return result


# -----------------------------
# Args / main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate generated videos with a Qwen3.5 OpenAI-compatible endpoint using single-question-per-request answering."
    )
    p.add_argument("--qa_json", type=str, required=True)
    p.add_argument("--video_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    p.add_argument("--api_key", type=str, default=DEFAULT_API_KEY)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--video_fps", type=float, default=2.0,
                   help="fps field attached to the video_url content item, matching the Qwen3.5 example format")
    p.add_argument("--max_answer_tokens", type=int, default=400)
    p.add_argument("--max_judge_tokens", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request_sleep", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--max_retries", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--id_list", type=str, default=None,
                   help="Comma-separated sample ids to evaluate, e.g. 1-a-1,1-a-3,1-a-10")
    p.add_argument("--save_every", type=int, default=1,
                   help="Also force-save every N processed items. default=1 means save after every item.")
    p.add_argument("--no_progress", action="store_true",
                   help="Disable terminal progress bars.")
    p.add_argument("--qa_mode", type=str, default="qa_pairs",
                   choices=["qa_pairs", "open_ended", "binary"],
                   help="Which QA field to use. 'qa_pairs' = legacy format, "
                        "'open_ended' = Gemini open_ended_qa, 'binary' = Gemini binary_qa. "
                        "For binary mode, Stage 2 uses direct yes/no match instead of LLM judge.")
    p.add_argument("--use_mm_processor_kwargs", action="store_true",
                   help="Pass video_fps via extra_body mm_processor_kwargs (for vLLM with "
                        "--media-io-kwargs). Without this flag, fps is set in the video_url content item.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    qa_json_path = Path(args.qa_json)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not qa_json_path.exists():
        raise SystemExit(f"QA JSON not found: {qa_json_path}")
    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    ensure_dir(output_dir)

    id_list = None
    if args.id_list:
        id_list = [x.strip() for x in args.id_list.split(",") if x.strip()]

    client = build_client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    result = evaluate_dataset(
        client=client,
        model_name=args.model,
        qa_json_path=qa_json_path,
        video_dir=video_dir,
        output_dir=output_dir,
        max_answer_tokens=args.max_answer_tokens,
        max_judge_tokens=args.max_judge_tokens,
        temperature=args.temperature,
        request_sleep=args.request_sleep,
        limit=args.limit,
        id_list=id_list,
        save_every=args.save_every,
        video_fps=args.video_fps,
        show_progress=not args.no_progress,
        qa_mode=args.qa_mode,
        use_mm_processor_kwargs=args.use_mm_processor_kwargs,
    )

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
