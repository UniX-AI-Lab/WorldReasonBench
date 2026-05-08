"""Shared helpers for lightweight WorldReason-Reward-Bench evaluation."""

from __future__ import annotations

import base64
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Union
from urllib.request import urlopen

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).parent


def load_template(task: str, template: str) -> str:
    template_path = ROOT / "templates" / task / f"{template}.txt"
    with template_path.open("r", encoding="utf-8") as f:
        return f.read()


def local_file_to_data_url(
    file_path: Union[str, Path],
    default_mime: str,
) -> str:
    path = Path(file_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = default_mime
    with path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def local_image_to_data_url(image_path: Union[str, Path]) -> str:
    return local_file_to_data_url(image_path, default_mime="image/png")


def local_video_to_data_url(video_path: Union[str, Path]) -> str:
    return local_file_to_data_url(video_path, default_mime="video/mp4")


def load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")

    image_str = str(image_input)
    if image_str.startswith(("http://", "https://")):
        with urlopen(image_str) as response:
            return Image.open(BytesIO(response.read())).convert("RGB")

    return Image.open(image_str).convert("RGB")


def pil_image_to_data_url(
    image: Image.Image,
    *,
    image_format: str = "JPEG",
    max_side: int | None = None,
    quality: int = 85,
) -> str:
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)!r}")

    image = image.convert("RGB")
    if max_side is not None and max(image.size) > max_side:
        scale = max_side / float(max(image.size))
        new_size = (
            max(1, int(round(image.width * scale))),
            max(1, int(round(image.height * scale))),
        )
        resample = (
            Image.Resampling.LANCZOS
            if hasattr(Image, "Resampling")
            else Image.LANCZOS
        )
        image = image.resize(new_size, resample)

    normalized_format = image_format.upper()
    mime_type = "image/jpeg" if normalized_format in {"JPEG", "JPG"} else "image/png"

    buffer = BytesIO()
    save_kwargs = {}
    if normalized_format in {"JPEG", "JPG"}:
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    image.save(buffer, format=normalized_format, **save_kwargs)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def sample_video_frames_with_metadata(
    video_path: Union[str, Path],
    max_num_frames: int = 12,
) -> List[Dict[str, Any]]:
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no readable frames: {path}")

    num_frames = min(max_num_frames, total_frames)
    raw_indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int).tolist()

    indices: List[int] = []
    seen_indices: set[int] = set()
    for idx in raw_indices:
        frame_idx = int(idx)
        if frame_idx in seen_indices:
            continue
        seen_indices.add(frame_idx)
        indices.append(frame_idx)

    target_set = set(indices)
    frames_by_index: Dict[int, Dict[str, Any]] = {}
    current_idx = 0
    while current_idx <= indices[-1]:
        success, frame = cap.read()
        if not success:
            break
        if current_idx in target_set:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_by_index[current_idx] = {
                "image": Image.fromarray(rgb),
                "frame_index": int(current_idx),
                "total_frames": total_frames,
                "fps": fps,
                "timestamp_sec": (float(current_idx) / fps) if fps > 0 else None,
                "relative_position": (
                    float(current_idx) / float(total_frames - 1)
                    if total_frames > 1
                    else 0.0
                ),
            }
            if len(frames_by_index) == len(indices):
                break
        current_idx += 1

    cap.release()

    frames: List[Dict[str, Any]] = []
    for order, idx in enumerate(indices, start=1):
        item = frames_by_index.get(idx)
        if item is None:
            continue
        item = dict(item)
        item["sample_order"] = order
        item["sample_count"] = len(indices)
        frames.append(item)

    if not frames:
        raise ValueError(f"Failed to sample frames from video: {path}")

    return frames


def sample_video_frames(
    video_path: Union[str, Path],
    max_num_frames: int = 12,
) -> List[Image.Image]:
    return [
        item["image"]
        for item in sample_video_frames_with_metadata(
            video_path,
            max_num_frames=max_num_frames,
        )
    ]
