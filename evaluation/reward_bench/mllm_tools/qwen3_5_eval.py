"""OpenAI-compatible Qwen3.5 judge wrapper.

This wrapper follows the SGLang/OpenAI-compatible path: local videos are
forwarded as ``video_url`` data URLs and decoded by the serving backend.
Unlike other wrappers, it does not perform client-side frame sampling and
intentionally ignores caller-provided frame-sampling hints such as ``fps``
or ``max_num_frames``.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from ..utils import local_image_to_data_url, local_video_to_data_url


class Qwen3_5:
    support_multi_image = True
    support_video_input = True
    merged_image_files = []

    def __init__(
        self,
        model_path: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_path = (
            model_path
            or os.environ.get("QWEN3_5_MODEL_NAME")
            or os.environ.get("OPENAI_MODEL_NAME")
            or "Qwen/Qwen3.5-27B"
        )
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or "http://localhost:8000/v1"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def __call__(self, inputs: List[dict]) -> str:
        messages = self.build_messages(inputs)
        response = self.create_completion(messages)
        return self.extract_response_text(response)

    def build_messages(self, inputs: List[dict]) -> List[dict]:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a strict multimodal judge for video generation. "
                            "Compare the provided visual content against the prompt, "
                            "reason carefully, and follow the requested output format exactly."
                        ),
                    }
                ],
            },
            {"role": "user", "content": []},
        ]

        for item in inputs:
            item_type = item["type"]
            if item_type == "image":
                messages[-1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": local_image_to_data_url(item["content"])},
                    }
                )
            elif item_type == "video":
                messages[-1]["content"].append(
                    {
                        "type": "video_url",
                        "video_url": {"url": local_video_to_data_url(item["content"])},
                    }
                )
            elif item_type == "text":
                messages[-1]["content"].append(
                    {"type": "text", "text": item["content"]}
                )
            else:
                raise NotImplementedError(f"Unsupported input type: {item_type}")
        return messages

    def create_completion(self, messages: List[dict]) -> Any:
        import openai

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=900.0,
            max_retries=2,
        )
        # No-Thinking mode: set QWEN3_5_NO_THINKING=1 to disable thinking chain
        no_thinking = os.environ.get("QWEN3_5_NO_THINKING", "0") == "1"
        extra_body: dict[str, Any] = {}
        if no_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        # vLLM fps control: set QWEN3_5_VIDEO_FPS to pass mm_processor_kwargs
        # Requires vLLM launched with --media-io-kwargs '{"video": {"num_frames": -1}}'
        video_fps_str = os.environ.get("QWEN3_5_VIDEO_FPS", "")
        if video_fps_str:
            try:
                video_fps = float(video_fps_str)
                extra_body["mm_processor_kwargs"] = {
                    "fps": video_fps,
                    "do_sample_frames": True,
                }
            except ValueError:
                pass

        return client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            max_tokens=int(os.environ.get("QWEN3_5_MAX_TOKENS", "16384")),
            temperature=0.1,
            top_p=0.8,
            **({"extra_body": extra_body} if extra_body else {}),
        )

    @staticmethod
    def extract_response_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        return Qwen3_5._extract_message_text(message)

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        if message is None:
            return ""

        chunks: List[str] = []

        def append_text(value: Any) -> None:
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    chunks.append(stripped)
                return
            if isinstance(value, list):
                for item in value:
                    text = getattr(item, "text", None)
                    if isinstance(text, str) and text.strip():
                        chunks.append(text.strip())
                    elif (
                        isinstance(item, dict)
                        and isinstance(item.get("text"), str)
                        and item["text"].strip()
                    ):
                        chunks.append(item["text"].strip())

        append_text(getattr(message, "content", None))
        append_text(getattr(message, "reasoning_content", None))
        append_text(getattr(message, "text", None))

        if not chunks and hasattr(message, "model_dump"):
            try:
                dumped = message.model_dump()
            except Exception:
                dumped = {}
            if isinstance(dumped, dict):
                append_text(dumped.get("content"))
                append_text(dumped.get("reasoning_content"))
                append_text(dumped.get("text"))

        return "\n".join(chunk for chunk in chunks if chunk).strip()

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(part for part in parts if part)
        return str(content)
