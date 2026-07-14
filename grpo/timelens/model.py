"""Model / config / processor loader (Qwen3-VL via transformers Auto classes)."""

from __future__ import annotations

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor


def _validate(model_path: str) -> None:
    low = model_path.lower()
    if "qwen3" not in low and "timelens" not in low:
        raise ValueError(f"Only Qwen3-VL / TimeLens checkpoints are supported, got {model_path!r}.")


def get_model_class(model_path: str):
    _validate(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    _validate(model_path)
    return AutoProcessor
