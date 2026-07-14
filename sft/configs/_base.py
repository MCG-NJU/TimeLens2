"""Shared XTuner configuration for the released TimeLens2 SFT runs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.datasets.mllm_tokenize_fn import OSSLoaderConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3VLDense4BConfig, Qwen3VLDense8BConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


ROOT = Path(__file__).resolve().parents[1]


def _read_data_config(path: str) -> dict:
    # Environment variables in the JSON keep the released config machine-independent.
    text = Path(path).expanduser().read_text(encoding="utf-8")
    return json.loads(os.path.expandvars(text))


def build_trainer(model_size: str) -> TrainerConfig:
    if model_size == "4b":
        model_cfg = Qwen3VLDense4BConfig(freeze_vision=True, freeze_language=False)
        default_model = "Qwen/Qwen3-VL-4B-Instruct"
    elif model_size == "8b":
        model_cfg = Qwen3VLDense8BConfig(freeze_vision=True, freeze_language=False)
        default_model = "Qwen/Qwen3-VL-8B-Instruct"
    else:
        raise ValueError(f"Unsupported model size: {model_size}")

    model_path = os.environ.get("MODEL_PATH", default_model)
    data_config_path = os.environ.get(
        "DATA_CONFIG",
        str(ROOT / "configs" / "data" / "timelens2_sft.json"),
    )
    work_dir = os.environ.get("WORK_DIR", str(ROOT / "outputs" / f"sft-{model_size}"))
    cache_dir = os.environ.get("DATASET_CACHE_DIR", str(ROOT / ".cache" / f"sft-{model_size}"))

    sample_max_length = int(os.environ.get("SAMPLE_MAX_LENGTH", "102400"))
    pack_max_length = int(os.environ.get("PACK_MAX_LENGTH", str(sample_max_length)))
    global_batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "256"))

    # Public Hugging Face data is local by default. Setting PETREL_CONF_PATH
    # restores the Petrel-backed I/O path used by the original training runs.
    petrel_conf_path = os.environ.get("PETREL_CONF_PATH")
    oss_loader_cfg = None
    if petrel_conf_path:
        oss_loader_cfg = OSSLoaderConfig(
            backend_kwargs={
                "conf_path": os.path.expanduser(os.path.expandvars(petrel_conf_path))
            }
        )

    dataset_config = []
    for name, data in _read_data_config(data_config_path).items():
        dataset_config.append(
            {
                "dataset": DatasetConfig(
                    name=name,
                    anno_path=data["anno_path"],
                    media_root=data.get("media_root", ""),
                    sample_ratio=data.get("sample_ratio", 1.0),
                    class_name="VLMJsonlDataset",
                    cache_dir=cache_dir,
                ),
                "tokenize_fn": Qwen3VLTokenizeFnConfig(
                    max_length=sample_max_length,
                    processor_path=model_path,
                    min_pixels=data.get("min_pixels", 4 * 32 * 32),
                    max_pixels=data.get("max_pixels", 480 * 480),
                    oss_loader_cfg=oss_loader_cfg,
                    video_min_total_pixels=data.get("video_min_total_pixels", 2 * 32 * 32),
                    video_max_total_pixels=data.get(
                        "video_max_total_pixels", int(sample_max_length * 0.8 * 2 * 32 * 32)
                    ),
                    video_min_frames=data.get("video_min_frames", 1),
                    video_max_frames=data.get("video_max_frames", 2048),
                    fps=data.get("fps", 2),
                    rand_video_max_frames=data.get("rand_video_max_frames", 32),
                    system_message=data.get("system_message"),
                    hash=data.get("hash"),
                ),
            }
        )

    return TrainerConfig(
        load_from=model_path,
        resume_cfg=ResumeConfig(auto_resume=True),
        tokenizer_path=model_path,
        fsdp_cfg=FSDPConfig(recompute_ratio=1.0, torch_compile=False),
        exp_tracker="tensorboard",
        model_cfg=model_cfg,
        optim_cfg=AdamWConfig(lr=5e-6, weight_decay=0.0, foreach=False),
        dataloader_cfg=DataloaderConfig(
            dataset_config_list=dataset_config,
            pack_max_length=pack_max_length,
            collator="qwen3_vl_sft_collator",
            num_workers=int(os.environ.get("DATALOADER_NUM_WORKERS", "2")),
            pack_extra_buffer_size=20,
        ),
        lr_cfg=LRConfig(lr_type="cosine", warmup_ratio=0.03, lr_min=5e-7),
        loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction="square"),
        global_batch_size=global_batch_size,
        total_epoch=int(os.environ.get("EPOCHS", "1")),
        hf_interval=int(os.environ.get("HF_INTERVAL", "5000")),
        checkpoint_interval=int(os.environ.get("CHECKPOINT_INTERVAL", "200")),
        checkpoint_maxkeep=1,
        hf_max_keep=1,
        work_dir=work_dir,
    )
