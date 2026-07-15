"""TimeLens2 SFT recipe for Qwen3-VL-8B-Instruct."""

import json
from pathlib import Path

from xtuner.v1.config import AdamWConfig, FSDPConfig, LRConfig
from xtuner.v1.datasets import Qwen3VLTokenizeFnConfig
from xtuner.v1.datasets.config import DataloaderConfig, DatasetConfig
from xtuner.v1.loss import CELossConfig
from xtuner.v1.model import Qwen3VLDense8BConfig
from xtuner.v1.train import ResumeConfig, TrainerConfig


# Paths
root = Path(__file__).resolve().parents[1]
model_path = "Qwen/Qwen3-VL-8B-Instruct"
meta_data_path = root / "configs/data/timelens2_sft.json"
work_dir = str(root / "outputs/sft-8b")
cache_dir = str(root / ".cache/sft-8b")

# Training recipe
sample_max_length = 102400
pack_max_length = 102400
global_batch_size = 256
total_epoch = 1
hf_interval = 5000
hf_max_keep = 1
checkpoint_interval = 200
checkpoint_maxkeep = 1

lr = 5e-6
weight_decay = 0.0
warmup_ratio = 0.03
lr_min = 5e-7
recompute_ratio = 1.0
loss_reduction = "square"
dataloader_num_workers = 2


model_cfg = Qwen3VLDense8BConfig(freeze_vision=True, freeze_language=False)

ds_collections = json.loads(meta_data_path.read_text(encoding="utf-8"))
dataset_config = []
for name, data in ds_collections.items():
    anno_path = Path(data["anno_path"]).expanduser()
    if not anno_path.is_absolute():
        anno_path = (root / anno_path).resolve()

    media_root = data.get("media_root", "")
    if media_root and "://" not in media_root:
        media_path = Path(media_root).expanduser()
        if not media_path.is_absolute():
            media_root = str((root / media_path).resolve())

    dataset_config.append(
        {
            "dataset": DatasetConfig(
                name=name,
                anno_path=str(anno_path),
                media_root=media_root,
                sample_ratio=data.get("sample_ratio", 1.0),
                class_name="VLMJsonlDataset",
                cache_dir=cache_dir,
            ),
            "tokenize_fn": Qwen3VLTokenizeFnConfig(
                max_length=sample_max_length,
                processor_path=model_path,
                min_pixels=data.get("min_pixels", 4 * 32 * 32),
                max_pixels=data.get("max_pixels", 480 * 480),
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

dataloader_config = DataloaderConfig(
    dataset_config_list=dataset_config,
    pack_max_length=pack_max_length,
    collator="qwen3_vl_sft_collator",
    num_workers=dataloader_num_workers,
    pack_extra_buffer_size=20,
)

trainer = TrainerConfig(
    load_from=model_path,
    resume_cfg=ResumeConfig(auto_resume=True),
    tokenizer_path=model_path,
    fsdp_cfg=FSDPConfig(recompute_ratio=recompute_ratio, torch_compile=False),
    exp_tracker="tensorboard",
    model_cfg=model_cfg,
    optim_cfg=AdamWConfig(lr=lr, weight_decay=weight_decay, foreach=False),
    dataloader_cfg=dataloader_config,
    lr_cfg=LRConfig(lr_type="cosine", warmup_ratio=warmup_ratio, lr_min=lr_min),
    loss_cfg=CELossConfig(mode="chunk", chunk_size=1024, loss_reduction=loss_reduction),
    global_batch_size=global_batch_size,
    total_epoch=total_epoch,
    hf_interval=hf_interval,
    checkpoint_interval=checkpoint_interval,
    checkpoint_maxkeep=checkpoint_maxkeep,
    hf_max_keep=hf_max_keep,
    work_dir=work_dir,
)
