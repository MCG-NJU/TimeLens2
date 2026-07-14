"""Argument dataclasses for GRPO training (model / data / GRPO)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="qwen3-vl-4b")
    model_name_or_path: Optional[str] = field(default=None)
    conv_type: Optional[str] = field(default="chatml")


@dataclass
class DataArguments:
    data_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a data config JSON (see timelens.data.sources)."},
    )

    # Generic filters (applied to every source).
    min_video_len: int = -1
    max_video_len: int = -1
    min_num_words: int = -1
    max_num_words: int = -1

    # Video frame budget (shared with the rollout stage to keep prompts consistent).
    min_tokens: int = 64
    total_tokens: int = 14336
    fps: float = 2.0
    fps_max_frames: Optional[int] = None

    sample_weight_std_power: float = field(
        default=2.0,
        metadata={
            "help": "Sample weight ~ (rollout_reward_std + eps) ** this. "
            "1.0 = linear in std; >1 emphasises high-disagreement samples.",
        },
    )
    sample_weight_mean_power: float = field(
        default=0.0,
        metadata={
            "help": "Sample weight also ~ (1 - rollout_reward_mean + eps) ** this. "
            "0 = disabled; >0 favours samples with lower rollout mean reward.",
        },
    )
    sample_weight_mean_spec: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward spec for rollout mean in sample weighting (e.g. 'tiou'). "
            "None = same as training reward spec.",
        },
    )


@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")

    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: Optional[str] = field(default=None)
    num_lora_modules: int = -1

    reward_funcs: str = field(
        default="tiou",
        metadata={
            "help": "Default reward spec when a source does not set its own 'reward'. "
            "Comma-separated names with optional ':weight' (see timelens.rewards.funcs).",
        },
    )
    reward_weights: Optional[list[float]] = field(default=None)
    scale_rewards: bool = field(default=True)
    loss_type: str = field(default="bnpo")
    beta: float = field(default=0.0)
    num_iterations: int = field(default=1)

    use_liger: bool = field(default=False)
    use_liger_loss: bool = field(default=False)
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: int = 512
    num_generations: int = 8

    save_train_rollouts: bool = field(
        default=False,
        metadata={
            "help": "Save all training rollouts (global_batch x num_generations per step) "
            "to output_dir/train_rollouts/step-XXXXXX.jsonl.",
        },
    )
    save_train_rollouts_every_n_steps: int = field(default=1)
