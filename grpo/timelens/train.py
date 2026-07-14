"""GRPO training entry for Qwen3-VL temporal grounding (and general MLLM RL)."""

from __future__ import annotations

import ast
import os
import pathlib
import shutil
import sys

WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, HfArgumentParser

from timelens.config import DataArguments, GRPOArguments, ModelArguments
from timelens.data import HybridDataset
from timelens.model import get_config_class, get_model_class, get_processor_class
from timelens.rewards import make_reward_func
from timelens.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    print_trainable_parameters,
    safe_save_model_for_hf_trainer,
)
from timelens.trainer import QwenvlGRPOTrainer

local_rank = None


def rank0_print(*args):
    if local_rank in (0, "0", None):
        print(*args)


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    lora_namespan_exclude = lora_namespan_exclude or []
    lora_module_names = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            lora_module_names.append(name)
    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(model.visual.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not training_args.freeze_merger)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    if not training_args.lora_enable:
        assert not training_args.vision_lora, "vision_lora requires lora_enable."
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
    else:
        training_args.lora_namespan_exclude = []
    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16 if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_args = {}
    if training_args.bits in (4, 8):
        bnb_args.update(
            dict(
                device_map={"": training_args.device},
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["visual"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    model_cls = get_model_class(model_args.model_name_or_path)
    processor_cls = get_processor_class(model_args.model_name_or_path)
    config_cls = get_config_class(model_args.model_name_or_path)

    config = config_cls.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = model_cls.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation="sdpa" if training_args.disable_flash_attn2 else "flash_attention_2",
        trust_remote_code=True,
        **bnb_args,
    )

    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    if training_args.bits in (4, 8):
        model.config.torch_dtype = compute_dtype
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": True},
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    peft_config = None
    if training_args.lora_enable:
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(
                model,
                lora_namespan_exclude=training_args.lora_namespan_exclude,
                num_lora_modules=training_args.num_lora_modules,
            ),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
        )
        if training_args.bits == 16:
            model.to(torch.bfloat16 if training_args.bf16 else torch.float16)

    if training_args.local_rank in (0, -1):
        print_trainable_parameters(model, training_args=training_args)

    processor = processor_cls.from_pretrained(
        model_args.model_name_or_path,
        do_resize=False,
        padding_side="left",
        trust_remote_code=True,
    )
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.pad_token = processor.tokenizer.pad_token

    reward_funcs = [make_reward_func(training_args.reward_funcs)]

    trainer = QwenvlGRPOTrainer(
        model=model,
        processing_class=processor,
        train_dataset=HybridDataset(
            processor, model.config, model_args, data_args, training_args
        ),
        reward_funcs=reward_funcs,
        args=training_args,
        peft_config=peft_config,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    if local_rank in (0, -1):
        final_step = int(trainer.state.global_step)
        dup_ckpt = os.path.join(training_args.output_dir, f"checkpoint-{final_step}")
        if os.path.isdir(dup_ckpt):
            print(f"Removing duplicate checkpoint (same step as final save): {dup_ckpt}")
            shutil.rmtree(dup_ckpt, ignore_errors=True)


if __name__ == "__main__":
    train()
