# ⏳ TimeLens2

**TimeLens2: Generalist Video Temporal Grounding with Multimodal LLMs**

> 🔎 **Find the moments that matter.** TimeLens2 turns natural-language queries
> into precise, traceable evidence intervals on the video timeline.

Yuhan Zhu<sup>&ast;1,2</sup>, Changlian Ma<sup>&ast;1,2</sup>,
Xiangyu Zeng<sup>&ast;1,2</sup>, Xinhao Li<sup>&ast;1</sup>,
Zhiqiu Zhang<sup>&ast;3,2</sup>, Jun Zhang<sup>1</sup>,
Ziang Yan<sup>4,2</sup>, Zikang Wang<sup>3,2</sup>,
Xinyu Chen<sup>1,2</sup>, Haoran Chen<sup>1,2</sup>,
Shaowei Zhang<sup>3,2</sup>, Limin Wang<sup>1,2,&dagger;</sup>

<sup>1</sup>Nanjing University &nbsp;
<sup>2</sup>Shanghai AI Laboratory &nbsp;
<sup>3</sup>Shanghai Jiao Tong University &nbsp;
<sup>4</sup>Zhejiang University

<sup>&ast;</sup>Equal contribution. <sup>&dagger;</sup>Corresponding author.

## 🔍 What is TimeLens2?

TimeLens2 is a generalist video temporal-grounding multimodal large language
model. Given a video and a natural-language description or question, it finds
**when the supporting visual evidence occurs** and returns one or more temporal
intervals. A single model handles short and long videos, single and repeated
events, descriptive and question-form queries, and both third-person and
egocentric footage through a unified generative interface.

At a glance:

- 🎬 **Flexible inputs:** short or long videos paired with descriptions or questions.
- ⏱️ **Grounded outputs:** one or more temporal intervals containing the visual evidence.
- 🌍 **Generalist capability:** single and repeated events, third-person footage,
  egocentric videos, and diverse query forms share one interface.
- 🧠 **Set-aware learning:** supervision and reinforcement learning both treat the
  answer as an interval set instead of an isolated timestamp string.

TimeLens2 treats temporal evidence as a set of intervals throughout training.
Its supervised stage uses verified single- and multi-span annotations from
TimeLens2-93K, while its GRPO stage combines temporal IoU with a matching-free
temporal Wasserstein reward. This provides useful optimization signals even for
non-overlapping predictions and naturally supports unequal numbers of predicted
and reference intervals. We release 4B and 8B checkpoints together with the SFT,
rollout, GRPO, and evaluation code in this repository.

## 🧩 Repository Structure

The official training and evaluation code is organized around the three stages
used in the project:

| Module | Contents | Main entry points |
| --- | --- | --- |
| 🧠 `sft/` | XTuner-based supervised fine-tuning | `scripts/train_sft_4b.sh`, `scripts/train_sft_8b.sh` |
| 🎯 `grpo/` | Off-policy rollout and GRPO | `scripts/rollout_timelens2_93k.sh`, `scripts/rollout_timelens_100k.sh`, `scripts/train_grpo_4b.sh`, `scripts/train_grpo_8b.sh` |
| 📊 `evaluation/` | VLMEvalKit with the TimeLens2 grounding entry | `scripts/srun_eval_all/run_grounding.sh` |

The runtime implementations are kept intact: the XTuner, TimeLens2 GRPO, and
VLMEvalKit packages retain their training/evaluation logic, acceleration paths,
tests, CI, dependency declarations, and documentation. The release surface only
removes historical experiment recipes, ablation launchers, generated artifacts,
and internal infrastructure wrappers. The remaining scripts correspond exactly
to the official SFT, rollout, GRPO, and evaluation flow.

## 🤗 Released Resources

- 🗂️ [TimeLens2 collection](https://huggingface.co/collections/MCG-NJU/timelens2)
- 🤖 [TimeLens2-4B](https://huggingface.co/MCG-NJU/TimeLens2-4B)
- 🤖 [TimeLens2-8B](https://huggingface.co/MCG-NJU/TimeLens2-8B)
- 🎞️ [TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K)
- 🎞️ [TimeLens-100K](https://huggingface.co/datasets/MCG-NJU/TimeLens-100K)

TimeLens2-93K contains 23,793 videos and 93,232 temporal grounding pairs. The
repository provides converters for its public conversation and raw-annotation
files under `sft/tools/` and `grpo/tools/`.

The final SFT mixture and the optional second rollout source also use the
Gemini-refined TimeLens-100K data. Its rollout script and data config are kept
separate from TimeLens2-93K so either source can be reproduced independently.

## 🚀 Reproduction Flow

1. Download and prepare TimeLens2-93K plus the other data sources used by SFT.
2. Run `sft/scripts/train_sft_4b.sh` or `sft/scripts/train_sft_8b.sh`.
3. Generate eight off-policy rollouts per sample with the corresponding script
   under `grpo/scripts/`.
4. Pass the emitted `train_data_config.json` to `train_grpo_4b.sh` or
   `train_grpo_8b.sh`.
5. Evaluate the released or local checkpoint with
   `evaluation/scripts/srun_eval_all/run_grounding.sh`.

Each module has a focused README with installation, data layout, environment
variables, and launch examples. Paths are configurable and may contain
environment variables; no PJLab-specific filesystem layout is required.

## ⚙️ Final Recipe at a Glance

SFT uses a 102,400-token packing budget, global batch size 256, one epoch,
learning rate `5e-6`, a frozen vision encoder, and square loss reduction. GRPO
uses 8 generations, 16K total tokens, up to 512 frames at 2 FPS, global batch
size 64, learning rate `1e-6`, KL coefficient `0.04`, and the
`twass1,tiou,parse_penalty` reward combination.

## 📜 License and Acknowledgements

The project is released under the Apache License 2.0. The SFT and evaluation
modules contain code derived from XTuner and VLMEvalKit respectively; see
`THIRD_PARTY.md` and the retained license files in each module.
