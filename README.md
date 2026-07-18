# TimeLens2: Generalist Video Temporal Grounding with Multimodal LLMs

> 🔎 **Find the moments that matter.** TimeLens2 turns natural-language queries
> into precise, traceable evidence intervals on the video timeline.

## 🔍 What is TimeLens2?

TimeLens2 is a generalist video temporal-grounding MLLM. Given a video and a natural-language description or question, it finds
**when the supporting visual evidence occurs** and returns one or more temporal
intervals. A single model handles short and long videos, single and repeated
events, descriptive and question-form queries, and both third-person and
egocentric footage through a unified generative interface.

TimeLens2 treats temporal evidence as a set of intervals throughout training.
Its supervised stage uses verified single- and multi-span annotations from
TimeLens2-93K, while its GRPO stage combines temporal IoU with a matching-free
temporal Wasserstein reward. We release the SFT,
GRPO, and evaluation code in this repository.

## 🧩 Repository Structure

The official training and evaluation code is organized around the three stages
used in the project:

| Module | Contents | Main entry points |
| --- | --- | --- |
| 🧠 `sft/` | XTuner-based supervised fine-tuning | `scripts/train_sft_4b.sh`, `scripts/train_sft_8b.sh` |
| 🎯 `grpo/` | Off-policy rollout and GRPO | `scripts/rollout_timelens2.sh`, `scripts/train_grpo_4b.sh`, `scripts/train_grpo_8b.sh` |
| 📊 `evaluation/` | VLMEvalKit with the TimeLens2 grounding entry | `scripts/srun_eval_all/run_grounding.sh` |

## 🤗 Released Resources

- 🤖 [TimeLens2-4B](https://huggingface.co/MCG-NJU/TimeLens2-4B)
- 🤖 [TimeLens2-8B](https://huggingface.co/MCG-NJU/TimeLens2-8B)
- 🎞️ [TimeLens2-93K](https://huggingface.co/datasets/MCG-NJU/TimeLens2-93K)

TimeLens2-93K contains 23,793 videos and 93,232 temporal grounding pairs. This
repository already includes the ready-to-use annotations and rollout data used
by the provided SFT and GRPO recipes. Video files are distributed separately
through the linked Hugging Face dataset and must be downloaded before training.
To use another framework for SFT or RL, download the public data from Hugging
Face and convert it to that framework's required format.

## 🚀 Reproduction Flow

1. Download and prepare videos for TimeLens2-93K, TimeLens-100K, and
   Ego4D-NLQ; the SFT and GRPO annotations are already included in this repository.
2. Run `sft/scripts/train_sft_4b.sh` or `sft/scripts/train_sft_8b.sh`.
3. Use the bundled rollout results directly, or regenerate them with
   `grpo/scripts/rollout_timelens2.sh`. Its single config includes both
   `timelens2-93k` and `timelens-100k`.
4. Run `train_grpo_4b.sh` or `train_grpo_8b.sh`. Both default to the bundled
   official rollout files for both sources, so the complete rollout need not be
   regenerated.
5. Evaluate the checkpoint with
   `evaluation/scripts/srun_eval_all/run_grounding.sh`.

Each module has a focused README with installation, data layout, environment
variables, and launch examples. Paths are configurable and may contain
environment variables.

## 📜 License and Acknowledgements

The project is released under the Apache License 2.0. The SFT, evaluation, and
GRPO modules contain code derived from [InternLM/xtuner](https://github.com/InternLM/xtuner),
[open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and
[TencentARC/TimeLens](https://github.com/TencentARC/TimeLens), respectively. We
thank their authors for open-sourcing these projects.
