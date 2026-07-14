# Third-party components

TimeLens2 includes modified source from the following Apache-2.0 projects:

- [TimeLens](https://github.com/TencentARC/TimeLens), the starting point for
  `grpo/`; the working snapshot was `1de5da65a328249d9592e0e26b680dede5bfea00`.
- [XTuner](https://github.com/InternLM/xtuner), used by `sft/`. The release was
  prepared from the TimeLens2 working snapshot `b094ed85924b5e54f4555e849a42a0c04bca9d95`.
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), used by
  `evaluation/`. The release was prepared from the TimeLens2 working snapshot
  `d378c43c3400a360cf376d47ec2c80137d14a695`.
- [Qwen3-VL](https://huggingface.co/Qwen), used as the base model family.

The original copyright and license notices remain applicable. Files were
trimmed and adapted for the TimeLens2 data, long-video training, grounding
metrics, and public model names. See `LICENSE`, `NOTICE`, `sft/LICENSE`, and
`evaluation/LICENSE`.
