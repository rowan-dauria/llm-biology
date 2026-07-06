# AGENTS.md

Guidance for AI coding agents working inside the `llm-biology` submission repo.

## Scope

This repo is the submitted code artifact for the MPhil project. Keep generated
data, model checkpoints, report figures, notebooks, and one-off analysis outputs
outside this repo unless the user explicitly asks to vendor a small source
artifact.

## Circuit-Tracing Constraints

- Use pretrained Qwen3-4B transcoders from `mwhanna/qwen3-4b-transcoders`.
- Use only `circuit_tracer.transcoder.single_layer_transcoder.SingleLayerTranscoder`
  and `load_transcoder` from `circuit-tracer`.
- Do not use `load_transcoder_set`, `TranscoderSet`, `attribute`,
  `ReplacementModel`, or other high-level circuit-tracer APIs.
- Keep attribution, graph construction, pruning, intervention, labelling, and
  export logic in project code.

## Dependencies

- Keep `pyproject.toml` as the source of truth for Python package dependencies.
- `environment.yml` is only a minimal CSD3 CUDA/PyTorch base environment.
- Do not reintroduce a local Mac conda environment file.
- Avoid dependencies used only by report plotting, notebooks, or historical
  experiments.
- After dependency changes, run `python -m pip check` in a clean environment
  when possible.

## Tests

- The default `python -m pytest` path must remain laptop-safe and skip large
  model downloads, network calls, and GPU-only checks.
- Mark expensive tests with `slow`, `network`, `gpu`, or `csd3` in
  `pyproject.toml`.
- Run full model/GPU coverage through `slurm/run_gpu.wilkes3` on CSD3.

## SLURM

Use `slurm/run_gpu.wilkes3` for submission-repo GPU jobs. It is based on the
project's required Wilkes3 template and includes the CSD3 account, ampere
partition, `qwen-sae` activation, and `LD_LIBRARY_PATH` fix.
