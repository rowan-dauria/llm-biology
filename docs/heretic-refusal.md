# Heretic Refusal Packaging

The refusal-model changes are packaged as an external companion fork rather than
vendored into `llm-biology`.

## Source

- Local fork: `../heretic-fork`
- Branch: `codex/heretic-touched-layers`
- Commit used for the current reproduction notes: `de4de5d`

The important project-specific changes are:

- Exclude the tracked transcoder MLP layers from Heretic MLP abliteration:
  `HERETIC_EXCLUDED_MLP_ABLITERATION_LAYERS="2,12,24,33"`.
- Keep attention changes allowed on all layers, because the attribution
  transcoders replace MLP outputs, not attention outputs.
- Disable Qwen thinking during the sweep: `HERETIC_ENABLE_THINKING=false`.
- Use the unquantized Qwen3-4B path (`quantization = "none"`).
- Record touched/excluded layers in trial metadata so the exported model can be
  audited against the attribution graph setup.

## Reproduction Shape

Run Heretic from the fork on CSD3 with `heretic-fork/slurm_qwen3_4b.sh`, export
the selected merged checkpoint, then point the llm-biology attribution wrapper at
that local model directory.

The current report-facing wrapper is:

```bash
sbatch scripts/refusal/attribution_like_notebook_heretic_trial114.wilkes3
```

That wrapper uses the merged Heretic checkpoint as `--model-id`, the base
`Qwen/Qwen3-4B` architecture as `--tl-model-id`, and the base tokenizer as
`--tokenizer-id`. The base-model comparison graph uses:

```bash
sbatch scripts/refusal/attribution_like_notebook_base_refusal.wilkes3
```

Then compare graph-surfaced feature activations with:

```bash
sbatch scripts/refusal/compare_cross_model_feature_activations.wilkes3
```

The generated graph JSONs and comparison outputs belong in the root project data
area (`../data/llm-biology/`), not in the submission repo.
