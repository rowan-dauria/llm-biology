# llm-biology

Custom TransformerLens circuit-tracing code for the MPhil Qwen3-4B attribution
experiments. This repo is the submission codebase; generated data lives outside
git, usually under CSD3 RDS/HPC storage.

The pipeline uses pretrained Qwen3-4B transcoders from
`mwhanna/qwen3-4b-transcoders`. It only uses circuit-tracer's per-layer
`SingleLayerTranscoder` and `load_transcoder`; graph construction, attribution,
pruning, interventions, labels, and exports are project code.

## Layout

- `llm_biology/attribution/`: attribution runner, graph export, and
  TransformerLens attribution helpers.
- `llm_biology/model/`: replacement-model loading, freeze, and forward hooks.
- `llm_biology/interventions/`: feature steering sweeps, random baselines, and
  steerable-ceiling diagnostics.
- `llm_biology/features/`: top-K activation windows, LLM labelling, graph target
  selection, and graph-label patching.
- `llm_biology/evaluation/`: replacement-fidelity (KL / delta-CE) measurement.
- `llm_biology/refusal/`: base-vs-Heretic feature panel and cross-model
  activation comparison tools.
- `llm_biology/viewer/`: viz-only Neuronpedia-compatible local graph viewer.
- `slurm/run_gpu.wilkes3`: generic CSD3 GPU launcher.
- `EXPERIMENTS.md`: exact replacements for the retired experiment wrappers.
- `data/neuronpedia-schemas/`: canonical frontend export schemas; do not edit
  them to fit new output.

Report-figure plotting and table-generation scripts are not part of this
submission codebase; they live in the parent project at `scripts/figures/`
(outside this repo) and read the JSON/CSV files this package writes.

## Environment

On CSD3, use the existing `qwen-sae` conda environment. It intentionally carries
the newer `transformers` / `huggingface_hub` pair needed by the Gemma labeller;
do not reinstall `circuit-tracer` or `transformer-lens` in that env just to
satisfy metadata constraints.

Install this package into that existing env without dependency resolution:

```bash
uv pip install -e . --no-deps
```

For local linting/dev hooks only:

```bash
python -m pip install "pre-commit" "ruff" "pyright[nodejs]"
pre-commit install
```

## Attribution Graphs

Generate a graph directly:

```bash
python -m llm_biology.attribution.run \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --dir-name csd3_attribution_graphs
```

By default outputs go under `./outputs`. Set `LLM_BIOLOGY_OUTPUT_ROOT` or pass
`--output-root /home/rd761/rds/hpc-work` for CSD3 RDS storage.

The default tracked layers are `2,12,24,33`. Use completion prompts by default;
pass `--chat-template` for refusal/chat cases.

## Feature Labels

Build top-K activation windows, label sampled features, and patch graph-surfaced
features:

```bash
python -m llm_biology.features.build_topk
python -m llm_biology.features.label_features --layer 2 --topk-dir data/feature_topk
python -m llm_biology.features.label_from_graph --slug <graph-slug>
```

Top-K and label outputs are generated under `data/` at runtime and are ignored
by git.

## Graph Viewer

The server is viz-only: it serves existing graph JSONs, lets you upload an
exported graph, and saves frontend `qParams`. It does not run model inference.

```bash
python -m llm_biology.viewer \
  --port 8041 \
  --graph-file-dir data/ui_graphs
```

The viewer expects the circuit-tracer frontend assets in a sibling
`../circuit-tracer` checkout, or pass `--frontend-dir` explicitly.

## Report Analyses

Run analysis helpers as package modules:

```bash
python -m llm_biology.evaluation.measure_replacement_fidelity
python -m llm_biology.interventions.sweep <graph.json> Texas
python -m llm_biology.interventions.bootstrap_random_supernode_baseline <graph.json> Texas
python -m llm_biology.interventions.steerable_ceiling <graph.json> Texas
python -m llm_biology.refusal.build_feature_panel_from_graph <graph.json> --direction base_to_jailbroken --output panel.json
python -m llm_biology.refusal.compare_cross_model_feature_activations panel.json --comparison-model-id Qwen/Qwen3-4B
```

Plot helpers consume the JSON/CSV files written by the analyses above. They
live outside this repo, in the parent project's `scripts/figures/` (run from
the parent project root):

```bash
python scripts/figures/plot_odds_vs_steering.py <baseline.json> <sweep.json>
python scripts/figures/plot_steering_top_logits.py <sweep.json>
python scripts/figures/plot_intervention_comparison.py <sweep-a.json> <sweep-b.json> --output out.png
python scripts/figures/plot_cross_model_feature_fate.py <forward.csv> <reverse.csv>
python scripts/figures/plot_cross_model_feature_fate_unsupervised.py <forward.csv> <reverse.csv>
```

The first three read the `.json` files from `sweep`/`bootstrap_random_supernode_baseline`;
the last two read the `.csv` (not `.json`) sibling written by
`compare_cross_model_feature_activations`.

## SLURM

Submit GPU jobs through the generic launcher:

```bash
sbatch -J attr_like_nb -t 0:15:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.attribution.run \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --dir-name csd3_attribution_graphs \
  --output-root /home/rd761/rds/hpc-work
```

See `EXPERIMENTS.md` for the exact replacements for the retired wrappers,
including refusal runs and cross-model comparison defaults.

## Tests

Do not run the full test suite on a laptop; several tests can load sizeable
models. On CSD3:

```bash
sbatch -J llm_bio_tests -t 01:00:00 slurm/run_gpu.wilkes3 \
  python -m pytest tests -q
```

For a CPU-safe local subset:

```bash
python -m pytest \
  tests/test_labels.py \
  tests/test_graph_targets.py \
  tests/test_patch_graph_labels.py \
  tests/test_circuit_graph_export.py \
  tests/test_biology_server.py \
  -q
```

## Heretic Refusal Runs

The Heretic (abliteration) changes are kept as a companion fork rather than
vendored here: [`rowan-dauria/heretic`](https://github.com/rowan-dauria/heretic)
(package `heretic-llm`), branch `codex/heretic-touched-layers`, pinned at
commit `de4de5d` — install with the optional `heretic` dependency group
(`pip install ".[heretic]"`). The methodologically important change on that
fork excludes the tracked transcoder MLP layers
(`HERETIC_EXCLUDED_MLP_ABLITERATION_LAYERS="2,12,24,33"`) from abliteration,
so the same transcoder basis stays valid on the abliterated model. The
reported results use Optuna trial `113` (KL divergence `0.081`, `3/100`
refusals vs. `97/100` base-model refusals).
