# llm-biology

Custom TransformerLens circuit-tracing code for the MPhil Qwen3-4B attribution
experiments. This repo is the submission codebase; generated data lives outside
it in the root project under `../data/llm-biology/`.

The pipeline uses pretrained Qwen3-4B transcoders from
`mwhanna/qwen3-4b-transcoders`. It only uses circuit-tracer's per-layer
`SingleLayerTranscoder` and `load_transcoder`; graph construction, attribution,
pruning, interventions, labels, and exports are project code.

## Layout

- `biology_server/`: TL-backed attribution runner plus a lightweight
  Neuronpedia-compatible graph viewer.
- `feature_lookup/`: top-K activation collection, window reconstruction,
  LLM labelling, and graph-label patching.
- `scripts/`: themed CSD3 wrappers and final report analysis scripts; see
  `scripts/README.md`.
- `data/neuronpedia-schemas/`: canonical frontend export schemas; do not edit
  them to fit new output.

## Environment

On CSD3, use the existing `qwen-sae` conda environment. It intentionally carries
the newer `transformers` / `huggingface_hub` pair needed by the Gemma labeller;
do not reinstall `circuit-tracer` or `transformer-lens` in that env just to
satisfy metadata constraints.

For local linting/dev hooks only:

```bash
python -m pip install "pre-commit" "ruff" "pyright[nodejs]"
pre-commit install
```

## Attribution Graphs

Generate a graph on CSD3 with the notebook-equivalent runner:

```bash
sbatch scripts/graphs/attribution_like_notebook.wilkes3
```

The direct Python entry point is:

```bash
python scripts/graphs/attribution_like_notebook.py \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --dir-name csd3_attribution_graphs
```

The default tracked layers are `2,12,24,33`. Use completion prompts by default;
pass `--chat-template` for refusal/chat cases.

## Feature Labels

Build top-K activation windows, then label and patch graph-surfaced features:

```bash
sbatch scripts/features/run_build_topk.slurm
sbatch scripts/features/run_label_features.slurm
SLUG=<graph-slug> sbatch scripts/features/run_label_from_graph.slurm
```

Top-K and label outputs are generated under `data/` at runtime and are ignored
by git. Existing project artifacts were moved to `../data/llm-biology/`.

## Graph Viewer

The server is viz-only: it serves existing graph JSONs, lets you upload an
exported graph, and saves frontend `qParams`. It does not run model inference.

```bash
python -m llm_biology.viewer \
  --port 8041 \
  --graph-file-dir ../data/llm-biology/ui_graphs
```

The viewer expects the circuit-tracer frontend assets in a sibling
`../circuit-tracer` checkout, or pass `--frontend-dir` explicitly.

## Report Analyses

Final-method wrappers:

```bash
sbatch scripts/evaluation/run_replacement_fidelity.slurm
sbatch scripts/interventions/sweep_supernode_interventions.wilkes3 <graph.json> Texas
sbatch scripts/interventions/bootstrap_random_supernode_baseline.wilkes3 <graph.json> Texas
sbatch scripts/interventions/steerable_ceiling.wilkes3 <graph.json> Texas
sbatch scripts/interventions/steer_supernode_top_logits.wilkes3 <graph.json> Texas
```

Plot helpers consume the JSON outputs:

```bash
python scripts/figures/plot_odds_vs_steering.py <baseline.json> <sweep.json>
python scripts/figures/plot_steering_top_logits.py <sweep.json>
python scripts/figures/plot_intervention_comparison.py <sweep-a.json> <sweep-b.json> --output out.png
```

## Heretic Refusal Runs

The Heretic changes are kept as a companion fork, not vendored here. See
`../docs/heretic-refusal.md` for the exact branch/commit, layer audit, and
reproduction notes.
The submission repo contains the graph/comparison wrappers:

```bash
sbatch scripts/refusal/attribution_like_notebook_base_refusal.wilkes3
sbatch scripts/refusal/attribution_like_notebook_heretic_trial114.wilkes3
sbatch scripts/refusal/compare_cross_model_feature_activations.wilkes3
```

## Tests

Do not run the full test suite on a laptop; several tests can load sizeable
models. Submit the CSD3 wrapper instead:

```bash
sbatch scripts/evaluation/run_tests.wilkes3
```

To narrow the run:

```bash
PYTEST_ARGS="tests/test_biology_server.py -q" sbatch scripts/evaluation/run_tests.wilkes3
```
