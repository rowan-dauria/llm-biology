# llm-biology
code repo for mphil project using SAEs to understand LLM reasoning

## Development

The full `environment.yml` is for the HPC Linux/CUDA environment. On an Apple
Silicon Mac, use the smaller Mac environment file:

```bash
conda activate qwen-sae-mac
conda env update -f environment-mac.yml
pre-commit install
```

If you only need the git hooks in an already-working environment, install the
dev tools directly:

```bash
python -m pip install "pre-commit==4.5.1" "ruff==0.15.11" "pyright[nodejs]==1.1.408"
pre-commit install
```

Run the hooks across the repo with:

```bash
pre-commit run --all-files
```

## Biology server

Start the prompt-preview backend and bundled circuit-tracer wrapper UI:

```bash
python -m biology_server --port 8041 --graph-file-dir data/ui_graphs
```

The equivalent script wrapper is:

```bash
python scripts/serve_biology_server.py --port 8041
```

Executable entry points live in `scripts/`; for example:

```bash
python scripts/generate_attribution_graph.py --prompt "The biological function of hemoglobin is to"
python scripts/label_from_graph.py --slug <slug> --dry-run
python scripts/check_neuronpedia_exports.py --graph-dir data/ui_graphs
```

For Neuronpedia upload targets, `generate_attribution_graph.py` and
`python -m biology_server` accept `--scan`, `--feature-dir-name`,
`--feature-json-base-url`, `--neuronpedia-source-set`, and
`--neuronpedia-lorsa-source-set`. The checker is read-only and reports schema
errors without rewriting existing graph artifacts.
