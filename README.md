# llm-biology
MPhil project using SAEs to understand LLM reasoning. Uses the [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) frontend and a custom attribution backend.

<img width="1728" height="753" alt="Screenshot 2026-05-28 at 16 13 35" src="https://github.com/user-attachments/assets/176a9fde-8446-45dd-83d8-21936b442670" />

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
```

## Causal interventions (CSD3 / Wilkes3)

Two SLURM wrappers steer transcoder features on the local replacement model for
a named `qParams` supernode in an exported attribution graph. Submit from the
repo on CSD3 (the wrappers `cd` into `llm-biology` and write output next to the
graph JSON unless `OUTPUT_JSON` is set). Paths are relative to `llm-biology/` or
absolute.

**Magnitude sweep** — steers the supernode's real constituents across
multiplicative factors (`m * clean_activation`) and records the target
logit/probability deltas:

```bash
sbatch scripts/sweep_supernode_interventions.wilkes3 <graph.json> "<Supernode label>"
# Optional env: MAGNITUDES='-1,0,1,2' OUTPUT_JSON=/path/out.json EXTRA_ARGS='--measure supernode'
```

**Random size-matched baseline** — the null for the sweep: draws `N` (default
100) random feature sets matching the supernode's `(layer, pos)` footprint and
size, sweeps each, and reports per-magnitude targeted-vs-baseline percentiles,
empirical p-value and z-score. Plot the targeted curve against the baseline band
(use the spread, not SEM):

```bash
sbatch scripts/bootstrap_random_supernode_baseline.wilkes3 <graph.json> "<Supernode label>"
# Optional env:
#   N_BOOTSTRAP=200          number of random draws (p-value resolution = 1/N)
#   SEED=0                    RNG seed for reproducible draws
#   MAGNITUDES='-1,0,1,2'     comma-separated steering factors
#   OUTPUT_JSON=/path/out.json
#   EXTRA_ARGS='--sampling global-active'   # or --match-magnitude-tol 1.0, --layers ...
```

Both wrappers default to `MAGNITUDES='-2,-1,0,0.5,1,1.5,...,8'` (0.5 steps over
0–8, plus −1 and −2); the supernode label
must match a `qParams.supernodes` entry exactly (case-insensitive). Run either
Python script directly with `--help` to see all options:

```bash
python scripts/bootstrap_random_supernode_baseline.py --help
```
