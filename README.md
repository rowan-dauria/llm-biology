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
