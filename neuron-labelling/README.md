# neuron-labelling

Feasibility study: label individual **MLP neurons** (not SAE or transcoder
features) of Qwen3-4B using the approach sketched in Goodfire's
[*Neural Networks Are Already Clean*](https://already-clean.goodfire.com/).
The essay doesn't publish its pipeline (a technical paper is forthcoming).
This is a minimal reimplementation of the parts it describes, for analysing
a handful of neurons.

This is a separate project from the `llm_biology` package in the repo root.
It reads raw SwiGLU neurons through Hugging Face `transformers` hooks and does
not use the transcoders or circuit-tracer.

## Method

Each Qwen3 MLP neuron computes `act = SiLU(gate) * value`. The pipeline:

1. **Collect** (`collect.py`): stream a corpus in fixed-length chunks and
   record `gate` and `value` for the chosen neurons at every token. Each
   neuron is projected out of the MLP input, so nothing of width
   `intermediate_size` is stored. Direct logit effects (`W_U · norm ·
   W_down[:, n]`) are saved too.
2. **Two poles, gate open only** (`examples.py`). The positive and negative
   poles are described separately. Only firings with `gate > 0` count. A closed
   gate times a negative value gives a small *positive* activation (the
   SwiGLU "leak"), which can fill a naive top-k list. Leaks are counted (share
   of a naive top-50, strongest leak ÷ pole max) and shown apart. For each
   pole, the evidence is:
   - the 20 strongest examples, one per chunk;
   - 8 weaker examples, sampled from 20–50% of max, to expose offshoots;
   - the strongest leaks.

   Contexts include tokens *after* the firing token, and every firing token in
   the window is scored 0–10.
3. **Steering** (`investigate.py`): greedy continuations of neutral prompts,
   with the neuron pushed to `±steer_scale × pole max`.
4. **Conjecture and test** (`investigate.py`). An investigator LLM sees the
   evidence and proposes a hypothesis per pole, with should-fire texts and
   near misses. These are run through the model and the scores are reported
   back. It revises for `--rounds` rounds, then writes a final explanation per
   pole, with a category (`one_job`, `two_related`, `several`, `unclear`),
   offshoots and confidence.
5. **Blind verification**. A fresh verifier sees *only* the explanation. It
   writes 5 examples, 5 paired near misses and one "blind maximum" text. The
   result is reported as hit rates (at `--strong` × corpus max, default 0.5),
   a win/tie/loss outcome, and the blind maximum as a fraction of the corpus
   maximum. This follows the essay's "explanations that can lose" and "blind
   maximum" tests.

Not implemented yet:
- clustering the attribution flowing into a neuron, so each bundle gets its
  own blind maximum;
- upstream and downstream neuron connections;
- using already-labelled neighbours as context;
- watching neurons form over training checkpoints;
- an agent that picks its own experiments.

## Install

From this directory, in the CSD3 `qwen-sae` environment or any Python 3.12+
environment with PyTorch:

```bash
python -m pip install -e ".[hf-corpus,dev]"
python -m pytest            # tiny random Qwen3; no downloads, GPU or network
```

The labeller calls any OpenAI-compatible chat-completions endpoint through
the standard library, so no SDK is needed:

```bash
export LLM_API_KEY=...                       # or OPENROUTER_API_KEY
export LLM_MODEL=<provider/model-slug>       # e.g. an OpenRouter model id
# optional; the default is OpenRouter (https://openrouter.ai/api/v1)
export LLM_API_BASE=https://generativelanguage.googleapis.com/v1beta/openai  # Gemini
```

## Run

```bash
# 1. activations for two random mid-network neurons (or --neurons L18/N1234,L24/N77)
python -m neuron_labelling.collect --random 2 --seed 0 \
  --n-chunks 4000 --seq-len 256 --output-dir outputs/run1

# 2. investigate, label and verify
python -m neuron_labelling.label outputs/run1 --rounds 2
```

On CSD3, submit from this directory with the repo's launcher:

```bash
sbatch -J neuron_collect -t 00:30:00 ../slurm/run_gpu.wilkes3 \
  python -u -m neuron_labelling.collect --random 2 --output-dir outputs/run1
```

Labelling needs the model (for tests and steering) *and* access to the LLM
API. If the GPU nodes can't reach the API, run `label` with `--device cpu` on
a node that can. It only runs a few dozen short texts.

`--text-file` (`.txt`: one document per line; `.jsonl`: a `text` field)
replaces the default streamed `HuggingFaceFW/fineweb-edu` `sample-10BT` corpus.

## Outputs

`<output-dir>/activations.npz`, `meta.json`, and under `labels/`:

- `L<l>_N<n>.json`: the evidence, steering samples, every test round, the
  final explanation, the verification, and the full investigator transcript;
- `L<l>_N<n>.md`: a short "neuron paper";
- `summary.md`: one row per pole.

`outputs/` is git-ignored.
