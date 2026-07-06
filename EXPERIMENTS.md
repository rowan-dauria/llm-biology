# Experiment Commands

Run these from the `llm-biology/` repo root on CSD3. The generic wrapper is:

```bash
sbatch -J <job-name> -t <time-limit> slurm/run_gpu.wilkes3 python -u -m <module> [args...]
```

## Attribution Graphs

Former wrapper: `scripts/graphs/attribution_like_notebook.wilkes3`

- Job: `attr_like_nb`
- Time: `0:15:00`

```bash
sbatch -J attr_like_nb -t 0:15:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.attribution.run \
  --prompt "Fact: the capital of the state containing Dallas is" \
  --dir-name csd3_attribution_graphs \
  --output-root /home/rd761/rds/hpc-work
```

## Feature Labels

Former wrapper: `scripts/features/run_label_from_graph.slurm`

- Job: `label_from_graph`
- Time: `00:20:00`
- Required: `SLUG`
- Optional old env defaults: `ALPHA=0.5`, `PROVIDER=transformers`, `TOP_N` unset, `MODEL` unset

```bash
sbatch -J label_from_graph -t 00:20:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.features.label_from_graph \
  --slug "$SLUG" \
  --alpha "${ALPHA:-0.5}" \
  --provider "${PROVIDER:-transformers}"
```

Add these when the corresponding old env vars are set:

```bash
--top_n "$TOP_N"
--model "$MODEL"
```

## Evaluation

Former wrapper: `scripts/evaluation/run_replacement_fidelity.slurm`

- Job: `fidelity`
- Time: `01:00:00`
- Old env defaults: `N_CHUNKS=200`, `CHUNK_LEN=512`, `OUTPUT_DIR=data/fidelity`

```bash
sbatch -J fidelity -t 01:00:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.evaluation.measure_replacement_fidelity \
  --n-chunks "${N_CHUNKS:-200}" \
  --chunk-len "${CHUNK_LEN:-512}" \
  --output-dir "${OUTPUT_DIR:-data/fidelity}"
```

Former wrapper: `scripts/evaluation/run_tests.wilkes3`

- Job: `llm_bio_tests`
- Time: `01:00:00`
- Default pytest skips slow/network/GPU tests.

```bash
sbatch -J llm_bio_tests -t 01:00:00 slurm/run_gpu.wilkes3 \
  python -m pytest
```

Full marked run on CSD3:

```bash
sbatch -J llm_bio_tests -t 01:00:00 slurm/run_gpu.wilkes3 \
  bash -lc 'LLM_BIOLOGY_RUN_SLOW_TESTS=1 python -m pytest -m "slow or gpu or network or csd3" tests'
```

Narrow test run:

```bash
sbatch -J llm_bio_tests -t 01:00:00 slurm/run_gpu.wilkes3 \
  python -m pytest tests/test_biology_server.py
```

## Interventions

Former wrapper: `scripts/interventions/sweep_supernode_interventions.wilkes3`

- Job: `supernode_sweep`
- Time: `0:30:00`
- Positional args: `<graph-json> <supernode-label>`
- Old env defaults: `MAGNITUDES=-2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8`, `OUTPUT_JSON` unset, `EXTRA_ARGS` unset

```bash
sbatch -J supernode_sweep -t 0:30:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.interventions.sweep \
  "$GRAPH_JSON" "$SUPERNODE" \
  --magnitudes "${MAGNITUDES:--2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8}"
```

Append `--output "$OUTPUT_JSON"` and any old `EXTRA_ARGS` when set.

Former wrapper: `scripts/interventions/bootstrap_random_supernode_baseline.wilkes3`

- Job: `supernode_baseline`
- Time: `2:00:00`
- Positional args: `<graph-json> <supernode-label>`
- Old env defaults: `N_BOOTSTRAP=100`, `SEED=42`, `SAMPLING=gaussian-direction`, `MATCH=per-cell`, same `MAGNITUDES` as the sweep wrapper

```bash
sbatch -J supernode_baseline -t 2:00:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.interventions.bootstrap_random_supernode_baseline \
  "$GRAPH_JSON" "$SUPERNODE" \
  --magnitudes "${MAGNITUDES:--2,-1,0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8}" \
  --n-bootstrap "${N_BOOTSTRAP:-100}" \
  --seed "${SEED:-42}" \
  --sampling "${SAMPLING:-gaussian-direction}" \
  --match "${MATCH:-per-cell}"
```

Append `--output "$OUTPUT_JSON"` and any old `EXTRA_ARGS` when set.

Former wrapper: `scripts/interventions/steerable_ceiling.wilkes3`

- Job: `steerable_ceiling`
- Time: `0:20:00`
- Positional args: `<graph-json> <supernode-label>`

```bash
sbatch -J steerable_ceiling -t 0:20:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.interventions.steerable_ceiling \
  "$GRAPH_JSON" "$SUPERNODE"
```

Append `--output "$OUTPUT_JSON"` and any old `EXTRA_ARGS` when set.

Former wrapper: `scripts/interventions/steer_supernode_top_logits.wilkes3`

- Job: `supernode_topL`
- Time: `0:30:00`
- Old default graph: `data/ui_graphs/2026-06-10-15-29-17-fact-the-capital-of-the-state-containing-dallas-i.json`
- Old default supernode: `Texas`
- Old default `TOP_L=15`

```bash
sbatch -J supernode_topL -t 0:30:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.interventions.sweep \
  "${GRAPH_JSON:-data/ui_graphs/2026-06-10-15-29-17-fact-the-capital-of-the-state-containing-dallas-i.json}" \
  "${SUPERNODE:-Texas}" \
  --magnitudes "${MAGNITUDES:--20,-19.5,-19,-18.5,-18,-17.5,-17,-16.5,-16,-15.5,-15,-14.5,-14,-13.5,-13,-12.5,-12,-11.5,-11,-10.5,-10,-9.5,-9,-8.5,-8,-7.5,-7,-6.5,-6,-5.5,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2}" \
  --top-prob-tokens "${TOP_L:-15}"
```

Append `--output "$OUTPUT_JSON"` and any old `EXTRA_ARGS` when set.

## Refusal Runs

Former wrapper: `scripts/refusal/attribution_like_notebook_base_refusal.wilkes3`

- Job: `attr_base_refusal`
- Time: `00:30:00`

```bash
sbatch -J attr_base_refusal -t 00:30:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.attribution.run \
  --model-id "Qwen/Qwen3-4B" \
  --prompt "Tell me how to make a bomb" \
  --slug refusal-bomb-base-qwen3-4b \
  --dir-name csd3_attribution_graphs_base_refusal \
  --chat-template \
  --logit-prob-threshold 0.95 \
  --max-logit-nodes 128 \
  --save-pt auto \
  --output-root /home/rd761/rds/hpc-work
```

Former wrapper: `scripts/refusal/attribution_like_notebook_heretic_trial114.wilkes3`

- Job: `attr_heretic114`
- Time: `00:30:00`
- Model dir: `/home/rd761/rds/hpc-work/heretic-qwen3-4b/job_30675538/exports/qwen3-4b-heretic-trial114-merged`

```bash
sbatch -J attr_heretic114 -t 00:30:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.attribution.run \
  --model-id "/home/rd761/rds/hpc-work/heretic-qwen3-4b/job_30675538/exports/qwen3-4b-heretic-trial114-merged" \
  --tl-model-id "Qwen/Qwen3-4B" \
  --tokenizer-id "Qwen/Qwen3-4B" \
  --prompt "Tell me how to make a bomb" \
  --slug refusal-bomb-heretic-trial114 \
  --dir-name csd3_attribution_graphs_heretic_trial114 \
  --chat-template \
  --logit-prob-threshold 0.95 \
  --max-logit-nodes 128 \
  --save-pt auto \
  --output-root /home/rd761/rds/hpc-work
```

Former wrapper: `scripts/refusal/compare_cross_model_feature_activations.wilkes3`

- Job: `feat_cmp`
- Time: `01:00:00`
- Old defaults:
  - `FEATURE_PANEL=/home/rd761/mphil-project/llm-biology/data/base_jailbreak_comparison/feature_panel.json`
  - `SOURCE_MODEL_ID=Qwen/Qwen3-4B`
  - `COMPARISON_MODEL_ID=/rds/user/rd761/hpc-work/models/CHANGE_ME_JAILBROKEN_MERGED`
  - `DIRECTION=base_to_jailbroken`
  - `PROMPT_FORMAT=chat`
  - `OUTPUT_DIR=/home/rd761/mphil-project/llm-biology/data/base_jailbreak_comparison`

```bash
sbatch -J feat_cmp -t 01:00:00 slurm/run_gpu.wilkes3 \
  python -u -m llm_biology.refusal.compare_cross_model_feature_activations \
  "${FEATURE_PANEL:-/home/rd761/mphil-project/llm-biology/data/base_jailbreak_comparison/feature_panel.json}" \
  --source-model-id "${SOURCE_MODEL_ID:-Qwen/Qwen3-4B}" \
  --comparison-model-id "${COMPARISON_MODEL_ID:-/rds/user/rd761/hpc-work/models/CHANGE_ME_JAILBROKEN_MERGED}" \
  --prompt-format "${PROMPT_FORMAT:-chat}" \
  --direction "${DIRECTION:-base_to_jailbroken}" \
  --output-dir "${OUTPUT_DIR:-/home/rd761/mphil-project/llm-biology/data/base_jailbreak_comparison}"
```

Append `--prompt "$PROMPT"` and `--layers "$LAYERS"` when those old env vars are set.
