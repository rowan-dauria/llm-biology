#!/bin/bash
#!
#! SLURM job: label graph-surfaced unlabelled features then patch the graph JSON.
#! Driven by env vars (with defaults):
#!   SLUG       slug of the graph under data/ui_graphs/<slug>.json   (required)
#!   TOP_N      cap on features labelled this run                     (default: unset = no cap)
#!   ALPHA      target-effect vs centrality weighting in [0, 1]       (default: 0.5)
#!   PROVIDER   transformers | ollama | openai | anthropic            (default: transformers)
#!   MODEL      provider model override                                (default: provider's default)
#!

#! sbatch directives begin here ###############################
#SBATCH -J label_from_graph
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mail-type=NONE
#SBATCH -o /home/rd761/mphil-project/llm-biology/slurm_logs/label_from_graph-%j.out
#SBATCH -e /home/rd761/mphil-project/llm-biology/slurm_logs/label_from_graph-%j.err

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here

numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp

source /home/rd761/miniconda3/bin/activate
conda activate qwen-sae

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/rd761/circuit-tracer:${PYTHONPATH:-}"
export HF_HOME="/rds/user/rd761/hpc-work/huggingface"
export OMP_NUM_THREADS=1

HF_TOKEN_FILE="/home/rd761/mphil-project/hf_token"
if [ -r "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN
    HF_TOKEN="$(tr -d '[:space:]' < "$HF_TOKEN_FILE")"
fi

PROJECT_ROOT="/home/rd761/mphil-project/llm-biology"
LOG_DIR="$PROJECT_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"

if [ -z "${SLUG:-}" ]; then
    echo "ERROR: SLUG env var is required (e.g. sbatch --export=ALL,SLUG=my-graph scripts/run_label_from_graph.slurm)" >&2
    exit 1
fi

ALPHA="${ALPHA:-0.5}"
PROVIDER="${PROVIDER:-transformers}"

workdir="$SLURM_SUBMIT_DIR"
cd "$workdir"
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID  Slug: $SLUG  Provider: $PROVIDER  Alpha: $ALPHA  Top_n: ${TOP_N:-(none)}\n======"
echo "Time: $(date)"
echo "Running on master node: $(hostname)"
echo "Current directory: $(pwd)"

if [ "$SLURM_JOB_NODELIST" ]; then
    export NODEFILE
    NODEFILE=$(generate_pbs_nodefile)
    cat "$NODEFILE" | uniq > "$LOG_DIR/machine.file.$JOBID"
    echo -e "\nNodes allocated:\n================"
    sed -e 's/\..*$//g' < "$LOG_DIR/machine.file.$JOBID"
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

CMD=(python -u "$PROJECT_ROOT/scripts/label_from_graph.py" --slug "$SLUG" --alpha "$ALPHA" --provider "$PROVIDER")
if [ -n "${TOP_N:-}" ]; then
    CMD+=(--top_n "$TOP_N")
fi
if [ -n "${MODEL:-}" ]; then
    CMD+=(--model "$MODEL")
fi

echo "Running: ${CMD[*]}"
srun "${CMD[@]}"
