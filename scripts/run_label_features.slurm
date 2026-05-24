#!/bin/bash
#!
#! SLURM array job: label features per layer with Gemma 4 E4B in transformers.
#! Each array task picks one layer from LAYERS, runs label_features.py against
#! the corresponding topk_layer_*.pt, and writes data/feature_labels/layer_<L>.jsonl.
#!

#! sbatch directives begin here ###############################
#SBATCH -J label_feats
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --mail-type=NONE
#SBATCH -o /home/rd761/mphil-project/llm-biology/slurm_logs/label_feats-%A_%a.out
#SBATCH -e /home/rd761/mphil-project/llm-biology/slurm_logs/label_feats-%A_%a.err

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
#! circuit-tracer is an editable install in the qwen-sae env (-> mphil-project/circuit-tracer).
#! Do NOT add the old /home/rd761/circuit-tracer checkout to PYTHONPATH; it shadows the
#! editable install with a pre-rename copy and breaks `from ... import load_transcoder`.
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

LAYERS=(2 12 24 33)
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$LAYER" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no LAYERS entry" >&2
    exit 1
fi

workdir="$SLURM_SUBMIT_DIR"
cd "$workdir"
echo -e "Changed directory to $(pwd).\n"

JOBID=$SLURM_JOB_ID
ARRAY_JOB="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo -e "JobID: $JOBID  Array: $ARRAY_JOB  Layer: $LAYER\n======"
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

srun python -u "$PROJECT_ROOT/feature_lookup/label_features.py" \
    --layer "$LAYER" \
    --provider transformers \
    --model google/gemma-4-E4B-it \
    --concurrency 8
