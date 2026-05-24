#!/bin/bash
#!
#! SLURM job script: per-feature top-K activating windows over the Pile.
#! Runs feature_lookup/build_topk.py for the 4 transcoder layers
#! [2, 12, 24, 33] in a single forward pass over the corpus.
#!

#! sbatch directives begin here ###############################
#SBATCH -J build_topk
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=NONE
#SBATCH -o /home/rd761/mphil-project/llm-biology/slurm_logs/build_topk-%j.out
#SBATCH -e /home/rd761/mphil-project/llm-biology/slurm_logs/build_topk-%j.err

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

PROJECT_ROOT="/home/rd761/mphil-project/llm-biology"
LOG_DIR="$PROJECT_ROOT/slurm_logs"
mkdir -p "$LOG_DIR"
#! ":all" streams the entire split. Set a number (e.g. ":10000") to cap docs per worker.
#! NOTE: results are saved ONLY after the full pass finishes (no checkpointing), so a
#! timeout loses everything. At ~27 docs/s, 150k docs ~= 93 min loop + ~2 min setup/save,
#! leaving ~25 min headroom under the 2h walltime. Raise this only if you also raise --time
#! or add incremental checkpointing.
CORPUS_SPEC="hf:monology/pile-uncopyrighted:train:150000"

workdir="$SLURM_SUBMIT_DIR"
cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID
echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > "$LOG_DIR/machine.file.$JOBID"
        echo -e "\nNodes allocated:\n================"
        echo `cat "$LOG_DIR/machine.file.$JOBID" | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

WORKERS="${WORKERS:-1}"
echo "Running with WORKERS=$WORKERS"

#! batch_size: tune up until VRAM is full (single A100 80GB has plenty of headroom).
#! prefetch: pre-tokenise batches on a background thread so the GPU never waits on ingest.
srun python -u "$PROJECT_ROOT/feature_lookup/build_topk.py" \
    --corpus_spec "$CORPUS_SPEC" \
    --k 8 \
    --batch_size 64 \
    --max_seq_len 256 \
    --workers "$WORKERS" \
    --prefetch 4
