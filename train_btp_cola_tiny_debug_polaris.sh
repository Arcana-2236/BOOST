#!/bin/bash -l
#PBS -N boost_cola_tiny_debug
#PBS -A TensorCompress
#PBS -q debug
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -euo pipefail

# --- Environment ---
module use /soft/modulefiles
module load conda
conda activate /home/$USER/conda-envs/nanotron-py310

cd /home/$USER/nanotron

# --- Cluster info from PBS ---
NNODES=$(wc -l < "$PBS_NODEFILE")
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=$((RANDOM + 1000))

echo "NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "PBS_NODEFILE:"
cat "$PBS_NODEFILE"

# --- Tiny debug configuration ---
TP=2
PP=1
EP=1
MODEL_ARCH=${MODEL_ARCH:-"cola"}
MODEL_SIZE=${MODEL_SIZE:-"llama_2l_h256_seq_4096"}
CONFIG_NAME=${MODEL_ARCH}_${MODEL_SIZE}

# Smaller training budget for quick parity/debug iteration
LR=${LR:-"0.003"}
BZ=${BZ:-"8"}
TBZ=${TBZ:-"32"}
TRAIN_STEPS=${TRAIN_STEPS:-"800"}
WU=${WU:-"30"}

# Smaller/default dataset (override via HF_DATASET env var)
# HF_DATASET=${HF_DATASET:-"stas/openwebtext-10k"}

DP=$((WORLD_SIZE / (TP * PP * EP)))
if [ "$DP" -lt 1 ]; then
  echo "Invalid parallelism: WORLD_SIZE=$WORLD_SIZE TP=$TP PP=$PP EP=$EP gives DP=$DP"
  exit 1
fi

CONTINUE=${CONTINUE:-"none"}
if [ "$CONTINUE" != "none" ]; then
  readonly continue_from_flag="--resume-checkpoint-path $CONTINUE"
else
  readonly continue_from_flag=""
fi

RUN_NAME=${RUN_NAME:-"BTP-${CONFIG_NAME}-tinydebug-embeddingcorrection-LR-${LR}-DP${DP}-TP${TP}-PP${PP}-EP${EP}"}
TAG=${TAG:-"none"}
if [ "$TAG" != "none" ]; then
  RUN_NAME="${TAG}-${RUN_NAME}"
fi

LOGDIR="/home/$USER/nanotron/.logging/$(date +%Y%m%d)"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID}.log"
echo "LOGFILE=$LOGFILE"

export TMPDIR=/tmp

mpiexec -n "$NNODES" -ppn 1 --hostfile "$PBS_NODEFILE" \
  bash -lc "
    conda activate /home/$USER/conda-envs/nanotron-py310
    cd /home/$USER/nanotron
    NODE_RANK=\${PMI_RANK:-0}

    export WANDB_API_KEY=\"8bd1f610ea32c79e0fbb32d3ec8511881a434c12\"
    wandb online
    wandb status

    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    torchrun \
      --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE \
      --node_rank=\$NODE_RANK \
      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
      -- examples/cola/train_cola.py --config-file examples/cola/config_${CONFIG_NAME}.yaml \
      --hf-dataset-or-datasets /eagle/TensorCompress/seq_len_4096 --run $RUN_NAME \
      --entity zhengyangwang-university-of-california-santa-barbara --project boost \
      --lr $LR --micro-batch-size $BZ --batch-accumulation-per-replica $((TBZ / (BZ * DP))) \
      --lr-warmup-steps $WU --tp $TP --pp $PP --dp $DP --train-steps $TRAIN_STEPS \
      $continue_from_flag
    " > "$LOGFILE" 2>&1

echo "DONE Running tiny CoLA Nanotron debug run"
