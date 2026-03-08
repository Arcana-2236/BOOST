#!/bin/bash -l
#PBS -N basic_cola_60m_1b_token
#PBS -A TensorCompress
#PBS -q debug-scaling
#PBS -l select=6:ncpus=64:ngpus=4
#PBS -l walltime=00:59:00
#PBS -l filesystems=home:eagle
#PBS -j oe

set -euo pipefail

# --- Environment ---
module use /soft/modulefiles
module load conda
conda activate /home/$USER/conda-envs/nanotron-py310

cd /home/$USER/nanotron


# --- Cluster info from PBS ---
NNODES=`wc -l < $PBS_NODEFILE`
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=$(( RANDOM + 1000 ))

echo "NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "PBS_NODEFILE:"
cat "$PBS_NODEFILE"


# --- Configuration ---
TP=1
PP=1
EP=1
MODEL_SIZE="llama_60m"
MODEL_ARCH="cola"

RUN_NAME=${RUN_NAME:-"None"}
CONFIG_NAME=${MODEL_ARCH}_${MODEL_SIZE}
LR=${LR:-"0.003"}
BZ=${BZ:-"8"}
TBZ=${TBZ:-"384"}
DP=$((WORLD_SIZE/(TP*PP*EP)))
TRAIN_STEPS=${TRAIN_STEPS:-"916"}
CONTINUE=${CONTINUE:-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly continue_from_flag="--resume-checkpoint-path $CONTINUE"
else
    readonly continue_from_flag=""
fi

RUN_NAME=$CONFIG_NAME-LR-$LR
TAG=${TAG:-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi
WU=${WU:-"180"}
if [ "${WU}" != "180" ]; then
    RUN_NAME=$RUN_NAME-WU-$WU
fi
RUN_NAME=$RUN_NAME-DP$DP-TP$TP-PP$PP-EP$EP

# --- (Optional) NCCL tuning via AWS OFI NCCL plugin
# ALCF notes this can improve performance but may cause hangs for some apps. Start without it if you want max safety. :contentReference[oaicite:2]{index=2}
# export NCCL_NET="AWS Libfabric"
# export FI_PROVIDER=cxi
# export FI_CXI_DISABLE_HOST_REGISTER=1


# --- Run CoLA Nanotron ---
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
      -- examples/cola/train_basic_cola.py --config-file examples/cola/config_${CONFIG_NAME}.yaml \
      --hf-dataset-or-datasets /eagle/TensorCompress/seq_len_4096 --run $RUN_NAME \
      --entity zhengyangwang-university-of-california-santa-barbara --project boost \
      --lr $LR --micro-batch-size $BZ --batch-accumulation-per-replica $((TBZ / (BZ * WORLD_SIZE))) \
      --lr-warmup-steps $WU --tp $TP --pp $PP --dp $DP --train-steps $TRAIN_STEPS \
      $continue_from_flag
    " > "$LOGFILE" 2>&1

echo "DONE Running CoLA Nanotron"