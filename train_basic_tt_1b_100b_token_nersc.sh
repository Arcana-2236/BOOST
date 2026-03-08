#!/bin/sh
#SBATCH -A m4788_g
#SBATCH -N 32
#SBATCH -q premium
#SBATCH -C gpu&hbm40g
#SBATCH -t 1:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1

cd /global/cfs/cdirs/m4788/alvinliu/repo/nanotron
module load python/3.10
module load cudatoolkit/12.4
source /global/cfs/cdirs/m4788/alvinliu/envs/nanotron/bin/activate

MASTER_PORT=$(( RANDOM + 1000 ))
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
WORLD_SIZE=$((SLURM_JOB_NUM_NODES * 4))

TP=${TP:-1}
PP=${PP:-1}

RUN_NAME=${RUN_NAME:-"None"}
CONFIG_NAME=${CONFIG_NAME:-"tt_llama_1b"}
LR=${LR:-"0.001"}
BZ=${BZ:-"2"}
TBZ=${TBZ:-"4096"}
CONTINUE=${CONTINUE:-"none"}
if [ "${CONTINUE}" != "none" ]; then
    readonly continue_from_flag="--resume-checkpoint-path=$CONTINUE"
else
    readonly continue_from_flag=""
fi

RUN_NAME=$CONFIG_NAME-LR-$LR
TAG=${TAG:-"none"}
if [ "${TAG}" != "none" ]; then
    RUN_NAME=$TAG-$RUN_NAME
fi
WU=${WU:-"1200"}
if [ "${WU}" != "1200" ]; then
    RUN_NAME=$RUN_NAME-WU-$WU
fi

HF_HOME="/global/cfs/cdirs/m4645/alvinliu/workspace/datasets/.cache/huggingface"

HF_HOME=$HF_HOME MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT WORLD_SIZE=$WORLD_SIZE \
    srun -u torchrun --nproc-per-node=4 --master-port=$MASTER_PORT --nnodes=$SLURM_JOB_NUM_NODES \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT -- examples/tensor/train_basic_tensor.py \
    --run $RUN_NAME \
    --config-file examples/tensor/config_$CONFIG_NAME.yaml \
    --hf-dataset-or-datasets /pscratch/sd/a/alvinliu/datasets/seq_len_4096 \
    --lr $LR \
    --micro-batch-size $BZ \
    --batch-accumulation-per-replica $((TBZ / (BZ * WORLD_SIZE))) \
    --lr-warmup-steps $WU \
    --tp $TP \
    --pp $PP \
    --dp $WORLD_SIZE \
    $continue_from_flag
