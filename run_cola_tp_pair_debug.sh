#!/bin/bash
set -euo pipefail

# Run TP=1 and TP=4 back-to-back with matched settings,
# then compare logs to find first divergence step.
#
# Usage:
#   bash run_cola_tp_pair_debug.sh
# Optional env overrides:
#   CONFIG_FILE, HF_DATASET, TRAIN_STEPS, LR, BZ, TBZ, WU, SEED, PROJECT, ENTITY
#   FIXED_BATCH_REPLAY, FIXED_BATCH_REPLAY_CLONE

cd /home/$USER/nanotron

PYTHON_BIN=${PYTHON_BIN:-"/home/$USER/conda-envs/nanotron-py310/bin/python"}
TORCHRUN_BIN=${TORCHRUN_BIN:-"/home/$USER/conda-envs/nanotron-py310/bin/torchrun"}

CONFIG_FILE=${CONFIG_FILE:-"examples/cola/config_cola_llama_2l_h256_seq_4096.yaml"}
HF_DATASET=${HF_DATASET:-"/eagle/TensorCompress/seq_len_4096"}
TRAIN_STEPS=${TRAIN_STEPS:-200}
LR=${LR:-0.003}
BZ=${BZ:-8}
TBZ=${TBZ:-32}
WU=${WU:-30}
SEED=${SEED:-42}
PROJECT=${PROJECT:-boost}
ENTITY=${ENTITY:-zhengyangwang-university-of-california-santa-barbara}
FIXED_BATCH_REPLAY=${FIXED_BATCH_REPLAY:-1}
FIXED_BATCH_REPLAY_CLONE=${FIXED_BATCH_REPLAY_CLONE:-1}
COLA_STRICT_DEBUG_STEP=${COLA_STRICT_DEBUG_STEP:-0}
COLA_STRICT_DEBUG_TOPK=${COLA_STRICT_DEBUG_TOPK:-12}
COLA_STRICT_TENSOR_DUMP_STEP=${COLA_STRICT_TENSOR_DUMP_STEP:-0}
COLA_STRICT_TENSOR_DUMP_ATOL=${COLA_STRICT_TENSOR_DUMP_ATOL:-1e-5}
COLA_STRICT_TENSOR_DUMP_RTOL=${COLA_STRICT_TENSOR_DUMP_RTOL:-1e-5}
COLA_ACT_TRACE_ENABLE=${COLA_ACT_TRACE_ENABLE:-0}
COLA_ACT_TRACE_LAYER=${COLA_ACT_TRACE_LAYER:-0}
COLA_ACT_TRACE_ATOL=${COLA_ACT_TRACE_ATOL:-1e-5}
COLA_ACT_TRACE_RTOL=${COLA_ACT_TRACE_RTOL:-1e-5}

PP=1
EP=1
NPROC_PER_NODE=$(nvidia-smi -L | wc -l)
WORLD_SIZE=$NPROC_PER_NODE
MASTER_ADDR=127.0.0.1

if [ "$WORLD_SIZE" -lt 4 ]; then
  echo "This script expects at least 4 GPUs on the node."
  echo "Detected WORLD_SIZE=$WORLD_SIZE"
  exit 1
fi

if [ $((TBZ % BZ)) -ne 0 ]; then
  echo "TBZ must be divisible by BZ. Got TBZ=$TBZ BZ=$BZ"
  exit 1
fi

PAIR_TAG=${PAIR_TAG:-"pair_$(date +%Y%m%d_%H%M%S)"}
PAIR_OUTPUT_ROOT=${PAIR_OUTPUT_ROOT:-"/home/$USER/nanotron/.logging/paired-tp"}
LOG_ROOT="${PAIR_OUTPUT_ROOT}/${PAIR_TAG}"
mkdir -p "$LOG_ROOT"
# Optional override for large .pt artifacts only.
# If set, .pt files are written under: ${PAIR_PT_ROOT}/${PAIR_TAG}/...
# Logs/reports remain under LOG_ROOT.
PAIR_PT_ROOT=${PAIR_PT_ROOT:-""}
if [ -n "$PAIR_PT_ROOT" ]; then
  PT_ROOT="${PAIR_PT_ROOT}/${PAIR_TAG}"
else
  PT_ROOT="$LOG_ROOT"
fi
mkdir -p "$PT_ROOT"

STRICT_DIR="${PT_ROOT}/strict-artifacts"
mkdir -p "$STRICT_DIR"
STRICT_INIT_PATH="${STRICT_DIR}/init_snapshot.pt"
STRICT_BATCH_PATH="${STRICT_DIR}/fixed_batch.pt"

run_one() {
  local TP="$1"
  local MASTER_PORT="$2"
  local STRICT_MODE="$3"
  local DP=$((WORLD_SIZE / (TP * PP * EP)))
  if [ $((WORLD_SIZE % (TP * PP * EP))) -ne 0 ]; then
    echo "Invalid parallelism for TP=$TP: WORLD_SIZE=$WORLD_SIZE"
    exit 1
  fi

  local ACC=$((TBZ / (BZ * DP)))
  if [ "$ACC" -lt 1 ]; then
    echo "Invalid batch setup for TP=$TP: TBZ=$TBZ BZ=$BZ DP=$DP -> ACC=$ACC"
    exit 1
  fi

  local RUN_NAME="PAIR-${PAIR_TAG}-TP${TP}-DP${DP}-LR${LR}-S${SEED}"
  local LOG_FILE="${LOG_ROOT}/${RUN_NAME}.log"

  echo "=== Running ${RUN_NAME} ==="
  echo "log: $LOG_FILE"

  # Deterministic-ish runtime knobs for reproducibility across runs.
  export PYTHONHASHSEED="$SEED"
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  export CUDA_DEVICE_MAX_CONNECTIONS=1

  COLA_FIXED_BATCH_REPLAY="$FIXED_BATCH_REPLAY" \
  COLA_FIXED_BATCH_REPLAY_CLONE="$FIXED_BATCH_REPLAY_CLONE" \
  COLA_STRICT_INIT_MODE="$STRICT_MODE" \
  COLA_STRICT_INIT_PATH="$STRICT_INIT_PATH" \
  COLA_STRICT_BATCH_MODE="$STRICT_MODE" \
  COLA_STRICT_BATCH_PATH="$STRICT_BATCH_PATH" \
  COLA_STRICT_DEBUG_STEP="$COLA_STRICT_DEBUG_STEP" \
  COLA_STRICT_DEBUG_TOPK="$COLA_STRICT_DEBUG_TOPK" \
  COLA_STRICT_TENSOR_DUMP_STEP="$COLA_STRICT_TENSOR_DUMP_STEP" \
  COLA_STRICT_TENSOR_DUMP_DIR="$PT_ROOT/tp${TP}" \
  COLA_STRICT_TENSOR_DUMP_INCLUDE_LOGITS=1 \
  COLA_ACT_TRACE_ENABLE="$COLA_ACT_TRACE_ENABLE" \
  COLA_ACT_TRACE_LAYER="$COLA_ACT_TRACE_LAYER" \
  COLA_ACT_TRACE_DIR="$PT_ROOT/tp${TP}" \
  COLA_ACT_TRACE_SAVE_ONCE=1 \
  "$TORCHRUN_BIN" \
    --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    -- examples/cola/train_cola.py --config-file "$CONFIG_FILE" \
    --hf-dataset-or-datasets "$HF_DATASET" \
    --run "$RUN_NAME" --project "$PROJECT" --entity "$ENTITY" \
    --seed "$SEED" \
    --lr "$LR" --micro-batch-size "$BZ" \
    --batch-accumulation-per-replica "$ACC" \
    --lr-warmup-steps "$WU" --tp "$TP" --pp "$PP" --dp "$DP" \
    --train-steps "$TRAIN_STEPS" \
    > "$LOG_FILE" 2>&1
}

# Run TP=1 first, then TP=4.
run_one 1 29601 save
run_one 4 29604 load

LOG_TP1=$(ls -t "${LOG_ROOT}"/PAIR-"${PAIR_TAG}"-TP1-*.log | head -n 1)
LOG_TP4=$(ls -t "${LOG_ROOT}"/PAIR-"${PAIR_TAG}"-TP4-*.log | head -n 1)
REPORT_NAME="compare_${PAIR_TAG}_tp1_vs_tp4.txt"
COMPARE_FAIL=0

echo "=== Comparing logs ==="
set +e
"$PYTHON_BIN" tests/boost/compare_cola_train_logs.py \
  --log-a "$LOG_TP1" \
  --log-b "$LOG_TP4" \
  --atol 1e-3 --rtol 1e-3 \
  --report-dir "$LOG_ROOT" \
  --report-name "$REPORT_NAME"
STATUS=$?
set -e
if [ $STATUS -ne 0 ]; then
  echo "WARNING: log comparison failed with status $STATUS"
  COMPARE_FAIL=1
fi

if [ "$COLA_STRICT_TENSOR_DUMP_STEP" -gt 0 ]; then
  DUMP_TP1="${PT_ROOT}/tp1/tensor_dump_step_${COLA_STRICT_TENSOR_DUMP_STEP}.pt"
  DUMP_TP4="${PT_ROOT}/tp4/tensor_dump_step_${COLA_STRICT_TENSOR_DUMP_STEP}.pt"
  TENSOR_REPORT="${LOG_ROOT}/compare_tensor_dump_step_${COLA_STRICT_TENSOR_DUMP_STEP}.txt"
  echo "=== Comparing tensor dumps (step ${COLA_STRICT_TENSOR_DUMP_STEP}) ==="
  set +e
  "$PYTHON_BIN" tests/boost/compare_cola_tensor_dumps.py \
    --dump-a "$DUMP_TP1" \
    --dump-b "$DUMP_TP4" \
    --atol "$COLA_STRICT_TENSOR_DUMP_ATOL" \
    --rtol "$COLA_STRICT_TENSOR_DUMP_RTOL" \
    > "$TENSOR_REPORT"
  STATUS=$?
  set -e
  echo "Tensor report: ${TENSOR_REPORT}"
  if [ $STATUS -ne 0 ]; then
    echo "WARNING: tensor comparison failed with status $STATUS"
    COMPARE_FAIL=1
  fi
fi

if [ "$COLA_ACT_TRACE_ENABLE" = "1" ] || [ "$COLA_ACT_TRACE_ENABLE" = "true" ]; then
  TRACE_TP1="${PT_ROOT}/tp1/activation_trace_layer${COLA_ACT_TRACE_LAYER}.pt"
  TRACE_TP4="${PT_ROOT}/tp4/activation_trace_layer${COLA_ACT_TRACE_LAYER}.pt"
  TRACE_REPORT="${LOG_ROOT}/compare_activation_trace_layer${COLA_ACT_TRACE_LAYER}.txt"
  echo "=== Comparing activation traces (layer ${COLA_ACT_TRACE_LAYER}) ==="
  set +e
  "$PYTHON_BIN" tests/boost/compare_cola_activation_traces.py \
    --trace-a "$TRACE_TP1" \
    --trace-b "$TRACE_TP4" \
    --atol "$COLA_ACT_TRACE_ATOL" \
    --rtol "$COLA_ACT_TRACE_RTOL" \
    > "$TRACE_REPORT"
  STATUS=$?
  set -e
  echo "Activation report: ${TRACE_REPORT}"
  if [ $STATUS -ne 0 ]; then
    echo "WARNING: activation comparison failed with status $STATUS"
    COMPARE_FAIL=1
  fi
fi

echo "Done."
echo "TP1 log: $LOG_TP1"
echo "TP4 log: $LOG_TP4"
echo "Report:  ${LOG_ROOT}/${REPORT_NAME}"
echo "PT root: ${PT_ROOT}"
if [ $COMPARE_FAIL -ne 0 ]; then
  exit 1
fi
