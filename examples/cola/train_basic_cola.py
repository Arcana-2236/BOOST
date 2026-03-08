"""
0. Work directory
cd /workspace/cola_nanotron/nanotron

1. Generate a config file
python examples/cola/config_cola_llama.py

2. Run the training
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_vanilla_cola.py --config-file examples/cola/config_cola_llama_7b.yaml

3. Run the training with config overrides
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_vanilla_cola.py \
    --config-file examples/cola/config_cola_llama_7b.yaml \
    --run my_custom_run_name \
    --project my_project \
    --checkpoints-path /path/to/checkpoints \
    --checkpoint-interval 100

Available config overrides (all optional):

  General config:
    --run: Override run name (general.run)
    --tag: Suffix for run name (general.tag)
    --entity: Override wandb entity name (general.entity)
    --project: Override project name (general.project)
    --seed: Override random seed (general.seed)

  Checkpoint config:
    --checkpoints-path: Override checkpoint save path (checkpoints.checkpoints_path)
    --checkpoint-interval: Override checkpoint interval (checkpoints.checkpoint_interval)
    --resume-checkpoint-path: Override resume checkpoint path (checkpoints.resume_checkpoint_path)
    --save-initial-state: Override save initial state (checkpoints.save_initial_state)
    --save-final-state: Override save final state (checkpoints.save_final_state)

  Optimizer config:
    --learning-rate, --lr: Override learning rate (optimizer.learning_rate_scheduler.learning_rate)
    --min-decay-lr: Override min decay learning rate (optimizer.learning_rate_scheduler.min_decay_lr)
    --lr-warmup-steps: Override learning rate warmup steps (optimizer.learning_rate_scheduler.lr_warmup_steps)

  Token config:
    --micro-batch-size: Override micro batch size (tokens.micro_batch_size)
    --batch-accumulation-per-replica: Override batch accumulation (tokens.batch_accumulation_per_replica)
    --train-steps: Override train steps (tokens.train_steps)
    --val-check-interval: Override validation check interval (tokens.val_check_interval)

  Parallelism config:
    --dp: Override data parallelism degree (parallelism.dp)
    --tp: Override tensor parallelism degree (parallelism.tp)
    --pp: Override pipeline parallelism degree (parallelism.pp)
"""
import argparse
import os
import sys

from nanotron import logging
from nanotron.config import get_config_from_file, apply_config_overrides
from nanotron.utils import get_args
from nanotron.trainer import DistributedTrainer
from config_basic_cola_llama import BasicColaLlamaConfig
from basic_cola_llama import BasicColaLlamaForTraining

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load config from file
    config = get_config_from_file(config_file, model_config_class=BasicColaLlamaConfig)

    # Apply command line overrides
    config = apply_config_overrides(config, args)

    # Load trainer with modified config
    trainer = DistributedTrainer(config, model_config_class=BasicColaLlamaConfig, model_class=BasicColaLlamaForTraining)
    dataloader = get_dataloader(trainer)

    # Train
    trainer.train(dataloader)