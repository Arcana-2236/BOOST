"""
0. Work directory
cd /workspace/cola_nanotron/nanotron

1. Generate a config file
python examples/cola/config_cola_llama.py

2. Run the training
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_cola.py --config-file examples/cola/config_cola_llama.yaml

3. Run the training with config overrides
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=4 examples/cola/train_cola.py \
    --config-file examples/cola/config_cola_llama.yaml \
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
from typing import Any, Mapping

import torch
import torch.distributed as torch_dist
from nanotron import logging
from nanotron.config import get_config_from_file, apply_config_overrides
from nanotron.utils import get_args
from nanotron.trainer import DistributedTrainer
from config_cola_llama import ColaLlamaConfig
from cola_llama import ColaLlamaForTraining

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from run_train import get_dataloader  # noqa

logger = logging.get_logger(__name__)


def _clone_batch_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, dict):
        return {k: _clone_batch_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clone_batch_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_batch_tree(v) for v in x)
    return x


def _to_cpu_tree(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_cpu_tree(v) for v in x)
    return x


def _move_tree_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device=device)
    if isinstance(x, dict):
        return {k: _move_tree_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_move_tree_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_tree_to_device(v, device) for v in x)
    return x


def _fixed_batch_replay_iter(dataloader, clone_each_step: bool = True):
    iterator = iter(dataloader)
    first_batch = next(iterator)
    if not isinstance(first_batch, Mapping):
        raise TypeError(
            f"Fixed replay expects batch mapping, got {type(first_batch)}. "
            "Use this mode only with standard train dataloaders."
        )
    while True:
        yield _clone_batch_tree(first_batch) if clone_each_step else first_batch


def _strict_fixed_batch_replay_iter(
    dataloader,
    clone_each_step: bool,
    strict_batch_path: str,
    strict_batch_mode: str,
    device: torch.device,
    dp_pg,
    local_accum_steps: int,
):
    iterator = iter(dataloader)
    local_first_batches = [next(iterator) for _ in range(local_accum_steps)]
    for local_first_batch in local_first_batches:
        if not isinstance(local_first_batch, Mapping):
            raise TypeError(
                f"Strict fixed replay expects batch mapping, got {type(local_first_batch)}."
            )

    is_dist = torch_dist.is_available() and torch_dist.is_initialized()
    world_rank = torch_dist.get_rank() if is_dist else 0
    dp_size = dp_pg.size() if dp_pg is not None else 1
    dp_rank = torch_dist.get_rank(group=dp_pg) if (is_dist and dp_pg is not None) else 0
    global_micro_count = dp_size * local_accum_steps

    if strict_batch_mode == "save":
        gathered = [None for _ in range(dp_size)] if dp_rank == 0 else None
        if dp_size > 1:
            torch_dist.gather_object(local_first_batches, gathered, dst=0, group=dp_pg)
        else:
            gathered = [local_first_batches]
        if dp_rank == 0 and world_rank == 0:
            global_microbatches = []
            for rank_batches in gathered:
                global_microbatches.extend(rank_batches)
            payload = {
                "meta": {
                    "dp_size": int(dp_size),
                    "local_accum_steps": int(local_accum_steps),
                    "global_micro_count": int(len(global_microbatches)),
                },
                "global_microbatches": _to_cpu_tree(global_microbatches),
            }
            os.makedirs(os.path.dirname(strict_batch_path), exist_ok=True)
            torch.save(payload, strict_batch_path)
        if is_dist:
            torch_dist.barrier()
    elif strict_batch_mode == "load":
        if not os.path.exists(strict_batch_path):
            raise FileNotFoundError(
                f"COLA_STRICT_BATCH_MODE=load but batch file not found: {strict_batch_path}"
            )
        if is_dist:
            torch_dist.barrier()
    else:
        raise ValueError(
            f"Invalid COLA_STRICT_BATCH_MODE={strict_batch_mode}. Expected save/load."
        )

    payload = torch.load(strict_batch_path, map_location="cpu")
    if not isinstance(payload, dict) or "global_microbatches" not in payload:
        # Backward compatibility with old single-batch strict replay artifact.
        payload = {
            "meta": {
                "dp_size": int(dp_size),
                "local_accum_steps": int(local_accum_steps),
                "global_micro_count": 1,
            },
            "global_microbatches": [payload],
        }
    global_microbatches = payload["global_microbatches"]
    if not isinstance(global_microbatches, list) or len(global_microbatches) == 0:
        raise TypeError("Strict replay payload has invalid 'global_microbatches'.")
    if len(global_microbatches) != global_micro_count:
        raise RuntimeError(
            f"Strict replay global microbatch count mismatch: saved={len(global_microbatches)} current={global_micro_count} (dp={dp_size}, acc={local_accum_steps})"
        )

    local_ids = [dp_rank * local_accum_steps + i for i in range(local_accum_steps)]
    local_microbatches = [_move_tree_to_device(global_microbatches[idx], device=device) for idx in local_ids]
    for batch in local_microbatches:
        if not isinstance(batch, Mapping):
            raise TypeError(
                f"Loaded strict batch element is not a mapping: got {type(batch)}."
            )

    if len(local_microbatches) == 1 and not isinstance(local_microbatches[0], Mapping):
        raise TypeError(
            f"Loaded strict batch is not a mapping: got {type(local_microbatches[0])}."
        )

    local_cursor = 0
    while True:
        batch = local_microbatches[local_cursor]
        local_cursor = (local_cursor + 1) % local_accum_steps
        yield _clone_batch_tree(batch) if clone_each_step else batch


def _enable_fixed_batch_replay(
    dataloader_or_dls,
    clone_each_step: bool = True,
    strict_batch_path: str = "",
    strict_batch_mode: str = "",
    device: torch.device = torch.device("cpu"),
    dp_pg=None,
    local_accum_steps: int = 1,
):
    use_strict = bool(strict_batch_path)

    # Preferred path for Nanotron's staged dataloader structure:
    # {stage_name: {"train": <dl_or_factory>, "validation": ...}}
    if isinstance(dataloader_or_dls, dict):
        wrapped = {}
        for stage_name, stage_data in dataloader_or_dls.items():
            if isinstance(stage_data, dict) and "train" in stage_data:
                train_src = stage_data["train"]
                stage_data_wrapped = dict(stage_data)
                if callable(train_src):
                    def _wrapped_train_factory(train_factory=train_src):
                        if use_strict:
                            return _strict_fixed_batch_replay_iter(
                                train_factory(),
                                clone_each_step=clone_each_step,
                                strict_batch_path=strict_batch_path,
                                strict_batch_mode=strict_batch_mode,
                                device=device,
                                dp_pg=dp_pg,
                                local_accum_steps=local_accum_steps,
                            )
                        return _fixed_batch_replay_iter(train_factory(), clone_each_step=clone_each_step)
                    stage_data_wrapped["train"] = _wrapped_train_factory
                else:
                    if use_strict:
                        stage_data_wrapped["train"] = _strict_fixed_batch_replay_iter(
                            train_src,
                            clone_each_step=clone_each_step,
                            strict_batch_path=strict_batch_path,
                            strict_batch_mode=strict_batch_mode,
                            device=device,
                            dp_pg=dp_pg,
                            local_accum_steps=local_accum_steps,
                        )
                    else:
                        stage_data_wrapped["train"] = _fixed_batch_replay_iter(
                            train_src, clone_each_step=clone_each_step
                        )
                wrapped[stage_name] = stage_data_wrapped
            else:
                wrapped[stage_name] = stage_data
        return wrapped

    # Fallback for plain iterator-style dataloaders.
    if use_strict:
        return _strict_fixed_batch_replay_iter(
            dataloader_or_dls,
            clone_each_step=clone_each_step,
            strict_batch_path=strict_batch_path,
            strict_batch_mode=strict_batch_mode,
            device=device,
            dp_pg=dp_pg,
            local_accum_steps=local_accum_steps,
        )
    return _fixed_batch_replay_iter(dataloader_or_dls, clone_each_step=clone_each_step)


def _save_init_snapshot(trainer, snapshot_path: str):
    if not snapshot_path:
        return
    if torch_dist.is_available() and torch_dist.is_initialized() and torch_dist.get_rank() != 0:
        torch_dist.barrier()
        return
    unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    payload = {name: p.detach().cpu().clone() for name, p in unwrapped.named_parameters()}
    os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
    torch.save(payload, snapshot_path)
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()


def _load_init_snapshot(trainer, snapshot_path: str):
    if not snapshot_path:
        return
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"COLA_STRICT_INIT_MODE=load but snapshot not found: {snapshot_path}")
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()

    full_state = torch.load(snapshot_path, map_location="cpu")
    if not isinstance(full_state, dict):
        raise TypeError(f"Invalid init snapshot format at {snapshot_path}")

    unwrapped = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    tp_pg = trainer.parallel_context.tp_pg
    tp_size = tp_pg.size() if tp_pg is not None else 1
    tp_rank = torch_dist.get_rank(group=tp_pg) if tp_size > 1 else 0

    with torch.no_grad():
        for name, p_local in unwrapped.named_parameters():
            if name not in full_state:
                continue
            p_full = full_state[name]
            if tuple(p_full.shape) == tuple(p_local.shape):
                p_local.copy_(p_full.to(device=p_local.device, dtype=p_local.dtype))
                continue

            shard_dims = [
                d
                for d in range(p_full.ndim)
                if p_full.shape[d] == p_local.shape[d] * tp_size
                and all(p_full.shape[k] == p_local.shape[k] for k in range(p_full.ndim) if k != d)
            ]
            if len(shard_dims) != 1:
                raise RuntimeError(
                    f"Cannot infer TP shard dim for {name}: local={tuple(p_local.shape)} full={tuple(p_full.shape)} tp={tp_size}"
                )
            shard_dim = shard_dims[0]
            local = p_local.shape[shard_dim]
            shard = p_full.narrow(shard_dim, tp_rank * local, local)
            p_local.copy_(shard.to(device=p_local.device, dtype=p_local.dtype))


if __name__ == "__main__":
    args = get_args()
    config_file = args.config_file

    # Load config from file
    config = get_config_from_file(config_file, model_config_class=ColaLlamaConfig)

    # Apply command line overrides
    config = apply_config_overrides(config, args)

    # Load trainer with modified config
    trainer = DistributedTrainer(config, model_config_class=ColaLlamaConfig, model_class=ColaLlamaForTraining)
    dataloader = get_dataloader(trainer)

    strict_init_mode = os.environ.get("COLA_STRICT_INIT_MODE", "").strip().lower()
    strict_init_path = os.environ.get("COLA_STRICT_INIT_PATH", "").strip()
    if strict_init_mode:
        if strict_init_mode == "save":
            logger.warning("COLA_STRICT_INIT_MODE=save -> saving init snapshot to %s", strict_init_path)
            _save_init_snapshot(trainer, strict_init_path)
        elif strict_init_mode == "load":
            logger.warning("COLA_STRICT_INIT_MODE=load -> loading init snapshot from %s", strict_init_path)
            _load_init_snapshot(trainer, strict_init_path)
        else:
            raise ValueError(
                f"Invalid COLA_STRICT_INIT_MODE={strict_init_mode}. Expected one of: save, load."
            )

    # Optional deterministic debug mode: replay the exact same first batch forever.
    # This isolates data-order/sharding effects when comparing TP runs.
    fixed_replay = os.environ.get("COLA_FIXED_BATCH_REPLAY", "0").lower() in {"1", "true", "yes", "on"}
    if fixed_replay:
        clone_each_step = os.environ.get("COLA_FIXED_BATCH_REPLAY_CLONE", "1").lower() in {"1", "true", "yes", "on"}
        strict_batch_mode = os.environ.get("COLA_STRICT_BATCH_MODE", "").strip().lower()
        strict_batch_path = os.environ.get("COLA_STRICT_BATCH_PATH", "").strip()
        if strict_batch_mode and strict_batch_mode not in {"save", "load"}:
            raise ValueError(
                f"Invalid COLA_STRICT_BATCH_MODE={strict_batch_mode}. Expected one of: save, load."
            )
        if strict_batch_mode and not strict_batch_path:
            raise ValueError("COLA_STRICT_BATCH_MODE is set but COLA_STRICT_BATCH_PATH is empty.")

        logger.warning(
            "COLA_FIXED_BATCH_REPLAY is enabled: replaying first training batch every step (clone_each_step=%s, strict_batch_mode=%s).",
            clone_each_step,
            strict_batch_mode if strict_batch_mode else "off",
        )
        if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
            replay_device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        elif torch.cuda.is_available():
            replay_device = torch.device("cuda")
        else:
            replay_device = torch.device("cpu")
        dataloader = _enable_fixed_batch_replay(
            dataloader,
            clone_each_step=clone_each_step,
            strict_batch_path=strict_batch_path,
            strict_batch_mode=strict_batch_mode,
            device=replay_device,
            dp_pg=trainer.parallel_context.dp_pg,
            local_accum_steps=config.tokens.batch_accumulation_per_replica,
        )

    # Train
    trainer.train(dataloader)