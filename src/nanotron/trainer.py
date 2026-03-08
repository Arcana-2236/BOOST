import datetime
import gc
import itertools
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import (
    Config,
    DatasetStageArgs,
    ExistingCheckpointInit,
    ParallelismArgs,
    RandomInit,
    SpectralMupInit,
    get_config_from_file,
)
from nanotron.constants import MODEL_CONFIG_FILE_NAME
from nanotron.dataloader import sanity_check_dataloader
from nanotron.helpers import (
    _vocab_size_with_padding,
    compute_remain_train_steps_of_a_data_stage_from_ckp,
    get_consumed_train_samples_of_a_data_stage_from_ckp,
    get_profiler,
    init_optimizer_and_grad_accumulator,
    init_random_states,
    log_throughput,
    lr_scheduler_builder,
)
from nanotron.logging import (
    LoggerWriter,
    LogItem,
    human_format,
    log_memory,
    log_rank,
    set_ranks_logging_level,
)
from nanotron.models import NanotronModel, build_model
from nanotron.models.base import check_model_has_grad
from nanotron.models.llama import LlamaForTraining, RotaryEmbedding
from nanotron.models.starcoder2 import Starcoder2ForTraining
from nanotron.optim.clip_grads import clip_grad_norm
from nanotron.parallel import ParallelContext
from nanotron.parallel.data_parallel.utils import sync_gradients_across_dp
from nanotron.parallel.parameters import NanotronParameter, sanity_check
from nanotron.parallel.pipeline_parallel.engine import (
    PipelineEngine,
    TensorPointer,
)
from nanotron.parallel.pipeline_parallel.utils import get_pp_rank_of
from nanotron.parallel.tensor_parallel.enum import TensorParallelLinearMode
from nanotron.parallel.tensor_parallel.nn import TensorParallelRowLinear
from nanotron.parallel.tied_parameters import (
    create_pg_for_tied_weights,
    get_tied_id_to_param,
    sync_tied_weights_gradients,
    tie_parameters,
)
from nanotron.random import set_random_seed
from nanotron.s3_checkpoints import S3Mover, check_path_is_local
from nanotron.sanity_checks import (
    after_optim_step_sanity_checks,
    after_tbi_sanity_checks,
    before_optim_step_sanity_checks,
    before_tbi_sanity_checks,
)
from nanotron.scaling.parametrization import ParametrizationMethod
from nanotron.serialize import (
    load_lr_scheduler,
    load_meta,
    load_weights,
    parse_ckpt_path,
    save,
    save_random_states,
)
from nanotron.serialize.metadata import DataStageMetadata, TrainingMetadata
from nanotron.serialize.optimizer import load_optimizer, state_dict_to_device

logger = logging.get_logger(__name__)

# Reduce the logging noise from torch.distributed when creating new process groups
dist_logger = logging.get_logger(dist.dist.__name__)
dist_logger.setLevel(logging.WARNING)

CONFIG_TO_MODEL_CLASS = {
    "LlamaConfig": LlamaForTraining,
    "Starcoder2Config": Starcoder2ForTraining,
}

try:
    import wandb
except ImportError:
    wandb = None


class DistributedTrainer:
    def __init__(
        self,
        config_or_config_file: Union[Config, str],
        config_class: Type[Config] = Config,
        model_config_class: Optional[Type] = None,
        model_class: Type[NanotronModel] = None,
    ):
        """
        Nanotron's distributed trainer.

        Args:
            config_or_config_file: Either a `Config` object or a path to a YAML file containing the config.
            config_class: The `Config` class to use.
            model_config_class: The `ModelConfig` class to use (for example `LlamaConfig`). Defaults to `None` which will use the model config class defined in the config.
            model_class: The `NanotronModel` class to use (for example `LlamaForTraining`). Defaults to `None` which will use the model class defined in the config.
        """

        super().__init__()
        self.config = get_config_from_file(
            config_or_config_file, config_class=config_class, model_config_class=model_config_class
        ) if not isinstance(config_or_config_file, Config) else config_or_config_file
        self.model_config = self.config.model.model_config
        if model_class is not None:
            CONFIG_TO_MODEL_CLASS[self.model_config.__class__.__name__] = model_class

        ########################################
        ## We start with setting up loggers and process groups
        ########################################

        # Initialise all process groups
        self.parallel_context = ParallelContext(
            tensor_parallel_size=self.config.parallelism.tp,
            pipeline_parallel_size=self.config.parallelism.pp,
            data_parallel_size=self.config.parallelism.dp,
            expert_parallel_size=self.config.parallelism.expert_parallel_size,
        )

        self.pre_init()

        # Set log levels
        set_ranks_logging_level(parallel_context=self.parallel_context, logging_config=self.config.logging)

        # Log benchmark info
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1":
            log_throughput(self.config, self.parallel_context)

        ########################################
        ## Setting up our model, optimizers, schedulers, etc.
        ########################################

        # Set random states
        set_random_seed(self.config.general.seed)

        # Init model and build on pp ranks
        self.random_states = init_random_states(
            parallel_config=self.config.parallelism, tp_pg=self.parallel_context.tp_pg
        )
        self.model = self.init_model()  # Defines self.model
        self.unwrapped_model: NanotronModel = (
            self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        )

        # TODO: find a better way to handle this
        parametrization_method = (
            ParametrizationMethod.SPECTRAL_MUP
            if hasattr(self.config.model.init_method, "use_mup") and self.config.model.init_method.use_mup
            else ParametrizationMethod.STANDARD
        )
        log_memory(logger=logger)
        # Init optimizer
        self.optimizer, self.grad_accumulator = init_optimizer_and_grad_accumulator(
            parametrization_method=parametrization_method,
            model=self.model,
            optimizer_args=self.config.optimizer,
            parallel_context=self.parallel_context,
        )
        # Init learning rate scheduler
        # HAS TO BE HERE (before loading the optimizer) to resume training deterministically
        self.lr_scheduler = lr_scheduler_builder(
            optimizer=self.optimizer,
            lr_scheduler_args=self.config.optimizer.learning_rate_scheduler,
            total_training_steps=self.config.tokens.train_steps,
        )
        log_memory(logger=logger)
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_optimizer:
            load_optimizer(
                optimizer=self.optimizer,
                parallel_context=self.parallel_context,
                root_folder=self.init_checkpoint_path,
                param_shard_metadata=self.param_shard_metadata,
                model=self.unwrapped_model,
                map_location="cpu",
            )

        if self.init_checkpoint_path is not None and self.config.checkpoints.load_lr_scheduler:
            load_lr_scheduler(
                lr_scheduler=self.lr_scheduler,
                is_zero=self.config.optimizer.zero_stage,
                parallel_context=self.parallel_context,
                root_folder=self.init_checkpoint_path,
            )

        # Define iteration start state
        if self.init_checkpoint_path is not None and self.config.checkpoints.load_lr_scheduler:
            checkpoint_metadata = load_meta(
                parallel_context=self.parallel_context, root_folder=self.init_checkpoint_path
            )
            assert isinstance(checkpoint_metadata.metas, TrainingMetadata)
            log_rank(str(checkpoint_metadata), logger=logger, level=logging.INFO, rank=0)
            self.metadata: TrainingMetadata = checkpoint_metadata.metas
            # NOTE: we should not change data stages
            assert (
                self.config.tokens.train_steps > self.metadata.last_train_step
            ), f"Loaded checkpoint has already trained {self.metadata.last_train_step} batches, you need to specify a higher `config.tokens.train_steps`"
        else:
            data_stages = [
                DataStageMetadata(
                    name=stage.name, start_training_step=stage.start_training_step, consumed_train_samples=0
                )
                for stage in self.config.data_stages
            ]
            self.metadata: TrainingMetadata = TrainingMetadata(
                consumed_train_samples=0, last_train_step=0, last_stage_idx=0, data_stages=data_stages
            )

        # Setup tensorboard write and log writers on output rank
        self.logger_ranks = self.parallel_context.get_global_rank(
            ep_rank=0, pp_rank=self.unwrapped_model.output_pp_rank, dp_rank=0, tp_rank=0
        ).flatten()
        self.loggerwriter = self.setup_log_writers()

        # Log where each module is instantiated
        self.unwrapped_model.log_modules(level=logging.DEBUG, group=self.parallel_context.world_pg, rank=0)

        self.micro_batch_size = self.config.tokens.micro_batch_size
        self.n_micro_batches_per_batch = self.config.tokens.batch_accumulation_per_replica
        self.global_batch_size = (
            self.micro_batch_size * self.n_micro_batches_per_batch * self.parallel_context.dp_pg.size()
        )
        self.sequence_length = self.config.tokens.sequence_length
        self.iteration_step = self.metadata.last_train_step
        self.limit_val_batches = self.config.tokens.limit_val_batches
        # NOTE: the dataloader currently in use for the current training stage
        self.current_dataloader: Optional[Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]] = None
        self._val_dataloader_factory: Optional[Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]] = None
        self._cola_strict_debug_step = int(os.environ.get("COLA_STRICT_DEBUG_STEP", "0") or "0")
        self._cola_strict_debug_topk = int(os.environ.get("COLA_STRICT_DEBUG_TOPK", "12") or "12")
        self._cola_tensor_dump_step = int(os.environ.get("COLA_STRICT_TENSOR_DUMP_STEP", "0") or "0")
        self._cola_tensor_dump_dir = os.environ.get("COLA_STRICT_TENSOR_DUMP_DIR", "").strip()

        self.post_init()

    def pre_init(self):
        self.init_checkpoint_path = parse_ckpt_path(config=self.config, parallel_context=self.parallel_context)

    def post_init(self):
        # S3 Mover and save initial state
        if self.config.s3_upload is not None:
            # NOTE: Only local rank 0 should upload
            dummy = bool(int(os.environ.get("LOCAL_RANK", None)) != 0)
            self.s3_mover = S3Mover(
                local_path=self.config.checkpoints.checkpoints_path,
                s3_path=self.config.s3_upload.upload_s3_path,
                remove_after_upload=self.config.s3_upload.remove_after_upload,
                s5cmd_numworkers=self.config.s3_upload.s5cmd_numworkers,
                s5cmd_concurrency=self.config.s3_upload.s5cmd_concurrency,
                s5cmd_path=self.config.s3_upload.s5cmd_path,
                dummy=dummy,
            )
        else:
            self.s3_mover = None

    def _should_emit_cola_strict_debug(self) -> bool:
        return self._cola_strict_debug_step > 0 and self.iteration_step == self._cola_strict_debug_step

    def _get_grad_tensor_for_debug(self, name: str, param: NanotronParameter) -> Optional[torch.Tensor]:
        if self.grad_accumulator is not None:
            try:
                return self.grad_accumulator.get_grad_buffer(name)
            except Exception:
                return None
        return param.grad

    def _compute_mp_param_grad_norms(
        self, named_parameters: List[Tuple[str, NanotronParameter]]
    ) -> List[Tuple[str, float]]:
        results: List[Tuple[str, float]] = []
        for name, param in named_parameters:
            grad = self._get_grad_tensor_for_debug(name=name, param=param)
            if grad is None:
                continue
            local_sq = grad.detach().float().pow(2).sum()
            mp_sq = local_sq.clone()
            dist.all_reduce(mp_sq, group=self.parallel_context.mp_pg, op=dist.ReduceOp.SUM)
            results.append((name, float(mp_sq.sqrt().item())))
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compute_mp_total_grad_norm(self, named_parameters: List[Tuple[str, NanotronParameter]]) -> float:
        local_total_sq = torch.zeros((), device="cuda", dtype=torch.float32)
        for name, param in named_parameters:
            grad = self._get_grad_tensor_for_debug(name=name, param=param)
            if grad is None:
                continue
            local_total_sq = local_total_sq + grad.detach().float().pow(2).sum()
        mp_total_sq = local_total_sq.clone()
        dist.all_reduce(mp_total_sq, group=self.parallel_context.mp_pg, op=dist.ReduceOp.SUM)
        return float(mp_total_sq.sqrt().item())

    def _compute_mp_update_norms(
        self,
        named_parameters: List[Tuple[str, NanotronParameter]],
        pre_step_snapshots: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        param_map = dict(named_parameters)
        mp_update_norms: Dict[str, float] = {}
        for name, pre_tensor_cpu in pre_step_snapshots.items():
            param = param_map[name]
            local_delta_sq_cpu = (param.detach().float().cpu() - pre_tensor_cpu).pow(2).sum()
            local_delta_sq = torch.tensor(float(local_delta_sq_cpu.item()), device=param.device, dtype=torch.float32)
            mp_delta_sq = local_delta_sq.clone()
            dist.all_reduce(mp_delta_sq, group=self.parallel_context.mp_pg, op=dist.ReduceOp.SUM)
            mp_update_norms[name] = float(mp_delta_sq.sqrt().item())
        return mp_update_norms

    def _log_cola_strict_debug_payload(self, payload: Dict[str, Union[int, float, str, list, dict]]) -> None:
        if dist.get_rank(self.parallel_context.world_pg) != 0:
            return
        log_rank(
            f"[COLA_STRICT_DEBUG] {json.dumps(payload, sort_keys=True)}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    def _should_emit_cola_tensor_dump(self) -> bool:
        return (
            self._cola_tensor_dump_step > 0
            and self.iteration_step == self._cola_tensor_dump_step
            and len(self._cola_tensor_dump_dir) > 0
        )

    def _is_dp0_rank(self) -> bool:
        return dist.get_rank(self.parallel_context.dp_pg) == 0

    def _infer_tp_shard_dim(
        self, local_shape: Tuple[int, ...], full_shape: Tuple[int, ...], tp_size: int
    ) -> Optional[int]:
        dims = [
            d
            for d in range(len(full_shape))
            if full_shape[d] == local_shape[d] * tp_size
            and all(full_shape[k] == local_shape[k] for k in range(len(full_shape)) if k != d)
        ]
        if len(dims) == 1:
            return dims[0]
        return None

    def _materialize_param_like_full_tensor(
        self, local_tensor: torch.Tensor, param: NanotronParameter
    ) -> Optional[torch.Tensor]:
        tp_pg = self.parallel_context.tp_pg
        tp_size = tp_pg.size()
        tp_rank = dist.get_rank(tp_pg)
        local = local_tensor.detach()

        # No TP to reconstruct.
        if tp_size == 1:
            return local.cpu().clone()

        unsharded_shape = tuple(local.shape)
        is_tp_sharded = False
        if param.is_sharded:
            sharded_info = param.get_sharded_info()
            is_tp_sharded = sharded_info.is_tp_sharded(self.parallel_context)
            if is_tp_sharded:
                unsharded_shape = tuple(sharded_info.unsharded_shape)

        # For TP-replicated tensors, rank0 already has the full tensor.
        if not is_tp_sharded:
            if tp_rank == 0:
                return local.cpu().clone()
            return None

        gathered = [torch.empty_like(local) for _ in range(tp_size)]
        dist.all_gather(gathered, local.contiguous(), group=tp_pg)
        if tp_rank != 0:
            return None

        shard_dim = self._infer_tp_shard_dim(tuple(local.shape), unsharded_shape, tp_size)
        if shard_dim is None:
            raise RuntimeError(
                f"Cannot infer TP shard dim for tensor local={tuple(local.shape)} full={unsharded_shape} tp={tp_size}"
            )
        full = torch.cat(gathered, dim=shard_dim)
        if tuple(full.shape) != unsharded_shape:
            raise RuntimeError(
                f"Reconstructed shape mismatch. got={tuple(full.shape)} expected={unsharded_shape}"
            )
        return full.cpu()

    def _capture_full_params_for_dump(
        self, named_parameters: List[Tuple[str, NanotronParameter]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._is_dp0_rank():
            return None
        out: Dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            full = self._materialize_param_like_full_tensor(param.detach(), param)
            if full is not None:
                out[name] = full
        return out if dist.get_rank(self.parallel_context.tp_pg) == 0 else None

    def _capture_full_grads_for_dump(
        self, named_parameters: List[Tuple[str, NanotronParameter]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._is_dp0_rank():
            return None
        out: Dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            grad = self._get_grad_tensor_for_debug(name=name, param=param)
            if grad is None:
                continue
            full = self._materialize_param_like_full_tensor(grad, param)
            if full is not None:
                out[name] = full
        return out if dist.get_rank(self.parallel_context.tp_pg) == 0 else None

    def _capture_first_microbatch_inputs(
        self, micro_batches: List[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._is_dp0_rank() or len(micro_batches) == 0:
            return None
        if dist.get_rank(self.parallel_context.tp_pg) != 0:
            return None
        first = micro_batches[0]
        out: Dict[str, torch.Tensor] = {}
        for k, v in first.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu().clone()
        return out

    def _capture_forward_output_for_dump(self, outputs: Iterable[Dict]) -> Optional[torch.Tensor]:
        if not self._is_dp0_rank():
            return None
        if dist.get_rank(self.parallel_context.tp_pg) != 0:
            # Non-root TP ranks still need to participate in gather below.
            tp_root = False
        else:
            tp_root = True

        outputs = list(outputs)
        if len(outputs) == 0:
            return None
        first = outputs[0]
        if "sharded_logits" not in first or not isinstance(first["sharded_logits"], torch.Tensor):
            return None

        local_logits = first["sharded_logits"].detach()
        tp_pg = self.parallel_context.tp_pg
        tp_size = tp_pg.size()
        if tp_size == 1:
            return local_logits.cpu().clone() if tp_root else None

        gathered = [torch.empty_like(local_logits) for _ in range(tp_size)]
        dist.all_gather(gathered, local_logits.contiguous(), group=tp_pg)
        if not tp_root:
            return None

        # For sharded logits, TP split is on vocab axis (last dim).
        return torch.cat(gathered, dim=-1).cpu()

    def _write_cola_tensor_dump(self, payload: Dict[str, object]) -> None:
        if not self._is_dp0_rank() or dist.get_rank(self.parallel_context.tp_pg) != 0:
            return
        os.makedirs(self._cola_tensor_dump_dir, exist_ok=True)
        path = Path(self._cola_tensor_dump_dir) / f"tensor_dump_step_{self.iteration_step}.pt"
        torch.save(payload, path)
        log_rank(
            f"[COLA_STRICT_TENSOR_DUMP] saved {path}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

    def _extract_compression_stats(self):
        """Extract compression statistics from CoLA or TensorTrain models for wandb logging."""
        stats = {}

        # Try to detect ColaLayer (from basic_cola_llama.py)
        cola_layers = []
        tt_layers = []

        for name, module in self.unwrapped_model.named_modules():
            # Detect ColaLayer by checking class name and attributes
            if module.__class__.__name__ == "ColaLayer":
                cola_layers.append(module)
            # Detect TensorTrainLinear by checking class name and attributes
            elif module.__class__.__name__ == "TensorTrainLinear":
                tt_layers.append(module)

        # Extract CoLA statistics
        if cola_layers:
            total_cola_params = 0
            total_dense_params = 0
            rank_info = {}

            for module in cola_layers:
                # CoLA parameters: cola_a + cola_b + optional bias
                cola_params = (
                    module.in_features * module.rank +
                    module.rank * module.out_features +
                    (module.out_features if hasattr(module, 'bias') and module.bias is not None else 0)
                )
                # Equivalent dense parameters
                dense_params = (
                    module.in_features * module.out_features +
                    (module.out_features if hasattr(module, 'bias') and module.bias is not None else 0)
                )
                total_cola_params += cola_params
                total_dense_params += dense_params

                # Track rank distribution
                rank_key = f"rank_{module.rank}"
                rank_info[rank_key] = rank_info.get(rank_key, 0) + 1

            compression_ratio = total_dense_params / total_cola_params if total_cola_params > 0 else 0
            stats.update({
                "compression/type": "CoLA",
                "compression/num_layers": len(cola_layers),
                "compression/total_params": total_cola_params,
                "compression/dense_params": total_dense_params,
                "compression/params_saved": total_dense_params - total_cola_params,
                "compression/ratio": compression_ratio,
                "compression/rank_distribution": rank_info,
            })

            # Add config-specific info if available
            if hasattr(self.model_config, 'mlp_rank'):
                stats["compression/mlp_rank"] = self.model_config.mlp_rank
            if hasattr(self.model_config, 'attn_rank'):
                stats["compression/attn_rank"] = self.model_config.attn_rank

        # Extract TensorTrain statistics
        elif tt_layers:
            total_tt_params = 0
            total_dense_params = 0
            rank_info = {}
            cores_info = {}

            for module in tt_layers:
                # TT parameters: sum of all cores + optional bias
                tt_params = sum(core.numel() for core in module.cores)
                if hasattr(module, 'bias') and module.bias is not None:
                    tt_params += module.bias.numel()

                # Equivalent dense parameters
                dense_params = module.in_features * module.out_features
                if hasattr(module, 'bias') and module.bias is not None:
                    dense_params += module.out_features

                total_tt_params += tt_params
                total_dense_params += dense_params

                # Track TT ranks (convert list to string for wandb)
                rank_key = str(module.tt_ranks)
                rank_info[rank_key] = rank_info.get(rank_key, 0) + 1

                # Track number of cores
                cores_key = f"cores_{module.tt_cores_count}"
                cores_info[cores_key] = cores_info.get(cores_key, 0) + 1

            compression_ratio = total_dense_params / total_tt_params if total_tt_params > 0 else 0
            stats.update({
                "compression/type": "TensorTrain",
                "compression/num_layers": len(tt_layers),
                "compression/total_params": total_tt_params,
                "compression/dense_params": total_dense_params,
                "compression/params_saved": total_dense_params - total_tt_params,
                "compression/ratio": compression_ratio,
                "compression/rank_distribution": rank_info,
                "compression/cores_distribution": cores_info,
            })

            # Add config-specific info if available
            if hasattr(self.model_config, 'tt_rank'):
                stats["compression/tt_rank"] = self.model_config.tt_rank
            if hasattr(self.model_config, 'tt_cores'):
                stats["compression/tt_cores"] = self.model_config.tt_cores
            if hasattr(self.model_config, 'tt_rank_ratio'):
                stats["compression/tt_rank_ratio"] = self.model_config.tt_rank_ratio

        return stats

    def pre_training(self, *args, **kwargs):
        self._print_training_plan()

        metadata: TrainingMetadata = self.metadata

        log_rank(
            f"[Start training] datetime: {datetime.datetime.now()} | mbs: {self.micro_batch_size} | grad_accum: {self.n_micro_batches_per_batch} | global_batch_size: {self.global_batch_size} | sequence_length: {self.sequence_length} | train_steps: {self.config.tokens.train_steps} | start_iteration_step: {metadata.last_train_step} | consumed_train_samples: {metadata.consumed_train_samples}",  # noqa
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        current_time = datetime.datetime.now().strftime("%d/%m/%Y_%H:%M:%S")
        if dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0] and wandb is not None:
            # Extract model compression statistics for wandb logging
            model_stats = self._extract_compression_stats()

            wandb.init(
                project=self.config.general.project,
                entity=self.config.general.entity,
                name=f"{self.config.general.run}_{current_time}",
                config={
                    "nanotron_config": self.config.as_dict(),
                    "model_compression": model_stats,
                },
            )

            # Log model compression statistics to wandb
            if model_stats:
                wandb.log(model_stats, step=0)

    def post_train_step(self):

        # Update our background upload/removal of checkpoints
        if self.s3_mover is not None:
            self.s3_mover.update()

    def post_training(self):
        if self.s3_mover is not None:
            self.s3_mover.distributed_wait_for_completion(group=self.parallel_context.world_pg)

    def _print_training_plan(self):
        if hasattr(self.config, "data_stages") and self.config.data_stages is not None:
            stages_info = "".join(
                f"[Stage {stage.name}] start from step {stage.start_training_step} \n"
                for stage in self.config.data_stages
            )
            full_log_message = (
                f"[Training Plan] There are {len(self.config.data_stages)} training stages \n{stages_info}"
            )
            log_rank(full_log_message, logger=logger, level=logging.INFO, rank=0)

    def _update_dataloader_based_on_training_stages(self, dataloaders: Union[List[DataLoader], DataLoader, Dict]):
        from collections.abc import Generator

        if not hasattr(self.config, "data_stages") or self.config.data_stages is None:
            if self.current_dataloader is None:
                if isinstance(dataloaders, tuple):
                    dataloader = dataloaders[0]
                elif isinstance(dataloaders, dict):
                    # Check if it's the new nested format
                    first_key = list(dataloaders.keys())[0]
                    stage_data = dataloaders[first_key]
                    if isinstance(stage_data, dict) and "train" in stage_data:
                        # New format: extract train and validation dataloaders
                        train_dl = stage_data["train"]
                        train_dl = train_dl() if callable(train_dl) else train_dl
                        self.current_dataloader = sanity_check_dataloader(
                            dataloader=train_dl, parallel_context=self.parallel_context, config=self.config
                        )
                        # Store validation dataloader factory (don't initialize yet)
                        self._val_dataloader_factory = stage_data.get("validation", None)
                        return
                    else:
                        # Old format
                        dataloader = stage_data
                else:
                    dataloader = dataloaders
                self.current_dataloader = sanity_check_dataloader(
                    dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
                )
                self._val_dataloader_factory = None
            return
        elif isinstance(dataloaders, Generator):
            # TODO(xrsrke): this is a hacky way to handle DoReMi's dataloader
            # remove this in the next PR
            self.current_dataloader = dataloaders
            self._val_dataloader_factory = None
            return

        assert len(dataloaders) > 0, "No dataloaders provided"
        assert len(dataloaders) == len(
            self.config.data_stages
        ), "Number of dataloaders should match the number of dataset stages"

        def clear_dataloader_from_memory(dataloader: DataLoader, stage_name: str):
            import gc

            log_rank(
                f"[Training Stage: {stage_name}] Clearing the previous training stage's dataloader and datasets from memory",
                logger=logger,
                level=logging.INFO,
            )

            # NOTE: Clear dataloader from memory
            del dataloader.dataset
            del dataloader.sampler
            del dataloader.batch_sampler

            gc.collect()

        dataloader = None

        def find_stage_idx_to_resume():
            reversed_data_stages = sorted(self.config.data_stages, key=lambda x: x.start_training_step, reverse=True)
            for idx, stage in enumerate(reversed_data_stages):
                if self.iteration_step >= stage.start_training_step:
                    return len(self.config.data_stages) - idx - 1
            return None

        stage_idx_to_resume = find_stage_idx_to_resume()

        for stage_idx, stage in enumerate(self.config.data_stages):
            if stage_idx < self.metadata.last_stage_idx:
                continue

            stage = cast(DatasetStageArgs, stage)

            is_resume_from_training = self.current_dataloader is None and stage_idx_to_resume == stage_idx
            if (stage.start_training_step == self.iteration_step) or is_resume_from_training:
                if self.current_dataloader is not None:
                    prev_stage_name = self.config.data_stages[stage_idx - 1].name
                    prev_stage_data = dataloaders[prev_stage_name]

                    # Handle both old and new formats for clearing memory
                    if isinstance(prev_stage_data, dict) and "train" in prev_stage_data:
                        # New format
                        prev_train_dl = prev_stage_data["train"]
                        if isinstance(prev_train_dl, DataLoader):
                            clear_dataloader_from_memory(prev_train_dl, stage_name=stage.name)
                    elif isinstance(prev_stage_data, DataLoader):
                        # Old format
                        clear_dataloader_from_memory(prev_stage_data, stage_name=stage.name)

                self.metadata.last_stage_idx = stage_idx

                if is_resume_from_training:
                    remaining_train_steps = compute_remain_train_steps_of_a_data_stage_from_ckp(
                        stage, self.config, self.metadata
                    )
                    consumed_train_steps = get_consumed_train_samples_of_a_data_stage_from_ckp(stage, self.metadata)
                    log_rank(
                        f"Resuming training from stage {stage.name}, it has trained for {consumed_train_steps} samples and has {remaining_train_steps} remaining train steps",
                        logger=logger,
                        level=logging.INFO,
                        rank=0,
                    )

                stage_data = dataloaders[stage.name]

                # Handle new nested dict format
                if isinstance(stage_data, dict) and "train" in stage_data:
                    # Extract train dataloader
                    dataloader = stage_data["train"]
                    dataloader = dataloader() if callable(dataloader) else dataloader

                    # Store validation dataloader factory (don't initialize)
                    self._val_dataloader_factory = stage_data.get("validation", None)
                else:
                    # Old format: single dataloader
                    dataloader = stage_data
                    dataloader = dataloader() if callable(dataloader) else dataloader
                    self._val_dataloader_factory = None
                break

        if dataloader is not None:
            self.current_dataloader = sanity_check_dataloader(
                dataloader=dataloader, parallel_context=self.parallel_context, config=self.config
            )

    def train(
        self,
        dataloader_or_dls: Dict[
            str, Union[Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]], Tuple[Iterator, ...]]
        ],
        **kwargs,
    ) -> None:
        self.pre_training(**kwargs)

        if self.config.checkpoints.save_initial_state and self.init_checkpoint_path is None:
            self.save_checkpoint()

        self.pipeline_engine: PipelineEngine = self.config.parallelism.pp_engine

        self.pipeline_engine.nb_microbatches = self.n_micro_batches_per_batch

        # TODO @nouamanetazi: refactor this
        # Useful mapping
        self.unwrapped_model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.unwrapped_model.module_id_to_prefix = {
            id(module): f"{module_name}." for module_name, module in self.unwrapped_model.named_modules()
        }
        # Fix the root_model
        self.unwrapped_model.module_id_to_prefix[id(self.unwrapped_model)] = ""

        self.initial_iter_step = self.metadata.last_train_step + 1
        self.last_iter_step = self.config.tokens.train_steps

        prof = get_profiler(config=self.config)
        # free memory
        gc.collect()
        torch.cuda.empty_cache()
        with prof:
            for self.iteration_step in range(self.initial_iter_step, self.last_iter_step + 1):
                if self.iteration_step == self.initial_iter_step + 5:
                    torch.cuda.cudart().cudaProfilerStart()

                if isinstance(prof, torch.profiler.profile):
                    prof.step()

                self.iteration_start_time = time.time()
                self._update_dataloader_based_on_training_stages(dataloader_or_dls)

                if self.iteration_step >= self.initial_iter_step + 5:
                    torch.cuda.nvtx.range_push("training_step{}".format(self.iteration_step))
                # Training step
                outputs, loss_avg = self.training_step(dataloader=self.current_dataloader)
                if self.iteration_step >= self.initial_iter_step + 5:
                    torch.cuda.nvtx.range_pop()

                # Training Logs
                # TODO(xrsrke): refactor using callbacks would be better
                self.metadata.consumed_train_samples += self.global_batch_size
                self.metadata.last_train_step = self.iteration_step
                self.metadata.data_stages[
                    self.metadata.last_stage_idx
                ].consumed_train_samples += self.global_batch_size

                if (self.iteration_step - 1) % self.config.logging.iteration_step_info_interval == 0:
                    self.train_step_logs(outputs=outputs, loss_avg=loss_avg)

                # Checkpoint
                if self.iteration_step % self.config.checkpoints.checkpoint_interval == 0:
                    self.save_checkpoint()

                # Validation
                if (self.config.tokens.val_check_interval > 0 and
                    self._val_dataloader_factory is not None and
                    self.iteration_step % self.config.tokens.val_check_interval == 0):
                    self.run_validation()

                if self.iteration_step == self.initial_iter_step + 5 + 3:
                    torch.cuda.cudart().cudaProfilerStop()

        dist.barrier()  # let's wait for everyone before leaving

        if self.config.checkpoints.save_final_state:
            self.save_checkpoint()

        self.post_training()

    def training_step(
        self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]
    ) -> Tuple[Iterable[Dict], Optional[torch.Tensor]]:
        before_tbi_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator, self.lr_scheduler
        )

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger)

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_push("train_batch_iter")
        micro_batches = [next(dataloader) for _ in range(self.n_micro_batches_per_batch)]
        outputs = self.pipeline_engine.train_batch_iter(
            model=self.model,
            pg=self.parallel_context.pp_pg,
            batch=iter(micro_batches),
            nb_microbatches=self.n_micro_batches_per_batch,
            grad_accumulator=self.grad_accumulator,
        )
        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_pop()

        if self.iteration_step < self.initial_iter_step + 5:
            log_memory(logger=logger)

        after_tbi_sanity_checks(self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator)

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_push("sync_gradients_across_dp")
        if isinstance(self.model, DistributedDataParallel) and self.grad_accumulator is not None:
            # Wait for fp32 grads allreduce to finish to make sure grads are synced across DP
            assert (
                self.grad_accumulator.fp32_grads_allreduce_handle is not None
            ), "No fp32_grads_allreduce_handle maybe you're using only a single training process"
            self.grad_accumulator.fp32_grads_allreduce_handle.wait()

        # Sync tied weights
        if not isinstance(self.model, DistributedDataParallel):
            # Manually sync across DP if it's not handled by DDP
            sync_gradients_across_dp(
                module=self.model,
                dp_pg=self.parallel_context.dp_pg,
                reduce_op=dist.ReduceOp.AVG,
                # TODO @thomasw21: This is too memory hungry, instead we run all_reduce
                reduce_scatter=False,  # optimizer.inherit_from(ZeroDistributedOptimizer),
                grad_accumulator=self.grad_accumulator,
            )

        # TODO @nouamane: Put this in hooks so we can overlap communication with gradient computation on the last backward pass.
        sync_tied_weights_gradients(
            module=self.unwrapped_model,
            parallel_context=self.parallel_context,
            grad_accumulator=self.grad_accumulator,
        )
        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_pop()

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_push("clip_gradients")
        named_parameters = [
            (name, param)
            for name, param in self.unwrapped_model.get_named_params_with_correct_tied()
            if param.requires_grad
        ]
        tensor_dump_now = self._should_emit_cola_tensor_dump()
        tensor_dump_payload: Optional[Dict[str, object]] = None
        if tensor_dump_now:
            tensor_dump_payload = {
                "step": int(self.iteration_step),
                "input_batch_before_iter0": self._capture_first_microbatch_inputs(micro_batches),
                "weights_after_init_before_iter0": self._capture_full_params_for_dump(named_parameters),
                "forward_output_before_loss": self._capture_forward_output_for_dump(outputs),
            }
        strict_debug_now = self._should_emit_cola_strict_debug()
        strict_debug_pre_clip_topk: List[Tuple[str, float]] = []
        strict_debug_pre_step_snapshots: Dict[str, torch.Tensor] = {}
        strict_debug_total_grad_norm_mp_pre_clip: Optional[float] = None
        if strict_debug_now:
            strict_debug_total_grad_norm_mp_pre_clip = self._compute_mp_total_grad_norm(named_parameters=named_parameters)
            strict_debug_pre_clip_topk = self._compute_mp_param_grad_norms(named_parameters=named_parameters)[
                : self._cola_strict_debug_topk
            ]
            param_map = dict(named_parameters)
            strict_debug_pre_step_snapshots = {
                name: param_map[name].detach().float().cpu().clone() for name, _ in strict_debug_pre_clip_topk
            }

        # Clip gradients
        if self.config.optimizer.clip_grad is not None:
            self.grad_norm_unclipped = clip_grad_norm(
                mp_pg=self.parallel_context.mp_pg,
                named_parameters=named_parameters,
                grad_accumulator=self.grad_accumulator,
                max_norm=self.config.optimizer.clip_grad,
            )
        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_pop()

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_push("loss_avg")
        # Compute DP average loss and overlap with optimizer step
        if isinstance(outputs[0]["loss"], torch.Tensor):
            # This is an average on only one data rank.
            loss_avg = torch.stack(
                [output["loss"] for output in outputs]
            ).sum()  # already divided by n_micro_batches_per_batch
            # sync loss across DP
            handle = dist.all_reduce(loss_avg, group=self.parallel_context.dp_pg, async_op=True, op=dist.ReduceOp.AVG)
        else:
            loss_avg = None
            handle = None
        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_pop()

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_push("optimizer_step")
        # Move optimizer states back to GPU before optimizer step
        if (
            self.init_checkpoint_path is not None
            and self.config.checkpoints.load_optimizer
            and self.iteration_step == self.initial_iter_step
        ):
            state_dict_to_device(self.optimizer.state_dict(), "cuda")

        before_optim_step_sanity_checks(
            self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator, self.optimizer
        )

        if tensor_dump_now and tensor_dump_payload is not None:
            tensor_dump_payload["grad_before_step0_update"] = self._capture_full_grads_for_dump(named_parameters)

        # Apply gradient
        self.optimizer.step()
        if strict_debug_now:
            param_map = dict(named_parameters)
            strict_debug_post_clip_topk = []
            for name, _ in strict_debug_pre_clip_topk:
                if name in param_map:
                    norms = self._compute_mp_param_grad_norms([(name, param_map[name])])
                    if len(norms) == 1:
                        strict_debug_post_clip_topk.append((name, norms[0][1]))
            strict_debug_update_norms = self._compute_mp_update_norms(
                named_parameters=named_parameters, pre_step_snapshots=strict_debug_pre_step_snapshots
            )
            grad_norm_unclipped = self.grad_norm_unclipped.item() if hasattr(self, "grad_norm_unclipped") else None
            clip_max_norm = self.config.optimizer.clip_grad
            clip_coef = None
            if grad_norm_unclipped is not None and clip_max_norm is not None:
                clip_coef = min(1.0, float(clip_max_norm) / (float(grad_norm_unclipped) + 1.0e-6))

            payload = {
                "step": int(self.iteration_step),
                "world_rank": int(dist.get_rank(self.parallel_context.world_pg)),
                "tp_rank": int(dist.get_rank(self.parallel_context.tp_pg)),
                "tp_size": int(self.parallel_context.tp_pg.size()),
                "dp_rank": int(dist.get_rank(self.parallel_context.dp_pg)),
                "dp_size": int(self.parallel_context.dp_pg.size()),
                "grad_norm_mp_pre_clip": strict_debug_total_grad_norm_mp_pre_clip,
                "grad_norm_unclipped": grad_norm_unclipped,
                "clip_max_norm": float(clip_max_norm) if clip_max_norm is not None else None,
                "clip_coef_clamped": clip_coef,
                "topk_grad_norms_mp_pre_clip": [
                    {"name": name, "value": value} for name, value in strict_debug_pre_clip_topk
                ],
                "topk_grad_norms_mp_post_clip": [
                    {"name": name, "value": value} for name, value in strict_debug_post_clip_topk
                ],
                "sentinel_update_norms_mp": strict_debug_update_norms,
            }
            self._log_cola_strict_debug_payload(payload)
        self.optimizer.zero_grad()

        # Update the learning rate
        self.lr_scheduler.step()

        if self.iteration_step >= self.initial_iter_step + 5:
            torch.cuda.nvtx.range_pop()

        after_optim_step_sanity_checks(self.config, self.parallel_context, self.unwrapped_model, self.grad_accumulator)

        if handle is not None:
            handle.wait()

        if tensor_dump_now and tensor_dump_payload is not None:
            tensor_dump_payload["lm_loss"] = float(loss_avg.item()) if loss_avg is not None else None
            tensor_dump_payload["grad_norm"] = (
                float(self.grad_norm_unclipped.item()) if hasattr(self, "grad_norm_unclipped") else None
            )
            tensor_dump_payload["lr"] = float(self.lr_scheduler.get_last_lr()[0])
            tensor_dump_payload["weight_after_step0_optimizer_update"] = self._capture_full_params_for_dump(named_parameters)
            self._write_cola_tensor_dump(tensor_dump_payload)

        self.post_train_step()

        return outputs, loss_avg

    def validation_step(self, dataloader: Iterator[Dict[str, Union[torch.Tensor, TensorPointer]]]) -> Iterable[Dict]:
        # Convert to list to count actual batches (validation dataset might be smaller than limit_val_batches)
        batch_list = list(itertools.islice(dataloader, self.limit_val_batches))
        actual_num_batches = len(batch_list)

        outputs = self.pipeline_engine.validate_batch_iter(
            model=self.model,
            batch=iter(batch_list),
            nb_microbatches=actual_num_batches,
        )
        return outputs

    def run_validation(self) -> None:
        """
        Execute validation and log metrics.
        Called from training loop when val_check_interval is reached.
        """
        log_rank(
            f"[Validation] Running validation at step {self.iteration_step}",
            logger=logger,
            level=logging.INFO,
            rank=0
        )

        # Reinitialize validation dataloader from factory to always start from beginning
        if self._val_dataloader_factory is not None:
            val_dl = self._val_dataloader_factory() if callable(self._val_dataloader_factory) else self._val_dataloader_factory
            val_dataloader = sanity_check_dataloader(
                dataloader=val_dl, parallel_context=self.parallel_context, config=self.config
            )

        # Set model to eval mode
        self.model.eval()

        # Run validation step
        val_outputs = self.validation_step(dataloader=val_dataloader)

        # Compute average loss across validation batches
        if isinstance(val_outputs[0]["loss"], torch.Tensor):
            # Sum losses from all micro-batches
            val_loss = torch.stack([output["loss"] for output in val_outputs]).sum()

            # Sync across DP ranks
            dist.all_reduce(val_loss, group=self.parallel_context.dp_pg, op=dist.ReduceOp.AVG)

            # Compute perplexity
            val_perplexity = torch.exp(val_loss)

            # Log validation metrics
            self.validation_step_logs(val_loss=val_loss, val_perplexity=val_perplexity)
        else:
            log_rank(
                "[Validation] No loss computed (not on output PP rank)",
                logger=logger,
                level=logging.INFO,
                rank=0
            )

        # Set model back to train mode
        self.model.train()

    def train_step_logs(
        self,
        outputs: Iterable[Dict[str, Union[torch.Tensor, TensorPointer]]],
        loss_avg: Optional[torch.Tensor],
    ) -> None:
        # TODO @nouamanetazi: Megatron-LM seems to be using a barrier to report their interval time. Check if this is necessary. https://github.com/NouamaneTazi/Megatron-LM/blob/e241a96c3085b18e36c6cee1d68a8155de77b5a6/megatron/training.py#L607
        dist.barrier()
        torch.cuda.synchronize()
        elapsed_time_per_iteration_ms = (time.time() - self.iteration_start_time) * 1000
        tokens_per_sec = (
            self.global_batch_size * self.sequence_length / (elapsed_time_per_iteration_ms / 1000)
        )  # tokens_per_sec is calculated using sequence_length
        model_tflops, hardware_tflops = self.unwrapped_model.get_flops_per_sec(
            iteration_time_in_sec=elapsed_time_per_iteration_ms / 1000,
            sequence_length=self.sequence_length,
            global_batch_size=self.global_batch_size,
        )

        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"

            lr = self.lr_scheduler.get_last_lr()[0]

            log_entries = [
                # LogItem("consumed_samples", self.consumed_train_samples, "human_format"),  # , "12d"),
                LogItem(
                    "consumed_tokens",
                    self.metadata.consumed_train_samples * self.config.tokens.sequence_length,
                    "human_format",
                ),  # , "12d"),
                LogItem("elapsed_time_per_iteration_ms", elapsed_time_per_iteration_ms, "human_format"),  # , ".1f"),
                LogItem("tokens_per_sec", tokens_per_sec, "human_format"),  # , "1.6E"),
                LogItem(
                    "tokens_per_sec_per_gpu", tokens_per_sec / self.parallel_context.world_pg.size(), "human_format"
                ),  # , "1.6E"),
                LogItem("global_batch_size", self.global_batch_size, "human_format"),  # , "5d"),
                LogItem("lm_loss", loss_avg.item(), "human_format"),  # , "1.6E"),
                LogItem("lr", lr, "human_format"),  # , ".3E"),
                LogItem("model_tflops_per_gpu", model_tflops, "human_format"),  # , ".2f"),
                LogItem("hardware_tflops_per_gpu", hardware_tflops, "human_format"),  # , ".2f"),
            ]

            if self.config.optimizer.clip_grad is not None:
                log_entries.append(LogItem("grad_norm", self.grad_norm_unclipped.item(), "human_format"))  # , ".3f"))

            # Log not too often the memory
            if self.iteration_step < 5 or (self.iteration_step - 1) % self.config.checkpoints.checkpoint_interval == 0:
                total, used, free = shutil.disk_usage("/")
                log_entries.extend(
                    [
                        LogItem(
                            "cuda_memory_allocated", torch.cuda.memory_allocated(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem(
                            "cuda_max_memory_reserved", torch.cuda.max_memory_reserved(), "human_format"
                        ),  #  / 1024**2, ".2f"),
                        LogItem("hd_total_memory_tb", total, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_used_memory_tb", used, "human_format"),  #  / (2**40), ".2f"),
                        LogItem("hd_free_memory_tb", free, "human_format"),  #  / (2**40), ".2f"),
                    ]
                )

            # NOTE: only one rank writes to wandb
            if dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0] and wandb is not None:
                wandb.log(
                    {
                        **{log_item.tag: log_item.scalar_value for log_item in log_entries},
                        "iteration_step": self.iteration_step,
                    }
                )

            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

        # Nanotron Benchmark mode: we log the throughput and exit
        if os.environ.get("NANOTRON_BENCHMARK", "0") == "1" and self.iteration_step == 3:
            log_throughput(
                self.config,
                self.parallel_context,
                model_tflops,
                hardware_tflops,
                tokens_per_sec,
            )
            log_rank("Throughput logging complete", logger=logger, level=logging.INFO)
            if "SLURM_JOB_ID" in os.environ:
                os.system("scancel " + os.environ["SLURM_JOB_ID"])
            else:
                exit(0)

    def validation_step_logs(
        self,
        val_loss: torch.Tensor,
        val_perplexity: torch.Tensor,
    ) -> None:
        """
        Log validation metrics to console, WandB, and TensorBoard.

        Args:
            val_loss: Average validation loss across all batches
            val_perplexity: Validation perplexity (exp(loss))
        """
        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            assert self.loggerwriter is not None, "loggerwriter should be defined on logger ranks"

            log_entries = [
                LogItem("val_loss", val_loss.item(), "human_format"),
                LogItem("val_perplexity", val_perplexity.item(), "human_format"),
            ]

            # Log to console via LoggerWriter
            log_rank(
                f"[Validation] Step {self.iteration_step}: val_loss={val_loss.item():.4f}, val_perplexity={val_perplexity.item():.4f}",
                logger=logger,
                level=logging.INFO,
            )

            # Log to WandB (only rank 0)
            if dist.get_rank(self.parallel_context.world_pg) == self.logger_ranks[0] and wandb is not None:
                wandb.log(
                    {
                        "val_loss": val_loss.item(),
                        "val_perplexity": val_perplexity.item(),
                        "iteration_step": self.iteration_step,
                    }
                )

            # Log to TensorBoard via LoggerWriter
            self.loggerwriter.add_scalars_from_list(log_entries, self.iteration_step)

    def init_model(self) -> Union[NanotronModel, DistributedDataParallel]:
        """Initialize the model and load weights from checkpoint if needed."""
        # TODO: add max_position_embeddings
        self.model_config.vocab_size = _vocab_size_with_padding(
            self.model_config.vocab_size,
            pg_size=self.parallel_context.tp_pg.size(),
            make_vocab_size_divisible_by=self.config.model.make_vocab_size_divisible_by,
        )

        if (
            getattr(self.model_config, "max_position_embeddings", None) is not None
            and self.model_config.max_position_embeddings != self.config.tokens.sequence_length
        ):
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                log_rank(
                    f"Finetuning a model with a sequence length {self.config.tokens.sequence_length} that is different from the checkpoint's max_position_embeddings {self.model_config.max_position_embeddings}.",  # noqa
                    logger=logger,
                    level=logging.WARNING,
                    rank=0,
                )
            else:
                if self.config.tokens.sequence_length != self.model_config.max_position_embeddings:
                    log_rank(
                        f"WARNING: The tokenizer's sequence length ({self.config.tokens.sequence_length}) does not match the model's maximum position embeddings ({self.model_config.max_position_embeddings}).",
                        logger=logger,
                        level=logging.WARNING,
                        rank=0,
                    )

        log_rank("Config:\n" + pformat(self.config), logger=logger, level=logging.INFO, rank=0)
        log_rank("Model Config:\n" + pformat(self.model_config), logger=logger, level=logging.INFO, rank=0)

        model = self._init_model_instance()
        model = self._load_model_checkpoint(model)
        return model

    def _init_model_instance(self) -> NanotronModel:
        model_config_cls = self.model_config.__class__.__name__
        assert (
            model_config_cls in CONFIG_TO_MODEL_CLASS
        ), f"Unsupported model config {model_config_cls}. Only {CONFIG_TO_MODEL_CLASS.keys()} are supported"

        model = self._init_model(
            model_builder=lambda: CONFIG_TO_MODEL_CLASS[model_config_cls](
                config=self.model_config,
                parallel_context=self.parallel_context,
                parallel_config=self.config.parallelism,
                random_states=self.random_states,
            ),
        )
        return model

    def _load_model_checkpoint(self, model: NanotronModel) -> NanotronModel:
        unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model

        # Load or initialize model weights
        reloaded_from_checkpoint = False
        if self.init_checkpoint_path is not None:
            # Load from a pre existing checkpoint
            if check_path_is_local(self.init_checkpoint_path):
                # Reload from a training checkpoint
                log_rank(
                    f"Loading weights from {self.init_checkpoint_path}", logger=logger, level=logging.INFO, rank=0
                )
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.init_checkpoint_path,
                )
            reloaded_from_checkpoint = True
        if not reloaded_from_checkpoint:
            log_rank("No checkpoint path provided.", logger=logger, level=logging.INFO, rank=0)
            if isinstance(self.config.model.init_method, ExistingCheckpointInit):
                # Initialize model from an pretrained model checkpoint (without optimizer, lr_scheduler...)
                self.param_shard_metadata = load_weights(
                    model=unwrapped_model,
                    parallel_context=self.parallel_context,
                    root_folder=self.config.model.init_method.path,
                )
            elif isinstance(self.config.model.init_method, (RandomInit, SpectralMupInit)):
                unwrapped_model.init_model_randomly(config=self.config)

                # Synchronize parameters so that the model is consistent
                # sync all params across dp
                for _, param in sorted(model.named_parameters(), key=lambda x: x[0]):
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=self.parallel_context.dp_pg)

                # sync tied params across tied groups
                for (_, group_ranks), param in sorted(
                    get_tied_id_to_param(
                        parameters=model.parameters(),
                        root_module=unwrapped_model,
                    ).items(),
                    key=lambda x: x[0],
                ):
                    group = self.parallel_context.world_ranks_to_pg[group_ranks]
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=group)
            else:
                raise ValueError(f"Unsupported {self.config.model.init_method}")

        return model

    def _init_model(
        self,
        model_builder: Callable[[], NanotronModel],
        target_pp_ranks: Optional[List[int]] = None,
    ) -> NanotronModel:
        config = self.config
        parallel_context = self.parallel_context

        parallel_config = config.parallelism
        make_ddp = parallel_context.data_parallel_size > 1 and not (
            config.optimizer.accumulate_grad_in_fp32 and config.optimizer.zero_stage > 0
        )

        # Build model and set pp ranks
        model = build_model(
            parallel_context=parallel_context,
            dtype=config.model.dtype,
            target_pp_ranks=target_pp_ranks,
            model_builder=model_builder,
        )

        # Initialize rotary embeddings
        for module in model.modules():
            if not isinstance(module, RotaryEmbedding):
                continue
            module.init_rotary_embeddings()

        # Mark some parameters as tied
        self._mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)

        # count number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        size_params = sum(p.numel() * p.element_size() for p in model.parameters())
        total_params = torch.tensor(num_params, device="cuda")
        total_size = torch.tensor(size_params, device="cuda")
        dist.all_reduce(total_params, group=parallel_context.tp_pg, async_op=False, op=dist.ReduceOp.SUM)  # TP
        dist.all_reduce(total_params, group=parallel_context.pp_pg, async_op=False, op=dist.ReduceOp.SUM)  # PP
        dist.all_reduce(total_size, group=parallel_context.tp_pg, async_op=False, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_size, group=parallel_context.pp_pg, async_op=False, op=dist.ReduceOp.SUM)

        # TODO @nouamanetazi: better memory logs
        log_rank(
            f"Total number of parameters: {human_format(total_params.item())} ({total_size.item() / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.world_pg,
            rank=0,
        )
        log_rank(
            f"Local number of parameters: {human_format(num_params)} ({size_params / 1024**2:.2f}MiB)",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.dp_pg,
            rank=0,
        )
        log_rank(
            f"[After model building] Memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f}MiB."
            f" Peak allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MiB"
            f" Peak reserved: {torch.cuda.max_memory_reserved() / 1024**2:.2f}MiB",
            logger=logger,
            level=logging.INFO,
            group=parallel_context.dp_pg,
            rank=0,
        )

        # Model make it DDP
        if make_ddp is True:
            # Check that the model has at least one grad. Necessary for DDP
            check_model_has_grad(model=model, parallel_context=parallel_context)
            # TODO @thomasw21: DDP doesn't support broadcasting complex buffers (and we don't really need that broadcasting anyway)
            model = DistributedDataParallel(
                model,
                process_group=parallel_context.dp_pg,
                broadcast_buffers=False,
                bucket_cap_mb=config.model.ddp_bucket_cap_mb,
            )

        # Sanity check the model, all parameters must be NanotronParameter (either tied or sharded)
        sanity_check(root_module=model)

        return model

    def setup_log_writers(
        self,
    ) -> Optional[LoggerWriter]:
        """Setup all log writers on the appropriate ranks

        Args:
            config (Config): The config object
            logger_ranks (Iterable[int]): The ranks that should log
            parallel_context (DistributedProcessGroups): The distributed process groups
        """
        if dist.get_rank(self.parallel_context.world_pg) in self.logger_ranks:
            loggerwriter = LoggerWriter(global_step=self.config.tokens.train_steps)
        else:
            loggerwriter = None

        return loggerwriter

    def pre_save_checkpoint(self) -> Path:
        if self.s3_mover is not None:
            self.s3_mover.distributed_wait_for_completion(self.parallel_context.world_pg)
            if self.s3_mover.post_upload_callback_outputs is not None:
                slurm_job_id, slurm_log = self.s3_mover.post_upload_callback_outputs
                self.log_object({"job_id": slurm_job_id, "log": slurm_log}, "slurm_eval")

    def post_save_checkpoint(self):
        # Upload to S3
        if self.s3_mover is not None:
            self.s3_mover.start_uploading()

    def save_checkpoint(self) -> Path:
        self.pre_save_checkpoint()

        checkpoints_path = self.config.checkpoints.checkpoints_path
        checkpoint_path = checkpoints_path / f"{self.iteration_step}"
        if self.config.checkpoints.checkpoints_path_is_shared_file_system:
            should_mkdir = dist.get_rank(self.parallel_context.world_pg) == 0
        else:
            should_mkdir = bool(int(os.environ.get("LOCAL_RANK", None)) == 0)
        if should_mkdir:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        dist.barrier(self.parallel_context.world_pg)

        log_rank(f"Saving checkpoint at {checkpoint_path}", logger=logger, level=logging.WARNING, rank=0)

        # Update step/samples numbers before we save the config
        self.config.general.step = self.metadata.last_train_step
        self.config.general.consumed_train_samples = self.metadata.consumed_train_samples

        save(
            model=self.unwrapped_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            should_save_model=bool(
                dist.get_rank(self.parallel_context.dp_pg) == 0
            ),  # We only save the weights on DP==0
            should_save_optimizer=True,
            should_save_lr_scheduler=True,
            should_save_config=bool(
                dist.get_rank(self.parallel_context.world_pg) == 0
            ),  # We only save the config on world_rank==0
            parallel_context=self.parallel_context,
            root_folder=checkpoint_path,
            training_metadata=self.metadata,
            config=self.config,
        )
        save_random_states(
            random_states=self.random_states, parallel_context=self.parallel_context, root_folder=checkpoint_path
        )
        with open(checkpoints_path / "latest.txt", mode="w") as fo:
            fo.write(f"{self.iteration_step}")

        if hasattr(self.model_config, "to_json_file"):
            self.model_config.to_json_file(checkpoint_path / MODEL_CONFIG_FILE_NAME)
        else:
            with open(checkpoint_path / MODEL_CONFIG_FILE_NAME, mode="w") as fo:
                fo.write(json.dumps(asdict(self.model_config)))

        self.post_save_checkpoint()

        return checkpoint_path

    def _mark_tied_parameters(
        self,
        model: NanotronModel,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs] = None,
    ):
        mark_tied_parameters(model=model, parallel_context=parallel_context, parallel_config=parallel_config)


def mark_tied_parameters(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: Optional[ParallelismArgs] = None
):
    # Tie embeddings
    embeddings_lm_head_tied_names = model.get_embeddings_lm_head_tied_names()
    if len(embeddings_lm_head_tied_names) > 0:
        shared_embeddings = [
            (
                target,
                (
                    parallel_context.get_global_rank(
                        ep_rank=dist.get_rank(parallel_context.expert_pg),
                        pp_rank=get_pp_rank_of(target, module=model),
                        dp_rank=dist.get_rank(parallel_context.dp_pg),
                        tp_rank=dist.get_rank(parallel_context.tp_pg),
                    ),
                ),
            )
            for target in embeddings_lm_head_tied_names
        ]
        tie_parameters(
            root_module=model, ties=shared_embeddings, parallel_context=parallel_context, reduce_op=dist.ReduceOp.SUM
        )

    # Tie custom params
    model.tie_custom_params()

    # Sync all parameters that have the same name and that are not sharded across TP and EXP
    assert not isinstance(model, DistributedDataParallel), "model shouldn't be DDP at this point"
    mark_unsharded_params_as_tied_across_tp(model, parallel_context, parallel_config)
    mark_unsharded_params_as_tied_across_expert(model, parallel_context, parallel_config)

    create_pg_for_tied_weights(root_module=model, parallel_context=parallel_context)


def mark_unsharded_params_as_tied_across_tp(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: "ParallelismArgs"
):
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along tp
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_tp_sharded(parallel_context=parallel_context):
                        continue

            if isinstance(module, TensorParallelRowLinear) and "bias" == param_name:
                # bias for TensorParallelRowLinear only exists on TP=0 so we don't need to tie it
                continue

            shared_weights = [
                (
                    name,
                    # sync across TP group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.tp_pg))),
                )
            ]

            if parallel_config is None or parallel_config.tp_mode is TensorParallelLinearMode.ALL_REDUCE:
                # We add `reduce_op=None` in order to signal that the weight are synced by design without needing to reduce
                # when TP=2 we have LN that is duplicated across TP, so by design it's tied
                reduce_op = None
            else:
                reduce_op = dist.ReduceOp.SUM

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )


def mark_unsharded_params_as_tied_across_expert(
    model: NanotronModel, parallel_context: ParallelContext, parallel_config: "ParallelismArgs"
):
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            name = f"{module_name}.{param_name}"

            if isinstance(param, NanotronParameter):
                # We skip tying if param already tied or sharded along expert
                if param.is_tied:
                    continue

                if param.is_sharded:
                    sharded_info = param.get_sharded_info()
                    if sharded_info.is_expert_sharded(parallel_context):
                        continue

            shared_weights = [
                (
                    name,
                    # sync across expert group
                    tuple(sorted(dist.get_process_group_ranks(parallel_context.expert_pg))),
                )
            ]

            # Besides MoE block which sees shards tokens, the rest of the model sees the full tokens
            # so we don't need to reduce the gradients across expert group
            reduce_op = None

            tie_parameters(
                root_module=model, ties=shared_weights, parallel_context=parallel_context, reduce_op=reduce_op
            )
