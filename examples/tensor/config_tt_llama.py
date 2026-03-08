""" Example python script to generate a YAML config file which can be used to run a training with nanotron. Refer to "examples" section in the `/README.md` for more information."""
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch

from nanotron.config import (
    AdamWOptimizerArgs,
    CheckpointsArgs,
    Config,
    DataArgs,
    DatasetStageArgs,
    GeneralArgs,
    LoggingArgs,
    LRSchedulerArgs,
    OptimizerArgs,
    ParallelismArgs,
    PretrainDatasetsArgs,
    RandomInit,
    TokenizerArgs,
    TokensArgs,
)
from nanotron.config.models_config import ExistingCheckpointInit, NanotronConfigs, RandomInit, SpectralMupInit
from nanotron.logging import human_format



@dataclass
class TTLlamaConfig():
    """Configuration for a Tensor-Train LLAMA model

    Be careful on having a coherent typing as we use it to reconstruct the model from yaml
    """

    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 11008
    is_tt_llama_config: bool = True  # We use this help differentiate models in yaml/python conversion
    max_position_embeddings: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    num_key_value_heads: Optional[int] = None
    pad_token_id: Optional[int] = None
    pretraining_tp: int = 1
    rms_norm_eps: float = 1e-6
    rope_scaling: Optional[dict] = None
    rope_theta: float = 10000.0
    rope_interleaved: bool = (
        False  # The default value has been True, but for loading Llama3 checkpoints you have to set it to False
    )
    tie_word_embeddings: bool = False
    use_cache: bool = True
    vocab_size: int = 32000
    tt_rank: int = 1024  # Tensor-Train rank (will be validated to a vector)
    tt_cores: int = 4  # Number of TT cores (controls the mode of the tensor)
    tt_rank_ratio: float = 1.0  # Rank ratio limit for TT validation (per_decomp_rank_ratio_limit)

    def __post_init__(self):
        # NOTE: user don't set self._init_method, ModelArgs will set it
        # then we only pass LlamaConfig around
        self._is_using_mup: bool = False
        # self._init_method: Optional[Union[RandomInit, SpectralMupInit, ExistingCheckpointInit]] = None

        # for backward compatibility
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        # Validate tt_cores
        if self.tt_cores < 2:
            raise ValueError(f"tt_cores must be at least 2, got {self.tt_cores}")

    @property
    def is_using_mup(self) -> bool:
        return self._is_using_mup


# model_config = ColaLlamaConfig(
#     # Config for a tiny model model with 1.62M parameters
#     bos_token_id=1,
#     eos_token_id=2,
#     hidden_act="silu",
#     hidden_size=16,
#     initializer_range=0.02,
#     intermediate_size=64,
#     max_position_embeddings=256,
#     num_attention_heads=4,
#     num_hidden_layers=2,
#     num_key_value_heads=4,
#     pretraining_tp=1,
#     rms_norm_eps=1e-05,
#     rope_scaling=None,
#     tie_word_embeddings=True,
#     use_cache=True,
#     vocab_size=256,
# )

# num_params = human_format(
#     model_config.vocab_size * model_config.hidden_size * 2
#     + model_config.num_hidden_layers
#     * (
#         3 * model_config.hidden_size * model_config.intermediate_size
#         + 4 * model_config.hidden_size * model_config.hidden_size
#     )
# ).replace(".", "p")

# print(f"Model has {num_params} parameters")

# seed = 42

# learning_rate = LRSchedulerArgs(
#     learning_rate=3e-4, lr_warmup_steps=2, lr_warmup_style="linear", lr_decay_style="cosine", min_decay_lr=1e-5
# )

# optimizer = OptimizerArgs(
#     zero_stage=0,
#     weight_decay=0.01,
#     clip_grad=1.0,
#     accumulate_grad_in_fp32=True,
#     learning_rate_scheduler=learning_rate,
#     optimizer_factory=AdamWOptimizerArgs(
#         adam_eps=1e-08,
#         adam_beta1=0.9,
#         adam_beta2=0.95,
#         torch_adam_is_fused=True,
#     ),
# )

# parallelism = ParallelismArgs(
#     dp=2,
#     pp=2,
#     tp=2,
#     pp_engine="1f1b",
#     tp_mode="REDUCE_SCATTER",
#     tp_linear_async_communication=True,
# )

# tokens = TokensArgs(sequence_length=256, train_steps=15, micro_batch_size=2, batch_accumulation_per_replica=1)

# data_stages = [
#     DatasetStageArgs(
#         name="Stable Training Stage",
#         start_training_step=1,
#         data=DataArgs(
#             dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text"),
#             seed=seed,
#         ),
#     ),
#     DatasetStageArgs(
#         name="Annealing Phase",
#         start_training_step=10,
#         data=DataArgs(
#             dataset=PretrainDatasetsArgs(hf_dataset_or_datasets="stas/openwebtext-10k", text_column_name="text"),
#             seed=seed,
#         ),
#     ),
# ]

# checkpoints_path = "./checkpoints"
# os.makedirs(checkpoints_path, exist_ok=True)

# config = Config(
#     general=GeneralArgs(project="debug", run="tiny_llama_%date_%jobid", seed=seed),
#     checkpoints=CheckpointsArgs(checkpoints_path=checkpoints_path, checkpoint_interval=10),
#     parallelism=parallelism,
#     model=ModelArgs(init_method=RandomInit(std=0.025), model_config=model_config),
#     tokenizer=TokenizerArgs("robot-test/dummy-tokenizer-wordlevel"),
#     optimizer=optimizer,
#     logging=LoggingArgs(),
#     tokens=tokens,
#     data_stages=data_stages,
#     profiler=None,
# )

# if __name__ == "__main__":
#     dir = os.path.dirname(__file__)

#     # Save config as YAML file
#     config.save_as_yaml(f"{dir}/config_cola_llama1.yaml")

    # You can now train a model with this config using `/run_train.py`
