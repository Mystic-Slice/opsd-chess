
from dataclasses import dataclass
from typing import Any

from tinker.types import LossFnType

# =============================================================================
# OPSDConfig
# =============================================================================
@dataclass
class OPSDConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    load_checkpoint_path: str | None = None
    lora_rank: int = 64
    learning_rate: float = 2e-5
    groups_per_batch: int = 32
    group_size: int = 1
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_coef: float = 1.0
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None
    num_substeps: int = 1
    save_every: int = 20
    log_path: str = "logs_v3/opsd_chess_recommended"
    base_url: str | None = None
    max_step: int | None = 100