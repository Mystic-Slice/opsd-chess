
from dataclasses import dataclass

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
    max_tokens: int = 1024
    temperature: float = 1.2
    kl_coef: float = 1.0
    ctx_len_penalty_ratio: float = 0.5
    use_frozen_teacher: bool = True
    save_every: int = 20
    log_path: str = "logs/opsd_countdown_sl_loop_frozen_teacher_ctx_len_penalty_ratio_0.5_fixed_length_bias"
    base_url: str | None = None
    max_step: int | None = 100