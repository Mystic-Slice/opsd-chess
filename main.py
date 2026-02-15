"""
Chess On-Policy Self-Distillation (OPSD) Training Script.

Trains a chess LLM by self-distilling from a "teacher" variant (same model,
prompted with the correct answer) to a "student" variant (no answer in prompt).
The teacher is frozen at initial weights; only the student updates.
No explicit rewards — only KL divergence between teacher/student distributions.
"""

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, cast

import tinker
import torch
from tinker.types import LossFnType

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.metrics import compute_kl_sample_train
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    gather_with_progress,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    RLDataset,
    StepResult,
    Trajectory,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook import checkpoint_utils

import data as chess_data
from eval import score_moves
from utils import extract_moves_strict

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


# =============================================================================
# 1. ChessDistillationEnv
# =============================================================================


class ChessDistillationEnv(Env):
    """Single-turn env: student prompt -> model generates reasoning + move -> reward = 0.0"""

    def __init__(
        self,
        student_messages: list[renderers.Message],
        renderer: renderers.Renderer,
    ):
        self.student_messages = student_messages
        self.renderer = renderer

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return (
            self.renderer.build_generation_prompt(self.student_messages),
            self.renderer.get_stop_sequences(),
        )

    async def step(self, action: Action) -> StepResult:
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.renderer.get_stop_sequences(),
        )


# =============================================================================
# 2. ChessGroupBuilder
# =============================================================================


@dataclass(frozen=True)
class ChessGroupBuilder(EnvGroupBuilder):
    """Builds a group of identical chess envs for GRPO-style advantage centering."""

    student_messages: list[renderers.Message]
    teacher_prompt_tokens: list[int]
    renderer: renderers.Renderer
    group_size: int
    fen: str
    best_move_uci: str

    async def make_envs(self) -> Sequence[Env]:
        return [
            ChessDistillationEnv(self.student_messages, self.renderer)
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["chess"]


# =============================================================================
# 3. ChessDataset
# =============================================================================


class ChessDataset(RLDataset):
    """Wraps the HuggingFace chess puzzle dataset for OPSD training."""

    def __init__(
        self,
        hf_dataset,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
    ):
        self.hf_dataset = hf_dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.hf_dataset))
        builders = []
        for i in range(start, end):
            row = self.hf_dataset[i]
            student_messages = row["prompt_student"]
            teacher_messages = row["prompt_teacher"]

            # Pre-tokenize the teacher prompt for KL computation
            teacher_prompt_tokens = self.renderer.build_generation_prompt(
                teacher_messages
            ).to_ints()

            builders.append(
                ChessGroupBuilder(
                    student_messages=student_messages,
                    teacher_prompt_tokens=teacher_prompt_tokens,
                    renderer=self.renderer,
                    group_size=self.group_size,
                    fen=row["fen"],
                    best_move_uci=row["best_move_uci"],
                )
            )
        return builders

    def __len__(self) -> int:
        return math.ceil(len(self.hf_dataset) / self.batch_size)


# =============================================================================
# 4. incorporate_kl_penalty_chess (core novel function)
# =============================================================================


async def incorporate_kl_penalty_chess(
    data_D: List[tinker.Datum],
    metadata_D: List[dict[str, int]],
    env_group_builders_P: Sequence[ChessGroupBuilder],
    teacher_sampling_client: tinker.SamplingClient,
    kl_coef: float,
) -> Dict[str, float]:
    """
    Compute KL divergence between teacher and student with DIFFERENT prompts.

    Unlike standard incorporate_kl_penalty, the teacher sees a different prompt
    (with the answer included). We extract the student's completion tokens,
    prepend the teacher's prompt, and compute teacher logprobs on that sequence.

    The advantage signal is: kl_coef * (teacher_lp - student_lp) per completion token.
    This encourages the student to match the teacher's distribution.

    Args:
        data_D: Training datums from assemble_training_data (student rollouts)
        metadata_D: Metadata with group_idx linking each datum to its builder
        env_group_builders_P: Builders containing teacher_prompt_tokens
        teacher_sampling_client: Frozen teacher model client
        kl_coef: KL penalty coefficient
    """
    env_group_builders_D = [env_group_builders_P[m["group_idx"]] for m in metadata_D]
    teacher_prompt_tokens = [env.teacher_prompt_tokens for env in env_group_builders_D]
    teacher_sequences = []

    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    for datum_idx, (datum, teacher_prompt) in enumerate(zip(data_D, teacher_prompt_tokens)):
        mask = float_masks[datum_idx]
        target_tokens = datum.loss_fn_inputs["target_tokens"].tolist()
        student_completion = [target_tokens[i] for i in range(len(mask)) if mask[i] > 0]
        teacher_sequences.append(
            tinker.ModelInput.from_ints(teacher_prompt + student_completion)
        )

    teacher_logprobs_D = await asyncio.gather(
        *[
            teacher_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in teacher_sequences
        ]
    )

    teacher_logprobs_D = [
        torch.tensor(teacher_logprobs[len(teacher_prompt):])
        for (teacher_logprobs, teacher_prompt) in safezip(teacher_logprobs_D, teacher_prompt_tokens)
    ]
    student_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]

    kl_sum = torch.tensor(0.0)
    teacher_lp_sum = torch.tensor(0.0)
    student_lp_sum = torch.tensor(0.0)
    total_mask = sum(mask.sum() for mask in float_masks)

    for datum, student_logprobs, teacher_logprobs, mask in safezip(data_D, student_logprobs_D, teacher_logprobs_D, float_masks):
        full_teacher_lps = torch.zeros_like(student_logprobs)
        full_teacher_lps[mask > 0] = teacher_logprobs
        kl_i = (full_teacher_lps - student_logprobs) * mask

        kl_sum += kl_i.sum()
        teacher_lp_sum += (full_teacher_lps * mask).sum()
        student_lp_sum += (student_logprobs * mask).sum()

        # Center advantages per-sequence to remove the negative bias.
        # Without centering, E[A_n] = -KL(p_S||p_T) < 0, which tells the loss
        # to reduce probability of ALL tokens, causing slow model degradation.
        num_completion_tokens = mask.sum()
        if num_completion_tokens > 0:
            kl_mean = kl_i.sum() / num_completion_tokens
            centered_kl = kl_i - kl_mean * mask
        else:
            centered_kl = kl_i
        kl_advantages = kl_coef * centered_kl
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

    metrics = {
        "teacher_kl": float(kl_sum / total_mask),
        "logprobs/teacher_mean": float(teacher_lp_sum / total_mask),
        "logprobs/student_mean": float(student_lp_sum / total_mask),
    }
    return metrics

# =============================================================================
# 5. OPSDConfig
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
    stockfish_time: float = 0.01


# =============================================================================
# 6. main() async training loop
# =============================================================================


async def main(cfg: OPSDConfig):
    """Main OPSD training loop."""

    os.makedirs(cfg.log_path, exist_ok=True)

    # --- Check for existing checkpoint to resume from ---
    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
        logger.info(f"Found checkpoint at batch {start_batch}, will resume from there")
    else:
        start_batch = 0

    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        config=cfg,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # --- Tinker clients (3-way: resume / load checkpoint / fresh) ---
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    # Always create a fresh LoRA client first so we can snapshot the initial
    # (untrained) weights as the frozen teacher before loading any checkpoint.
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )
    tokenizer = training_client.get_tokenizer()

    # --- Renderer ---
    renderer_name = get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # --- Frozen teacher: snapshot initial (base model) weights BEFORE loading any checkpoint ---
    teacher_sampling_client = (
        await training_client.save_weights_and_get_sampling_client_async("teacher_init")
    )
    logger.info("Created frozen teacher sampling client at initial weights")

    # Now load checkpoint / resume state on top of the training client
    if resume_info:
        future = await training_client.load_state_with_optimizer_async(
            resume_info["state_path"]
        )
        _ = await future.result_async()
        logger.info(f"Resumed training from {resume_info['state_path']}")
    elif cfg.load_checkpoint_path:
        future = await training_client.load_state_with_optimizer_async(
            cfg.load_checkpoint_path
        )
        _ = await future.result_async()
        logger.info(f"Loaded weights from {cfg.load_checkpoint_path}")

    # --- Initial student sampling client ---
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every, start_batch
    )

    # --- Dataset ---
    ds = chess_data.get_data()
    dataset = ChessDataset(
        hf_dataset=ds["train"],
        batch_size=cfg.groups_per_batch,
        group_size=cfg.group_size,
        renderer=renderer,
    )
    num_batches = len(dataset)
    if cfg.max_step is not None:
        num_batches = min(cfg.max_step, num_batches)
    logger.info(f"Training for {num_batches} batches (starting from batch {start_batch})")

    # --- Training loop ---
    for i_batch in range(start_batch, num_batches):
        metrics: dict[str, Any] = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # 1. Get batch of env group builders
        env_group_builders_P = dataset.get_batch(i_batch)

        # 2. Sample rollouts from student
        with timed("sample", metrics):
            trajectory_groups_or_none = await asyncio.gather(
                *[
                    do_group_rollout_and_filter_constant_reward(
                        sampling_client,
                        builder,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                        do_remove_constant_reward_groups=False,
                    )
                    for builder in env_group_builders_P
                ]
            )
        # Filter out failed rollouts, keeping builders aligned with their groups
        valid_indices = [i for i, tg in enumerate(trajectory_groups_or_none) if tg is not None]
        trajectory_groups_P: list[TrajectoryGroup] = [trajectory_groups_or_none[i] for i in valid_indices]
        env_group_builders_P = [env_group_builders_P[i] for i in valid_indices]

        if not trajectory_groups_P:
            logger.warning(f"Batch {i_batch}: no valid trajectory groups, skipping")
            continue

        # --- Token and sample metrics ---
        all_completion_lengths = []
        for tg in trajectory_groups_P:
            for traj in tg.trajectories_G:
                n_tokens = sum(len(t.ac.tokens) for t in traj.transitions)
                all_completion_lengths.append(n_tokens)

        total_completion_tokens = sum(all_completion_lengths)
        metrics["tokens/total_completion"] = total_completion_tokens
        metrics["tokens/mean_completion_len"] = (
            total_completion_tokens / len(all_completion_lengths)
            if all_completion_lengths else 0
        )
        metrics["tokens/max_completion_len"] = max(all_completion_lengths, default=0)
        metrics["tokens/min_completion_len"] = min(all_completion_lengths, default=0)
        metrics["batch/num_groups"] = len(trajectory_groups_P)
        metrics["batch/num_trajectories"] = len(all_completion_lengths)

        # --- Score completions with Stockfish ---
        score_results = []
        for g_idx, tg in enumerate(trajectory_groups_P):
            builder = cast(ChessGroupBuilder, env_group_builders_P[g_idx])
            for traj in tg.trajectories_G:
                completion_toks = []
                for t in traj.transitions:
                    completion_toks.extend(t.ac.tokens)
                text = tokenizer.decode(completion_toks)
                model_move = extract_moves_strict(text)
                score_results.append({
                    "fen": builder.fen,
                    "best_move": builder.best_move_uci,
                    "model_move": model_move,
                    "parse_success": model_move is not None,
                    "score": None,
                })

        num_total = len(score_results)
        num_parse_ok = sum(1 for r in score_results if r["parse_success"])

        score_results = score_moves(score_results, cfg.stockfish_time)

        valid_scores = [r["score"] for r in score_results if r["parse_success"] and r["score"] != -1000]
        max_scores = [r["max_score"] for r in score_results]
        num_illegal = sum(1 for r in score_results if r["parse_success"] and r["score"] == -1000)
        num_exact_match = sum(1 for r in score_results if r["model_move"] == r["best_move"])

        metrics["eval/num_parse_ok"] = num_parse_ok
        metrics["eval/num_illegal"] = num_illegal
        metrics["eval/num_exact_match"] = num_exact_match
        metrics["eval/num_total"] = num_total
        metrics["eval/avg_score"] = (
            sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        )
        metrics["eval/avg_max_score"] = (
            sum(max_scores) / len(max_scores) if max_scores else 0.0
        )
        metrics["eval/score_gap"] = metrics["eval/avg_max_score"] - metrics["eval/avg_score"]

        # Log 5 sample completions
        num_samples_to_log = min(5, len(trajectory_groups_P))
        sample_parts = []
        for s_idx in range(num_samples_to_log):
            tg = trajectory_groups_P[s_idx]
            builder = cast(ChessGroupBuilder, env_group_builders_P[s_idx])
            question = builder.student_messages[0]["content"]
            traj = tg.trajectories_G[0]
            completion_tokens = []
            for t in traj.transitions:
                completion_tokens.extend(t.ac.tokens)
            completion_text = tokenizer.decode(completion_tokens)
            sample_parts.append(
                f"--- Sample {s_idx + 1} ---\n"
                f"FEN: {builder.fen}\n"
                f"Best: {builder.best_move_uci}\n"
                f"A: {completion_text}\n"
                f"Tokens: {len(completion_tokens)}"
            )
        samples_text = "\n\n".join(sample_parts)
        logger.info(f"Batch {i_batch} sample completions:\n{samples_text}")
        with open(os.path.join(cfg.log_path, "samples.log"), "a", encoding="utf-8") as f:
            f.write(f"=== Batch {i_batch} ===\n{samples_text}\n\n")

        # Log move score metrics
        logger.info(
            f"Batch {i_batch} metrics: "
            f"{metrics['eval/avg_score']:.2f} avg_score, "
            f"{metrics['eval/avg_max_score']:.2f} avg_max_score, "
            f"{metrics['eval/score_gap']:.2f} score_gap"
            f" {num_parse_ok} parse_ok,"
            f" {num_illegal} illegal moves,"
            f" {num_total} total samples"
        )

        # 3. Compute advantages (all zeros since reward=0)
        advantages_P = compute_advantages(trajectory_groups_P)

        # 4. Assemble training data
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

        # 5. Incorporate KL penalty (the OPSD signal)
        with timed("kl_penalty", metrics):
            kl_metrics = await incorporate_kl_penalty_chess(
                data_D,
                metadata_D,
                env_group_builders_P,
                teacher_sampling_client,
                cfg.kl_coef,
            )
        metrics.update(kl_metrics)

        # 6. Train step
        with timed("train", metrics):
            training_logprobs_D = await train_step(
                data_D=data_D,
                training_client=training_client,
                learning_rate=cfg.learning_rate,
                num_substeps=cfg.num_substeps,
                loss_fn=cfg.loss_fn,
                loss_fn_config=cfg.loss_fn_config,
                metrics=metrics,
            )

        # 7. Post-step metrics + new sampling client
        sampling_client, full_batch_metrics = (
            await compute_full_batch_metrics_and_get_sampling_client(
                training_client,
                i_batch + 1,
                data_D,
                training_logprobs_D,
                cfg.log_path,
                cfg.save_every,
                do_compute_post_kl=False,
            )
        )
        metrics.update(full_batch_metrics)

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)

    # --- Final checkpoint ---
    if start_batch < num_batches:
        await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("Training completed successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main(OPSDConfig()))
