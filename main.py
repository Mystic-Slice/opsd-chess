"""
On-Policy Self-Distillation (OPSD) Training Script.

Trains a LLM by self-distilling from a "teacher" variant (same model,
prompted with the correct answer) to a "student" variant (no answer in prompt).
No explicit rewards — only KL divergence between teacher/student distributions.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence, cast

import tinker
import torch

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import TrajectoryGroup
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook import checkpoint_utils

from data import CustomDataset, CustomEnvGroupBuilder, get_data, eval_samples
from config import OPSDConfig

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# incorporate_kl_penalty (core novel function)
# =============================================================================
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    metadata_D: List[dict[str, int]],
    env_group_builders_P: Sequence[CustomEnvGroupBuilder],
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
# main() async training loop
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

    # --- Frozen teacher (optional): snapshot initial weights BEFORE loading any checkpoint ---
    if cfg.use_frozen_teacher:
        teacher_sampling_client = (
            await training_client.save_weights_and_get_sampling_client_async("teacher_init")
        )
        logger.info("Created frozen teacher sampling client at initial weights")
    else:
        teacher_sampling_client = None

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
    ds = get_data()
    dataset = CustomDataset(
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

        # --- Score completions ---
        sample_metrics = eval_samples(trajectory_groups_P, env_group_builders_P, tokenizer)
        metrics.update(sample_metrics)

        # Log 5 sample completions
        num_samples_to_log = min(5, len(trajectory_groups_P))
        sample_parts = []
        for s_idx in range(num_samples_to_log):
            tg = trajectory_groups_P[s_idx]
            builder = cast(CustomEnvGroupBuilder, env_group_builders_P[s_idx])
            question = builder.student_messages[0]["content"]
            traj = tg.trajectories_G[0]
            completion_tokens = []
            for t in traj.transitions:
                completion_tokens.extend(t.ac.tokens)
            completion_text = tokenizer.decode(completion_tokens)
            logging_info_env = builder.logging_info()
            sample_parts.append(
                f"--- Sample {s_idx + 1} ---\n"
                + "\n".join(f"{field}: {value}" for field, value in logging_info_env.items()) + "\n"
                + f"Q: {question}\n"
                + f"A: {completion_text}\n"
                f"Tokens: {len(completion_tokens)}"
            )
        samples_text = "\n\n".join(sample_parts)
        logger.info(f"Batch {i_batch} sample completions:\n{samples_text}")
        with open(os.path.join(cfg.log_path, "samples.log"), "a", encoding="utf-8") as f:
            f.write(f"=== Batch {i_batch} ===\n{samples_text}\n\n")

        # Log move score metrics
        logger.info(
            f"Batch {i_batch} metrics: "
            + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sample_metrics.items())
        )

        # 3. Compute advantages (all zeros since reward=0)
        advantages_P = compute_advantages(trajectory_groups_P)

        # 4. Assemble training data
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

        # 5. Incorporate KL penalty (the OPSD signal)
        with timed("kl_penalty", metrics):
            kl_metrics = await incorporate_kl_penalty(
                data_D,
                metadata_D,
                env_group_builders_P,
                teacher_sampling_client if cfg.use_frozen_teacher else sampling_client,
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
