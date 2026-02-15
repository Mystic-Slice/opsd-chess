"""
Chess Model Evaluation Script.

Samples moves from a model (base or trained checkpoint), evaluates them
with Stockfish, and saves results as JSON.
"""

import asyncio
import json
import logging
import os
import statistics
from dataclasses import asdict, dataclass

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.rl.train import gather_with_progress

import data as chess_data
from utils import extract_moves_strict, extract_moves

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    checkpoint_path: str | None = None
    num_samples: int = 50
    max_tokens: int = 4096
    temperature: float = 0.0
    output_dir: str = "eval_results_v2"
    base_url: str | None = None
    stockfish_time: float = 0.01
    use_teacher_prompt: bool = False


def score_moves(results: list[dict], stockfish_time: float) -> list[dict]:
    """Score each model move using Stockfish.

    Takes the list of result dicts (with 'fen', 'model_move', 'expected_move'
    fields) and fills in 'score' for each one.

    Args:
        results: List of result dicts from evaluation. Each has:
            - fen: The board position (FEN string)
            - model_move: The move the model predicted (str or None)
            - expected_move: The ground-truth best move (str)
            - parse_success: Whether a move was extracted from model output
        stockfish_time: Time limit in seconds for Stockfish analysis per move.

    Returns:
        The same list with 'score' filled in for each result.
    """
    import chess.engine
    from stockfish import Stockfish

    sf = Stockfish(limit=chess.engine.Limit(time=stockfish_time))
    try:
        for result in results:
            result['max_score'] = sf.get_score(result["fen"], result["best_move"])
            if not result["parse_success"] or result["model_move"] is None:
                result["score"] = -1000
            else:
                result["score"] = sf.get_score(result["fen"], result["model_move"])
    finally:
        del sf

    return results


async def evaluate_puzzle(
    idx: int,
    row: dict,
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    sampling_params: tinker.types.SamplingParams,
    cfg: EvalConfig,
) -> dict:
    """Evaluate a single puzzle: generate a move and extract it."""
    if cfg.use_teacher_prompt:
        student_messages = row["prompt_teacher"]
    else:
        student_messages = row["prompt_student"]
    prompt = renderer.build_generation_prompt(student_messages)

    response = await sampling_client.sample_async(
        prompt=prompt, num_samples=1, sampling_params=sampling_params
    )

    tokens = response.sequences[0].tokens
    message = renderer.parse_response(tokens)[0]
    text = renderers.get_text_content(message)

    model_move = extract_moves_strict(text)

    return {
        "puzzle_idx": idx,
        "fen": row["fen"],
        "best_move": row["best_move_uci"],
        "model_move": model_move,
        "model_response": text,
        "score": None,
        "parse_success": model_move is not None,
    }


async def main(cfg: EvalConfig):
    # Print config
    logger.info(f"Evaluation config:\n{json.dumps(asdict(cfg), indent=2)}")

    os.makedirs(cfg.output_dir, exist_ok=True)

    # --- Load test data ---
    ds = chess_data.get_data()
    test_ds = ds["test"]
    num_puzzles = min(cfg.num_samples, len(test_ds))
    logger.info(f"Evaluating {num_puzzles} puzzles from test split")

    # --- Create sampling client ---
    service_client = tinker.ServiceClient(base_url=cfg.base_url)

    if cfg.checkpoint_path is not None:
        from tinker_cookbook import checkpoint_utils

        ckpt = checkpoint_utils.get_last_checkpoint(
            cfg.checkpoint_path, required_key="sampler_path"
        )
        if ckpt is None:
            raise ValueError(
                f"No checkpoint with sampler_path found in {cfg.checkpoint_path}"
            )
        sampler_path = ckpt["sampler_path"]
        logger.info(f"Loading checkpoint: {sampler_path}")
        sampling_client = service_client.create_sampling_client(
            model_path=sampler_path
        )
    else:
        logger.info(f"Using base model: {cfg.model_name}")
        sampling_client = service_client.create_sampling_client(
            base_model=cfg.model_name
        )

    # --- Renderer ---
    tokenizer = get_tokenizer(cfg.model_name)
    renderer_name = get_recommended_renderer_name(cfg.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    # --- Sampling params ---
    sampling_params = tinker.types.SamplingParams(
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        stop=renderer.get_stop_sequences(),
    )

    # --- Generate moves (batched async) ---
    logger.info("Generating model responses...")
    tasks = [
        evaluate_puzzle(i, test_ds[i], sampling_client, renderer, sampling_params, cfg)
        for i in range(num_puzzles)
    ]
    results = await gather_with_progress(tasks, "Generating model responses")

    # --- Score with Stockfish ---
    logger.info("Scoring moves with Stockfish...")
    results = score_moves(list(results), cfg.stockfish_time)

    # --- Compute summary ---
    scores = [r["score"] for r in results if r["parse_success"] and r["score"] != -1000]
    max_scores = [r["max_score"] for r in results]
    num_illegal = sum(
        1 for r in results if r["parse_success"] and r["score"] == -1000
    )
    num_parse_failures = sum(1 for r in results if not r["parse_success"])

    summary = {
        "num_puzzles": num_puzzles,
        "num_valid_moves": len(scores),
        "num_illegal_moves": num_illegal,
        "num_parse_failures": num_parse_failures,
        "avg_score": round(statistics.mean(scores), 1) if scores else None,
        "median_score": round(statistics.median(scores), 1) if scores else None,
        "avg_max_score": round(statistics.mean(max_scores), 1) if max_scores else None,
        "median_max_score": round(statistics.median(max_scores), 1) if max_scores else None,
    }

    # --- Save ---
    output = {
        "config": asdict(cfg),
        "summary": summary,
        "results": results,
    }

    tag = cfg.checkpoint_path.split("\\")[-1] if cfg.checkpoint_path else "base"
    if cfg.use_teacher_prompt:
        tag += "_teacher_prompt"
    out_path = os.path.join(cfg.output_dir, f"eval_{tag}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {out_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Evaluate chess model")
    parser.add_argument("--model_name", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output_dir", default="eval_results_v2")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--stockfish_time", type=float, default=0.01)
    parser.add_argument("--use_teacher_prompt", action="store_true")

    args = parser.parse_args()
    cfg = EvalConfig(**vars(args))
    asyncio.run(main(cfg))
