import pandas as pd
import datasets
import chess
import re

import math
from dataclasses import dataclass
from typing import Any, Dict, Sequence, cast

import tinker

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
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

def extract_moves_strict(completion):
    matches = re.findall(r"<best_move>([a-h][1-8][a-h][1-8][qrbnQRBN]?)</best_move>", completion)
    if matches:
        return matches[-1].strip()
    return None

def extract_moves(completion):
    matches = re.findall(r"<best_move>(.*?)</best_move>", completion)
    if matches:
        return matches[-1].strip()
    return None

def stringify_board(board):
    lst = ["  a b c d e f g h"]
    for i, row in enumerate(board.__str__().split("\n")):
        lst.append(f"{8-i} " + row)
    return "\n".join(lst)

# =============================================================================
# CustomEnv
# =============================================================================
class CustomEnv(Env):
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
# CustomEnvGroupBuilder
# =============================================================================
@dataclass(frozen=True)
class CustomEnvGroupBuilder(EnvGroupBuilder):
    """Builds a group of identical chess envs for GRPO-style advantage centering."""

    student_messages: list[renderers.Message]
    teacher_prompt_tokens: list[int]
    renderer: renderers.Renderer
    group_size: int
    fen: str
    best_move_uci: str

    async def make_envs(self) -> Sequence[Env]:
        return [
            CustomEnv(self.student_messages, self.renderer)
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["chess"]
    
    def logging_info(self) -> dict:
        return {
            "FEN": self.fen,
            "Best Move": self.best_move_uci,
        }


# =============================================================================
# CustomDataset
# =============================================================================
class CustomDataset(RLDataset):
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
                CustomEnvGroupBuilder(
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

def process_example(sample):
    fen_string = sample['fen']

    board = chess.Board(fen_string)
    board_string = str(board)

    best_move = sample['best_move_uci']

    # PROMPT_MESSAGES_TEACHER = [
    #     {
    #         'role': 'user',
    #         'content': "You are a chess teacher. You should suggest the next best move (also provided to you) from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. But in your output, you shouldn't reveal that you are given the answer. Try explaining the reasoning by yourself. You should state your reasoning between <reasoning> and </reasoning> symbols. Output the final move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])"\
    #             f"<board>{board_string}</board>" \
    #             f"<best_move>{best_move}</best_move>"
    #     }
    # ]

    # PROMPT_MESSAGES_STUDENT = [
    #     {
    #         'role': 'user',
    #         'content': "You are a chess player. You should suggest the next best move from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. Output your reasoning between <reasoning> and </reasoning> symbols. Output the final move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])"\
    #             f"<board>{board_string}</board>" \
    #     }
    # ]

        
    # # v2
    # board_string = stringify_board(board)
    # PROMPT_MESSAGES_TEACHER = [
    #     {
    #         'role': 'user',
    #         'content': "You are a chess teacher. You should suggest the next best move (also provided to you) from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. But in your output, you should not reveal that you are given the answer. Try explaining the reasoning by yourself. You should state your reasoning between <reasoning> and </reasoning> symbols. Output the final move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])."\
    #             f"<board>{board_string}</board>" \
    #             f"The best move in this position is {best_move}. Using this as reference, explain the reasoning behind this move and why other moves are worse."
    #     }
    # ]

    # PROMPT_MESSAGES_STUDENT = [
    #     {
    #         'role': 'user',
    #         'content': "You are a chess player. You should suggest the next best move from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. Output your reasoning between <reasoning> and </reasoning> symbols. Output the final move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])."\
    #             f"<board>{board_string}</board>" \
    #     }
    # ]

        
    # v3
    board_string = stringify_board(board)
    PROMPT_MESSAGES_TEACHER = [
        {
            'role': 'user',
            'content': "You are a chess teacher. You should suggest the next best move (also provided to you) from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. But in your output, you should not reveal that you are given the answer. Try explaining the reasoning by yourself. Finally, output your choice for the next move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])."\
                f"<board>{board_string}</board>" \
                f"The best move in this position is {best_move}. Using this as reference, explain the reasoning behind this move and why other moves are worse."
        }
    ]

    PROMPT_MESSAGES_STUDENT = [
        {
            'role': 'user',
            'content': "You are a chess player. You should suggest the next best move from a given position. The letters on the board mean different pieces: P -> Pawn, N -> Knight, B -> Bishop, R -> Rook, Q -> Queen, K -> King, . -> empty cell. Uppercase letters mean White pieces and lowercase letters are black pieces. Finally, output your choice for the next move between <best_move> and </best_move> symbols. The move should be in UCI format (i.e., [from square][to square][promotion piece])."\
                f"<board>{board_string}</board>" \
        }
    ]

    return {
        'prompt_teacher': PROMPT_MESSAGES_TEACHER,
        'prompt_student': PROMPT_MESSAGES_STUDENT,
    }

def get_data():
    df = pd.read_csv("puzzles_with_reasoning.csv")
    ds = datasets.Dataset.from_pandas(df)

    ds = ds.map(process_example, num_proc=8)

    ds = ds.train_test_split(test_size=0.01, seed=42)

    return ds

def eval_samples(trajectory_groups_P: list[TrajectoryGroup], env_group_builders_P: list[CustomEnvGroupBuilder], tokenizer: tinker.Tokenizer) -> Dict[str, Any]:
    score_results = []
    for g_idx, tg in enumerate(trajectory_groups_P):
        builder = cast(CustomEnvGroupBuilder, env_group_builders_P[g_idx])
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

    score_results = score_moves(score_results)

    valid_scores = [r["score"] for r in score_results if r["parse_success"] and r["score"] != -1000]
    max_scores = [r["max_score"] for r in score_results]
    num_illegal = sum(1 for r in score_results if r["parse_success"] and r["score"] == -1000)
    num_exact_match = sum(1 for r in score_results if r["model_move"] == r["best_move"])

    metrics = {}

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
    return metrics    

def score_moves(results: list[dict], stockfish_time: float = 0.01) -> list[dict]:
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