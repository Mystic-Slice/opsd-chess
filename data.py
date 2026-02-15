import pandas as pd
import datasets
import fen_tokenizer
import chess
from utils import stringify_board

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

    
