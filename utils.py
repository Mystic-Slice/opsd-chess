
import re
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

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