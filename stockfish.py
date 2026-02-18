import os

import chess
import chess.engine

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

class Stockfish():
  """The classical version of stockfish."""

  def __init__(
      self,
      limit: chess.engine.Limit,
  ) -> None:
    self._limit = limit
    self._skill_level = None
    bin_path = os.path.join(
        os.getcwd(),
        '..\\stockfish_x86-64-bmi2.exe',
    )
    print(bin_path)
    self._raw_engine = chess.engine.SimpleEngine.popen_uci(bin_path)

  def __del__(self) -> None:
    self._raw_engine.close()

  @property
  def limit(self) -> chess.engine.Limit:
    return self._limit

  @property
  def skill_level(self) -> int | None:
    return self._skill_level

  @skill_level.setter
  def skill_level(self, skill_level: int) -> None:
    self._skill_level = skill_level
    self._raw_engine.configure({'Skill Level': self._skill_level})

  def analyse(self, board: chess.Board):
    """Returns analysis results from stockfish."""
    return self._raw_engine.analyse(board, limit=self._limit)

  def get_score(self, fen, move):
    to_move = fen.split()[1]
    board = chess.Board(fen)

    try:
      board.push_san(move)
    except:
      print("Illegal move")
      return -1000

    info = self.analyse(board)

    if info['score'].is_mate():
      return 1000

    if to_move == 'b':
      score = info['score'].black()
    else:
      score = info['score'].white()
    return clamp(score.score(), -1000, 1000)

  def play(self, board: chess.Board) -> chess.Move:
    """Returns the best move from stockfish."""
    best_move = self._raw_engine.play(board, limit=self._limit).move
    if best_move is None:
      raise ValueError('No best move found, something went wrong.')
    return best_move