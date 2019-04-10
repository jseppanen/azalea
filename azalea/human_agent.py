
from typing import Any, Dict, Optional

from .game.hex import HexGame


class HumanAgent:
    """Interactive human player."""

    def __init__(self):
        self.game = HexGame()
        self.last_move = None
        self.settings: Dict[str, Any] = {}

    def reset(self) -> None:
        """Start a new game."""
        self.game.reset()

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed random number generator."""
        pass

    def choose_action(self) -> int:
        """Plan next move."""
        print_board(self.game.io, self.game.state.board, self.last_move)
        move = ask_move(self.game.io,
                        self.game.state.board,
                        self.game.state.legal_moves)
        return move

    def execute_action(self, move: int) -> int:
        """Execute move.
        :returns: Game result
        """
        self.game.step(move)
        self.last_move = move
        return self.game.state.result


def ask_move(io, board, moves):
    while True:
        move_txt = input('Your move? ')
        try:
            move = io.parse_move(board, moves, move_txt)
            if move in moves:
                return move
        except io.ParseError:
            print('illegal move:', move_txt)


def print_board(io, board, computer_move):
    print('\033[2J')
    io.print_board(board)
    if computer_move:
        move_txt = io.format_move(board, computer_move)
        #print(f'value: {value:.2f}')
        print(f'last move: {move_txt}')
