
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numba import njit, jitclass, int32

from . import hex_io


@dataclass
class HexGameState:
    color: int  # 0=first player (red), 1=second player (blue)
    legal_moves: np.ndarray
    result: int
    board: np.ndarray


class HexGame:
    """Game of Hex
    See https://en.wikipedia.org/wiki/Hex_(board_game)
    """

    # printing boards etc.
    io = hex_io

    def __init__(self, board_size: int = 11) -> None:
        self.board_size = board_size
        self.impl = HexGameImpl(self.board_size)
        self._game_snapshot = None
        self.reset()

    def __getstate__(self):
        """Pickling support"""
        return (self.impl.board.copy(),
                self.impl.color,
                self.impl.winner)

    def __setstate__(self, state):
        """Pickling support"""
        board, color, winner = state
        self.__init__(board.shape[0])
        self.impl.board[:] = board
        self.impl.color = color
        self.impl.winner = winner

    def reset(self):
        self.impl = HexGameImpl(self.board_size)
        self._game_snapshot = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed random number generator."""
        pass

    @property
    def state(self) -> HexGameState:
        return HexGameState(self.impl.color - 1,
                            self.impl.legal_moves(),
                            self.impl.result(),
                            self.impl.board.copy())

    def step(self, move: int) -> None:
        self.impl.step(move)

    def snapshot(self) -> None:
        self._game_snapshot = self.__getstate__()

    def restore(self) -> None:
        assert self._game_snapshot
        self.__setstate__(self._game_snapshot)

    @staticmethod
    def flip_player_board(board: np.ndarray) -> np.ndarray:
        """Flip board to opponent perspective.
        Change both color and attack direction.
        :param board: One game board (M x M) or batch of boards (B x M x M)
        """
        assert isinstance(board, np.ndarray)
        if len(board.shape) == 2:
            return HexGame.flip_player_board(board[None, :, :])
        assert len(board.shape) == 3, 'expecting batch of boards'
        assert board.shape[-2] == board.shape[-1], 'board must be square'
        # flip color
        board = (board > 0) * (3 - board)
        # flip attack direction: mirror board along diagonal
        board = np.flip(np.rot90(board, axes=(-2, -1)), axis=-1)
        return board

    @staticmethod
    def flip_player_board_moves(board: np.ndarray, moves: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        """Flip board and legal moves to opponent perspective.
        Change both color and attack direction.
        :param board: One game board (M x M) or batch of boards (B x M x M)
        :param moves: Legal moves (K) or padded batch of moves (B x K)
        """
        assert isinstance(board, np.ndarray)
        assert isinstance(moves, np.ndarray)
        if len(board.shape) == 2:
            assert len(moves.shape) == 1, 'expecting 1D moves array'
            return HexGame.flip_player_board_moves(board[None, :, :],
                                                   moves[None, :])
        board = HexGame.flip_player_board(board)
        assert isinstance(moves, np.ndarray)
        assert len(moves.shape) == 2, 'expecting batch of moves'
        assert len(moves) == len(board), 'board and moves batch sizes differ'
        board_size = board.shape[-2:]
        # remove padding and collapse ragged rows
        moves_size = moves.shape
        flat_moves = moves.ravel().copy()
        mask = (flat_moves > 0)
        tiles = flat_moves[mask] - 1
        # calculate new move coordinates mirrored along diagonal
        tile_ids = np.unravel_index(tiles, board_size)
        tiles = np.ravel_multi_index(
            (board_size[1] - 1 - tile_ids[1], board_size[0] - 1 - tile_ids[0]),
            board_size)
        # restore padded moves batch
        flipped_moves = np.zeros_like(moves).ravel()
        flipped_moves[mask] = tiles.astype(np.int32) + 1
        flipped_moves = flipped_moves.reshape(moves_size)
        return board, flipped_moves

    @staticmethod
    def random_reflect(board, moves=None, rng=None):
        """Rotate/flip board at random while keeping same player perspective.
        """
        # b - same direction
        # np.rot90(b,2) - same direction
        # np.fliplr(np.rot90(b)) - flipped direction
        # np.flipud(np.rot90(b)) - flipped direction
        if moves is not None:
            return board, moves
        return board


hex_game_spec = [
    ('board', int32[:, :]),
    ('color', int32),
    ('winner', int32),
]


@jitclass(hex_game_spec)
class HexGameImpl:
    def __init__(self, board_size):
        self.board = np.zeros((board_size, board_size), dtype=np.int32)
        self.color = 1  # 1:X, 2:O
        self.winner = 0

    def legal_moves(self):
        if self.winner:
            return np.empty(0, dtype=np.int32)
        # reserve zero for padding
        #moves = np.flatnonzero(self.board == 0).astype(np.int32) + 1
        moves = [i + 1
                 for i in range(self.board.size) if self.board.flat[i] == 0]
        assert len(moves), 'winner not detected'
        return np.array(moves, dtype=np.int32)

    def result(self):
        """Has game terminated.
        :returns: 0: game continues,
                  1: second player (O) won,
                  3: first player (X) won
        """
        if self.winner:
            result = 1 if self.winner == 2 else 3
            return result
        return 0

    def step(self, move):
        tile = move - 1
        assert tile >= 0, 'illegal move'
        assert self.board.flat[tile] == 0, 'illegal move'
        assert self.winner == 0, 'illegal move'
        self.board.flat[tile] = self.color
        self.color = 3 - self.color
        self.winner = check_win(self.board, tile)


@njit('int32[:](int32, UniTuple(int64, 2))')
def neighbors(tile, size):
    """Enumerate neighboring tiles on the hex board.
    :param tile: Center tile
    :param size: Board size
    :returns: Array of neighboring tiles
    """
    ti, tj = divmod(tile, size[1])
    neigs = np.empty(6, dtype=np.int32)
    n = 0
    for ni, nj in [(ti - 1, tj),
                   (ti - 1, tj + 1),
                   (ti, tj - 1),
                   (ti, tj + 1),
                   (ti + 1, tj - 1),
                   (ti + 1, tj)]:
        if 0 <= ni < size[0] and 0 <= nj < size[1]:
            neigs[n] = ni * size[1] + nj
            n = n + 1
    return neigs[:n]


@njit('int32(int32[:, :], int32)')
def check_win(board, tile):
    """Check if move terminated the game.
    :param board: Game board after move
    :param tile: Tile of most recent move
    :returns: Color of winner or 0 if none
    """
    color = board.flat[tile]
    assert color, 'cannot check empty tile'
    connected = set()
    border = [tile]
    imin, imax = 9999, -9999
    while border:
        tile = border.pop()
        connected.add(tile)
        # get Y (color=1) or X (color=2) coordinate of tile
        i = divmod(tile, board.shape[1])[color - 1]
        # check if connected region reaches both ends of board
        # vertically (color=1) or horizontally (color=2)
        imin, imax = min(imin, i), max(imax, i)
        if imin == 0 and imax == board.shape[color - 1] - 1:
            return color
        for neig in neighbors(tile, board.shape):
            if neig in connected:
                continue
            if board.flat[neig] == color:
                border.append(neig)
    return 0


if __name__ == '__main__':
    game = HexGame(board_size=11)
    game.snapshot()
    for g in range(100):
        game.restore()
        for m in range(1000):
            moves = game.state.legal_moves
            move = np.random.choice(moves)
            print('Move %d %s: %s' % (
                m,
                ['x', 'o'][game.state.color],
                game.io.format_move(game.state.board, move)))
            game.move(move)
            game.io.print_board(game.state.board)
            if game.state.result:
                break
