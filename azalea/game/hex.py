
import sys
from contextlib import contextmanager

import numpy as np
from numba import njit, jitclass, int32

from .term import gray, blue, red, Color, termcolor


game_state_spec = [
    ('board', int32[:, :]),
    ('color', int32),
    ('winner', int32),
]


@jitclass(game_state_spec)
class HexGameStateImpl:
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
        :returns: 0: game continues, 1: O won, 3: X won
        """
        if self.winner:
            result = 1 if self.winner == 2 else 3
            return result
        return 0

    def move(self, move):
        tile = move - 1
        assert tile >= 0, 'illegal move'
        assert self.board.flat[tile] == 0, 'illegal move'
        assert self.winner == 0, 'illegal move'
        self.board.flat[tile] = self.color
        self.color = 3 - self.color
        self.winner = check_win(self.board, tile)


class GameState:
    def __init__(self, startpos=None, board_size=11):
        assert startpos is None, 'start position not supported'
        self.impl = HexGameStateImpl(board_size)

    def board(self):
        return self.impl.board.copy()

    def color(self):
        """Current player color.
        :returns: 0=first player (red), 1=second player (blue)
        """
        return self.impl.color - 1

    def legal_moves(self):
        return self.impl.legal_moves()

    def result(self):
        return self.impl.result()

    def move(self, move):
        self.impl.move(move)

    @contextmanager
    def snapshot(self):
        b = self.impl.board.copy()
        c = self.impl.color
        w = self.impl.winner
        try:
            yield
        finally:
            self.impl.board[:] = b
            self.impl.color = c
            self.impl.winner = w


def flip_player(board, moves=None):
    """Flip board to opponent perspective.
    Change both color and attack direction.
    """
    # flip color
    board = (board > 0) * (3 - board)
    # flip attack direction
    board = np.fliplr(np.rot90(board))
    if moves is not None:
        m, n = board.shape
        tiles = moves - 1
        ii, jj = np.unravel_index(tiles, board.shape)
        tiles = np.ravel_multi_index((n - 1 - jj, m - 1 - ii), board.shape)
        moves = tiles.astype(np.int32) + 1
        return board, moves
    return board


def random_reflect(board, moves=None):
    """Rotate/flip board at random while keeping same player perspective.
    """
    # b - same direction
    # np.rot90(b,2) - same direction
    # np.fliplr(np.rot90(b)) - flipped direction
    # np.flipud(np.rot90(b)) - flipped direction
    if moves is not None:
        return board, moves
    return board


@njit('int32[:](int32, UniTuple(int64, 2))')
def neighbors(tile, size):
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
    color = board.flat[tile]
    assert color, 'cannot check empty tile'
    connected = set()
    border = [tile]
    imin, imax = 9999, -9999
    while border:
        tile = border.pop()
        connected.add(tile)
        i = divmod(tile, board.shape[1])[color - 1]
        imin, imax = min(imin, i), max(imax, i)
        if imin == 0 and imax == board.shape[color - 1] - 1:
            return color
        for neig in neighbors(tile, board.shape):
            if neig in connected:
                continue
            if board.flat[neig] == color:
                border.append(neig)
    return 0


def print_board(board):
    for i in range(board.shape[0]):
        if i < 9:
            txt = ' ' * i
        elif i == 9:
            txt = termcolor('  x', fg=Color(red)) + ' ' * 6
        else:
            txt = termcolor(' o', fg=Color(blue)) \
                  + termcolor(r'\\', fg=Color(gray)) + ' ' * 6
        txt += format_rank(board[i])
        sys.stdout.write(txt)
        sys.stdout.write('\n')


def print_moves(board, moves, probs):
    print_board(board)


def format_rank(rank):
    txt = ''
    for j in range(len(rank)):
        col = rank[j]
        pie = '.' if col == 0 else \
              'X' if col == 1 else \
              'O'
        fg = Color(gray if col == 0 else
                   red if col == 1 else
                   blue)
        txt += termcolor(pie, fg=fg)
        txt += ' '
    return txt


def format_move(board, move):
    tile = move - 1
    i, j = np.unravel_index(tile, board.shape)
    return '{}{}'.format(chr(j + ord('a')), i + 1)


class ParseError(BaseException):
    pass


def parse_move(board, moves, move_txt):
    try:
        x = move_txt[0]
        y = int(move_txt[1:])
    except (IndexError, ValueError):
        raise ParseError(move_txt)
    i = y - 1
    j = ord(x) - ord('a')
    if not (0 <= i < board.shape[0]
            and 0 <= j < board.shape[1]):
        raise ParseError(move_txt)
    tile = np.ravel_multi_index((i, j), board.shape)
    return tile + 1


if __name__ == '__main__':
    game = GameState(board_size=11)
    for g in range(100):
        with game.snapshot():
            for m in range(1000):
                moves = game.legal_moves()
                move = np.random.choice(moves)
                print('Move %d %s: %s' % (
                    m,
                    ['x', 'o'][game.color()],
                    format_move(game.board(), move)))
                game.move(move)
                print_board(game.board())
                if game.result():
                    break
