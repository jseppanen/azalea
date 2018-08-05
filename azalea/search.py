
import numpy as np
from numba import jit

from .fnv1a import fnv1a


#@jit
def get_game_states(utils, game, path_ids, path_lens, path_checksums):
    """Run batch of game simulations.
    Return leaf board states
    """
    board_size = game.board().shape[0]
    boards = np.zeros((len(path_lens), board_size * board_size),
                      dtype=np.int32)
    results = np.zeros(len(path_lens), dtype=np.int32)
    checksums = np.zeros(len(path_lens), dtype=np.uint32)
    moves = []  # need padding
    p = 0
    for i in range(len(path_lens)):
        pl = path_lens[i]
        path = path_ids[p:p+pl]
        check = path_checksums[p:p+pl]
        p += pl
        with game.snapshot():
            for j, c in zip(path, check):
                mov = game.legal_moves()
                assert fnv1a(mov) == c
                assert not game.result()
                game.move(mov[j])
            norm_board, norm_moves, norm_result = \
                normalize_position(utils, game, randomize=True)
            boards[i] = norm_board.ravel()
            results[i] = norm_result
            if norm_result:
                checksums[i] = 0
                moves.append([])
            else:
                checksums[i] = fnv1a(game.legal_moves())
                moves.append(norm_moves)
    # pad moves
    maxlen = max(len(m) for m in moves)
    padded_moves = np.zeros((len(path_lens), maxlen), dtype=np.int32)
    for i, m in enumerate(moves):
        padded_moves[i, :len(m)] = m
    return boards, results, padded_moves, checksums


def normalize_position(utils, game, randomize=False):
    # normalize board and moves always to white's perspective
    norm_moves = game.legal_moves()
    norm_board = game.board()
    norm_result = game.result()
    if game.color():
        norm_board, norm_moves = utils.flip_player(
            norm_board, norm_moves)
        norm_result = 4 - norm_result if norm_result else 0
    if randomize:
        norm_board, norm_moves = utils.random_reflect(
            norm_board, norm_moves)
    return norm_board, norm_moves, norm_result
