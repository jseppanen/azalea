
import sys
import numpy as np

from .term import gray, blue, red, darkgray, darkblue, darkred
from .term import Color, termcolor


def print_state(state):
    print_board(state.board)


def print_board(board):
    hdr = format_header(board)
    sys.stdout.write(' ' * 3 + hdr + '\n')
    for i in range(board.shape[0]):
        txt = ' ' * (i + 1)
        txt += format_rank(board[i], i)
        sys.stdout.write(txt)
        sys.stdout.write('\n')
    sys.stdout.write(' ' * 15 + hdr + '\n')


def print_moves(board, moves, probs):
    print_board(board)


def format_rank(rank, i):
    hdr = termcolor(f'{i + 1:<2d}', fg=Color(darkblue))
    txt = hdr + ' '
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
    txt += hdr
    return txt


def format_header(board):
    alpha = 'abcdefghijk'
    txt = ''.join(
        f'{alpha[j]} ' for j in range(board.shape[1])
    )
    col = Color(darkred)
    return termcolor(txt, fg=col)


def format_move(board, move):
    tile = move - 1
    i, j = np.unravel_index(tile, board.shape)
    return '{}{}'.format(chr(j + ord('a')), i + 1)


class ParseError(Exception):
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
