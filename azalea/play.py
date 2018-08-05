
import logging

import numpy as np
import torch
import click

import azalea as az


def play_game(config, policy, human=1, game_max_length=300):
    # start new game
    gamelib = az.import_game(config['game'])
    game = gamelib.GameState(board_size=config['board_size'])
    node = policy.reset(game)
    computer_move = None
    for m in range(game_max_length):
        moves = game.legal_moves()
        if human:
            print_board(gamelib, game.board(), computer_move)
            move = ask_move(gamelib, game.board(), moves)
            action = list(moves).index(move)
        else:
            # computer plays
            action, node, probs, metrics = \
                policy.choose_action(game, node, m,
                                     temperature=0, noise_scale=0)
            computer_move = move = moves[action]
        game.move(move)
        if game.result():
            print_board(gamelib, game.board(), computer_move)
            return game.result()
        if human:
            node = policy.make_action(game, node, action, moves)
        human = 1 - human


def ask_move(utils, board, moves):
    while True:
        move_txt = input('Your move? ')
        try:
            move = utils.parse_move(board, moves, move_txt)
            if move in moves:
                return move
        except utils.ParseError:
            print('illegal move:', move_txt)


def print_board(utils, board, computer_move):
    print('\033[2J')
    if computer_move:
        utils.print_board(board)  # , highlight=computer_move)
        print('last move: {}'.format(utils.format_move(
            board, computer_move)))
    else:
        utils.print_board(board)


@click.command()
@click.argument('model')
@click.option('--game', default='hex', help='Game, chess or hex')
@click.option('--first/--second', default=False, help='Human plays first or second')
def main(model, game, first):
    """Play against chess AI.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # runtime configuration (no hyperparameters)
    config = {
        'game': game,
        'seed': 0xBAD5EED5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    policy = az.Policy.load(config, model)
    config['board_size'] = policy.board_size
    res = play_game(config, policy, human=int(first))
    if not first:
        res = 4 - res
    if res == 3:
        print('You win')
    if res == 2:
        print('Draw')
    else:
        print('You lose')


if __name__ == '__main__':
    main()
