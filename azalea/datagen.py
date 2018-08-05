
import logging
import time
from collections import defaultdict

import numpy as np

from .game import import_game
from .replaybuf import Replaydata
from .monitor import monitor
from .search_tree import SearchTreeFull


def play_game(config, policy, print_moves=False):
    """Play one game, where policy is playing against itself.
    Generate training data with self play.
    :returns: Replay data and search metrics from one game
    """
    start_position = config.get('start_position', None)
    game_max_length = config.get('game_max_length', 300)
    # start new game
    t0 = time.time()
    game_data = defaultdict(list)
    gamelib = import_game(config['game'])
    game = gamelib.GameState(board_size=config['board_size'],
                             startpos=start_position)
    assert not game.result(), 'game started in terminal state'
    node = policy.reset(game)
    metrics = {}
    for m in range(game_max_length):
        try:
            action, node, probs, search_metrics = policy.choose_action(
                game, node, m)
        except SearchTreeFull:
            no_data = Replaydata.empty(config['board_size'], 0)
            no_metrics = {'moves_per_game': 0,
                          'seconds_per_game': time.time() - t0,
                          'game_error': 1}
            return no_data, no_metrics
        moves = game.legal_moves()
        put_game_data(gamelib, game_data,
                      game.board(), game.color(), moves, probs, m)
        search_metrics['action_logprob'] = np.log(probs[action])
        for m in search_metrics:
            metrics[m] = metrics.get(m, 0) + search_metrics[m]
        if print_moves:
            #print('\033[2J')
            print('Move %d (%s): %s %.2f' % (
                m + 1,
                ['white', 'black'][game.color()],
                gamelib.format_move(game.board(), moves[action]),
                probs[action]))
        game.move(moves[action])
        if print_moves:
            gamelib.print_moves(game.board(), moves, probs)
        if game.result():
            break
    result = game.result()
    if not result:
        logging.warning("game didn't terminate in %d moves", game_max_length)
        result = 2  # draw
    # compute reward for each player in turn
    reward = result - 2.
    reward = np.ones(len(game_data['board']), dtype=np.float32) \
        * reward
    reward[1::2] *= -1
    game_data['reward'] = reward
    for m in metrics:
        metrics[m] /= len(game_data['board'])
    metrics.update(policy.tree_stats())
    metrics['reward'] = float(reward[-1])
    metrics['moves_per_game'] = len(game_data['board'])
    metrics['seconds_per_game'] = time.time() - t0
    metrics['game_error'] = 0
    return Replaydata(game_data), metrics


def put_game_data(utils, game_data, board, color, moves, probs, depth):
    # represent board and moves as if the white player was in turn
    if color:
        board, moves = utils.flip_player(board, moves)
    board, moves = utils.random_reflect(board, moves)
    game_data['board'].append(board.ravel())
    game_data['moves'].append(moves.astype(np.int64))
    game_data['probs'].append(probs.astype(np.float32))


def play_game_startpos(config, policy):
    """Play game from random start position.
    """
    if 'start_positions' in config:
        config = dict(config,
                      startpos=np.random.choice(config['start_positions']))
    return play_game(config, policy)


class Datagen:
    def __init__(self, policy, config):
        self.policy = policy
        self.config = config

    def play_game(self):
        """Self play data generator.
        Run policy against itself and return game data.
        :returns: Replay data from one self play game
        """
        game_data, game_stats = play_game_startpos(self.config, self.policy)
        monitor.add_scalar(
            'datagen/moves_per_second',
            game_stats['moves_per_game'] / game_stats['seconds_per_game'])
        for m in game_stats:
            monitor.add_scalar('datagen/{}'.format(m), game_stats[m])
        return game_data

    def close(self):
        pass
