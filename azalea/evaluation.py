
import logging
from collections import defaultdict

import numpy as np
import torch

from .game import import_game
from .policy import Policy
from .random_policy import RandomPolicy
from .parallel import ParallelRunner


def play_game(config, policies, game_max_length=300):
    """Play one game between two policies.
    :returns: +1 first player wins, 0 draw, -1 first player loses
    """
    gamelib = import_game(config['game'])
    game = gamelib.GameState(board_size=config['board_size'])
    nodes = [p.reset(game) for p in policies]
    for m in range(game_max_length):
        action, nodes[0], moves = \
            play_move(gamelib, game, policies[0], nodes[0], m)
        if game.result():
            break
        nodes[1] = policies[1].make_action(game, nodes[1], action, moves)
        # switch turns
        policies = policies[::-1]
        nodes = nodes[::-1]
    result = game.result()
    if not result:
        logging.warning("game didn't terminate in %d moves", game_max_length)
        return 0  # draw
    return result - 2


def play_move(utils, game, policy, node, m, print_moves=False):
    action, node, probs, metrics = policy.choose_action(
        game, node, m, temperature=1, noise_scale=0)
    moves = game.legal_moves()
    if print_moves:
        print('Move %d (%s): %s %.2f' % (
            m,
            ['white', 'black'][m % 2],
            utils.format_move(game.board(), moves[action]),
            probs[action]))
    game.move(moves[action])
    if print_moves:
        utils.print_moves(game.board(), moves, probs)
    return action, node, moves


def evaluate(policies, config, num_rounds):
    """Run round robin tournament between policies.
    :param num_rounds: Number of rounds in tournament
    :returns: Dict of outcome triplets
    """
    outcomes = defaultdict(lambda: [0, 0, 0])
    pairs = gen_pairs(len(policies))
    num_games = num_rounds * len(pairs)
    game = 1
    for r in range(num_rounds):
        for pair, res in parallel_compare(pairs, policies, config):
            outcomes[pair][0] += (res > 0)   # first player wins
            outcomes[pair][1] += (res == 0)  # draws
            outcomes[pair][2] += (res < 0)   # second player wins
            winrate = outcomes[pair][0] / sum(outcomes[pair])
            logging.info(f'game {game}/{num_games}: pair {pair}: '
                         f'outcomes {outcomes[pair]} (wins {winrate:.2f})')
            game += 1
    return outcomes


def gen_pairs(num_players):
    """generate round robin tournament pair ordering"""
    pairs = [(i, j)
             for j in range(num_players)
             for i in range(j)]
    return pairs


def parallel_compare(pairs, policies, config):
    runner = ParallelRunner(parallel_play,
                            num_workers=config.get('num_workers'))
    for seed, pair in enumerate(pairs):
        state_pair = [policies[pair[0]].state_dict(),
                      policies[pair[1]].state_dict()]
        runner.submit(pair, state_pair, config, seed)
        if runner.full():
            yield runner.result()
    while not runner.empty():
        yield runner.result()
    runner.close()


def parallel_play(pair, states, config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    def restore(state):
        if state:
            policy = Policy(config)
            policy.load_state_dict(state)
        else:
            policy = RandomPolicy(config)
        return policy

    policies = [restore(s) for s in states]
    # choose first player by coin flip
    order = 1 if np.random.random() < .5 else -1
    outcome = order * play_game(config, policies[::order])
    return pair, outcome
