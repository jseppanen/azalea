
import logging
import time
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .typing import Agent


# self play with one policy and store training data
# play against human (interactively)
# play two policies and return outcome


def play_game(agents: Sequence[Agent], *,
              game_max_length: int = 300,
              print_moves: bool = False,
              collect_data: bool = False) \
        -> Tuple[int, Optional[Dict], Dict]:
    """Play one game with given agents.
    MAY RAISE SearchTreeFull
    :param collect_data: Generate training data with self play
    :returns: Replay data and search metrics from one game
    """
    # start new game
    for a in agents:
        a.reset()

    game_data: Optional[Dict[str, Sequence]] = None
    if collect_data:
        game_data = {
            'state': [],       # input
            'moves_prob': [],  # target
        }

    metrics: Dict[str, float] = defaultdict(int)
    start_time = time.time()
    result = 0
    for ply in range(game_max_length):
        move, info = choose_move(agents[0],
                                 game_data=game_data,
                                 print_moves=print_moves)
        for name in info['metrics']:
            metrics[name] += info['metrics'][name]
        metrics['action_logprob'] += np.log(info['prob'])
        res = [ag.execute_action(move) for ag in agents]
        result = res[0]
        assert all(r == result for r in res), "conflicting game states"
        if result:
            break
        # switch turns
        agents = agents[::-1]

    game_length = ply + 1

    assert result in (0, 1, 2, 3)
    if not result:
        logging.warning("game didn't terminate in %d moves", game_max_length)
        result = 2  # draw

    # compute reward for each player in turn
    reward = np.full(game_length, result - 2., dtype=np.float32)
    reward[1::2] *= -1
    if collect_data:
        game_data['reward'] = reward.tolist()

    for name in metrics:
        metrics[name] /= max(1, game_length)

    metrics['games'] = 1
    metrics['reward'] = float(reward[-1])
    metrics['moves_per_game'] = game_length
    metrics['seconds_per_game'] = time.time() - start_time
    metrics['game_error'] = 0

    return result, game_data, metrics


def choose_move(agent: Agent,
                game_data: Optional[Dict[str, Sequence]],
                print_moves: bool) -> Tuple[int, Dict]:
    move = agent.choose_action()
    info = agent.info

    if game_data:
        game_data['state'].append(agent.game.state)
        game_data['moves_prob'].append(
            info['moves_prob'].astype(np.float32))

    if print_moves:
        #print('\033[2J\033[1;1H')  # clear screen
        color = ['white', 'black'][agent.game.state.color]
        move_txt = agent.game.io.format_move(agent.game.state.board, move)
        prob = info['prob']
        ply = agent.ply
        print(f'Move {ply+1} ({color}): {move_txt} {prob:.2f}')

    return move, info
