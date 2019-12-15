
import logging
import time
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .replay_buffer import ReplayDataFrame
from .typing import Agent


# self play with one policy and store training data
# play against human (interactively)
# play two policies and return outcome


def play_game(agents: Sequence[Agent], *,
              game_max_length: int = 300,
              print_moves: bool = False,
              collect_data: bool = False) \
        -> Tuple[int, ReplayDataFrame, Dict]:
    """Play one game with given agents.
    MAY RAISE SearchTreeFull
    :param collect_data: Generate training data with self play
    :returns: Replay data and search metrics from one game
    """
    # start new game
    for a in agents:
        a.reset()

    game_data = ReplayDataFrame()
    if collect_data:
        agents = [agent_collect_gameplay(agent, game_data)
                  for agent in agents]

    metrics: Dict[str, float] = defaultdict(int)
    agents = [agent_track_metrics(agent, metrics)
              for agent in agents]

    if print_moves:
        agents = [agent_print_moves(agent) for agent in agents]

    start_time = time.time()
    result = 0
    for ply in range(game_max_length):
        move = agents[0].choose_action()
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
        game_data.reward = list(reward)  # keep as float32

    for name in metrics:
        metrics[name] /= max(1, game_length)

    metrics['games'] = 1
    metrics['reward'] = float(reward[-1])
    metrics['moves_per_game'] = game_length
    metrics['seconds_per_game'] = time.time() - start_time
    metrics['game_error'] = 0

    return result, game_data, metrics


def agent_collect_gameplay(agent: Agent, game_data: ReplayDataFrame) -> Agent:
    """Wrap agent with data collection."""

    class CollectGameplayWrapper:
        """Wrap agent with data collection."""

        def __getattr__(self, name):
            return getattr(agent, name)

        def choose_action(self) -> int:
            move = agent.choose_action()

            game_data.state.append(agent.game.state)
            game_data.moves_prob.append(
                agent.info['moves_prob'].astype(np.float32))
            return move

    return CollectGameplayWrapper()


def agent_track_metrics(agent: Agent, metrics: Dict[str, float]) -> Agent:
    """Wrap agent with metrics tracking."""

    class TrackMetricsWrapper:
        """Wrap agent with metrics tracking."""

        def __getattr__(self, name):
            return getattr(agent, name)

        def choose_action(self) -> int:
            move = agent.choose_action()
            for name in agent.info['metrics']:
                metrics[name] += agent.info['metrics'][name]
            metrics['action_logprob'] += np.log(agent.info['prob'])
            return move

    return TrackMetricsWrapper()


def agent_print_moves(agent: Agent) -> Agent:
    """Wrap agent with move printing."""

    class PrintMovesWrapper:
        """Wrap agent with move printing."""

        def __getattr__(self, name):
            return getattr(agent, name)

        def choose_action(self) -> int:
            move = agent.choose_action()
            #print('\033[2J\033[1;1H')  # clear screen
            color = ['white', 'black'][agent.game.state.color]
            move_txt = agent.game.io.format_move(agent.game.state.board, move)
            prob = agent.info['prob']
            ply = agent.ply
            print(f'Move {ply+1} ({color}): {move_txt} {prob:.2f}')
            return move

    return PrintMovesWrapper()
