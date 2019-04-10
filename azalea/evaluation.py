
import logging
from collections import defaultdict
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np

from .play_game import play_game
from .process_pool import ProcessPool
from .typing import Agent


Pair = Tuple[int, int]
OutcomeCounts = List[int]


def evaluate(agents: List[Agent], num_rounds: int,
             num_workers: Optional[int] = None) \
        -> Dict[Pair, OutcomeCounts]:
    """Run round robin tournament between policies.
    :param num_rounds: Number of rounds in tournament
    :returns: Dict of outcome triplets
    """
    outcomes: Dict[Pair, OutcomeCounts] = defaultdict(lambda: [0, 0, 0])
    pairs = gen_pairs(len(agents))
    num_games = num_rounds * len(pairs)
    game = 1
    pool = ProcessPool(num_workers=num_workers)
    for r in range(num_rounds):
        for pair, res in parallel_compare(pool, pairs, agents, r):
            outcomes[pair][0] += (res > 0)   # first player wins
            outcomes[pair][1] += (res == 0)  # draws
            outcomes[pair][2] += (res < 0)   # second player wins
            winrate = outcomes[pair][0] / sum(outcomes[pair])
            logging.info(f'game {game}/{num_games}: pair {pair}: '
                         f'outcomes {outcomes[pair]} (wins {winrate:.2f})')
            game += 1
    return outcomes


def gen_pairs(num_players: int) -> List[Pair]:
    """generate round robin tournament pair ordering"""
    pairs = [(i, j)
             for j in range(num_players)
             for i in range(j)]
    return pairs


def parallel_compare(pool: ProcessPool,
                     pairs: List[Pair], agents: List[Agent],
                     seed: int) \
        -> Generator[Tuple[Pair, int], None, None]:
    tasks = (
        (pair, (agents[pair[0]], agents[pair[1]]), 10000 * seed + s)
        for s, pair in enumerate(pairs)
    )
    for pair, outcome in pool.map_unordered_lowlat(worker, tasks):
        yield pair, outcome


def worker(pair: Pair, agents: Tuple[Agent, Agent], seed: int) \
        -> Tuple[Pair, int]:
    """Parallel evaluation worker.
    """
    # does nothing if logging is already configured
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # choose first player by coin flip
    rng = np.random.RandomState(seed)
    order = rng.choice([-1, 1])

    # re-seed agents
    for a in agents:
        a.seed(rng.randint(1 << 32))

    result, _, _ = play_game(agents[::order])
    outcome = order * (result - 2)
    return pair, outcome
