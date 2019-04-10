
import logging
from collections import defaultdict
from typing import Dict, Generator, Optional, Sequence, Tuple

from .play_game import play_game
from .process_pool import ProcessPool
from .replay_buffer import ReplayData
from .search_tree import SearchTreeFull
from .typing import Agent


class Player:
    def __init__(self, pool: ProcessPool,
                 agents: Sequence[Agent]):
        self.gen = parallel_gameplay_gen(pool, agents, collect_data=True)

    def read(self, size: int):
        """Play self-play games and generate boards.
        :param size: Number of game positions to return
        """
        return batch_examples(self.gen, size)


def batch_examples(gameplay_gen, size: int):
    """Batch replay data to given size.
    """
    examples = ReplayData()
    metrics: Dict[str, float] = defaultdict(int)
    while len(examples) < size:
        _, game_data, game_metrics = next(gameplay_gen)
        examples.append(game_data)
        for name in game_metrics:
            metrics[name] += game_metrics[name]
    return examples, metrics


def parallel_gameplay_gen(pool: ProcessPool, agents: Sequence[Agent], *,
                          collect_data: bool = False) \
        -> Generator[Tuple[int, Optional[Dict], Dict], None, None]:
    """Play games in parallel with process pool.
    Policies (network) state are copied to worker processes repeatedly.
    """
    def gen_args():
        while True:
            yield (agents, collect_data)

    for res in pool.map_unordered_lowlat(worker, gen_args()):
        yield res


def worker(agents, collect_data):
    """Parallel game playing worker.
    """
    # does nothing if logging is already configured
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # FIXME should have cleaner way to put agent in inference mode
    for a in agents:
        try:
            a.policy.net.eval()
        except AttributeError:
            pass

    try:
        return play_game(agents, collect_data=collect_data)
    except SearchTreeFull:
        logging.warning('game failed because of SearchTreeFull '
                        '(skipped)')
        return None, {}, {}
