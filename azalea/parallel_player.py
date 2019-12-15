
import logging
from collections import defaultdict
from typing import Dict, Generator, Sequence, Tuple

from .play_game import play_game
from .process_pool import ProcessPool
from .replay_buffer import ReplayDataFrame
from .search_tree import SearchTreeFull
from .typing import Agent


Metrics = Dict[str, float]
GameplayGen = Generator[Tuple[int, ReplayDataFrame, Metrics], None, None]


class Player:
    def __init__(self, pool: ProcessPool,
                 agents: Sequence[Agent]):
        self.agents = agents
        self.running = True
        self.gen = pool.map_unordered_lowlat(worker, self._gen_worker_args())

    def read(self, size: int) -> Tuple[ReplayDataFrame, Metrics]:
        """Play self-play games and generate boards.
        :param size: Number of game positions to return
        """
        return batch_examples(self.gen, size)

    def stop(self) -> None:
        """Clean up processes after self-play."""
        self.running = False
        for _ in self.gen:
            pass

    def _gen_worker_args(self):
        while self.running:
            yield (self.agents,)


def batch_examples(gameplay_gen: GameplayGen, size: int) \
        -> Tuple[ReplayDataFrame, Metrics]:
    """Batch replay data to given size.
    """
    examples = ReplayDataFrame()
    metrics: Metrics = defaultdict(int)
    while len(examples) < size:
        _, game_data, game_metrics = next(gameplay_gen)
        examples.append(game_data)
        for name in game_metrics:
            metrics[name] += game_metrics[name]
    return examples, metrics


def worker(agents: Sequence[Agent]) \
        -> Tuple[int, ReplayDataFrame, Metrics]:
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
        return play_game(agents, collect_data=True)
    except SearchTreeFull:
        logging.warning('game failed because of SearchTreeFull '
                        '(skipped)')
        return -1, ReplayDataFrame(), {}
