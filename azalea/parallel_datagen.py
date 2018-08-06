
import time

import numpy as np
import torch

from .datagen import play_game_startpos
from .monitor import monitor
from .policy import Policy
from .random_policy import RandomPolicy
from .parallel import ParallelRunner


class ParallelDatagen:
    def __init__(self, policy, config):
        num_workers = config.get('num_workers')
        self.runner = ParallelRunner(parallel_play, num_workers=num_workers)
        self.policy = policy
        self.config = config
        self.random_play = True
        self.seed = config['seed']
        self.t0 = time.time()
        self.submit()

    def set_random_play(self, random):
        """Choose between normal or random policy
        """
        self.random_play = random
        self.runner.flush()
        self.submit()

    def submit(self):
        """Submit new tasks for parallel execution"""
        state = self.policy.state_dict() if not self.random_play else None
        while not self.runner.full():
            self.runner.submit(state, self.config, self.seed)
            self.seed += 1

    def play_game(self):
        game_data, game_stats = self.runner.result()
        self.submit()
        # total moves/second in parallel
        dur = time.time() - self.t0
        self.t0 = time.time()
        monitor.add_scalar('datagen/moves_per_second',
                           game_stats['moves_per_game'] / dur)
        for m in game_stats:
            monitor.add_scalar('datagen/{}'.format(m), game_stats[m])
        return game_data

    def close(self):
        self.runner.close()


def parallel_play(policy_state, config, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if policy_state:
        policy = Policy(config)
        policy.load_state_dict(policy_state)
        policy.net.eval()
    else:
        policy = RandomPolicy(config)
    return play_game_startpos(config, policy)
