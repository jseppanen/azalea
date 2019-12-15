
from typing import Dict, Optional

import numpy as np


class RandomPolicy:
    def __init__(self):
        self.rng = np.random.RandomState()
        self.settings = {}
        self.seed()

    def reset(self):
        pass

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng.seed(seed)

    def load_state_dict(self, state: Dict) -> None:
        pass

    def state_dict(self) -> Dict:
        return {}

    def choose_action(self, game):
        """Choose next move at random.
        :param game: Current game state
        :returns: move - chosen move
                  probs - 1d array of action probabilities
        """
        assert not game.state.result
        moves = game.state.legal_moves
        action = self.rng.randint(len(moves))
        move = moves[action]
        probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        info = dict(move_id=action,
                    moves=moves,
                    moves_prob=probs,
                    prob=probs[action],
                    metrics={})
        return move, info

    def execute_action(self, move, moves):
        """Update search tree with opponent action.
        can raise SearchTreeFull
        """
        pass

    def tree_metrics(self):
        return {}
