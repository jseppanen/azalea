
import numpy as np


class RandomPolicy:
    def __init__(self, config):
        pass

    def reset(self, game):
        pass

    def load_state_dict(self, state):
        pass

    def state_dict(self):
        return {}

    def choose_action(self, game, node, depth, **kwargs):
        """Choose next move at random.
        :param game: Current game state
        :returns: action - action id,
                  node - child node following action,
                  probs - 1d array of action probabilities
        """
        assert not game.result()
        moves = game.legal_moves()
        action = np.random.randint(len(moves))
        probs = np.ones(len(moves), dtype=np.float32) / len(moves)
        metrics = {}
        node = None
        return action, node, probs, metrics

    def make_action(self, game, node, action, moves):
        """Update search tree with opponent action.
        can raise SearchTreeFull
        """
        pass

    def tree_stats(self):
        return {}
