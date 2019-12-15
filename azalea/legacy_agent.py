
from typing import Optional


class LegacyAgent:
    """Agent interface for Azalea version 0.1.0 policies"""

    def __init__(self, *, path: str = None,
                 device='cpu'):
        # deferred import of legacy azalea 0.1.0
        import azalea01

        config = {'game': 'hex',
                  'device': device}
        self.policy = azalea01.Policy.load(config, path)
        self.settings = {'move_sampling': False}
        self.reset()

    def reset(self) -> None:
        """Start a new game."""
        # deferred import of legacy azalea 0.1.0
        import azalea01.game.hex

        self.game = azalea01.game.hex.GameState()
        self.node = self.policy.reset(self.game)
        self.update_node = True
        self.ply = 0

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed random number generator."""
        #XXX self.policy.seed(seed)

    def choose_action(self) -> int:
        """Plan next move."""
        temperature = 1.0 if self.settings['move_sampling'] else 0.0
        move_id, node, probs, metrics = \
            self.policy.choose_action(
                self.game, self.node, self.ply,
                temperature=temperature,
                noise_scale=0.)
        self.node = node
        self.update_node = False
        move = self.game.legal_moves()[move_id]
        self.info = {
            'prob': probs[move_id],
            'metrics': metrics
        }
        return move

    def execute_action(self, move: int) -> int:
        """Execute move."""
        moves = self.game.legal_moves()
        self.game.move(move)
        if self.update_node:
            move_id = list(moves).index(move)
            self.node = self.policy.make_action(
                self.game, self.node, move_id, moves)
        self.update_node = True
        self.ply += 1
        return self.game.result()
