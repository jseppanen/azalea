
from typing import Optional

import torch

from .game.hex import HexGame
from .policy import Policy
from .random_policy import RandomPolicy


class AzaleaAgent:
    """Top-level game playing agent/engine interface."""

    def __init__(self, *, path: str = None, policy=None,
                 device=None, game=HexGame):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if path is not None and policy is not None:
            raise ValueError('cannot give both path and policy')

        if path is None and policy is None:
            policy = RandomPolicy()
        elif path:
            policy = Policy.load(path, device=device)

        self.game = game()
        self.policy = policy
        self.ply = 0
        self.info = None
        self.settings = {'exploration_temperature': 0.0}

        self.seed()

    def reset(self) -> None:
        """Start a new game."""
        self.game.reset()
        self.policy.reset()
        self.ply = 0
        self.info = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed random number generator."""
        self.game.seed(seed)
        self.policy.seed(None if seed is None else seed + 1)

    def choose_action(self) -> int:
        """Plan next move."""
        move, info = self.policy.choose_action(
            self.game, self.ply,
            temperature=self.settings['exploration_temperature'],
            noise_scale=0.)
        self.info = info
        return move

    def execute_action(self, move: int) -> int:
        """Execute move."""
        self.policy.execute_action(move, self.game.state.legal_moves)
        self.game.step(move)
        self.ply += 1
        return self.game.state.result
