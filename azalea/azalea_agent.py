
from typing import Callable, Dict, Optional

import torch

from .policy import Policy
from .random_policy import RandomPolicy
from .typing import SearchableEnv


class AzaleaAgent:
    """Top-level game playing agent/engine interface."""

    def __init__(self, game_factory: Callable[[], SearchableEnv], *,
                 path: str = None,
                 policy=None,
                 device=None):

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if path is not None and policy is not None:
            raise ValueError('cannot give both path and policy')

        if path is None and policy is None:
            policy = RandomPolicy()
        elif path:
            policy = Policy.load(path, device=device)

        self.game = game_factory()  # own game for each agent
        self.policy = policy
        self.info = None
        self.seed()

    def reset(self) -> None:
        """Start a new game."""
        self.game.reset()
        self.policy.reset()
        self.info = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Seed random number generator."""
        self.game.seed(seed)
        self.policy.seed(None if seed is None else seed + 1)

    @property
    def ply(self) -> int:
        return self.policy.ply

    @property
    def settings(self) -> Dict:
        return self.policy.settings

    def choose_action(self) -> int:
        """Plan next move."""
        move, info = self.policy.choose_action(self.game)
        self.info = info
        return move

    def execute_action(self, move: int) -> int:
        """Execute move."""
        self.policy.execute_action(move, self.game.state.legal_moves)
        self.game.step(move)
        return self.game.state.result
