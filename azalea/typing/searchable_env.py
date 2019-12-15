
from typing import Optional
from typing_extensions import Protocol

import numpy as np


class SearchableEnv(Protocol):

    def reset(self, *args, **kwargs) -> None:
        """Start new game
        """
        ...

    def seed(self, seed: Optional[int]) -> None:
        """Seed random number generator."""
        ...

    def step(self, action: int) -> None:
        """Make one move in game.
        """
        ...

    @property
    def state(self) -> 'GameState':
        """Get current game state.
        """
        ...

    def snapshot(self) -> None:
        """Store snapshot of current game state.
        """
        ...

    def restore(self) -> None:
        """Restore game state from previous snapshot.
        """
        ...


class GameState(Protocol):
    color: int
    legal_moves: np.ndarray
    result: int
    board: np.ndarray
