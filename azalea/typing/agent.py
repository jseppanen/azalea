
from enum import IntEnum
from typing import Any, Mapping, Optional
from typing_extensions import Protocol


class Agent(Protocol):
    """Game playing agent interface"""

    def reset(self, *args, **kwargs) -> None:
        """Start a new game."""
        ...

    def seed(self, seed: Optional[int]) -> None:
        """Seed random number generator."""
        ...

    @property
    def settings(self) -> Mapping[str, Any]:
        """Access agent settings (read/write)."""
        ...

    def choose_action(self) -> int:
        """Plan next move."""
        ...

    def execute_action(self, action: int) -> 'GameResult':
        """Execute move.
        :returns: Game result
        """
        ...


class GameResult(IntEnum):
    """Game result
    Wins/losses are from first player perspective.
    """
    ONGOING = 0  # game hasn't terminated
    LOSS = 1     # first player lost
    DRAW = 2
    WIN = 3      # first player won
