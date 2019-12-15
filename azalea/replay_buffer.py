
from dataclasses import dataclass, field, fields
from typing import Dict, List, overload

import numpy as np
from torch.utils.data import Dataset

from .typing import GameState


@dataclass
class ReplayRecord:
    """Single record of gameplay data."""

    # game state
    state: GameState

    # move probabilities
    moves_prob: np.ndarray

    # reward
    reward: np.float32


@dataclass
class ReplayDataFrame(Dataset):
    """Data frame of gameplay data"""

    # game states
    state: List[GameState] = field(default_factory=list)

    # move probabilities
    moves_prob: List[np.ndarray] = field(default_factory=list)

    # rewards
    reward: List[np.float32] = field(default_factory=list)

    def __len__(self) -> int:
        """Return number of training examples in replay buffer.
        """
        return len(self.state)

    @overload
    def __getitem__(self, idx: int) -> ReplayRecord:
        """Return one training example from buffer.
        :param idx: Buffer index
        :returns: Replay data row
        """
        ...

    @overload
    def __getitem__(self, idx: slice) -> 'ReplayDataFrame':
        """Return many training examples from buffer.
        :param idx: Buffer slice
        :returns: Replay data rows
        """
        ...

    def __getitem__(self, idx):
        """Return one or many training examples from buffer.
        :param idx: Buffer index or slice
        :returns: Replay data rows
        """
        rows = [
            getattr(self, f.name)[idx]
            for f in fields(self)
        ]
        if isinstance(idx, int):
            return ReplayRecord(*rows)
        elif isinstance(idx, slice):
            return ReplayDataFrame(*rows)
        else:
            raise TypeError(idx)

    @overload
    def __setitem__(self, idx: int, rows: ReplayRecord) -> None:
        """Overwrite one training example in buffer.
        :param idx: Buffer index
        :param rows: New example
        """
        ...

    @overload
    def __setitem__(self, idx: slice, rows: 'ReplayDataFrame') -> None:
        """Overwrite many training examples in buffer.
        :param idx: Buffer slice
        :param rows: New examples
        """
        ...

    def __setitem__(self, idx, rows):
        """Overwrite one or many training examples in buffer.
        :param idx: Buffer index or slice
        :param rows: New examples
        """
        len_before = len(self)
        for f in fields(self):
            getattr(self, f.name)[idx] = getattr(rows, f.name)
        assert len(self) == len_before

    def append(self, rows: 'ReplayDataFrame') -> None:
        """Append one or many training examples."""
        for f in fields(self):
            getattr(self, f.name).extend(getattr(rows, f.name))


class ReplayBuffer(ReplayDataFrame):
    """Replay buffer is a circular buffer of gameplay data.
    Data is structured as dict of lists/tensors.
    Minibatches are sampled randomly from buffer contents.
    """

    def __init__(self, contents: ReplayDataFrame):
        """
        :param data: Initial replay data
        """
        super().__init__(contents.state, contents.moves_prob, contents.reward)
        self.write_idx = 0
        self.fresh_counter = 0

    def consume(self, num_examples: int, player: 'Player') -> Dict[str, float]:
        """Consume training data and optionally refill.
        :returns: Self play metrics
        """
        self.fresh_counter -= num_examples
        refill = max(0, num_examples - self.fresh_counter)
        if refill:
            replays, metrics = player.read(refill)
            self.put(replays)
            return metrics
        else:
            return {}

    def put(self, new_data: ReplayDataFrame) -> None:
        """Write new replay data to buffer in FIFO order.
        :param new_data: New replay data frame
        """
        # implement wraparound
        if self.write_idx + len(new_data) > len(self):
            space = len(self) - self.write_idx
            self.put(new_data[:space])
            self.put(new_data[space:])
            return

        ids = slice(self.write_idx, self.write_idx + len(new_data))
        assert ids.stop <= len(self), 'overflow'
        self[ids] = new_data
        self.write_idx = ids.stop % len(self)
        self.fresh_counter += len(new_data)

    def state_dict(self) -> Dict:
        """Return replaybuf contents
        """
        return {
            'examples': self.examples,
            'write_idx': self.write_idx,
            'fresh_counter': self.fresh_counter,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load replaybuf contents
        """
        self.examples = state['examples']
        self.write_idx = state['write_idx']
        self.fresh_counter = state['fresh_counter']
