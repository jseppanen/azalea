
from typing import Dict, List, Optional, Union

from torch.utils.data import Dataset


# FIXME make proper (data)class for this
Examples = Dict[str, List]


class ReplayData(Dataset):

    def __init__(self, examples: Optional[Union['ReplayData', Examples]] = None):
        """
        :param examples: Initial replay data dict
        """
        super().__init__()

        if examples is None:
            self.examples: Examples = {
                'state': [],
                'moves_prob': [],
                'reward': [],
            }
        elif isinstance(examples, ReplayData):
            self.examples = examples.examples
        elif isinstance(examples, dict):
            self.examples = examples
        else:
            raise TypeError(examples)

    def __len__(self) -> int:
        """Return number of training examples in replay buffer.
        """
        return len(self.examples['state'])

    def __getitem__(self, idx: Union[int, slice]) -> Examples:
        """Return one or many training examples from buffer.
        :param idx: Buffer index or slice
        :returns: dict
        """
        examples = {key: self.examples[key][idx] for key in self.examples}
        return examples

    def __setitem__(self, idx: Union[int, slice],
                    examples: Examples) -> None:
        """Overwrite one or many training examples in buffer.
        :param idx: Buffer index or slice
        :param value: New examples
        """
        len_before = len(self)
        for key in self.examples:
            self.examples[key][idx] = examples[key]
        assert len(self) == len_before

    def append(self, examples: Examples) -> None:
        """Append one or many training examples."""
        assert isinstance(examples, dict)
        for key in self.examples:
            self.examples[key].extend(examples[key])


class ReplayBuffer(ReplayData):
    """Replay buffer is a circular buffer of gameplay data.
    Data is structured as dict of lists/tensors.
    Minibatches are sampled randomly from buffer contents.
    """

    def __init__(self, examples):
        """
        :param data: Initial replay data as dict
        """
        super().__init__(examples)
        self.write_idx = 0
        self.fresh_counter = 0

    def consume(self, num_examples, player):
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

    def put(self, new_data):
        """Write new replay data to buffer in FIFO order.
        :param new_data: New replay data dict
        """
        if not isinstance(new_data, ReplayData):
            new_data = ReplayData(new_data)

        # implement wraparound
        if self.write_idx + len(new_data) > len(self):
            space = len(self) - self.write_idx
            self.put(new_data[:space])
            self.put(new_data[space:])
            return

        ids = slice(self.write_idx, self.write_idx + len(new_data))
        assert ids.stop <= len(self), 'overflow'
        self[ids] = new_data.examples
        self.write_idx = ids.stop % len(self)
        self.fresh_counter += len(new_data)

    def state_dict(self):
        """Return replaybuf contents
        """
        return {
            'examples': self.examples,
            'write_idx': self.write_idx,
            'fresh_counter': self.fresh_counter,
        }

    def load_state_dict(self, state):
        """Load replaybuf contents
        """
        self.examples = state['examples']
        self.write_idx = state['write_idx']
        self.fresh_counter = state['fresh_counter']
