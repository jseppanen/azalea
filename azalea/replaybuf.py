
import numpy as np
from torch.utils.data import Dataset, DataLoader

from .monitor import monitor


class Replaydata:
    def __init__(self, data=None, board_size=None):
        if data is None:
            assert board_size is not None, 'board_size missing'
            data = Replaydata.empty(board_size, 0).data
        assert set(data.keys()) == {'board', 'moves', 'probs', 'reward'}
        self.data = dict(data)

    @staticmethod
    def empty(board_size, size):
        return Replaydata({
            'board': np.zeros((size, board_size * board_size), dtype=np.int32),
            'moves': [np.zeros(0, dtype=np.int32) for i in range(size)],
            'probs': [np.zeros(0, dtype=np.float32) for i in range(size)],
            'reward': np.zeros(size, dtype=np.float32)
        })

    def __len__(self):
        return len(self.data['board'])

    def __getitem__(self, ii):
        res = {}
        for k in self.data:
            res[k] = self.data[k][ii]
        if isinstance(ii, slice):
            return Replaydata(res)
        else:
            return res

    def __setitem__(self, ii, other):
        if isinstance(other, Replaydata):
            other = other.data
        for k in self.data:
            self.data[k][ii] = other[k]

    def concat(self, other):
        return Replaydata({
            'board': np.vstack([self.data['board'], other.data['board']]),
            'moves': self.data['moves'] + other.data['moves'],
            'probs': self.data['probs'] + other.data['probs'],
            'reward': np.concatenate([self.data['reward'],
                                      other.data['reward']])
        })

    def state_dict(self):
        """Return replaybuf contents
        """
        return self.data

    def load_state_dict(self, state):
        """Load replaybuf contents
        """
        self.data = dict(state)


class Replaybuf(Dataset):
    def __init__(self, board_size, size, randomize=None):
        """
        :param randomize: Flip examples by random during sampling
        """
        super().__init__()
        self.data = Replaydata.empty(board_size, size)
        self.randomize = randomize
        self.write_idx = 0
        self.epoch = 0
        self.board_size = board_size

    def __len__(self):
        """Return amount of data in replay buffer.
        """
        if self.ready():
            return len(self.data)
        # partially filled
        return self.write_idx

    def capacity(self):
        """Return maximum capacity of data in replay buffer.
        """
        return len(self.data)

    def __getitem__(self, i):
        assert self.ready()
        row = self.data[i]
        board, moves = row['board'], row['moves']
        if self.randomize:
            # assume board is square
            board = board.reshape((self.board_size, self.board_size))
            board, moves = self.randomize(board, moves.astype(np.int32))
            board = board.ravel()
        input = {'board': board.astype('int64'),
                 'moves': moves.astype('int64')}
        target = {'reward': row['reward'],
                  'moves_prob': row['probs']}
        return input, target

    def put(self, new_data):
        if len(new_data) > len(self.data):
            new_data = new_data[-len(self.data):]
        if self.write_idx + len(new_data) > len(self.data):
            n = len(self.data) - self.write_idx
            self.put(new_data[:n])
            new_data = new_data[n:]
        n = len(new_data)
        if not n:
            return
        assert self.write_idx + n <= len(self.data), 'overflow'
        self.data[self.write_idx:self.write_idx + n] = new_data
        self.write_idx = (self.write_idx + n) % len(self.data)
        self.epoch += (self.write_idx == 0)

    def ready(self):
        return self.epoch > 0

    def state_dict(self):
        """Return replaybuf contents
        """
        return {
            'buf': self.data[:len(self)].state_dict(),
            'board_size': self.board_size,
            'randomize': self.randomize
        }

    def load_state_dict(self, state):
        """Load replaybuf contents
        """
        self.data.load_state_dict(state['buf'])
        self.board_size = state['board_size']
        self.randomize = state['randomize']
        self.write_idx = 0
        self.epoch = 1


class ReplaybufLoader(DataLoader):
    def __init__(self, datagen, board_size, replaybuf_size, replaybuf_resample,
                 randomize=None, **kwargs):
        """Initialize buffering data loader.
        :param datagen: Data generator
        :param replaybuf_size: Replay buffer size
        :param replaybuf_resample: How many times to resample data
        """
        self.replaybuf = Replaybuf(board_size,
                                   replaybuf_size,
                                   randomize=randomize)
        self.replaybuf_resample = replaybuf_resample
        super().__init__(self.replaybuf, **kwargs)
        self._sampler = loop(super(ReplaybufLoader, self).__iter__)
        self.datagen = datagen
        self.fresh_data_counter = 0
        self.fill()

    def __iter__(self):
        # prevent accidental use because semantics differ from DataLoader
        raise NotImplementedError()

    def fill(self):
        """Fill replay buffer with new data.
        """
        self.datagen.set_random_play(True)
        while not self.replaybuf.ready():
            game_data = self.datagen.play_game()
            self.replaybuf.put(game_data)
        self.datagen.set_random_play(False)

    def sample(self):
        """Sample batch from replay buffer and refill with fresh data.
        Data generation is done after each batch, to use most up-to-date model.
        """
        assert self.replaybuf.ready()
        while self.fresh_data_counter < self.batch_size:
            game_data = self.datagen.play_game()
            self.replaybuf.put(game_data)
            self.fresh_data_counter += self.replaybuf_resample * len(game_data)
            monitor.add_scalar('sampler/replaybuf_write_position',
                               self.replaybuf.write_idx)
        batch = next(self._sampler)
        self.fresh_data_counter -= self.batch_size
        monitor.add_scalar('sampler/abs_reward',
                           batch[1]['reward'].abs().mean())
        return batch

    def state_dict(self):
        """Return replaybuf contents
        """
        return {
            'replaybuf': self.replaybuf.state_dict(),
            'replaybuf_resample': self.replaybuf_resample
        }

    def load_state_dict(self, state):
        """Load replaybuf contents
        """
        self.replaybuf.load_state_dict(state['replaybuf'])
        self.replaybuf_resample = state['replaybuf_resample']
        self.fresh_data_counter = 0


def loop(make_seq):
    while True:
        for x in make_seq():
            yield x
