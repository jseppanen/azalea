
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple

from .search_tree import SearchTree
from .typing import SearchableEnv
from .utils import import_and_get


def create_network(network_type, board_size, num_blocks, base_chans):
    # backward compat
    if network_type in ('ChessNetwork', 'HexNetwork'):
        network_type = 'azalea.network.' + network_type
    Net = import_and_get(network_type)
    return Net(board_size=board_size,
               num_blocks=num_blocks,
               base_chans=base_chans)


class Policy:
    """Game playing policy, combination of MCTS and network
    """

    def __init__(self):
        """Construct new policy"""
        # do greedy & deterministic inference by default
        self.settings = {
            'move_sampling': False,
            'move_exploration': False,
        }

        self.rng = np.random.RandomState()
        self.seed()

    def initialize(self, config):
        """Initialize policy for training"""
        device = torch.device(config['device'])
        if device.type == 'cuda':
            # enable cudnn auto-tuner
            torch.backends.cudnn.benchmark = True
        self.net = create_network(config['network'],
                                  config['board_size'],
                                  config['num_blocks'],
                                  config['base_chans'])
        self.net.to(device)
        # don't train anything by default
        self.net.eval()
        # network params
        self.network_type = config['network']
        self.board_size = config['board_size']
        self.num_blocks = config['num_blocks']
        self.base_chans = config['base_chans']
        # search params
        self.simulations = config['simulations']
        self.search_batch_size = config['search_batch_size']
        self.exploration_coef = config['exploration_coef']
        self.exploration_depth = config['exploration_depth']
        self.exploration_noise_alpha = config['exploration_noise_alpha']
        self.exploration_noise_scale = config['exploration_noise_scale']
        self.exploration_temperature = config['exploration_temperature']
        if 'seed' in config:
            self.seed(config['seed'])

    @property
    def net(self):
        try:
            return self._net
        except AttributeError:
            raise RuntimeError('Policy must be initialized or loaded before use')

    @net.setter
    def net(self, net):
        self._net = net

    def reset(self):
        """Start new game
        """
        self.tree = SearchTree()
        self.ply = 0

    def seed(self, seed: Optional[int] = None) -> None:
        self.rng.seed(seed)

    def load_state_dict(self, state):
        """Load model state
        """
        # load network architecture and params
        self.network_type = state['network_type']
        self.board_size = state['board_size']
        self.num_blocks = state['num_blocks']
        self.base_chans = state['base_chans']
        self.net = create_network(self.network_type,
                                  self.board_size,
                                  self.num_blocks,
                                  self.base_chans)
        self.net.load_state_dict(state['net'])

        # load search params
        self.simulations = state['simulations']
        self.search_batch_size = state['search_batch_size']
        self.exploration_coef = state['exploration_coef']
        self.exploration_depth = state['exploration_depth']
        self.exploration_noise_alpha = state['exploration_noise_alpha']
        self.exploration_noise_scale = state['exploration_noise_scale']
        self.exploration_temperature = state['exploration_temperature']

        # load random number generator state
        if 'rng' in state:
            self.rng.__setstate__(state['rng'])

    def state_dict(self):
        """Return model state
        Only serializes the (hyper)parameters, not ongoing game state (search tree etc)
        """
        return {
            'net': self.net.state_dict(),
            'network_type': self.network_type,
            'board_size': self.board_size,
            'num_blocks': self.num_blocks,
            'base_chans': self.base_chans,
            'simulations': self.simulations,
            'search_batch_size': self.search_batch_size,
            'exploration_coef': self.exploration_coef,
            'exploration_depth': self.exploration_depth,
            'exploration_noise_alpha': self.exploration_noise_alpha,
            'exploration_noise_scale': self.exploration_noise_scale,
            'exploration_temperature': self.exploration_temperature,
            'rng': self.rng.__getstate__(),
        }

    def choose_action(self, game: SearchableEnv) \
            -> Tuple[int, Dict[str, Any]]:
        """Choose next move.
        can raise SearchTreeFull
        :param game: Current game environment
        :returns: move - chosen move
                  info - auxiliary information
        """
        assert not game.state.result

        temperature = 0.0
        noise_scale = 0.0
        if self.settings['move_sampling']:
            temperature = self.exploration_temperature
            if self.settings['move_exploration']:
                noise_scale = self.exploration_noise_scale
        if self.ply >= self.exploration_depth:
            temperature = 0.

        probs, value, metrics = self.tree.search(
            game, self.net,
            temperature=temperature,
            exploration_noise_scale=noise_scale,
            num_simulations=self.simulations,
            batch_size=self.search_batch_size,
            exploration_coef=self.exploration_coef,
            exploration_noise_alpha=self.exploration_noise_alpha,
            rng=self.rng)
        move_id = np.argmax(self.rng.multinomial(1, probs))
        move = game.state.legal_moves[move_id]
        info = dict(prob=probs[move_id],
                    value=value,
                    moves=game.state.legal_moves,
                    moves_prob=probs,
                    move_id=move_id,
                    metrics=metrics)
        return move, info

    def execute_action(self, move: int, legal_moves: np.ndarray) -> None:
        """Update search tree with own or opponent action.
        can raise SearchTreeFull
        """
        move_id = legal_moves.tolist().index(move)
        self.tree.move(move_id)
        self.ply += 1

    def tree_metrics(self):
        return self.tree.metrics()

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> 'Policy':
        """Create policy and load weights from checkpoint
        Paths can be local filenames or s3://... URL's (please install
        smart_open library for S3 support).
        Loads tensors according to device
        :param path: Either local or S3 path of policy file
        """
        policy = cls()
        if device:
            device = torch.device(device)
            location = device.type
            if location == 'cuda':
                location += f':{device.index or 0}'
        else:
            location = None
        if path.startswith('s3://'):
            # smart_open is optional dependency
            import smart_open
            with smart_open.smart_open(path) as f:
                state = torch.load(f, map_location=location)
        else:
            state = torch.load(path, map_location=location)
        policy.load_state_dict(state['policy'])
        policy.net.eval()
        if device:
            policy.net.to(device)
        return policy
