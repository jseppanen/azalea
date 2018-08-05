
import logging
import numpy as np
import torch

from . import network as networks
from .game import import_game
from .search_tree import SearchTree


def create_network(network_type, board_size, num_blocks, base_chans):
    Net = getattr(networks, network_type)
    return Net(board_size=board_size,
               num_blocks=num_blocks,
               base_chans=base_chans)


class Policy:
    def __init__(self, config):
        """Construct new policy"""
        self.gamelib = import_game(config['game'])
        self.device = torch.device(config['device'])
        if self.device.type == 'cuda':
            # enable cudnn auto-tuner
            torch.backends.cudnn.benchmark = True

    def initialize(self, config):
        """Initialize policy for training"""
        self.net = create_network(config['network'],
                                  config['board_size'],
                                  config['num_blocks'],
                                  config['base_chans'])
        self.net.to(self.device)
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

    @property
    def net(self):
        try:
            return self._net
        except AttributeError:
            raise RuntimeError('Policy must be initialized or loaded before use')

    @net.setter
    def net(self, net):
        self._net = net

    def reset(self, game):
        """Start new game
        :returns: Initial node
        """
        self.tree = SearchTree(self.gamelib,
                               self.net,
                               self.device,
                               self.simulations,
                               self.search_batch_size,
                               self.exploration_coef,
                               self.exploration_noise_alpha)
        self.tree.reset(game)
        return self.tree.root

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
        self.net.to(self.device)

        # load search params
        self.simulations = state['simulations']
        self.search_batch_size = state['search_batch_size']
        self.exploration_coef = state['exploration_coef']
        self.exploration_depth = state['exploration_depth']
        self.exploration_noise_alpha = state['exploration_noise_alpha']
        self.exploration_noise_scale = state['exploration_noise_scale']
        self.exploration_temperature = state['exploration_temperature']

    def state_dict(self):
        """Return model state
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
        }

    def choose_action(self, game, node, depth,
                      temperature=None,
                      noise_scale=None):
        """Choose next move.
        can raise SearchTreeFull
        :param game: Current game state
        :returns: action - action id,
                  node - child node following action,
                  probs - 1d array of action probabilities
        """
        assert not game.result()
        moves = game.legal_moves()
        if temperature is None:
            temperature = self.exploration_temperature
        if noise_scale is None:
            noise_scale = self.exploration_noise_scale
        if depth >= self.exploration_depth:
            temperature = 0.
        probs, metrics = self.tree.search(game, node, temperature, noise_scale)
        action = np.argmax(np.random.multinomial(1, probs))
        node = self.tree.get_child(node, action, moves)
        assert node
        return action, node, probs, metrics

    def make_action(self, game, node, action, moves):
        """Update search tree with opponent action.
        can raise SearchTreeFull
        """
        node = self.tree.move(game, node, action, moves)
        return node

    def tree_stats(self):
        return self.tree.stats()

    @classmethod
    def load(cls, config, path):
        """Create policy and load weights from checkpoint
        Paths can be local filenames or s3://... URL's (please install
        smart_open library for S3 support).
        Loads tensors according to config['device']
        :param path: Either local or S3 path of policy file
        """
        policy = cls(config)
        location = policy.device.type
        if location == 'cuda':
            location += f':{policy.device.index or 0}'
        if path.startswith('s3://'):
            # smart_open is optional dependency
            import smart_open
            with smart_open.smart_open(path) as f:
                state = torch.load(f, map_location=location)
        else:
            state = torch.load(path, map_location=location)
        policy.load_state_dict(state['policy'])
        policy.net.eval()
        return policy
