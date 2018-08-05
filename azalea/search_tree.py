
import numpy as np
import torch
from numba import jit, njit, jitclass
from numba import int32, float32, uint32

from .search import get_game_states, normalize_position
from .fnv1a import fnv1a

MAX_NODES = 350000    # 300 moves x 1000 evaluations
MAX_EDGES = 10000000  # 30 legal moves per turn


class SearchTreeFull(BaseException):
    pass


search_tree_spec = [
    ('num_nodes', int32),
    ('num_edges', int32),
    ('first_child', int32[:]),
    ('num_children', int32[:]),
    ('moves_checksum', uint32[:]),
    ('children', int32[:]),
    ('num_visits', float32[:]),
    ('total_value', float32[:]),
    ('prior_prob', float32[:]),
]


@jitclass(search_tree_spec)
class SearchTreeData(object):
    def __init__(self):
        self.num_nodes = 0
        self.num_edges = 0
        self.first_child = np.zeros(MAX_NODES, dtype=np.int32)
        self.num_children = np.zeros(MAX_NODES, dtype=np.int32)
        self.moves_checksum = np.zeros(MAX_NODES, dtype=np.uint32)
        self.children = np.zeros(MAX_EDGES, dtype=np.int32)
        self.num_visits = np.zeros(MAX_EDGES, dtype=np.float32)
        self.total_value = np.zeros(MAX_EDGES, dtype=np.float32)
        self.prior_prob = np.zeros(MAX_EDGES, dtype=np.float32)


@jit
def reset(tree, net, device, board, moves):
    """Reset search tree at the start of a new game"""
    junk, num_children, prior_prob = evaluate(
        tree,
        net,
        device,
        np.expand_dims(board, 0),
        np.expand_dims(moves, 0),
        np.zeros(1, dtype=np.int32))
    expand(
        tree,
        np.zeros(1, dtype=np.int32),
        num_children,
        prior_prob,
        np.array([fnv1a(moves)], dtype=np.uint32))


@njit
def select(tree, root_id, batch_size, exploration_coef,
           noise_scale, noise_alpha):
    """Select batch of leaf edges for expansion
    :param root_id: search start position
    :param batch_size: how many positions to select
    :param exploration_coef: exploration coefficient c_puct
    :param noise_scale: Dirichlet exploration noise scale
    :param noise_alpha: Dirichlet exploration noise concentration
    """
    assert root_id >= 0 and root_id < tree.num_nodes, 'invalid root node'
    assert tree.num_children[root_id], 'terminal node'
    path_child_ids = []  # index of each traversed edge/child
    path_checksums = []  # checksum of moves of each traversed edge
    path_lens = []
    path_leafs = []  # leaf edges sampled
    for i in range(batch_size):
        node_id = root_id
        edge_id = -1
        path_len = 0
        root = True
        while (root or node_id) and tree.num_children[node_id]:
            e = tree.first_child[node_id]
            f = e + tree.num_children[node_id]
            t = np.sqrt(np.sum(tree.num_visits[e:f])) \
                / (1. + tree.num_visits[e:f])
            p = tree.prior_prob[e:f]
            if root and noise_scale:
                # explore random actions at root node
                # numba doesn't support np.random.dirichlet
                noise = np.random.gamma(noise_alpha, 1.0, len(p)) \
                    .astype(np.float32)
                noise = noise / np.sum(noise)
                p = np.float32(1. - noise_scale) * p \
                    + np.float32(noise_scale) * noise
            u = exploration_coef * p * t
            q = tree.total_value[e:f] / np.maximum(1., tree.num_visits[e:f])
            s = q + u
            child_id = np.argmax(s)
            edge_id = e + child_id
            # virtual loss: assume selected path leads to loss
            # assumption is reverted in backup(.)
            tree.num_visits[edge_id] += 1
            tree.total_value[edge_id] -= 1
            path_child_ids.append(child_id)
            path_checksums.append(tree.moves_checksum[node_id])
            node_id = tree.children[edge_id]
            path_len += 1
            root = False
        # sampled new path
        assert path_len, 'no path found'
        assert edge_id != -1, 'no leaf found'
        path_lens.append(path_len)
        path_leafs.append(edge_id)
        #print('select', i, 'root', root_id, 'leaf', edge_id, 'path', *path_child_ids[-path_len:])
    path_child_ids2 = np.array(path_child_ids, dtype=np.int32)
    path_checksums2 = np.array(path_checksums, dtype=np.uint32)
    path_lens2 = np.array(path_lens, dtype=np.int32)
    path_leafs2 = np.array(path_leafs, dtype=np.int32)
    return path_child_ids2, path_lens2, path_leafs2, path_checksums2


@jit
def evaluate(tree, net, device, boards, moves, results):
    """Evaluate batch of new child nodes
    Boards, moves, and results always from same player orientation
    """
    # evaluate batch
    value = np.zeros(results.shape, dtype=np.float32)
    prior_prob = np.zeros(moves.shape, dtype=np.float32)
    if any(results == 0):
        batch = {
            'board': torch.tensor(boards[results == 0]),
            'moves': torch.tensor(moves[results == 0], dtype=torch.int64)
        }
        if device.type == 'cuda':
            batch['board'] = batch['board'].pin_memory().to(device)
            batch['moves'] = batch['moves'].pin_memory().to(device)
        # numba doesn't like torch.no_grad() context manager
        prev_grad_enable = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        output = net.run(batch)
        torch.set_grad_enabled(prev_grad_enable)
        value[results == 0] = output['value'].cpu().numpy()
        prior_prob[results == 0] = np.exp(output['moves_logprob']
                                          .cpu().numpy())
    value[results != 0] = results[results != 0] - 2.
    num_children = (moves > 0).sum(1)
    return value, num_children, prior_prob


@njit
def expand(tree, path_leafs, num_children, prior_prob, moves_checksum):
    """Create nodes for batch of new child nodes"""
    # expand tree
    for i in range(len(path_leafs)):
        edge = path_leafs[i]
        n = num_children[i]
        # unpad prior_prob
        tree.children[edge] = add_node(tree, n, prior_prob[i, :n],
                                       moves_checksum[i])


@njit
def backup(tree, root_id, path_child_ids, path_lens, values):
    """Update tree with batch of evaluation results"""
    assert root_id >= 0 and root_id < tree.num_nodes, 'invalid root node'
    assert tree.num_children[root_id], 'terminal node'
    p = 0
    for i in range(len(path_lens)):
        pl = path_lens[i]
        v = values[i]
        # value from root's perspective
        v *= 1 - 2 * (pl % 2)
        node_id = root_id
        for j in range(p, p + pl):
            child_id = path_child_ids[j]
            edge_id = tree.first_child[node_id] + child_id
            # num_visits already incremented at select (virtual loss)
            tree.total_value[edge_id] += (1 + v)
            node_id = tree.children[edge_id]
            v = -v
        assert v == values[i], 'backup: wrong value'
        p += pl
    assert p == len(path_child_ids), 'backup: wrong path length'


@njit
def add_node(tree, num_children, prior_prob, moves_checksum):
    """
    :returns: New node id
    """
    node_id = tree.num_nodes
    first_child = tree.num_edges
    end_child = first_child + num_children
    if tree.num_nodes == MAX_NODES:
        raise SearchTreeFull('too many nodes')
    if tree.num_edges == MAX_EDGES:
        raise SearchTreeFull('too many edges')
    tree.num_nodes += 1
    assert tree.num_nodes <= MAX_NODES, 'tree nodes buffer overflow'
    tree.num_edges += num_children
    assert tree.num_edges <= MAX_EDGES, 'tree edges buffer overflow'
    tree.first_child[node_id] = first_child
    tree.num_children[node_id] = num_children
    tree.moves_checksum[node_id] = moves_checksum
    tree.children[first_child:end_child] = 0
    tree.num_visits[first_child:end_child] = 0
    tree.total_value[first_child:end_child] = 0
    tree.prior_prob[first_child:end_child] = prior_prob
    #print('add node:', node_id, num_children, hex(moves_checksum))
    return node_id


@jit
def sample_paths(tree, node_id, utils, game, net, device,
                 num_simulations, batch_size, exploration_coef,
                 exploration_noise_scale, exploration_noise_alpha):
    """Sample paths and update tree statistics in place.
    """
    num_batches = num_simulations // batch_size + 1
    search_depth = 0
    search_value = 0
    for i in range(num_batches):
        #print('sample paths: batch %d' % i)
        child_ids, lens, leafs, leaf_checksums = select(
            tree, node_id, batch_size, exploration_coef,
            exploration_noise_scale, exploration_noise_alpha)
        boards, results, moves, checksums = get_game_states(
            utils, game, child_ids, lens, leaf_checksums)
        values, num_children, prior_prob = evaluate(
            tree, net, device, boards, moves, results)
        expand(tree, leafs, num_children, prior_prob, checksums)
        backup(tree, node_id, child_ids, lens, values)
        search_depth += np.sum(lens)
        search_value += np.sum(values)
    metrics = {
        'search_depth': search_depth / (num_batches * batch_size),
        'search_value': search_value / (num_batches * batch_size)
    }
    return metrics


class SearchTree(object):
    def __init__(self, utils, net, device,
                 num_simulations, batch_size, exploration_coef,
                 exploration_noise_alpha):
        """Initialize parallel search tree from starting position.
        :param exploration_coef: Exploration coefficient c_puct
        :param exploration_noise_alpha: Exploration noise concentration
        """
        self.tree = SearchTreeData()
        self.utils = utils
        self.net = net
        self.device = device
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.exploration_coef = exploration_coef
        self.exploration_noise_alpha = exploration_noise_alpha

    def reset(self, game):
        """Start new game.
        :param game: Initial game state
        """
        board = game.board()
        moves = game.legal_moves()
        assert not game.result(), 'game is terminal'
        assert not game.color(), 'expecting white turn'
        self.tree = SearchTreeData()
        reset(self.tree, self.net, self.device, board.ravel(), moves)

    @property
    def root(self):
        """Get tree root.
        :returns: Root node id
        """
        assert self.tree.num_nodes > 0, 'tree not initialized'
        return 0

    def get_child(self, node_id, child_id, moves):
        """Get node child.
        :returns: Child node id or 0 if leaf node
        """
        assert node_id < self.tree.num_nodes, 'illegal node'
        # check moves are same as during search
        assert self.tree.moves_checksum[node_id] == fnv1a(moves), 'checksum error'
        return self.tree.children[self.tree.first_child[node_id] + child_id]

    def search(self, game, node_id, temperature, noise_scale):
        """Search for next moves.
        :param game: Current game state
        :param node_id: Node representing current game state
        :param temperature: Exploration temperature
        :param noise_scale: Exploration noise scale
        :return: Next move probabilities
        """
        metrics = sample_paths(self.tree, node_id,
                               self.utils, game, self.net, self.device,
                               self.num_simulations, self.batch_size,
                               self.exploration_coef,
                               noise_scale,
                               self.exploration_noise_alpha)
        # play
        e = self.tree.first_child[node_id]
        f = e + self.tree.num_children[node_id]
        v = self.tree.num_visits[e:f]
        log_pi = np.log(v.clip(min=1))
        log_pi[v == 0] = -999
        if temperature:
            log_pi = log_pi / temperature
        else:
            log_pi[log_pi < log_pi.max()] = -999
        # avoid numerical issues
        log_pi = log_pi.astype(np.float64)
        log_z = np.logaddexp.reduce(log_pi)
        pi = np.exp(log_pi - log_z)
        metrics['search_root_width'] = np.sum(v > 0)
        metrics['search_root_visits'] = np.mean(v)
        metrics['search_root_children'] = len(v)
        return pi, metrics

    def move(self, game, node_id, action, moves):
        """Make move without searching.
        :param game: Game state **after** move has been made
        :param action: Action id of move that was made
        :param moves: Legal moves **before** move was made
        """
        child = self.get_child(node_id, action, moves)
        if child:
            return child
        board_after, moves_after, result_after = \
            normalize_position(self.utils, game)
        checksum = fnv1a(game.legal_moves())
        value, num_children, prior_prob = evaluate(
            self.tree,
            self.net,
            self.device,
            np.expand_dims(board_after.ravel(), 0),
            np.expand_dims(moves_after, 0),
            np.array([result_after], dtype=np.int32))
        edge_id = self.tree.first_child[node_id] + action
        expand(
            self.tree,
            np.array([edge_id], dtype=np.int32),
            num_children,
            prior_prob,
            np.array([checksum], dtype=np.uint32))
        # no need to backup because node will be new root
        child = self.tree.children[edge_id]
        assert child, 'wrong node'
        return child

    def stats(self):
        return {
            'search_tree_nodes': self.tree.num_nodes,
            'search_tree_edges': self.tree.num_edges
        }
