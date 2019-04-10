
from contextlib import contextmanager
from dataclasses import dataclass, astuple
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from numpy import int32, float32
from numpy.random import RandomState

from .fnv1a import fnv1a
from .network import Network
from .typing import SearchableEnv, GameState

DEBUG = False

# 300 moves/game x 1000 evaluations/move x 30 next moves/eval
MAX_NODES = 10000000


class SearchTreeFull(Exception):
    pass


class SearchTree:
    """
    SearchTree stores the explored game tree and its MCTS statistics.
    Tree nodes are board positions and edges are moves of the game.
    Terminal leaf nodes have

        `num_children[node_id] = 0`.

    Unevaluated leaf nodes have

        `num_children[node_id] = -1`.

    * total_value is stored from current node's active player's perspective
      (opponent of parent node's player)
    * prior_prob is stored from parent node's player's perspective
      (opponent of current node's player)
    """

    def __init__(self):
        self.num_nodes = 0
        self.root_id: int32 = 0

        # tree edges
        self.parent = -np.ones(MAX_NODES, dtype=int32)
        self.first_child = -np.ones(MAX_NODES, dtype=int32)
        self.num_children = -np.ones(MAX_NODES, dtype=int32)

        # MCTS statistics
        self.num_visits = -np.ones(MAX_NODES, dtype=float32)
        self.total_value = -np.ones(MAX_NODES, dtype=float32)
        self.prior_prob = -np.ones(MAX_NODES, dtype=float32)

        self.reset()

    def reset(self):
        """Clear tree.
        """
        self.num_nodes = 1
        self.root_id = 0

        # create unevaluated root node
        self.parent[0] = -1
        self.first_child[0] = -1
        self.num_children[0] = -1
        self.num_visits[0] = 0
        self.total_value[0] = 0
        self.prior_prob[0] = 1.0

    def search(self, game: SearchableEnv, network: Network, *,
               num_simulations: int = 100,
               temperature: float = 1.0,
               exploration_coef: float = 1.0,
               exploration_noise_scale: float = 1.0,
               exploration_noise_alpha: float = 1.0,
               batch_size: int = 10,
               rng: Optional[RandomState] = None) \
            -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Plan for next moves with Monte Carlo tree search (AZ flavor).
        The search is deterministic by default, but you can provide seeded
        random number generators for stochastic results.
        :param temperature: Exploration temperature
        :param exporation_noise_scale: Exploration noise scale
        :param rng: Random number generator (by default seeded with 0)
        :return: Next move probabilities, game value, and debug metrics
        """
        # avoid circular dependency
        from . import mcts

        if rng is None:
            rng = RandomState(0)

        with torch.no_grad():
            network.eval()
            metrics = mcts.sample_paths(
                self, game, network,
                num_simulations, batch_size,
                exploration_coef,
                exploration_noise_scale,
                exploration_noise_alpha,
                rng)
        # play
        stats = self.root.move_stats
        move_probs = as_distribution(stats.num_visits, temperature)
        value = self.root.total_value / self.root.num_visits
        metrics['search_root_width'] = np.sum(stats.num_visits > 0)
        metrics['search_root_visits'] = np.mean(stats.num_visits)
        metrics['search_root_children'] = len(stats.num_visits)
        metrics['search_tree_nodes'] = self.num_nodes
        return move_probs, value, metrics

    def move(self, move_id: int) -> None:
        """Commit move and pick new root node.
        Set new tree root to one of current root's child nodes.
        Forget about ancestor and sibling nodes.
        :param move_id: Action id of move that was made
        """
        if not self.root.is_evaluated():
            # step to the unknown, nothing to remember
            self.reset()
            return

        assert self.root_id < self.num_nodes, 'root node is unevaluated'
        node = self.root.child(move_id)
        if not node.is_evaluated():
            # step to the unknown, nothing to remember
            self.reset()
        else:
            self.root_id = node.id

    @property
    def root(self) -> 'SearchNode':
        """Access root node.
        """
        return SearchNode(self, self.root_id)

    @contextmanager
    def search_forward(self, game: SearchableEnv) \
            -> Generator['ForwardSearchIterator', None, None]:
        """Snapshot game state and start search.
        :returns: Iterator for root node
        """
        assert self.root_id >= 0 and self.root_id < self.num_nodes, \
            'invalid root node'
        assert self.root.is_evaluated(), 'unevaluated root'
        assert not self.root.is_terminal(), 'terminal root'
        game.snapshot()
        try:
            yield ForwardSearchIterator(self.root, game)
        finally:
            game.restore()


class SearchNode:
    """Search tree node, incl. MCTS statistics."""

    def __init__(self, tree: SearchTree, node_id: int) -> None:
        assert node_id >= 0, 'illegal node'
        assert node_id < tree.num_nodes, 'illegal node'
        self.tree = tree
        self._id = node_id

    def __eq__(self, other: 'SearchNode') -> bool:
        return self.id == other.id

    @property
    def id(self) -> int32:
        """Node id (read only).
        """
        return self._id

    @property
    def parent(self) -> 'SearchNode':
        """Get parent node in search tree.
        """
        assert not self.is_root(), 'root has no parent'
        return SearchNode(self.tree, self.tree.parent[self.id])

    def child(self, move_id: int32) -> 'SearchNode':
        """Get N'th child node.
        :param move_id: Index of child node
        """
        assert self.is_evaluated(), 'unevaluated node'
        assert move_id >= 0, 'illegal child'
        assert move_id < self.num_children, 'illegal child'
        node_id = self.tree.first_child[self.id] + move_id
        return SearchNode(self.tree, node_id)

    @property
    def move_stats(self) -> 'SearchStats':
        """Get vectorized MCTS statistics for all moves from this node.
        :returns: MCTS stats for moves from current player's perspective
        """
        assert self.is_evaluated(), 'unevaluated nodes have no stats'
        first_child = self.tree.first_child[self.id]
        node_ids = slice(first_child, first_child + self.num_children)
        # priors are already stored from parent's perspective, but
        # values are from child nodes' own perspective, so need to be negated
        return SearchStats(self.tree.num_visits[node_ids],
                           -self.tree.total_value[node_ids],
                           self.tree.prior_prob[node_ids])

    @property
    def num_children(self) -> int32:
        return self.tree.num_children[self.id]

    @property
    def num_visits(self) -> float32:
        return self.tree.num_visits[self.id]

    @num_visits.setter
    def num_visits(self, visits: float32) -> None:
        self.tree.num_visits[self.id] = visits
    
    @property
    def total_value(self) -> float32:
        return self.tree.total_value[self.id]

    @total_value.setter
    def total_value(self, value: float32) -> None:
        self.tree.total_value[self.id] = value
    
    @property
    def prior_prob(self):
        return self.tree.prior_prob[self.id]

    def is_evaluated(self) -> bool:
        """Test for evaluated nodes.
        Unevaluated nodes are leaves, whose children and MCTS stats
        are undefined.
        """
        return self.num_children >= 0

    def is_terminal(self) -> bool:
        """Test for terminal nodes.
        Terminal nodes are leaves which have been evaluated and
        have no children.
        """
        return self.num_children == 0

    def is_root(self) -> bool:
        """Test for root node.
        """
        return self == self.tree.root

    def is_leaf(self) -> bool:
        """Test for leaf nodes.
        """
        return (not self.is_evaluated()) or self.is_terminal()

    def create_child_nodes(self, num_children: int,
                           prior_prob: np.ndarray) -> None:
        """Create new child nodes and initialize MCTS statistics.
        """
        if self.tree.num_nodes + num_children > MAX_NODES:
            raise SearchTreeFull('too many nodes')

        first_node_id = self.tree.num_nodes
        self.tree.num_nodes += num_children
        assert self.tree.num_nodes <= MAX_NODES, 'tree nodes buffer overflow'

        self.tree.first_child[self.id] = first_node_id
        self.tree.num_children[self.id] = num_children

        node_ids = slice(first_node_id, first_node_id + num_children)
        self.tree.parent[node_ids] = self.id
        self.tree.first_child[node_ids] = -1
        self.tree.num_children[node_ids] = -1  # mark as unevaluated
        self.tree.num_visits[node_ids] = 0
        self.tree.total_value[node_ids] = 0
        self.tree.prior_prob[node_ids] = prior_prob


@dataclass
class SearchStats:
    num_visits: np.ndarray  # dim=1, dtype=float32
    total_value: np.ndarray  # dim=1, dtype=float32
    prior_prob: np.ndarray  # dim=1, dtype=float32


class ForwardSearchIterator:
    """Combined game state and tree node when searching forward in game.
    Search forward in the game and track search tree node while searching.
    """

    def __init__(self, node: SearchNode, game: SearchableEnv) -> None:
        self._game = game
        self._node = node
        if DEBUG:
            print(f'search: node {self._node.id} '
                  f'nch {self._node.num_children} '
                  f'game {hash(str(astuple(self._game.state)))} '
                  f'nmv {len(self._game.state.legal_moves)}')

    def step(self, move_id: int) -> None:
        """Move one step forward in game and tree.
        :param move_id: Ordinal move id among legal moves in current node
        """
        state = self._game.state
        assert move_id < len(state.legal_moves), \
            f"move id {move_id} out of range"
        assert not state.result, 'game ended in select'
        move = state.legal_moves[move_id]
        self._game.step(move)
        self._node = self._node.child(move_id)
        if DEBUG:
            print(f'step {move}: node {self._node.id} '
                  f'nch {self._node.num_children} '
                  f'game {hash(str(astuple(self._game.state)))} '
                  f'nmv {len(self._game.state.legal_moves)}')
            #self._game.io.print_state(self._game.state)

    @property
    def state(self) -> GameState:
        """Retrieve game state for current search position."""
        return self._game.state

    @property
    def node(self) -> SearchNode:
        """Retrieve tree node for current search position."""
        return self._node


def as_distribution(counts: np.ndarray, temperature: float = 1.0) \
        -> np.ndarray:
    """Normalize count vector as a multinomial distribution.
    Apply temperature change if requested.
    :returns: Multinomial probability distribution
    """
    assert all(counts >= 0)
    log_pi = np.log(counts.clip(min=1))
    log_pi[counts == 0] = -np.inf
    if temperature:
        log_pi = log_pi / temperature
    else:
        log_pi[log_pi < log_pi.max()] = -np.inf
    # avoid numerical issues
    log_pi = log_pi.astype(np.float64)
    log_z = np.logaddexp.reduce(log_pi)
    pi = np.exp(log_pi - log_z)
    return pi


def print_tree(tree: SearchTree) -> None:
    """Pretty print search tree.
    0
    ├── 5 [0] move 393240 value -0.00/3 (0.43)
    │   ├── [22] move 458757 value 0.00/0 (1.00)
    │   └── [34] move 196636 value 0.00/0 (0.00)
    ├── 3 [1] move 393241 value 0.51/2 (0.00)
    │   ├── 7 [10] move 262149 value 0.69/1 (0.61)
    │   │   ├── [39] move 327682 value 0.00/0 (0.00)
    """
    for line in format_tree_lines(tree):
        print(line)


def format_tree_lines(tree: SearchTree) -> Generator[str, None, None]:
    yield format_node(tree.root)
    for line in format_subtree_lines(tree, tree.root, 1, []):
        yield line


def format_subtree_lines(tree: SearchTree, node: SearchNode, indent: int,
                         last: List[bool]) -> Generator[str, None, None]:
    indent += 1
    for i in range(node.num_children):
        child = node.child(i)
        last.append(i == node.num_children - 1)
        prefix = format_tree_prefix(last)
        suffix = format_node(child, move_id=i)
        yield prefix + suffix
        if not child.is_leaf():
            for line in format_subtree_lines(tree, child, indent, last):
                yield line
        last.pop()


def format_tree_prefix(last: List[bool]) -> str:
    if not last:
        return ''

    prefix = ''.join('    ' if x else '│   ' for x in last[:-1])
    prefix += '└── ' if last[-1] else '├── '
    return prefix


def format_node(node: SearchNode, move_id: Optional[int] = None) -> str:
    num_visits = node.num_visits
    value = node.total_value / max(1, num_visits)
    prior = node.prior_prob
    txt = f'{node.id} '
    if move_id is not None:
        txt += f'[{move_id}] '
    txt += f'value {value:.2f} vis {num_visits:.0f} (pri {prior:.2f})'
    return txt
