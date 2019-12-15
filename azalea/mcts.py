
from typing import Dict, Sequence, Set, Tuple

import numpy as np
import torch
from numpy.random import RandomState

from . import prep
from .network import Network
from .search_tree import SearchNode, SearchTree, SearchStats
from .search_tree import ForwardSearchIterator
from .search_tree import print_tree
from .typing import GameState, SearchableEnv

DEBUG = False


def evaluate_root(tree: SearchTree,
                  game: SearchableEnv,
                  net: Network,
                  rng: RandomState) -> None:
    """Evaluate root node.
    """
    check_game_state(game.state)
    value, _, prior_prob = evaluate_batch(game, net, [game.state], rng)
    tree.root.create_child_nodes(len(prior_prob[0]), prior_prob[0])
    if DEBUG: print('root value:', value[0])


def check_game_state(state):
    assert isinstance(state.board, np.ndarray), 'board type error'
    assert state.board.dtype == np.int32, 'board type error'
    assert len(state.board.shape) == 2, 'board type error'
    assert isinstance(state.legal_moves, np.ndarray), 'moves type error'
    assert state.legal_moves.dtype == np.int32, 'moves type error'
    assert len(state.legal_moves.shape) == 1, 'moves type error'
    assert all(state.legal_moves > 0), 'moves value error'
    assert isinstance(state.color, int), 'color type error'
    assert isinstance(state.result, int), 'result type error'
    assert len(state.legal_moves) or state.result, \
        'no legal moves but game is continuing'
    assert not (len(state.legal_moves) and state.result), \
        'legal moves but game ended'


def select_batch(tree: SearchTree, game: SearchableEnv,
                 batch_size: int, exploration_coef: float,
                 noise_scale: float, noise_alpha: float,
                 rng: RandomState) \
        -> Tuple[Sequence[SearchNode], Sequence[GameState]]:
    """Select batch of leaf nodes for expansion
    Replay moves in game and collect resulting game states.
    :param batch_size: how many positions to select
    :param exploration_coef: exploration coefficient c_puct
    :param noise_scale: Dirichlet exploration noise scale
    :param noise_alpha: Dirichlet exploration noise concentration
    Return leaf board states
    """
    leaf_nodes = []   # leaf nodes in each path
    leaf_states = []  # game states in each path

    for i in range(batch_size):
        with tree.search_forward(game) as root_iter:
            leaf_node, leaf_state = select_leaf(
                root_iter, exploration_coef, noise_scale, noise_alpha, rng)
            # add virtual loss to enforce search diversity:
            # assume selected path leads to loss
            apply_virtual_loss([leaf_node], 1)
            leaf_states.append(leaf_state)
            leaf_nodes.append(leaf_node)
    # undo above virtual losses
    apply_virtual_loss(leaf_nodes, -1)

    # remove duplicate leaves
    leaf_nodes, leaf_states = deduplicate_leaves(leaf_nodes, leaf_states)
    return leaf_nodes, leaf_states


def apply_virtual_loss(leaf_nodes: Sequence[SearchNode], amount: int) -> None:
    """Edit search tree in place to add virtual loss.
    Add virtual losses to both players on all paths from root to given leafs.
    :param amount: +1 to apply virtual loss, -1 to undo it
    """
    for node in leaf_nodes:
        while not node.is_root():
            node.num_visits += amount
            # we want the modification to look like a loss from the
            # perspective of each node's move_stats, and therefore, the values
            # should look like wins from each node's own perspective
            # (assuming a two-player game here)
            node.total_value += amount
            node = node.parent


def select_leaf(search_iter: ForwardSearchIterator,
                exploration_coef: float,
                noise_scale: float,
                noise_alpha: float,
                rng: RandomState) -> Tuple[SearchNode, GameState]:
    """Sample path of moves from current (root) position.
    Walk one path and return final leaf node, which is either:
    1. unevaluated: game continues but number of children is not known
    2. terminal: game terminates and node has zero children
    """
    while not search_iter.node.is_leaf():
        scores = score_actions(search_iter.node.move_stats,
                               exploration_coef=exploration_coef,
                               noise_scale=noise_scale,
                               noise_alpha=noise_alpha,
                               rng=rng)
        assert len(scores) == len(search_iter.state.legal_moves)
        child_id = np.argmax(scores)
        search_iter.step(child_id)
        noise_scale = 0.0  # only explore at root
    # access full state of new leaf
    return search_iter.node, search_iter.state


def score_actions(stats: SearchStats,
                  exploration_coef: float,
                  noise_scale: float,
                  noise_alpha: float,
                  rng: RandomState) -> np.ndarray:
    """Score actions (child nodes) according to UCT heuristic (AZ flavor)"""
    prior = stats.prior_prob  # P(s, a)
    if noise_scale:
        # explore random actions at root node
        noise = rng.dirichlet(np.full(len(prior), noise_alpha))
        prior = (
            (1.0 - noise_scale) * prior + noise_scale * noise
        ).astype(np.float32)
    visit_gap = np.sqrt(np.sum(stats.num_visits)) / (1.0 + stats.num_visits)
    action_uct_boost = exploration_coef * prior * visit_gap  # U(s, a)
    action_value = stats.total_value / stats.num_visits.clip(min=1)  # Q(s, a)
    score = action_value + action_uct_boost  # Q(s, a) + U(s, a)
    return score


def deduplicate_leaves(leaf_nodes: Sequence[SearchNode],
                       leaf_states: Sequence[GameState]) \
        -> Tuple[Sequence[SearchNode], Sequence[GameState]]:
    """Remove duplicate leafs"""
    seen: Set[int] = set()

    dedup_nodes = []
    dedup_states = []
    for node, state in zip(leaf_nodes, leaf_states):
        if node.id not in seen:
            dedup_nodes.append(node)
            dedup_states.append(state)
            seen.add(node.id)
    return dedup_nodes, dedup_states


def evaluate_batch(game: SearchableEnv,
                   net: Network,
                   states: Sequence[GameState],
                   rng: RandomState) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate batch of board positions (new tree nodes)
    Run network to evaluate value and move probabilities of
    each board position, from the perspective of each player in turn.
    :returns tuple of arrays:
        value - game values (of active player) for each position,
        num_children - number of moves (nodes) for each position,
        prior_prob - move priors for each position
    """
    batch = prep.batch_games(states)

    # FIXME necessary?
    batch['board'] = batch['board'].copy()
    batch['legal_moves'] = batch['legal_moves'].copy()

    # normalize boards and moves for the network so that it can always
    # evaluate positions from the same perspective (first player/color=0)
    # note that the resulting values and move priors are still in the
    # active player's perspective
    flip_mask = batch['color'] == 1
    batch['board'][flip_mask], batch['legal_moves'][flip_mask] = \
        game.flip_player_board_moves(batch['board'][flip_mask],
                                     batch['legal_moves'][flip_mask])

    # data augmentation by random input variations
    batch['board'], batch['legal_moves'] = \
        game.random_reflect(batch['board'], batch['legal_moves'],
                            rng=rng)

    value = np.zeros_like(batch['color'], dtype=np.float32)
    prior_prob = np.zeros_like(batch['legal_moves'], dtype=np.float32)
    num_children = (batch['legal_moves'] > 0).sum(1)

    # for each terminal game position, value is -1 from the active
    # (losing) player perspective
    term_mask = batch['result'] != 0
    value[term_mask] = -1.0

    # assert that terminal results are always losses
    own_result = batch['result'].copy()
    own_result[flip_mask] = flip_result(own_result[flip_mask])
    assert all(own_result[term_mask] == 1)

    if any(~term_mask):
        tbatch = {}
        for k in batch:
            tbatch[k] = torch.tensor(batch[k][~term_mask])
            if net.device.type == 'cuda':
                tbatch[k] = tbatch[k].pin_memory().to(net.device)
        output = net.run(tbatch)
        nonterm_value = output['value'].cpu().numpy()
        nonterm_prior_prob = np.exp(output['moves_logprob'].cpu().numpy())
        assert all(nonterm_prior_prob.flat >= 0.0), 'negative prior prob'
        assert all(abs(nonterm_prior_prob.sum(1) - 1.0).flat < 1e-4), \
            'prior probs normalized incorrectly'
        value[~term_mask] = nonterm_value
        prior_prob[~term_mask] = nonterm_prior_prob

    return value, num_children, prior_prob


def flip_result(result):
    #njit fails at all()
    #assert all((result >= 0) & (result <= 3)), 'result out of range'
    return (result > 0) * (4 - result)


def expand_batch(leaf_nodes: Sequence[SearchNode],
                 num_children: np.ndarray,
                 prior_probs: np.ndarray) -> None:
    """Initialize MCTS statistics for batch of new leaf nodes.
    :param leaf_nodes: List of leaf nodes to expand
    :param num_children: number of child nodes for each leaf, 1d array
    :param prior_probs: 2d array of prior probability vectors for
                        each node
    """
    for i, node in enumerate(leaf_nodes):
        assert node.is_leaf(), 'expanded node is not leaf'
        if not node.is_terminal():
            m = num_children[i]
            node.create_child_nodes(m, prior_probs[i, :m])


def backup_batch(leaf_nodes: Sequence[SearchNode],
                 values: np.ndarray) -> None:
    """Update tree MCTS statistics with batch of evaluation results.
    :param values: Game value from leaf node player's perspective
    """
    for node, value in zip(leaf_nodes, values):
        while True:
            node.total_value += value
            node.num_visits += 1
            # flip value to opponent's perspective
            value = -value
            if node.is_root():
                break
            node = node.parent


def sample_paths(tree: SearchTree, game: SearchableEnv, net: Network,
                 num_simulations: int,
                 batch_size: int,
                 exploration_coef: float,
                 exploration_noise_scale: float,
                 exploration_noise_alpha: float,
                 rng: RandomState) -> Dict[str, float]:
    """Sample paths and update tree statistics in place.
    :returns: Search tree metrics
    """
    num_batches = num_simulations // batch_size + 1
    search_depth = 0
    search_value = 0

    if not tree.root.is_evaluated():
        evaluate_root(tree, game, net, rng)

    for i in range(num_batches):
        if DEBUG:
            print('sample paths: batch %d' % i)
            print_tree(tree)
        leaf_nodes, leaf_states = select_batch(
            tree, game, batch_size, exploration_coef,
            exploration_noise_scale, exploration_noise_alpha, rng)
        values, num_children, prior_prob = evaluate_batch(
            game, net, leaf_states, rng)
        expand_batch(leaf_nodes, num_children, prior_prob)
        backup_batch(leaf_nodes, values)
        #search_depth += np.sum(path_lens)
        search_value += np.sum(values)

    metrics = {
        #'search_depth': search_depth / (num_batches * batch_size),
        'search_value': search_value / (num_batches * batch_size)
    }
    return metrics
