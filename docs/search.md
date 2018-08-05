
# Search tree representation for jit

Arrays of nodes and edges.

# Nodes

Tree nodes can be in three states:
1. evaluated and non-terminal
2. unevaluated, pending evaluation (leaf)
3. terminal (leaf)

Each tree search simulation expands new leaf nodes and assigns them
into unevaluated state, until they have been evaluated.

Unevaluated nodes don't actually exist (node_id is zero in parent's
`children` array) until they have been evaluated.

# Node arrays

Indexed with node_id:

* num_children: number of edges by node id (zero for terminal nodes)
* first_child: offset of node's first edge in edge arrays

# Edge arrays

Indexed with edge_id:

* children: array of child node ids (zero for unevaluated nodes)
* num_visits, total_value, prior_prob: edge MCTS arrays
