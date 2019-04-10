
import dataclasses as dc
from typing import Any, Dict, Optional, Sequence, Union, overload

import numpy as np
import torch

Dataclass = Any
ScalarOrArray = Union[float, np.ndarray]


def batch(seq: Sequence[Dataclass]) -> Dict[str, np.ndarray]:
    """Batch sequence of dataclass objects into arrays
    :param states: Sequence of dataclasses with scalar or array members
    :returns: Dict of arrays
    """
    df = transpose_dataclass(seq)
    return {name: pad(df[name]) for name in df}


def torch_batch(seq: Sequence[Dataclass]) -> Dict[str, torch.Tensor]:
    nb = batch(seq)
    tb = {k: torch.from_numpy(nb[k]) for k in nb}
    return tb


def transpose_dataclass(seq: Sequence[Dataclass]) \
        -> Dict[str, Sequence[ScalarOrArray]]:
    """Transpose sequence of dataclass objects
    :param states: Sequence of dataclasses with scalar or array members
    :returns: Dict of lists
    """
    names = [f.name for f in dc.fields(seq[0])]
    values = zip(*map(dc.astuple, seq))
    # {'board': [...], 'legal_moves': [...], ...}
    return dict(zip(names, values))


@overload
def pad(mats: Sequence[float]) -> np.ndarray:
    """Cast list of scalars as tensor.
    """
    ...


@overload
def pad(mats: Sequence[np.ndarray],
        size: Optional[np.ndarray]) -> np.ndarray:
    """Pad n-dimensional matrices with zeros.
    """
    ...


#@jit
def pad(mats, size=None):
    """Pad n-dimensional matrices with zeros.
    """
    if isinstance(mats[0], (int, float)):
        assert size is None
        return np.array(mats)
    max_size = np.amax([m.shape for m in mats], 0)
    if size is None:
        size = max_size
    else:
        assert all(max_size <= size)
    padded_size = (len(mats),) + tuple(size)
    padded = np.zeros(padded_size, dtype=mats[0].dtype)
    for i, m in enumerate(mats):
        ids = [i] + [slice(s) for s in m.shape]
        padded[tuple(ids)] = m
    return padded
