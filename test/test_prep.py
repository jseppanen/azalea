
import numpy as np
from numpy.testing import assert_array_equal

from azalea import prep


def test_pad_1d_int():
    seq = np.arange(10, dtype=int).tolist()
    bat = prep.pad(seq)
    assert bat.dtype == np.int64
    assert bat.shape == (10,)
    assert_array_equal(bat, seq)


def test_pad_1d_float():
    seq = np.arange(10, dtype=float).tolist()
    bat = prep.pad(seq)
    assert bat.dtype == np.float64
    assert bat.shape == (10,)
    assert_array_equal(bat, seq)


def test_pad_2d():
    seq = [np.arange(i) for i in range(2, 6)]
    bat = prep.pad(seq)
    assert bat.dtype == np.int64
    assert bat.shape == (4, 5)
    assert_padded_equal(bat, seq)


def test_pad_2d_size():
    seq = [np.arange(i) for i in range(2, 6)]
    bat = prep.pad(seq, size=np.array([4, 7]))
    assert bat.dtype == np.int64
    assert bat.shape == (4, 7)
    assert_padded_equal(bat, seq)


def test_pad_3d():
    seq = [np.arange(3 * i).reshape((3, i))
           for i in range(2, 6)]
    bat = prep.pad(seq)
    assert bat.dtype == np.int64
    assert bat.shape == (4, 3, 5)
    assert_padded_equal(bat, seq)


def assert_padded_equal(padded, seq):
    for i, row in enumerate(seq):
       m = len(row)
       assert_array_equal(padded[i, :m], row)
       assert_array_equal(padded[i, m:], 0)
