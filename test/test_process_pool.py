
import random
import time

from azalea.process_pool import ProcessPool


def test_process_pool():
    args = [(i,) for i in range(100)]
    pool = ProcessPool(num_workers=10)
    res1 = list(pool.map_unordered_lowlat(work, args))
    res2 = list(pool.map_unordered_lowlat(work, args))
    assert sorted(res1) == sorted(res2)


def work(n):
    time.sleep(.001 * random.random())
    return n
