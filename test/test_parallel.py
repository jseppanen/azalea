
from azalea.parallel import ParallelRunner


def test_parallel():
    # check determinism
    runner = ParallelRunner(work, num_workers=5)
    assert runner.empty()
    assert not runner.full()
    i = 0
    while not runner.full():
        runner.submit(i)
        i += 1
    assert not runner.empty()
    assert runner.full()
    j = 0
    while not runner.empty():
        assert j == runner.result()
        j += 1
    assert i == j
    assert runner.empty()
    assert not runner.full()
    runner.close()


def work(n):
    import random
    import time
    time.sleep(.3 * random.random())
    return n
