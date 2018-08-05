
import os

from torch import multiprocessing as mp


class ParallelRunner:
    def __init__(self, fun, num_workers=None):
        self.fun = fun
        if num_workers is None:
            num_workers = os.cpu_count()
        if num_workers > 0:
            # CUDA requires spawn mode
            self.pool = mp.get_context('spawn').Pool(num_workers)
        else:
            self.pool = None
        self.tasks = []
        self.max_tasks = num_workers or 1

    def empty(self):
        return not self.tasks

    def full(self):
        return len(self.tasks) == self.max_tasks

    def submit(self, *args, **kwargs):
        assert not self.full(), 'too many tasks submitted'
        if self.pool is None:
            res = self.fun(*args, **kwargs)
        else:
            res = self.pool.apply_async(self.fun, args, kwargs)
        self.tasks.append(res)

    def result(self):
        assert not self.empty(), 'no tasks submitted'
        res = self.tasks.pop(0)
        if self.pool is None:
            return res
        else:
            return res.get()

    def flush(self):
        self.tasks = []

    def close(self):
        if self.pool is None:
            return
        self.pool.close()
        self.pool.join()
