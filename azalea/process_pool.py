
import os
from multiprocessing.pool import AsyncResult
from typing import Callable, Generator, Iterator
from typing import List, Sequence, TypeVar

from torch import multiprocessing as mp


Arg = TypeVar('Arg')
Args = List[Arg]
Result = TypeVar('Result')


class ProcessPool:
    def __init__(self, *, num_workers=None, seed=None):
        if num_workers is None:
            num_workers = os.cpu_count()
        if num_workers > 0:
            # CUDA requires spawn mode
            self.pool = mp.get_context('spawn').Pool(num_workers)
            if seed is not None:
                seeds = [seed + i for i in range(num_workers)]
                self.pool.map(set_seed, seeds, 1)
        else:
            self.pool = None

        # have one extra task waiting at all times
        self.num_idle = num_workers + 1

    def map_unordered_lowlat(self, fun: Callable[..., Result],
                             seq: Sequence[Args]) \
            -> Generator[Result, None, None]:
        """Process sequence with given function.
        Like multiprocessing.imap_unordered but minimizing the latency
        between reading from input sequence and returning output
        item. Try to keep all workers busy but not caching any
        results if the reader is not keeping up.
        """
        seq_iter = iter(seq)
        pending: List[AsyncResult] = []

        self._submit_many(fun, seq_iter, pending)
        while pending:
            res: Result = wait_for_result(pending)
            self._submit_many(fun, seq_iter, pending)
            yield res

    @property
    def is_multiprocess(self) -> bool:
        return self.pool is not None

    def close(self) -> None:
        if not self.is_multiprocess:
            return
        self.pool.close()
        self.pool.join()

    def _submit_one(self, fun: Callable[..., Result], args: Args) \
            -> AsyncResult:
        """Submit one task to workers."""
        assert self.num_idle >= 1
        self.num_idle -= 1

        def done(ignored=None):
            self.num_idle += 1

        if not self.is_multiprocess:
            return apply_deferred(fun, args, done)
        else:
            return self.pool.apply_async(fun, args, {}, done, done)

    def _submit_many(self, fun: Callable[..., Result],
                     args_iter: Iterator[Args],
                     queue: List[AsyncResult]) -> None:
        """Submit as many tasks to workers as needed to maximize throughput."""
        while self.num_idle:
            try:
                args = next(args_iter)
            except StopIteration:
                return
            queue.append(self._submit_one(fun, args))


def apply_deferred(fun, args, done):
    class DeferredResult:
        def get(self, timeout):
            try:
                return fun(*args)
            finally:
                done()

        def ready(self):
            return True

    return DeferredResult()


def wait_for_result(results: List[AsyncResult]) -> Result:
    """Wait for first parallel result
    :param results: List of AsyncResult objects
    """
    while True:
        for i, f in enumerate(results):
            if f.ready():
                results.pop(i)
                return f.get(None)
        try:
            r = results[0].get(0.2)
            results.pop(0)
            return r
        except mp.TimeoutError:
            pass


def set_seed(seed):
    import numpy as np
    import torch
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
