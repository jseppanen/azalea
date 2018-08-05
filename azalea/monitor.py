
from functools import partial

from tensorboardX import SummaryWriter


class Monitor:
    def __init__(self):
        self._writer = None
        self.step = 0

    def init(self, logdir, log_steps=1):
        self._writer = SummaryWriter(logdir)
        self._log_steps = log_steps

    def close(self):
        self._writer.close()

    def __getattr__(self, name):
        if name in ['init', 'step', 'close']:
            raise AttributeError()
        if self.step % self._log_steps == 0:
            fun = getattr(self._writer, name)
            return partial(fun, global_step=self.step)
        else:
            return nop


def nop(*args, **kwargs):
    pass


# singleton
monitor = Monitor()
