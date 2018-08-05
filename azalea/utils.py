
import os
import atexit
import logging


def redirect_all_output(path):
    '''Redirect stdout and stderr to file.
    :param path: Path to log file
    '''
    logging.info('redirecting logs to %s', path)

    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)

    def close():
        os.close(fd)
    atexit.register(close)

    os.dup2(fd, 1)  # stdout
    os.dup2(fd, 2)  # stderr
