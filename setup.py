#!/usr/bin/env python

import subprocess as sp
from contextlib import contextmanager

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


@contextmanager
def set_version():
    # get __version__
    exec(open('azalea/version.py').read())
    version = locals()['__version__']

    if 'dev' in version and '+' not in version:
        # append git hash
        proc = sp.run(['git', 'rev-parse', 'HEAD'],
                      stdout=sp.PIPE, stderr=sp.DEVNULL)
        if proc.stdout:
            git_rev = proc.stdout.decode().strip()
            version = f'{version}+{git_rev[:7]}'
            with open('azalea/version.py', 'w') as f:
                f.write(f"__version__ = '{version}'\n")

    try:
        yield version
    finally:
        # restore default version.py
        sp.run(['git', 'checkout', '--', 'azalea/version.py'],
               stderr=sp.DEVNULL)


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

with set_version() as version:
    setup(
        name='azalea',
        version=version,
        license='Apache-2.0',
        description='Hex board game AI with self-play learning based on the AlphaZero algorithm',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author='Jarno Seppänen',
        author_email='azalea@meit.si',
        url='https://github.com/jseppanen/azalea',
        packages=[
            'azalea',
            'azalea.game',
            'azalea.typing',
        ],
        install_requires=[
            'numba',
            'numpy',
            'torch',
            'scipy',
            'pyyaml',
            'click',
            'tensorboardX',
        ],
        entry_points={
            'console_scripts': [
                'azalea-compare = azalea.compare_cli:main',
                'azalea-play = azalea.play_cli:main',
            ]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: OS Independent",
        ],
    )
