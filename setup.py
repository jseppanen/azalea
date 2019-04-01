#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='azalea',
    version='0.1.0',
    license='GPL-3.0-only',
    description='Hex board game AI with self-play learning based on the AlphaZero algorithm',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Jarno Seppänen',
    author_email='azalea@meit.si',
    url='https://github.com/jseppanen/azalea',
    packages=[
        'azalea',
        'azalea.game',
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
            'azalea-play = azalea.play:main',
        ]
    },
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: OS Independent",
    ),
)
