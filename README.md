
# Azalea

> playing to learn to play

Azalea is a reinterpretation of the [AlphaZero game AI](https://en.wikipedia.org/wiki/AlphaZero)
learning algorithm for the [Hex board game](https://en.wikipedia.org/wiki/Hex_(board_game)).

## Features

* Straightforward reimplementation of the AlphaZero algorithm except
  for MCTS parallelization (see below)
* Pre-trained model for Hex board game
* Fast MCTS implementation through Numba JIT acceleration.
* Fast Hex game move generation implementation through Numba.
* Parallelized self play to saturate Nvidia V100 GPU during training
* AI policy evaluation through round robin tournament, also parallelized
* Tested on Ubuntu 16.04
* Requires Python 3.6 and PyTorch 0.4

## Differences to published AlphaZero

* Single GPU implementation only - tested on Nvidia V100, with 8 CPU's
  for move generation and MCTS, and 1 GPU for the policy network.
* Only Hex game is implemented, though the code supports adding more
  games. Two components are needed for a new game: move generator and
  policy network, with board input and moves output adjusted to the
  new game.
* MCTS simulations are not run in parallel threads, but instead,
  self-play games are played in parallel processes. This is to avoid
  the need for a multi-threaded MCTS implementation while still
  maintaining fast training speed and saturating the GPU.
* MCTS simulation and board evaluations are batched according to
  `search_batch_size` config parameter. "Virtual loss" is used
  as in AlphaZero, to increase search diversity.

# Installation

Clone the repository and install dependencies with Conda:

    git clone https://github.com/jseppanen/azalea.git
    conda env create -n azalea
    source activate azalea

The default `environment.yml` installs GPU packages but you can choose
`environment-cpu.yml` for testing on a laptop.

## Playing against pretrained model

    python play.py models/hex11-20180712-3362.policy.pth

This will load the model and start playing, asking for your move. The
columns are labeled a–k and rows 1–11. The first player, playing `X`'s,
is trying to draw a vertical connected path through the board, while the
second player, with `O`'s, is drawing a horizontal path.

```
O O O O X . . . . . . 
 . . . . . . . . . . . 
  . . . . . . . . . . . 
   . . . . X . . . . . . 
    . . . . . X . . . . . 
     . . . . . . . . . . . 
      . . . . X . . . . . . 
       . . . . . . . . . . . 
        . . . X . . . . . . . 
  x      . . . . . . . . . . . 
 o\\      . . . . . . . . . . . 
last move: e1
Your move? 
```

## Model training

    python train.py --config config/hex11_train_config.yml --rundir runs/train

## Model comparison

    python compare.py --config config/hex11_eval_config.yml --rundir runs/compare <mode1> <model2> [model3] ...

## Model selection

    python tune.py

## References

* [Mastering the Game of Go without Human Knowledge](https://deepmind.com/documents/119/agz_unformatted_nature.pdf)
* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
