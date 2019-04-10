
import logging

import numpy as np
import azalea as az
import torch


# random hyperparameter search for hex

# runtime configuration (no hyperparameters)
config = {
    'log_interval': 500,
    'model_checkpoint_interval': 10000,      # 4MB/30mins
    'replaybuf_checkpoint_interval': 50000,  # 160MB/150mins
    'game': 'hex11',
    'seed': 0xBAD5EED5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

config_variants = {
    'num_blocks': [3, 5, 7],
    'base_chans': [32, 64, 128],
    'replaybuf_size': 100000,
    'replaybuf_resample': 10,
    'lr_initial': lambda params: 10 ** np.random.uniform(-3, -.5),
    'lr_decay_steps': [100000, 300000, 1000000],
    'total_steps': 5000000,
    'lr_decay': 0.1,
    'momentum': [0.9, 0.95],
    'l2_regularization': 1e-4,
    'batch_size': [64, 128, 256],
    'exploration_coef': lambda params: np.exp(np.random.uniform(0, np.log(6))) - 1,
    'exploration_temperature': 1.,
    'exploration_depth': 15,
    'exploration_noise_alpha': 0.03,
    'exploration_noise_scale': 0.25,
    'simulations': 800,
    'search_batch_size': 10,
}

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    rundir = 'runs/tune-hex11'
    num_experiments = 100
    timeout = 12 * 3600
    ranking = az.tuning.random_search(rundir, config, config_variants,
                                      num_experiments, timeout)
    for rank, (score, path) in enumerate(ranking):
        print(f'{rank+1}. {score:.1f} {path}')
