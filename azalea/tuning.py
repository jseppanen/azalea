
import logging
import time
import json
import random
from collections import defaultdict

import numpy as np

from .experiment import Experiment
from .evaluation import evaluate
from .policy import Policy
from .random_policy import RandomPolicy
from .ranking import compute_ranking, RankingError


def random_search(logdir, config, variants, total_experiments, timeout,
                  prev_results_path=None):
    """Tune hyperparameters by random search.
    :returns: list of (score, path) pairs in ranking order
    """
    checkpoints = ['<random>']
    outcomes = defaultdict(lambda: [0, 0, 0])
    scores = [0.0]
    if prev_results_path:
        with open(prev_results_path) as f:
            prev_results = json.load(f)
        checkpoints = prev_results['policies']
        outcomes.update((tuple(k), v)
                        for k, v in prev_results['outcomes'])
        num_outcomes = sum(x for p in outcomes for x in outcomes[p])
        logging.info(f'loaded {len(checkpoints)} checkpoints and {num_outcomes} outcomes from {prev_results_path}')

    for t in range(total_experiments):
        var_config = dict(config, **sample_variant_params(variants))
        path, step = train(f'{logdir}/expt-{t:03d}', var_config, timeout)
        checkpoints.append(path)
        scores = update_ranking(checkpoints, var_config, outcomes)
        best = ''
        if scores and scores[-1] >= max(scores):
            best = '*'
        logging.info(f'experiment {t} steps {step} score {scores and scores[-1]} {best}')
        with open(f'{logdir}/results.json', 'w') as f:
            res = {
                'outcomes': list(outcomes.items()),
                'scores': scores,
                'policies': checkpoints
            }
            json.dump(res, f)
    ranking = list(zip(scores, checkpoints))
    ranking.sort(reverse=True)
    return ranking


def sample_variant_params(variants):
    params = {}
    for name in variants:
        value = variants[name]
        if isinstance(value, list):
            params[name] = random.choice(value)
        elif callable(value):
            params[name] = value(params)
        else:
            params[name] = value
    return params


def train(logdir, config, timeout):
    """Train one new policy from scratch.
    """
    expt = Experiment(logdir)
    expt.restart(config)
    t0 = time.time()
    res = {'step': 0}
    while time.time() - t0 < timeout:
        res = expt.train(steps=100)  # ~20 secs
    path = expt.save_checkpoint('final')
    expt.close()
    return path, res['step']


def update_ranking(policy_checkpoints, config, outcomes):
    """Run evaluation tournament between newest policy and all older policies.
    """
    # random policy is reference point at 0 elo
    assert policy_checkpoints[0] == '<random>'
    policies = [RandomPolicy()]
    policies += [Policy.load(config, p)
                 for p in policy_checkpoints[1:]]
    num_battles = 100
    new_outcomes = evaluate(policies, config, num_battles,
                            start_index=len(policies)-1)
    for pair in new_outcomes:
        o = np.array(outcomes[pair]) + new_outcomes[pair]
        outcomes[pair] = o.tolist()
    try:
        ranking = compute_ranking(len(policies), outcomes)
        return ranking.tolist()
    except RankingError as err:
        logging.error('ranking failed: %s', err)
        return []
