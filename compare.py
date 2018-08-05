# -*- encoding: utf-8 -*-

import logging

import numpy as np
import torch
import json
import yaml
import click

import azalea as az


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='YAML configuration file')
@click.option('--rundir', type=click.Path(), required=True,
              help='Directory to save results from evaluation run')
@click.argument('models', nargs=-1, type=click.Path())
def main(config, rundir, models):
    """Compare game AI models.
    """

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    az.redirect_all_output(f'{rundir}/compare.log')

    config = yaml.load(open(config))
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if 'eval_models' in config:
        models = list(models) + config['eval_models']
    if len(models) < 2:
        raise click.UsageError('need at least two models to compare')

    scores, outcomes = compare(models, config)

    with open(f'{rundir}/results.json', 'w') as f:
        res = {
            'outcomes': list(outcomes.items()),
            'scores': scores,
            'policies': models
        }
        json.dump(res, f)

    for i in np.argsort(scores)[::-1][:10]:
        print(f'{scores[i]:4.0f} {models[i]}')


def compare(models, config):
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    device = torch.device(config['device'])
    policies = []
    for m in models:
        logging.info(f'loading {m}')
        p = az.Policy.load(config, m)
        p.net.to(device)
        policies.append(p)

    # add random policy as reference point at 0 elo
    models = ['<random>'] + models
    policies = [az.RandomPolicy(config)] + policies

    outcomes = az.evaluate(policies, config, config['eval_rounds'])
    scores = az.ranking.compute_ranking(len(policies), outcomes).tolist()

    # remove random policy
    scores = scores[1:]
    outcomes = dict(((i - 1, j - 1), outcomes[i, j])
                    for i, j in outcomes)
    return scores, outcomes


if __name__ == '__main__':
    main()
