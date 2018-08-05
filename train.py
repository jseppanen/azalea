
import logging

import torch
import yaml
import click

import azalea as az


@click.command()
@click.option('--config', type=click.Path(exists=True), required=True,
              help='YAML configuration file')
@click.option('--rundir', type=click.Path(), required=True,
              help='Directory to save results from training run')
@click.option('--startpos', type=click.Path(),
              help='YAML file of board starting positions')
@click.option('--model', type=click.Path(),
              help='Warm start training from model checkpoint')
@click.option('--replaybuf', type=click.Path(),
              help='Warm start training from replay buffer checkpoint')
def main(config, rundir, model, replaybuf, startpos):
    """Train a chess AI model.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    az.redirect_all_output(f'{rundir}/train.log')
    logging.info(f'azalea {az.__version__}')

    expt = az.Experiment(rundir)

    config = yaml.load(open(config))
    if config['device'] == 'auto':
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    if startpos:
        ss = yaml.load(open(startpos))
        config['start_positions'] = [s['fen'] for s in ss]

    expt.restart(config)

    if model:
        expt.load_checkpoint(model)

    if replaybuf:
        expt.load_replaybuf(replaybuf)

    expt.train()
    expt.save_checkpoint('final')
    expt.close()


if __name__ == '__main__':
    main()
