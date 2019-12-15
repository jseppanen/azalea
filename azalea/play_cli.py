
import logging
import warnings
from typing import List

import click
import azalea as az
from azalea.game.hex import HexGame


def play_game_interactive(agents: List[az.typing.Agent], *,
                          game_max_length: int = 300) -> int:
    # start new game
    for ag in agents:
        ag.seed(0)
        ag.reset()
    for ply in range(game_max_length):
        agent = agents[ply % len(agents)]
        move = agent.choose_action()
        res = [ag.execute_action(move) for ag in agents]
        result = res[0]
        assert all(r == result for r in res), "conflicting game states"
        if result:
            return result
    warnings.warn('game reached max length {game_max_length}')
    return 2  # draw


@click.command()
@click.argument('model')
@click.option('--first/--second', default=False,
              help='Human plays first or second')
def main(model, first):
    """Play against game AI.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    computer = az.AzaleaAgent(HexGame, path=model)
    human = az.HumanAgent()
    if first:
        agents = [human, computer]
    else:
        agents = [computer, human]
    res = play_game_interactive(agents)
    if not first:
        res = 4 - res
    if res == 3:
        print('You win')
    elif res == 2:
        print('Draw')
    else:
        print('You lose')


if __name__ == '__main__':
    main()
