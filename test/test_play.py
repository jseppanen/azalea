
import azalea as az
from azalea.play_cli import play_game_interactive


def test_play():
    config = {
        'game': 'azalea.game.hex.HexGame',
        'network': 'azalea.network.HexNetwork',
        'board_size': 11,
        'seed': 1337,
        'device': 'cpu',
        'num_workers': 0,

        'exploration_coef': 1.0,
        'exploration_temperature': 1.0,
        'exploration_depth': 15,
        'exploration_noise_alpha': 0.03,
        'exploration_noise_scale': 0.25,
        'num_blocks': 6,
        'base_chans': 64,
        'simulations': 10,
        'search_batch_size': 10,
    }

    policy = az.Policy(config)
    policy.initialize(config)
    res = play_game_interactive(config, policy, human=None)
    assert res in (1, 2, 3)
