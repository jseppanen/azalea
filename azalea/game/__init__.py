
def import_game(name):
    """Import game by name.
    """
    # from . import {name}
    import importlib
    gamelib = importlib.import_module(f'.{name}', 'azalea.game')
    return gamelib
