
import importlib
from typing import Any


def import_and_get(name: str) -> Any:
    """Import module and return value from within.
    """
    if '.' not in name:
        raise ImportError(f'name is not like <module>.<name>: {name}')
    module_name, attr_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, attr_name):
        raise ImportError(f'name not found in module: {name}')
    return getattr(module, attr_name)
