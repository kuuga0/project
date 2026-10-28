"""Discover task packages; each package registers its own Gym environments."""

from importlib import import_module
from pkgutil import iter_modules


def register_task_packages() -> None:
    """Import immediate task packages without eagerly importing their config files."""
    for module in sorted(iter_modules(__path__), key=lambda item: item.name):
        if module.ispkg and not module.name.startswith("_"):
            import_module(f"{__name__}.{module.name}")


register_task_packages()
