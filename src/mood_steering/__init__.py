"""Démonstrateur d'expression d'humeur d'un SLM par steering d'activations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mood-steering")
except PackageNotFoundError:  # sources lues sans installation du paquet
    __version__ = "0.0.0"

__all__ = ["__version__"]
