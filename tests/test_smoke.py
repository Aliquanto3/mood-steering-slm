"""Test de fumée : le paquet s'importe et expose une version cohérente avec le pyproject."""

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import mood_steering

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_version_matches_pyproject():
    expected = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert mood_steering.__version__ == expected
    assert version("mood-steering") == expected


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", mood_steering.__version__)
