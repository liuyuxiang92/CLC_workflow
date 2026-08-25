"""Screening workflow for perovskites applied in chemical looping beyond combustion."""
from importlib.metadata import PackageNotFoundError, version as _version

__author__ = "Yuxiang Liu"

try:
    # Read the version off the install rather than restating it here, so `clc --version`
    # and pyproject.toml cannot disagree.
    __version__ = _version("clc-workflow")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"
