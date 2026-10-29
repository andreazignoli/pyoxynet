"""The Python package of the Oxynet project"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('pyoxynet')
except PackageNotFoundError:
    __version__ = 'unknown'

from . import utilities
from .utilities import *

from . import testing
from .testing import *
