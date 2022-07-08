

__version__ = "0.0.1"
__author__ = 'Tim Schaefer'

"""
Meshlearn high-level API functions.
"""

# The next line makes the listed functions show up in sphinx documentation directly under the package (they also show up under their real sub module, of course)
__all__ = []

from . import tfdata
from .tfdata import VertexPropertyDataset
