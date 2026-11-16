"""
hmmcraft
========

``hmmcraft`` is a set of algorithms for learning and inference of
Hidden Markov Models.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("hmm-craft")
except ImportError:
    __version__ = "0+unknown"
