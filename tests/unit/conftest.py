"""
Mock heavy ML dependencies at import time so unit tests run without
installing sentence-transformers, transformers, or ReFinED.
"""

import sys
from unittest.mock import MagicMock

# Stub out all heavy ML packages before any test module imports them
for mod in [
    "sentence_transformers",
    "transformers",
    "refined",
    "refined.inference",
    "refined.inference.processor",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
]:
    sys.modules.setdefault(mod, MagicMock())
