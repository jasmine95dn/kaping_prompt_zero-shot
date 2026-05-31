"""
Stub heavy ML packages and pre-import kaping/qa submodules so that
patch() can resolve dotted attribute paths during e2e tests.
"""

import sys
from unittest.mock import MagicMock

for mod in [
    "sentence_transformers",
    "transformers",
    "refined",
    "refined.inference",
    "refined.inference.processor",
    "sklearn",
    "sklearn.metrics",
    "sklearn.metrics.pairwise",
    "bs4",
    "requests",
]:
    sys.modules.setdefault(mod, MagicMock())

# Pre-import submodules so patch("kaping.entity_verbalization.pipeline") etc. resolve correctly
