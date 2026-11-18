"""Locations of the inputs that are not set in ``config/config.yaml``.

Each location can be set with an environment variable; relative defaults are resolved
against the repository root.

``BTB_ROOT``       the Brain Treebank release (transcripts, metadata, neural recordings)
``BTB_ALIGNMENT``  a table with one row per timed transcript token of every Brain Treebank
                   session and its ``neural_sample``, which places words on each session's
                   neural clock
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _path(var: str, default: str) -> Path:
    p = Path(os.environ.get(var, default))
    return p if p.is_absolute() else REPO / p


BTB_ROOT = _path("BTB_ROOT", "data/braintreebank")
BTB_ALIGNMENT = _path("BTB_ALIGNMENT", "data/braintreebank/aligned_words.csv")
