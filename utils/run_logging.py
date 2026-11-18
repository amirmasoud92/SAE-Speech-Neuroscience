"""Run logging: config digest, package versions, per-stage runtime and warnings.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_TRACKED = ["numpy", "scipy", "pandas", "sklearn", "statsmodels", "mne", "nilearn",
            "h5py", "himalaya", "torch", "transformers", "matplotlib"]


def setup_logging(log_dir: Path, name: str, config_digest: str = "") -> logging.Logger:
    """Configure root logging to both a timestamped file and stderr.

    Captures ``warnings.warn`` into the log as well, so a silent degradation leaves a
    trace in the run record.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logfile = log_dir / f"{name}_{stamp}.log"

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)-18s  %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    log = logging.getLogger(name)
    log.info("=" * 88)
    log.info("run %s | log %s", name, logfile.name)
    log.info("config digest (cache key, compute sections only): %s",
             config_digest or "<none>")
    log.info("python %s on %s", sys.version.split()[0], platform.platform())
    for mod in _TRACKED:
        try:
            m = __import__(mod)
            log.info("  %-14s %s", mod, getattr(m, "__version__", "?"))
        except Exception:
            log.info("  %-14s <not importable>", mod)
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                log.info("  gpu[%d] %s %.1f GB sm_%d%d", i, p.name,
                         p.total_memory / 1e9, p.major, p.minor)
        else:
            log.warning("  no CUDA device visible - fits will run on CPU")
    except Exception:
        pass
    log.info("=" * 88)
    return log


def log_data_versions(log: logging.Logger, bids_root: Path) -> None:
    """Record which version of the dataset produced a run."""
    for rel in ("dataset_description.json", "CHANGES"):
        f = bids_root / rel
        if f.exists():
            first = f.read_text(encoding="utf-8").strip().splitlines()
            log.info("dataset %s: %s", rel, " | ".join(first[:3]))
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if rev.returncode == 0:
            log.info("code git rev: %s", rev.stdout.strip())
    except Exception:
        pass


@contextmanager
def stage(log: logging.Logger, label: str) -> Iterator[None]:
    """Time a pipeline stage and log its duration, including on failure."""
    log.info("[start] %s", label)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        log.exception("[FAIL ] %s after %.1f s", label, time.perf_counter() - t0)
        raise
    log.info("[done ] %s in %.1f s", label, time.perf_counter() - t0)
