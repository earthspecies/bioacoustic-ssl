"""Run-scoped, on-the-fly per-item dataset cache backed by :mod:`diskcache`.

The cache lives in a unique directory created at run start and is removed when the
run ends (normal completion or unhandled exception). It is shared across all
DataLoader worker processes via the filesystem, so each item is decoded / fetched
once per run regardless of how many workers or epochs touch it.
"""

import atexit
import logging
import os
import shutil
import uuid

import diskcache
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

_DEFAULT_BASE = os.path.expanduser("~/.cache/soundscape_ssl")
_CLEANUPS: list = []


def cleanup_all() -> None:
    """Run every registered run-cache cleanup (idempotent)."""
    while _CLEANUPS:
        _CLEANUPS.pop()()


def open_run_cache(base_dir: str | None, size_limit_gb: float | None) -> diskcache.Cache:
    """Create a fresh run-scoped :class:`diskcache.Cache` and register its cleanup.

    Args:
        base_dir: Parent directory for the run cache; ``None`` falls back to
            ``~/.cache/soundscape_ssl``.
        size_limit_gb: Soft size cap in GB; diskcache evicts (LRU) past this.
            ``None`` disables eviction (effectively unlimited).

    Returns:
        An open :class:`diskcache.Cache`. Its backing directory is removed on
        interpreter exit.
    """
    base = base_dir or _DEFAULT_BASE
    run_id = os.environ.get("SLURM_JOB_ID") or uuid.uuid4().hex
    run_dir = os.path.join(base, f"run-{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    # diskcache has no native "unlimited"; a huge cap means eviction never fires.
    size_limit = 2**62 if size_limit_gb is None else int(size_limit_gb * 1e9)
    cache = diskcache.Cache(run_dir, size_limit=size_limit)
    log.info(f"[cache] run cache at {run_dir} (limit {size_limit_gb or 'unlimited'} GB)")

    done = False

    def cleanup() -> None:
        nonlocal done
        if done:
            return
        done = True
        cache.close()
        shutil.rmtree(run_dir, ignore_errors=True)
        log.info(f"[cache] removed {run_dir}")

    atexit.register(cleanup)
    _CLEANUPS.append(cleanup)
    return cache


class CachedDataset(Dataset):
    """Memoises ``dataset[idx]`` in a shared :class:`diskcache.Cache`.

    The wrapped dataset's ``__getitem__`` output must be deterministic for the
    cached value to be correct. Attribute access other than ``__len__`` /
    ``__getitem__`` is delegated to the wrapped dataset.
    """

    def __init__(self, dataset: Dataset, cache: diskcache.Cache, namespace: str) -> None:
        super().__init__()
        self.dataset = dataset
        self.cache = cache
        self.namespace = namespace

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        key = f"{self.namespace}:{idx}"
        value = self.cache.get(key)
        if value is None:
            value = self.dataset[idx]
            self.cache.set(key, value)
        return value

    def __getattr__(self, name: str) -> object:
        # Only called when normal attribute lookup fails. Guard the core
        # attributes so unpickling (spawn workers) doesn't recurse before
        # __dict__ is restored.
        if name in ("dataset", "cache", "namespace"):
            raise AttributeError(name)
        return getattr(self.dataset, name)
