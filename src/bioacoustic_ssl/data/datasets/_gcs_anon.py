"""Force anonymous GCS access for alp_data's filesystem factory.

Our GCS-backed datasets (XenoCanto, AudioSet, iNaturalist, Beans — all under
``gs://esp-data-274503`` — plus the public NOAA bucket) live in *public* buckets.
By default :func:`alp_data.io.filesystem.filesystem` builds ``GCSFileSystem()``
with ambient application-default credentials, which expire mid-run and force a
re-authentication that breaks long training jobs.

Importing this module reassigns the module-level ``filesystem`` factory so that
the ``gcs`` / ``gs`` protocol is always built with ``token="anon"``. Because
``filesystem_from_path`` (used both by our datasets and by
``alp_data.io.read_utils``) resolves ``filesystem`` via module-global lookup at
call time, this single reassignment covers every fs-based GCS read. Other
protocols (``r2``, ``local``) are delegated to the original factory unchanged.

The patch is idempotent and applied at import, so it is re-established in every
spawned DataLoader worker.
"""

import sys
from functools import cache
from typing import Literal

from gcsfs import GCSFileSystem

import alp_data.io.filesystem  # noqa: F401  (ensure the submodule is in sys.modules)

# ``alp_data.io.__init__`` rebinds the name ``filesystem`` to the *function*, so
# ``alp_data.io.filesystem`` is not the module. Fetch the real module object,
# whose namespace is the one ``filesystem_from_path`` reads its global from.
_fsmod = sys.modules["alp_data.io.filesystem"]

_PATCH_FLAG = "_bioacoustic_ssl_anon_gcs"


def _install() -> None:
    if getattr(_fsmod, _PATCH_FLAG, False):
        return

    _original = _fsmod.filesystem

    @cache
    def _anon_filesystem(
        protocol: Literal["gcs", "gs", "r2", "local"] = "local",
        **kwargs: dict,
    ):
        if protocol in ("gcs", "gs"):
            kwargs.setdefault("token", "anon")
            return GCSFileSystem(**kwargs)
        return _original(protocol, **kwargs)

    _fsmod.filesystem = _anon_filesystem
    setattr(_fsmod, _PATCH_FLAG, True)


_install()
