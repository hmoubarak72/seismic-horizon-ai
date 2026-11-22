"""Provenance records: input hashes, parameters, software and algorithm versions.

Every product written by the platform carries a :class:`Provenance` record so that a
result can be traced back to the exact inputs, parameters and code that produced it.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import getpass
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from . import ALGORITHM_VERSIONS, __version__


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks (safe for large SEG-Y)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_array(arr: Any) -> str:
    """Hash a NumPy array's bytes together with its dtype and shape."""
    import numpy as np

    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except Exception:  # pragma: no cover - git missing is not an error
        return None


def _library_versions() -> dict[str, str]:
    """Installed versions of key libraries, read from package metadata (no heavy imports)."""
    from importlib import metadata

    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "segyio", "h5py", "zarr", "numba", "torch"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


@dataclasses.dataclass
class Provenance:
    """Traceability record attached to every derived product."""

    product: str
    parameters: dict[str, Any] = dataclasses.field(default_factory=dict)
    inputs: dict[str, str] = dataclasses.field(default_factory=dict)  # name -> sha256
    notes: list[str] = dataclasses.field(default_factory=list)
    created_utc: str = dataclasses.field(
        default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat()
    )
    software_version: str = __version__
    algorithm_versions: dict[str, str] = dataclasses.field(
        default_factory=lambda: dict(ALGORITHM_VERSIONS)
    )
    library_versions: dict[str, str] = dataclasses.field(default_factory=_library_versions)
    python: str = dataclasses.field(default_factory=platform.python_version)
    platform: str = dataclasses.field(default_factory=platform.platform)
    git_commit: str | None = dataclasses.field(default_factory=_git_commit)
    user: str = dataclasses.field(default_factory=lambda: _safe_user())

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=_json_default, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover
        return "unknown"


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return str(obj)
