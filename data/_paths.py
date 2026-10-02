"""What every build script under ``data/`` starts from.

A script finds this file by walking up from its own location, then takes from it:

``REPO``, ``DATA``
    the repository root and its ``data/`` folder;
``layout``
    the data layout, ``sacsma/paths.py``, loaded by file location so that it also works in an
    environment where the package is not installed (the ``sacsma-gis`` one).  A script asks it
    for every path under ``data/``: ``layout.grid_cells(DATA)``, ``layout.entities(DATA)``, ...
``local_path``, ``local_value``
    where this machine keeps the inputs that are not in the repository.

Importing it also puts the repository root on ``sys.path``, so a script run by file path can
``import sacsma``.

Local inputs
------------
The build scripts read large or licensed inputs from outside the repo: the raw GIS stage, the
forcing master, the WGEN releases, the observation downloads, a sibling repository.  Where they
sit differs by machine, so no script names a location.  Each one is looked up by key, in order:

1. the environment variable ``SACSMA_<KEY>`` (upper case), then
2. ``data/local_paths.toml`` (not tracked: copy ``local_paths.example.toml``, which lists
   every key and what it points at, and fill in the ones you need), then
3. the default the script passes, if it has one (a folder inside the repo).

A key that is set nowhere and has no default resolves to a path named ``UNSET_<KEY>...``, so
importing a script or asking for its ``--help`` never fails; the script stops with that name in
the error when it first reads the input.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
REPO = DATA.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("sacsma_paths", REPO / "sacsma" / "paths.py")
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)

_FILE = DATA / "local_paths.toml"
_cache: dict[str, str] | None = None


def _config() -> dict[str, str]:
    global _cache
    if _cache is None:
        _cache = {}
        if _FILE.exists():
            try:
                import tomllib

                with open(_FILE, "rb") as f:
                    _cache = {k: str(v) for k, v in tomllib.load(f).items()}
            except ImportError:      # Python 3.10: read the flat ``key = 'value'`` lines
                for line in _FILE.read_text(encoding="utf-8").splitlines():
                    key, sep, val = line.partition("=")
                    if sep and not key.strip().startswith("#"):
                        _cache[key.strip()] = val.split(" #")[0].strip().strip("'\"")
    return _cache


def local_value(key: str, default: str | None = None) -> str | None:
    """The configured value of ``key`` (environment, then ``local_paths.toml``), else ``default``."""
    return os.environ.get(f"SACSMA_{key.upper()}") or _config().get(key) or default


def local_path(key: str, default: str | Path | None = None) -> Path:
    """The configured location of ``key`` as a path; see the module docstring for the order."""
    val = local_value(key)
    if val:
        return Path(val)
    if default is not None:
        return Path(default)
    return Path(f"UNSET_{key.upper()}__set_it_in_data_local_paths.toml")
