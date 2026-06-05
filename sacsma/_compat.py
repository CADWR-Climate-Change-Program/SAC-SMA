"""Optional Numba acceleration.

If Numba is installed, ``njit`` is the real decorator; otherwise it is a
no-op so the pure-Python/NumPy path still runs (slower).  The physics
cores are written in a Numba-friendly style (scalar locals, preallocated
arrays, day-of-year / leap-year passed in as arrays) so they JIT cleanly.
"""

from __future__ import annotations

try:  # pragma: no cover - exercised by environment, not unit tests
    from numba import njit as _njit

    def njit(*args, **kwargs):
        # Default to cache=True unless the caller overrides it.
        kwargs.setdefault("cache", True)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _njit(cache=True)(args[0])
        return _njit(*args, **kwargs)

    HAVE_NUMBA = True
except Exception:  # pragma: no cover

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(func):
            return func

        return deco

    HAVE_NUMBA = False
