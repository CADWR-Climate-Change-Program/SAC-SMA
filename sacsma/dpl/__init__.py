"""Differentiable parameter learning (dPL) for SAC-SMA — the 15cdec application.

A PyTorch re-implementation of the per-HRU pipeline (Hamon PET -> Snow-17 ->
SAC-SMA -> Lohmann routing) that is differentiable end-to-end, plus a network
mapping per-HRU static attributes (and optionally climate indices) to the
physical parameters, trained against the daily observed CDEC gage FNF.

The frozen NumPy/Numba kernels (``sacsma.pet``/``snow17``/``sma``/``routing``)
remain the ONLY parity reference: everything reported as model skill is scored
by pushing exported parameter tables back through ``sacsma.model.run_basin``.
The torch physics here necessarily deviates from the reference numerics (fixed
substep count, branch blends) — the fidelity benchmark
(``sacsma dpl benchmark``) quantifies that gap before any training.

Torch is an optional dependency (``pip install sacsma[dpl]`` or the conda env);
importing :mod:`sacsma` itself never touches it.
"""

from __future__ import annotations

__all__ = [
    "BOUNDS",
    "DplConfig",
    "FIXED_PARAMS",
    "fidelity_benchmark",
    "train",
]


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sacsma.dpl requires PyTorch. Install the 'sacsma' conda env from "
            "environment.yml (conda-forge pytorch/cuda130) or 'pip install sacsma[dpl]'."
        ) from exc


def __getattr__(name: str):
    if name in {"BOUNDS", "FIXED_PARAMS", "DplConfig"}:
        from . import config

        return getattr(config, name)
    if name == "fidelity_benchmark":
        _require_torch()
        from .evaluate import fidelity_benchmark

        return fidelity_benchmark
    if name == "train":
        _require_torch()
        from .train import train

        return train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
