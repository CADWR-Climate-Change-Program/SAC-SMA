"""Parity test: full pipeline vs the MATLAB simulated flow.

Runs the native ``data/`` store (forcing + params + HRU table) and compares the
simulated BND gauge flow against the committed MATLAB reference flow in
``data/reference/``.  Skipped automatically unless the forcing store is present
(it is large and gitignored).

Tolerances are placeholders — tighten once parity is validated.  Differences
may stem from open items (HRU temperature lapse adjustment, is_outlet rule,
exact per-basin run window).
"""

import os

import pytest

from sacsma.metrics import kge, nse  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _have_data() -> bool:
    need = [
        os.path.join(DATA_DIR, "forcing", "historical_15cdec.nc"),
        os.path.join(DATA_DIR, "hru", "hruinfo_15cdec.parquet"),
        os.path.join(DATA_DIR, "params", "ga_optimum_15cdec.parquet"),
        os.path.join(DATA_DIR, "reference", "simflow_15cdec.parquet"),
    ]
    return all(os.path.exists(p) for p in need)


pytestmark = pytest.mark.skipif(
    not _have_data(), reason="native data/ store (forcing/params/reference) not present"
)


@pytest.mark.slow
def test_bnd_parity():
    import pandas as pd

    from sacsma.io import load_reference
    from sacsma.model import run_basin

    ref = load_reference(DATA_DIR, basin="BND")
    start = ref["date"].min().strftime("%Y-%m-%d")
    end = ref["date"].max().strftime("%Y-%m-%d")

    sim = run_basin("BND", data_dir=DATA_DIR, start=start, end=end)
    merged = pd.merge(sim, ref, on="date", suffixes=("_sim", "_ref"))
    assert len(merged) > 0

    s = merged["flow_sim"].to_numpy()
    r = merged["flow_ref"].to_numpy()
    # The Python port reproduces the MATLAB simflow exactly (differences are just
    # the reference text rounded to 8 decimals).
    import numpy as np

    assert kge(s, r) > 0.9999, f"BND KGE vs MATLAB simflow not exact: {kge(s, r):.5f}"
    assert np.abs(s - r).max() < 0.05, f"BND max daily diff too large: {np.abs(s - r).max():.4f}"
