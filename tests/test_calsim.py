"""Smoke test for the CalSim catchment mapping + run.

Exercises the spatial HRU->catchment assignment and a short-window inflow run on
the native ``data/`` store.  Skipped unless the forcing store and GeoPackage are
present (both large / one git-LFS).
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def _have_data() -> bool:
    need = [
        os.path.join(DATA_DIR, "forcing", "historical_15cdec.nc"),
        os.path.join(DATA_DIR, "hru", "hruinfo_15cdec.csv"),
        os.path.join(DATA_DIR, "params", "ga_optimum_15cdec.csv"),
        os.path.join(DATA_DIR, "gis", "calsim3.gpkg"),
    ]
    return all(os.path.exists(p) for p in need)


pytestmark = pytest.mark.skipif(
    not _have_data(), reason="native data/ store (forcing + calsim3.gpkg) not present"
)


def test_mapping_and_coverage():
    """Rim catchments split into covered/partial/outside; coverage fractions sane."""
    from sacsma.calsim import (
        COVERED_FRAC,
        load_catchments,
        load_hru_cells,
        map_hrus_to_catchments,
    )

    catch = load_catchments(DATA_DIR, rim_only=True)
    cells = load_hru_cells(DATA_DIR)
    mapping, cov = map_hrus_to_catchments(catch, cells)

    # footprint overlap: an HRU may feed several catchments, but each (cid, key) once
    assert not mapping.duplicated(["cid", "key"]).any()
    assert set(cov["status"]) <= {"covered", "partial", "outside"}
    assert (cov["status"] == "covered").sum() > 50      # many real basins covered
    assert (cov["status"] == "outside").sum() > 0       # valley/non-CDEC are outside

    # every covered catchment clears the threshold; the grid-footprint measure no
    # longer pins to 1.0 (15cdec's coarse grid leaves real gaps), but stays well above it
    covered = cov[cov["status"] == "covered"]
    assert (covered["cov_frac"] >= COVERED_FRAC).all()
    assert covered["cov_frac"].median() > 0.7

    # an edge-only catchment must NOT be spuriously covered (the buffer-overreach bug):
    # THM028 has 15cdec HRUs only grazing its rim -> partial/outside, never covered
    thm = cov[cov["node"] == "THM028"]
    assert (thm["status"] != "covered").all()

    # small interior arcs must NOT be spuriously excluded (the centroid-method bug):
    # the American system (NFA/MFA/FOLSM...) should be essentially fully covered
    amer = cov[cov["node"].astype(str).str.match(r"(NFA|MFA|FOLSM|LOONL|LKVLY)")]
    assert (amer["status"] == "covered").mean() >= 0.85


def test_flowlens():
    """Each catchment has exactly one outlet (flowlen 0); others positive."""
    from sacsma.calsim import (
        assign_flowlens, load_catchments, load_hru_cells, map_hrus_to_catchments,
    )

    catch = load_catchments(DATA_DIR, rim_only=True)
    cells = load_hru_cells(DATA_DIR)
    mapping, _ = map_hrus_to_catchments(catch, cells)
    fl = assign_flowlens(mapping, cells)

    # one outlet per catchment, at flowlen 0; all lengths finite and >= 0
    outl = fl[fl["is_outlet"] == 1]
    assert outl.groupby("cid").size().eq(1).all()
    assert (outl["flowlen_m"] == 0).all()
    assert (fl["flowlen_m"] >= 0).all()
    assert fl["flowlen_m"].max() > 0


@pytest.mark.slow
def test_short_run_local_and_routed():
    """Both run modes give finite per-catchment series; routing conserves volume."""
    import numpy as np

    from sacsma.calsim import run_calsim

    W = dict(start="2000-10-01", end="2001-09-30")
    loc, _cov, mapping = run_calsim(DATA_DIR, route=False, progress=False, **W)
    rou, _cov2, _m2 = run_calsim(DATA_DIR, route=True, progress=False, **W)

    assert loc["cid"].nunique() == mapping["cid"].nunique()
    assert np.isfinite(loc["flow_cfs"].to_numpy()).all()
    assert (loc["flow_mmday"].to_numpy() >= 0).all()
    # routing redistributes timing but conserves each catchment's mean (volume);
    # the tiny residual is Lohmann UH truncation at the finite window edges.
    ml = loc.groupby("cid")["flow_cfs"].mean()
    mr = rou.groupby("cid")["flow_cfs"].mean().reindex(ml.index)
    assert np.allclose(ml.to_numpy(), mr.to_numpy(), rtol=5e-3, atol=1e-3)
