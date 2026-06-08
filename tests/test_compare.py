"""Smoke test for the consolidated CalSim cross-compare.

Scores the SAC-SMA calibration sets and VIC against the CalSim3 historical
inflow (the actual).  Runs ``calsim.run_calsim`` live for each set, so it needs
the native ``data/`` store (forcing + GeoPackage + params) plus the reference
CSV reference tables.  Marked slow (a full-period domain run) and skipped if data is missing.
"""

import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

#: a single set keeps the smoke test to one domain run.
SET = "15cdec"


def _have() -> bool:
    need = [
        os.path.join(DATA_DIR, "forcing", f"historical_{SET}.nc"),
        os.path.join(DATA_DIR, "hru", f"hruinfo_{SET}.csv"),
        os.path.join(DATA_DIR, "params", f"ga_optimum_{SET}.csv"),
        os.path.join(DATA_DIR, "gis", "calsim3.gpkg"),
        os.path.join(DATA_DIR, "reference", "vic_routed_monthly.csv"),
        os.path.join(DATA_DIR, "reference", "calsim3_inflow_monthly.csv"),
        os.path.join(DATA_DIR, "reference", "calsim_crosswalk.csv"),
    ]
    return all(os.path.exists(p) for p in need)


pytestmark = pytest.mark.skipif(not _have(), reason="native data/ store or reference tables missing")


@pytest.mark.slow
def test_cross_compare_structure():
    from sacsma.compare import build_calsets_long, calset_metrics

    sets = (SET,)
    long, matched, coverage = build_calsets_long(DATA_DIR, sets)

    assert len(matched) > 50
    assert set(long["source"].unique()) <= {SET, "calsim3", "vic"}

    # common period: every source shares the same min/max date (global clip)
    bounds = long.groupby("source")["date"].agg(["min", "max"])
    assert bounds["min"].nunique() == 1
    assert bounds["max"].nunique() == 1

    # per-set basin->node table accompanies the long frame (now carries kind)
    assert set(coverage["set"]) == {SET}
    assert {"cid", "node", "arc", "basin", "kind", "cov_frac"} <= set(coverage.columns)
    assert coverage["basin"].notna().all()
    # cumulative systems are now WHOLE catchments (merged layer) -> scored/shown, not grey
    assert "I_MCLRE" in set(coverage["arc"])
    # below-rim valley reaches are never scored
    assert not {"I_SJR258", "I_SJR265", "I_TUO054", "I_TUO105"} & set(coverage["arc"])

    met = calset_metrics(long, matched, list(sets) + ["vic"])
    sc = met[met["set"] == SET]
    assert len(sc) > 50
    assert sc["kge"].median() > 0.3   # reasonable monthly agreement vs the actual


@pytest.mark.slow
def test_derive_basin_nodes_anchor_hybrid():
    """Hybrid anchor+geographic mapping: rim systems from the crosswalk (incl. nesting &
    cumulative single nodes); secondary watersheds geographic; valley reaches excluded."""
    import pandas as pd
    from sacsma.calsim import derive_basin_nodes

    c3 = set(pd.read_csv(os.path.join(DATA_DIR, "reference", "calsim3_inflow_monthly.csv"))
             ["arc"].astype(str))
    df = derive_basin_nodes(DATA_DIR, "15cdec", calsim3_arcs=c3)
    by_basin = df.groupby("basin")["arc"].apply(set)
    # nesting: Bend Bridge (SRBB) inherits Shasta's I_SHSTA; Shasta scored alone too
    assert "I_SHSTA" in by_basin.get("SHA", set())
    assert "I_SHSTA" in by_basin.get("BND", set())
    # cumulative single-node systems: whole basin = one anchor node
    assert by_basin.get("MRC") == {"I_MCLRE"}
    assert by_basin.get("MIL") == {"I_MLRTN"}
    assert df[df["basin"] == "MRC"]["kind"].eq("rim_cumulative").all()
    # below-rim valley reaches never assigned to anyone
    assert not {"I_SJR258", "I_SJR265", "I_TUO054", "I_TUO105"} & set(df["arc"])
    # distributed rim system pulls in its full anchor membership (American is large)
    assert len(by_basin.get("FOL", set())) > 30
    # Tulare/Kern basins are outside the CalSim3 inflow domain -> no nodes
    assert "ISB" not in set(df["basin"])


@pytest.mark.slow
def test_derive_basin_nodes_secondary_alias():
    """Secondary watersheds resolve geographically; the BRYSA->I_PTH070 alias works."""
    import pandas as pd
    from sacsma.calsim import derive_basin_nodes

    c3 = set(pd.read_csv(os.path.join(DATA_DIR, "reference", "calsim3_inflow_monthly.csv"))
             ["arc"].astype(str))
    df = derive_basin_nodes(DATA_DIR, "9unimp", calsim3_arcs=c3)
    by_basin = df.groupby("basin")["arc"].apply(set)
    assert df["basin"].nunique() == 9                       # all 9 watersheds mapped
    # Putah/Berryessa: GIS polygon BRYSA aliases to the CalSim3 series I_PTH070
    assert by_basin.get("PutahCreek") == {"I_PTH070"}
    # Mokelumne recovers its forks
    assert {"I_NFM010", "I_MFM008", "I_SFM005"} <= by_basin.get("MokelumneRiver", set())


@pytest.mark.slow
def test_vic_full_metrics_spans_all():
    """The full VIC-vs-CalSim3 table covers far more than any one set (incl. San Luis)."""
    from sacsma.compare import vic_full_metrics

    vm = vic_full_metrics(DATA_DIR)
    assert len(vm) > 150                       # ~206 VIC-comparable CalSim3 arcs
    assert "I_SLUIS" in set(vm["arc"])         # valley node no SAC-SMA set covers


@pytest.mark.slow
def test_anchor_basin_level():
    """Basin-level anchor comparison: each basin vs the sum of its sub-nodes scores well."""
    from sacsma.compare import anchor_metrics, build_anchor_long

    if not os.path.exists(os.path.join(DATA_DIR, "forcing", f"historical_{SET}.nc")):
        pytest.skip("forcing store not present")
    long = build_anchor_long(DATA_DIR, sets=(SET,))
    assert set(long["source"].unique()) <= {SET, "calsim3", "vic"}
    met = anchor_metrics(long)
    sc = met[(met["set"] == SET) & (met["source"] == SET)]
    assert sc["basin"].nunique() >= 9
    assert sc["kge"].median() > 0.6   # basin-level aggregation is high-skill


def _have_unimp() -> bool:
    return os.path.exists(os.path.join(DATA_DIR, "reference", "calsim_unimpaired_monthly.csv")) and \
        os.path.exists(os.path.join(DATA_DIR, "reference", "vic_routed_monthly.csv"))


def test_main_rivers_are_the_8_index():
    """MAIN_RIVERS is the CA 8-River Index (Shasta/Trinity/Whiskeytown excluded)."""
    from sacsma.compare import MAIN_RIVERS

    systems = [s for s, _ in MAIN_RIVERS]
    assert len(MAIN_RIVERS) == 8
    assert set(systems) == {"SRBB", "OROV", "YUBA", "FOLS", "ST", "TU", "ME", "SJ"}
    assert not ({"SHAS", "TRIN", "WH"} & set(systems))


@pytest.mark.skipif(not _have_unimp(), reason="unimpaired reference table missing")
def test_unimp_reference_present():
    """The CalSim FLOW-UNIMPAIRED reference (used by the anchor) resolves for all systems."""
    from sacsma.compare import UNIMP_MAP, load_unimpaired_monthly

    u = load_unimpaired_monthly(DATA_DIR)
    assert set(u["system"]) == set(UNIMP_MAP)            # all 11 rim systems present
    assert u["flow_taf"].notna().all()


@pytest.mark.slow
@pytest.mark.skipif(not _have_unimp(), reason="unimpaired / vic reference table missing")
def test_anchor_uses_unimpaired_for_main_rivers():
    """For the 8 main-river basins the anchor reference is the FLOW-UNIMPAIRED series."""
    from sacsma.calsim import BASIN_RIM_SYSTEM
    from sacsma.compare import MAIN_RIVERS, anchor_metrics, build_anchor_long

    if not os.path.exists(os.path.join(DATA_DIR, "forcing", f"historical_{SET}.nc")):
        pytest.skip("forcing store not present")
    long = build_anchor_long(DATA_DIR, sets=(SET,))
    met = anchor_metrics(long)
    river_systems = {s for s, _ in MAIN_RIVERS}
    sys_of = BASIN_RIM_SYSTEM[SET]
    sac = met[(met["set"] == SET) & (met["source"] == SET)]
    rim = sac[sac["basin"].map(lambda b: sys_of.get(b) in river_systems)]
    assert len(rim) >= 8                                  # 15cdec touches 8 of the main rivers
    assert (rim["ref_kind"] == "unimp").all()             # all scored vs FLOW-UNIMPAIRED


def test_anchor_area_scale_table():
    """The anchor area-nudge table is within the cap and never degrades KGE/NSE."""
    import pandas as pd

    p = os.path.join(DATA_DIR, "reference", "anchor_area_scale.csv")
    if not os.path.exists(p):
        pytest.skip("anchor_area_scale.csv not present")
    t = pd.read_csv(p)
    # scale within +/-10% cap, and area_after = area_before * scale
    assert (t["scale"] >= 0.90 - 1e-9).all() and (t["scale"] <= 1.10 + 1e-9).all()
    assert ((t["area_after_mi2"] - t["area_before_mi2"] * t["scale"]).abs() < 1.0).all()
    # the nudge does not degrade KGE/NSE (tol 0.005) and does not worsen |pbias|
    assert (t["kge_after"] >= t["kge_before"] - 0.005 - 1e-9).all()
    assert (t["nse_after"] >= t["nse_before"] - 0.005 - 1e-9).all()
    assert (t["pbias_after"].abs() <= t["pbias_before"].abs() + 1e-9).all()


@pytest.mark.slow
def test_compute_anchor_area_scale_respects_cap():
    """compute_anchor_area_scale honours the cap and reduces median |pbias|."""
    from sacsma.compare import compute_anchor_area_scale

    if not os.path.exists(os.path.join(DATA_DIR, "forcing", f"historical_{SET}.nc")):
        pytest.skip("forcing store not present")
    df = compute_anchor_area_scale(DATA_DIR, sets=(SET,), cap=0.10)
    assert (df["scale"].between(0.90 - 1e-9, 1.10 + 1e-9)).all()
    assert df["pbias_after"].abs().median() <= df["pbias_before"].abs().median()


@pytest.mark.slow
def test_subarc_validation_improves_on_test_period():
    """Per-sub-arc train/test bias-correction + anchor mass-balance improves held-out skill."""
    from sacsma.compare import subarc_validation_metrics

    if not os.path.exists(os.path.join(DATA_DIR, "forcing", f"historical_{SET}.nc")):
        pytest.skip("forcing store not present")
    m = subarc_validation_metrics(DATA_DIR, sets=(SET,))
    assert len(m) > 50                                          # the distributed-rim sub-arcs
    # corrected sub-arcs are scored on the held-out test period and beat the raw estimate
    assert m["kge_corr"].median() > m["kge_raw"].median()
    assert m["pbias_corr"].abs().median() < m["pbias_raw"].abs().median()
