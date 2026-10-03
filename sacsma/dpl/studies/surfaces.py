"""What the (Δprecip, ΔT) response-surface studies share: the grid, the four metrics, the
hydroclimate regimes, the evaluation window and the reductions.

For each point of the grid a study recomputes daily basin flow and reduces it to four
metrics per watershed, each reported as the % change against the (0, 0) point of the same
model: total annual runoff and the April–July freshet volume (from the mean-monthly regime),
and the daily 99.9th percentile (flood peak) and 30th percentile (low flow).  Each enters only
as a ratio, so the % change does not depend on the area.  Used by ``adaptive`` (the physics)
and ``hybrids`` (the hybrid family).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...cdec15 import CAL_END
from .climatology import _WY, _monthly_taf

DOMAIN = "15cdec_grid"

#: metric rows: (key, label).  ``q999``/``q30`` are DAILY-flow percentiles.
#: ``q999`` = the 99.9th percentile = the FLOOD PEAK (~top 37 days of the 1915-2018
#: record) — deliberately the extreme tail, not Q98: in snow basins Q98 tracks the
#: snowmelt-freshet shoulder (which the freshet row already carries and which
#: *declines* under warming), whereas the flood peak *intensifies* (snow→rain +
#: rain-on-snow), the complementary half of the warming story.  ``q30`` = low flow.
METRICS: list[tuple[str, str]] = [
    ("annual", "Total annual runoff"),
    ("freshet", "Apr–Jul freshet"),
    ("q999", "Daily Q99.9 (flood peak)"),
    ("q30", "Daily Q30 (low flow)"),
]

#: response-surface grid — nodes sit exactly on the ±10% / +3 °C points, and bracket
#: them at ±20% / +4 °C.  contourf interpolates between the nodes; the 9×9 grid (step
#: 5% / 0.5 °C) gives smooth surfaces at a tractable per-point (frozen noah-lite ~4 s)
#: cost.
DP = np.round(np.arange(-0.20, 0.2001, 0.05), 4)         # Δprecip fraction (9)
DT = np.round(np.arange(0.0, 4.0001, 0.5), 4)            # ΔT degC (9)

#: hydroclimate regimes — freshet-fraction terciles (Apr–Jul runoff / annual of
#: the ``noah`` physics baseline, the snowmelt-timing signature; 5 basins each,
#: snowmelt-strongest → weakest).  Shared by the hybrid-family and physics
#: regime-aggregate figures.
REGIMES: dict[str, list[str]] = {
    "snow": ["PNF", "MIL", "TLG", "ISB", "MRC"],
    "mix":  ["NML", "TRM", "MKM", "SCC", "FOL"],
    "rain": ["YRS", "ORO", "SHA", "BND", "NHG"],
}
REGIME_TITLE = {"snow": "SNOW-dominated", "mix": "MIXED", "rain": "RAIN-dominated"}

#: response-surface EVALUATION window — the metrics reduce over this subset of the
#: daily record: WY1951-1988 (pre-cal) + WY2004-2018 (val).  It
#:   (1) EXCLUDES the WY1989-2003 CAL window the hybrids and their response loss
#:       trained on, so the reported response is OUT-OF-SAMPLE; and
#:   (2) DROPS the 1915-1950 lead-in.  The physics runs cold-start from 1915 (SMA
#:       [0,0,100,100,100,0], Snow-17 zeros); ~35 yr equilibrates every store
#:       (incl. the slow multi-year lztwc) before WY1951, and the LSTM's 365-day
#:       lookback + its 1915-based sim channel are likewise warm.
#: Baseline and perturbed share the identical 1915 spin-up, so it cancels in the
#: %Δ regardless — the window just makes the eval OOS + spin-up-transient-free.
_CAL_START = "1988-10-01"       # WY1989 — the hybrids' training-window start
_EVAL_START = "1950-10-01"      # WY1951 — drop the 1915-1950 cold-start lead-in


def eval_mask(idx) -> np.ndarray:
    """Boolean row mask for the response-evaluation window (see :data:`_EVAL_START`)."""
    ts = pd.DatetimeIndex(idx)
    in_cal = (ts >= pd.Timestamp(_CAL_START)) & (ts <= pd.Timestamp(CAL_END))
    return np.asarray((ts >= pd.Timestamp(_EVAL_START)) & ~in_cal)


def metrics_from_daily(daily: pd.DataFrame, areas: dict[str, float]) -> pd.DataFrame:
    """Per-basin (annual, freshet, q999, q30) from the full record (index=basin,
    cols=the 4 metric keys).

    ``annual``/``freshet`` are TAF from the mean-monthly regime; ``q999``/``q30``
    are the 99.9th (flood peak) / 30th (low flow) percentiles of the DAILY flow
    (mm/day).  Percentiles enter the surfaces only as a % change vs the (0,0)
    baseline, so the mm/day unit is immaterial (the area cancels).

    Reduced over the out-of-sample, spin-up-free window :func:`eval_mask`
    (WY1951-1988 + WY2004-2018; excludes the CAL window + the 1915-1950 lead-in)."""
    daily = daily[eval_mask(daily.index)]                # OOS + spin-up-free window
    m = _monthly_taf(daily, areas)                       # date x basin monthly TAF
    out = {}
    for b in m.columns:
        s = m[b].dropna()
        reg = s.groupby(s.index.month).mean().reindex(_WY)  # 12-pt WY regime
        d = daily[b].dropna().to_numpy()                    # daily mm/day
        out[b] = dict(annual=float(reg.sum()),
                      freshet=float(reg.loc[[4, 5, 6, 7]].sum()),
                      q999=float(np.percentile(d, 99.9)) if d.size else float("nan"),
                      q30=float(np.percentile(d, 30)) if d.size else float("nan"))
    return pd.DataFrame(out).T


def aggregate_regime(tbl: pd.DataFrame, basins: list[str],
                     areas: dict[str, float]) -> pd.DataFrame:
    """Area-weighted mean of each model's per-basin % change over ``basins`` —
    one pooled surface per (model, dp, dt).  Model-agnostic: works for the 2-col
    physics table and the 4-col hybrid table alike."""
    w = np.array([areas[b] for b in basins], float)
    sub = tbl[tbl.basin.isin(basins)]
    rows = []
    for (model, dp, dt), g in sub.groupby(["model", "dp", "dt"]):
        g = g.set_index("basin").reindex(basins)
        r: dict[str, object] = {"basin": "AGG", "model": model, "dp": dp, "dt": dt}
        for k, _ in METRICS:
            r[f"pct_{k}"] = float(np.average(g[f"pct_{k}"].to_numpy(), weights=w))
        rows.append(r)
    return pd.DataFrame(rows)
