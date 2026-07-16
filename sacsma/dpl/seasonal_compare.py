"""A dPL arm vs the canonical noah / noah_ft / Hybrid series — did the arm
recover the LSTM's seasonal-timing correction without its volume-bias downside?

Given an arm's daily sim (dumped by ``evaluate.score_noah_torch`` ->
``daily_sim_<label>.csv``), this scores it head to head against the frozen
``noah`` baseline, the canonical ``noah_ft`` seasonal-melt fine-tune, and the
canonical Hybrid ensemble on the metrics that matter for the seasonal question:

* **daily-gage validation** (vs the daily CDEC gage FNF, dates > ``CAL_END``):
  the KGE decomposition r / alpha / beta / pbias and the monthly seasonal-mismatch
  (``metrics.seasonal_mismatch`` on the val-period mean-monthly regime);
* **monthly vs CalSim3 FNF** over the climatology's combined out-of-cal period
  (WY1950-1987 + WY2004-2018) — the basis the climatology figure uses, where the
  already-good basins (NML/MRC/ORO) drop under the LSTM;
* the **arm-minus-noah correction** decomposed into its fixed 12-month climatology
  vs interannual remainder (``seas_frac`` — how much a static seasonal lever could
  ever absorb).

The noah / LSTM daily series and the CalSim3 monthly FNF are read from the cached
climatology artifacts (``artifacts/calsim/compare/_climatology_cache``); run
``sacsma.dpl.climatology.make_cdec15_climatology`` first if they are absent.

Writes ``seasonal_compare_<label>.csv`` + ``seasonal_compare_<label>.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cdec15 import CAL_END, load_gage
from ..io import mmday_to_cfs
from ..metrics import seasonal_mismatch

_CACHE = Path("artifacts/calsim/compare/_climatology_cache")
#: combined out-of-calibration period (mirrors climatology._PERIOD).
_PERIOD = [("1949-10-01", "1987-09-30"), ("2003-10-01", "2018-12-31")]
_AF_PER_CFS_DAY = 1.98347


def _kge_decomp(sim: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    """KGE plus its r / alpha (std ratio) / beta (mean ratio) / pbias parts."""
    m = np.isfinite(sim) & np.isfinite(obs)
    s, o = sim[m], obs[m]
    if s.size < 30:
        return dict(kge=np.nan, r=np.nan, alpha=np.nan, beta=np.nan, pbias=np.nan)
    mus, muo, sds, sdo = s.mean(), o.mean(), s.std(), o.std()
    r = float(np.mean((s - mus) * (o - muo)) / (sds * sdo))
    alpha, beta = sds / sdo, mus / muo
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    pbias = 100.0 * (s.sum() - o.sum()) / o.sum()
    return dict(kge=float(kge), r=r, alpha=float(alpha), beta=float(beta),
                pbias=float(pbias))


def _monthly_mean(s: pd.Series) -> pd.Series:
    return s.resample("MS").mean()


def _in_period(idx: pd.PeriodIndex | pd.DatetimeIndex) -> np.ndarray:
    ts = idx.to_timestamp() if isinstance(idx, pd.PeriodIndex) else pd.DatetimeIndex(idx)
    mask = np.zeros(len(ts), bool)
    for lo, hi in _PERIOD:
        mask |= (ts >= pd.Timestamp(lo)) & (ts <= pd.Timestamp(hi))
    return mask


def _to_taf(daily: pd.Series, area: float) -> pd.Series:
    cfs = pd.Series(mmday_to_cfs(daily.to_numpy(), area), index=daily.index)
    mc = cfs.resample("MS").mean()
    return pd.Series(mc.to_numpy() * mc.index.days_in_month.to_numpy()
                     * _AF_PER_CFS_DAY / 1000.0, index=mc.index)


def _calsim3_kge(daily: pd.Series, c3: pd.Series, area: float) -> float:
    s = _to_taf(daily, area)
    s.index, o = s.index.to_period("M"), c3.copy()
    o.index = pd.DatetimeIndex(o.index).to_period("M")
    df = pd.concat({"s": s, "o": o}, axis=1).dropna()
    df = df[_in_period(df.index)]
    if len(df) < 12:
        return np.nan
    return _kge_decomp(df["s"].to_numpy(), df["o"].to_numpy())["kge"]


def _correction_seas_frac(corr_daily: pd.Series, obs_mask: np.ndarray
                          ) -> tuple[float, float]:
    """Fraction of the monthly arm-minus-noah correction variance explained by a
    fixed 12-month climatology (R^2 of month-of-year), + the mean correction."""
    cm = corr_daily[obs_mask].resample("MS").mean().dropna()
    if len(cm) < 24:
        return np.nan, np.nan
    clim = cm.groupby(cm.index.month).mean()
    pred = clim.reindex(cm.index.month).to_numpy()
    ss_tot = float(np.sum((cm.to_numpy() - cm.to_numpy().mean()) ** 2))
    ss_res = float(np.sum((cm.to_numpy() - pred) ** 2))
    seas_frac = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return seas_frac, float(cm.to_numpy().mean())


def seasonal_physics_report(
    arm_daily_csv: str | Path,
    out_dir: str | Path,
    *,
    label: str = "seasonal",
    data_dir: str = "data",
    cache_dir: str | Path = _CACHE,
) -> pd.DataFrame:
    """Score the arm vs noah + the two LSTM ensembles; write CSV + figure."""
    from ..calsim.catchments import basin_areas

    cache = Path(cache_dir)
    need = {"noah": cache / "sim_noah.csv",
            "noah_ft": Path("artifacts/dpl/noah_ft/daily_sim_noah_ft.csv"),
            "hybrid": cache / "sim_hybrid.csv",
            "c3": cache / "calsim3_fnf_monthly.csv"}
    for k, p in need.items():
        if not p.exists():
            raise FileNotFoundError(
                f"missing cached {k} ({p}); run "
                "sacsma.dpl.climatology.make_cdec15_climatology first")
    sims = {
        "arm": pd.read_csv(arm_daily_csv, parse_dates=["date"]).set_index("date"),
        "noah": pd.read_csv(need["noah"], parse_dates=["date"]).set_index("date"),
        "noah_ft": pd.read_csv(need["noah_ft"],
                               parse_dates=["date"]).set_index("date"),
        "hybrid": pd.read_csv(need["hybrid"],
                              parse_dates=["date"]).set_index("date"),
    }
    c3 = pd.read_csv(need["c3"], parse_dates=["date"]).set_index("date")
    gage = load_gage(data_dir)
    areas = basin_areas(data_dir, domain="15cdec")
    idx = sims["noah"].index
    basins = [b for b in sims["arm"].columns if b in sims["noah"].columns]

    rows = []
    for b in basins:
        o = gage[gage.basin == b].set_index("date")["flow"].reindex(idx)
        val = (idx > pd.Timestamp(CAL_END)) & np.isfinite(o.to_numpy())
        obs_all = np.isfinite(o.to_numpy())
        rec: dict[str, float | str] = {"basin": b}
        for m in ("noah", "noah_ft", "arm", "hybrid"):
            s = sims[m][b].reindex(idx)
            d = _kge_decomp(s.to_numpy()[val], o.to_numpy()[val])
            # monthly seasonal mismatch over the val period vs the gage
            mm = pd.DataFrame({"s": s, "o": o})[idx > pd.Timestamp(CAL_END)]
            mm = mm[np.isfinite(mm["o"])]
            sm = _monthly_mean(mm["s"]).to_frame("s")
            sm["o"] = _monthly_mean(mm["o"])
            smis = (seasonal_mismatch(sm.index, sm["s"].to_numpy(), sm["o"].to_numpy())
                    if len(sm) >= 12 else np.nan)
            c3kge = (_calsim3_kge(s, c3[b], areas[b]) if b in c3.columns
                     and b in areas else np.nan)
            rec.update({f"{m}_val_kge": d["kge"], f"{m}_r": d["r"],
                        f"{m}_alpha": d["alpha"], f"{m}_beta": d["beta"],
                        f"{m}_pbias": d["pbias"], f"{m}_seas_mis": smis,
                        f"{m}_c3_kge": c3kge})
        corr = sims["arm"][b].reindex(idx) - sims["noah"][b].reindex(idx)
        sf, mc = _correction_seas_frac(corr, obs_all)
        rec["arm_vs_noah_seas_frac"] = sf
        rec["arm_vs_noah_mean_corr"] = mc
        rows.append(rec)

    df = pd.DataFrame(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv = out / f"seasonal_compare_{label}.csv"
    df.round(4).to_csv(csv, index=False)
    print(f"wrote {csv}", flush=True)
    _plot(df, basins, out / f"seasonal_compare_{label}.png", label)
    # console summary: means
    for m in ("noah", "noah_ft", "arm", "hybrid"):
        print(f"  {m:5s}: mean val_KGE={df[f'{m}_val_kge'].mean():.3f}  "
              f"mean seas_mis={df[f'{m}_seas_mis'].mean():.3f}  "
              f"mean CalSim3_KGE={df[f'{m}_c3_kge'].mean():.3f}", flush=True)
    print(f"  arm-vs-noah correction seas_frac (mean) = "
          f"{df['arm_vs_noah_seas_frac'].mean():.3f}", flush=True)
    return df


_COLORS = {"noah": "#bcbd22", "noah_ft": "#d62728", "arm": "#17becf",
           "hybrid": "#9467bd"}
_NAMES = {"noah": "dPL noah", "noah_ft": "dPL noah_ft", "arm": "arm",
          "hybrid": "Hybrid"}


def _plot(df: pd.DataFrame, basins: list[str], path: Path, label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = ["noah", "noah_ft", "arm", "hybrid"]
    rows = [("daily-gage val KGE", "val_kge", (0.0, 1.0)),
            ("seasonal mismatch (val)", "seas_mis", None),
            ("monthly KGE vs CalSim3 FNF", "c3_kge", (0.0, 1.0))]
    n = len(models)
    width = 0.86 / n
    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.0), sharex=True)
    seen: dict[str, object] = {}
    for ax, (name, key, ylim) in zip(axes, rows, strict=True):
        for i in range(len(basins)):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f4f4", zorder=0)
        for k, m in enumerate(models):
            xs = [i + (k - (n - 1) / 2) * width for i in range(len(basins))]
            ys = df.set_index("basin").reindex(basins)[f"{m}_{key}"].to_numpy()
            bc = ax.bar(xs, ys, width=width * 0.92, color=_COLORS[m], alpha=0.9,
                        zorder=2, label=_NAMES[m])
            seen[m] = bc
        ax.set_ylabel(name, fontsize=10.5)
        ax.set_xlim(-0.5, len(basins) - 0.5)
        ax.grid(axis="y", alpha=0.3, lw=0.5, zorder=1)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    axes[-1].set_xticks(range(len(basins)))
    axes[-1].set_xticklabels(basins, fontsize=11, fontweight="bold")
    fig.legend([seen[m] for m in models], [_NAMES[m] for m in models],
               loc="lower center", ncol=n, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Arm ({label}) vs noah / noah_ft / Hybrid",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}", flush=True)
