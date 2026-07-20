"""Per-HRU features for the parameter network: physical statics + climate indices.

Two variants (the study's ablation):

* ``static`` — physical statics only: elev, lat, lon, flowlen (z-scored) plus
  one-hot ``soil_class`` / ``veg_class`` (the same information class the GA's
  soil/veg zonal regionalization used);
* ``climate`` — statics PLUS forcing-derived climatology, computed per HRU
  from its grid cell over a configurable ``(product, window)``.

Optional spatial Fourier terms (net-v2, ``fourier_k > 0``): sin/cos of
``2*pi*f*(lat, lon)`` normalized to the training-domain extent, ``f = 1..k`` —
low-frequency coordinate features that let the net express smooth regional
parameter fields (the role of the GA's hand-drawn SMA zones) which raw lat/lon
through a small MLP cannot bend into.  The extent is stored in the
:class:`FeatureSet` so checkpoint evaluation reproduces them exactly.

Non-stationarity: the climate indices are FUNCTIONS OF THE FORCING WINDOW and
are recomputable under any future climate product — under warming,
``snow_frac`` falls and ``aridity`` rises, so a climate-variant parameter
field ADAPTS when the indices are recomputed (snow/ET-controlled parameters
move), whereas the statics-only field is climate-frozen.  The window/product
used at training time is stored with the normalization stats so an evaluation
under a new climate is an explicit, documented choice.

Indices (per HRU):
    p_mean       mean annual precipitation (mm/yr)
    aridity      sum(raw coefficient-free Hamon PET) / sum(P)  (Kpet-independent)
    snow_frac    sum(P on days tavg <= 0 degC) / sum(P)   (0 degC = fixed PXTEMP)
    seasonality  Walsh & Lawler SI = sum_m |P_m - P/12| / P  (0..1.83)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .pet import hamon_raw_pet_numpy

CONTINUOUS_STATICS = ("elev", "lat", "lon", "flowlen")
CLIMATE_INDICES = ("p_mean", "aridity", "snow_frac", "seasonality")

# -- physical variant: continuous soil/veg/terrain/LAI, sampled per HRU from the
#    CA raster stack (data/raw_gis; see sacsma.io.soilveg_path).  These REPLACE
#    the opaque one-hot soil_class/veg_class of the static/climate variants. --
#: POLARIS depth zones aggregated to SAC-SMA storage layers (depth weights, cm).
_POLARIS_PROPS = ("sand", "clay", "ksat", "theta_s")
_SURF_ZONE = {"0_5": 5.0, "5_15": 10.0, "15_30": 15.0}          # ~ upper zone
_DEEP_ZONE = {"30_60": 30.0, "60_100": 40.0, "100_200": 100.0}  # ~ lower zone
PHYSICAL_FEATURES = (
    "sand_surf", "sand_deep", "clay_surf", "clay_deep",
    "ksat_surf", "ksat_deep", "thetas_surf", "thetas_deep",   # soil (log10 ksat)
    "evc_cover", "evh_height",                                # LANDFIRE veg
    "slope", "aspect_sin", "aspect_cos", "curvature", "relief",  # 3DEP terrain
    "lai_mean", "lai_amp", "lai_peak_sin", "lai_peak_cos",    # MODIS LAI
)


@dataclass
class FeatureSet:
    """Feature matrix + everything needed to rebuild it identically."""

    x: np.ndarray                  # (N, F) float32, z-scored continuous cols
    names: list[str]
    mean: np.ndarray               # (F,) normalization stats (0/1 for one-hots)
    std: np.ndarray
    soil_categories: list[int]
    veg_categories: list[int]
    variant: str                   # "static" | "climate"
    climate_window: tuple[str, str] | None
    climate_product: str | None
    fourier_k: int = 0             # spatial Fourier order (0 = off)
    #: (lat_min, lat_max, lon_min, lon_max) normalization extent for the
    #: Fourier terms (None when fourier_k == 0)
    coord_bounds: tuple[float, float, float, float] | None = None


def climate_indices(
    hrus: pd.DataFrame,
    forcing,                        # model.DomainForcing
    *,
    window: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """The four indices per HRU row, computed over ``window`` (default: full record)."""
    dates = forcing.dates
    if window is not None:
        sel = (dates >= pd.Timestamp(window[0])) & (dates <= pd.Timestamp(window[1]))
    else:
        sel = np.ones(len(dates), dtype=bool)
    idx = np.array([forcing.pos[k] for k in hrus["key"]], dtype=np.int64)
    prcp = forcing.prcp[idx][:, sel].astype(np.float64)
    tavg = forcing.tavg[idx][:, sel].astype(np.float64)
    doy = forcing.doy[sel].astype(np.float64)
    lat_rad = np.deg2rad(hrus["lat"].to_numpy(np.float64))

    n_years = sel.sum() / 365.25
    p_sum = prcp.sum(axis=1)
    p_mean = p_sum / n_years
    raw_pet = hamon_raw_pet_numpy(tavg, doy, lat_rad)
    aridity = raw_pet.sum(axis=1) / np.maximum(p_sum, 1e-9)
    snow_frac = (prcp * (tavg <= 0.0)).sum(axis=1) / np.maximum(p_sum, 1e-9)

    months = pd.DatetimeIndex(dates[sel]).month.to_numpy()
    pm = np.zeros((len(hrus), 12))
    for m in range(1, 13):
        pm[:, m - 1] = prcp[:, months == m].sum(axis=1)
    seasonality = np.abs(pm - p_sum[:, None] / 12.0).sum(axis=1) / np.maximum(p_sum, 1e-9)

    return pd.DataFrame({"p_mean": p_mean, "aridity": aridity,
                         "snow_frac": snow_frac, "seasonality": seasonality},
                        index=hrus.index)


def _zone_mean(sv: pd.DataFrame, prop: str, zone: dict[str, float]) -> np.ndarray:
    """Depth-weighted mean of a POLARIS property over a depth zone.  ``ksat`` is
    stored as log10(cm/hr), so its arithmetic depth-mean is the (correct)
    depth-geometric mean of conductivity."""
    num = np.zeros(len(sv))
    den = 0.0
    for depth, w in zone.items():
        num += w * sv[f"{prop}_{depth}"].to_numpy(np.float64)
        den += w
    return num / den


def physical_features(sv: pd.DataFrame) -> pd.DataFrame:
    """Derive the :data:`PHYSICAL_FEATURES` from the raw ``soilveg_continuous``
    columns: POLARIS depth-zone aggregates, LANDFIRE cover/height, 3DEP terrain,
    MODIS-LAI mean/amplitude and circular peak-timing (sin/cos of the peak DOY)."""
    out = {}
    for prop, tag in zip(_POLARIS_PROPS, ("sand", "clay", "ksat", "thetas"), strict=True):
        out[f"{tag}_surf"] = _zone_mean(sv, prop, _SURF_ZONE)
        out[f"{tag}_deep"] = _zone_mean(sv, prop, _DEEP_ZONE)
    out["evc_cover"] = sv["EVC_cover_pct"].to_numpy(np.float64)
    out["evh_height"] = sv["EVH_height_m"].to_numpy(np.float64)
    out["slope"] = sv["slope_deg"].to_numpy(np.float64)
    out["aspect_sin"] = sv["aspect_sin"].to_numpy(np.float64)
    out["aspect_cos"] = sv["aspect_cos"].to_numpy(np.float64)
    out["curvature"] = sv["curvature"].to_numpy(np.float64)
    out["relief"] = sv["relief_m"].to_numpy(np.float64)
    out["lai_mean"] = sv["lai_mean"].to_numpy(np.float64)
    out["lai_amp"] = sv["lai_amp"].to_numpy(np.float64)
    doy = sv["lai_peak_doy"].to_numpy(np.float64)
    out["lai_peak_sin"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["lai_peak_cos"] = np.cos(2.0 * np.pi * doy / 366.0)
    return pd.DataFrame(out, index=sv.index)[list(PHYSICAL_FEATURES)]


def load_physical(hrus: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Load the per-HRU soilveg table and align it to ``hrus`` by ``key``.  The
    sampled values depend only on lat/lon (= key), so shared cells map to
    identical rows; alignment is key-based (robust to any row reordering)."""
    sv = pd.read_csv(path).drop_duplicates("key").set_index("key")
    aligned = sv.reindex(hrus["key"].to_numpy())
    if aligned.isna().any().any():
        miss = int(aligned.isna().any(axis=1).sum())
        raise ValueError(f"{miss} HRU keys absent from {path} (rebuild the "
                         f"soilveg table for this domain)")
    aligned.index = hrus.index
    return aligned


def build_features(
    hrus: pd.DataFrame,
    *,
    variant: str = "static",
    forcing=None,
    climate_window: tuple[str, str] | None = None,
    climate_product: str | None = None,
    fourier_k: int = 0,
    physical_path: str | Path | None = None,
    stats: FeatureSet | None = None,
) -> FeatureSet:
    """Assemble the (N, F) matrix.  Pass a previous :class:`FeatureSet` as
    ``stats`` to reuse its categories, z-scoring, and Fourier extent
    (checkpoint evaluation)."""
    if variant not in ("static", "climate", "physical", "physical_climate"):
        raise ValueError(f"variant {variant!r}")

    cols: list[np.ndarray] = []
    names: list[str] = []
    for c in CONTINUOUS_STATICS:
        cols.append(hrus[c].to_numpy(np.float64))
        names.append(c)
    if variant in ("climate", "physical_climate"):
        if forcing is None:
            raise ValueError(f"{variant} variant needs the DomainForcing")
        ci = climate_indices(hrus, forcing, window=climate_window)
        for c in CLIMATE_INDICES:
            cols.append(ci[c].to_numpy(np.float64))
            names.append(c)
    if variant in ("physical", "physical_climate"):
        if physical_path is None:
            raise ValueError(f"{variant} variant needs physical_path "
                             "(sacsma.io.soilveg_path)")
        phys = physical_features(load_physical(hrus, physical_path))
        for c in PHYSICAL_FEATURES:
            cols.append(phys[c].to_numpy(np.float64))
            names.append(c)
    n_cont = len(names)                       # z-scored columns end here

    fk = stats.fourier_k if stats is not None else fourier_k
    bounds = None
    if fk > 0:
        lat = hrus["lat"].to_numpy(np.float64)
        lon = hrus["lon"].to_numpy(np.float64)
        if stats is not None and stats.coord_bounds is not None:
            bounds = stats.coord_bounds
        else:
            bounds = (float(lat.min()), float(lat.max()),
                      float(lon.min()), float(lon.max()))
        u = (lat - bounds[0]) / max(bounds[1] - bounds[0], 1e-9)
        v = (lon - bounds[2]) / max(bounds[3] - bounds[2], 1e-9)
        for f in range(1, fk + 1):
            for tag, w in (("lat", u), ("lon", v)):
                cols.append(np.sin(2.0 * np.pi * f * w))
                names.append(f"sin{f}_{tag}")
                cols.append(np.cos(2.0 * np.pi * f * w))
                names.append(f"cos{f}_{tag}")

    # one-hot soil/veg — the static/climate zonal encoding; the physical variants
    # REPLACE these with continuous soil/veg/terrain columns (added above).
    if variant in ("physical", "physical_climate"):
        soil_cats: list[int] = []
        veg_cats: list[int] = []
    else:
        soil_cats = (stats.soil_categories if stats is not None
                     else sorted(hrus["soil_class"].unique().tolist()))
        veg_cats = (stats.veg_categories if stats is not None
                    else sorted(hrus["veg_class"].unique().tolist()))
        for cat in soil_cats:
            cols.append((hrus["soil_class"] == cat).to_numpy(np.float64))
            names.append(f"soil_{cat}")
        for cat in veg_cats:
            cols.append((hrus["veg_class"] == cat).to_numpy(np.float64))
            names.append(f"veg_{cat}")

    x = np.stack(cols, axis=1)
    if stats is not None:
        mean, std = stats.mean, stats.std
    else:
        # only the physical/climate columns are z-scored; Fourier terms are
        # already in [-1, 1] and one-hots are left as 0/1
        mean = np.zeros(x.shape[1])
        std = np.ones(x.shape[1])
        mean[:n_cont] = x[:, :n_cont].mean(axis=0)
        std[:n_cont] = x[:, :n_cont].std(axis=0).clip(min=1e-9)
    x = (x - mean) / std

    return FeatureSet(x=x.astype(np.float32), names=names, mean=mean, std=std,
                      soil_categories=soil_cats, veg_categories=veg_cats,
                      variant=variant, climate_window=climate_window,
                      climate_product=climate_product,
                      fourier_k=fk, coord_bounds=bounds)
