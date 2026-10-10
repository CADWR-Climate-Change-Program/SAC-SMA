"""Per-HRU features for the parameter network, and the climate indices of the hybrids.

Two variants:

* ``physical`` — the continuous statics elev, lat, lon and the flow length to the
  basin outlet (:data:`CONTINUOUS_STATICS`) plus the continuous soil, vegetation,
  terrain and LAI features :data:`PHYSICAL_FEATURES`, sampled per HRU from the CA
  raster stack (see ``sacsma.io.soilveg_path``), every column z-scored;
* ``aef_u`` — AlphaEarth Foundations satellite embeddings (Google/DeepMind,
  ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``, CC-BY 4.0; the 2017-2025 mean 64-d
  vector per region cell, ``data/inputs/grid/aef_cell_mean.npz``) in place of every
  other input, so the parameter field is climate-frozen: the first :data:`AEF_PCS`
  principal components of the unit DIRECTIONS plus the mean vector's LENGTH (how
  consistent the land surface is across the cell and the years).  The PCA is fit
  once on all 4410 region cells — no training data involved — and stored in the
  :class:`FeatureSet`, so an evaluation projects exactly as the training did.  The
  PCs are NOT z-scored one by one: PCk keeps ``(sd_k / sd_1) ** AEF_PC_POWER`` of
  PC1's spread (1 = the PCA's own geometry), so the low-variance PCs, the least
  stable from one embedding year to the next, do not get PC1's input scale; the
  length is z-scored.

The climate indices (:func:`climate_indices`) are no network input; the hybrids take
them as statics (:mod:`sacsma.dpl.hybrid`).

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

from .. import paths
from .physics.pet import hamon_raw_pet_numpy

CONTINUOUS_STATICS = ("elev", "lat", "lon", "flowlen")
CLIMATE_INDICES = ("p_mean", "aridity", "snow_frac", "seasonality")

# -- physical variant: continuous soil/veg/terrain/LAI, sampled per HRU from the
#    CA raster stack (data/inputs/grid/raw_gis; see sacsma.io.soilveg_path). --
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

#: the AlphaEarth store under the data directory (data/inputs/grid/gee_aef_region.py)
AEF_PCS = 16                       # principal components of the unit directions
AEF_PC_POWER = 1.0                 # aef_u: PCk spread = (sd_k / sd_1) ** this; PC1's = 1
VARIANTS = ("physical", "aef_u")


@dataclass
class FeatureSet:
    """Feature matrix + everything needed to rebuild it identically."""

    x: np.ndarray                  # (N, F) float32, z-scored
    names: list[str]
    mean: np.ndarray               # (F,) normalization stats
    std: np.ndarray
    variant: str                   # one of VARIANTS
    #: ``aef_u``: the region-cell PCA of the unit embedding directions — mean (64,) and
    #: components (AEF_PCS, 64)
    aef_mean: np.ndarray | None = None
    aef_basis: np.ndarray | None = None


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


def aef_store(data_dir: str | Path = "data") -> Path:
    """The AlphaEarth cell-mean store under ``data_dir``."""
    return paths.alphaearth(data_dir)


def aef_pca(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """PCA of the unit embedding directions over every cell of the store:
    ``(mean (64,), components (AEF_PCS, 64))``, each component's sign fixed so
    its largest loading is positive (SVD signs are otherwise arbitrary)."""
    e = np.load(path, allow_pickle=True)["emb"].astype(np.float64)
    u = e / np.linalg.norm(e, axis=1, keepdims=True)
    mean = u.mean(axis=0)
    _, _, vt = np.linalg.svd(u - mean, full_matrices=False)
    basis = vt[:AEF_PCS].copy()
    basis *= np.sign(basis[np.arange(AEF_PCS), np.abs(basis).argmax(axis=1)])[:, None]
    return mean, basis


def aef_u_std(std: np.ndarray) -> np.ndarray:
    """``aef_u``'s input scales from the columns' standard deviations ``std``: PCk keeps
    ``(sd_k / sd_1) ** AEF_PC_POWER`` of PC1's spread; the length keeps its own."""
    s = np.array(std, dtype=np.float64)
    s[:AEF_PCS] = s[:AEF_PCS] ** (1.0 - AEF_PC_POWER) * s[0] ** AEF_PC_POWER
    return s


def aef_features(hrus: pd.DataFrame, path: str | Path, mean: np.ndarray,
                 basis: np.ndarray) -> pd.DataFrame:
    """Per HRU row: the cell's unit embedding direction projected on ``basis``
    (``aef_pc01``..) and the length of its mean vector (``aef_len``)."""
    z = np.load(path, allow_pickle=True)
    pos = {k: i for i, k in enumerate(z["keys"].astype(str))}
    keys = hrus["key"].astype(str).to_numpy()
    miss = sorted(set(keys) - pos.keys())
    if miss:
        raise ValueError(f"{len(miss)} HRU keys absent from {path} (e.g. {miss[0]}; "
                         "the embeddings exist only on the 1/16-deg region grid)")
    e = z["emb"].astype(np.float64)[[pos[k] for k in keys]]
    n = np.linalg.norm(e, axis=1)
    pcs = (e / n[:, None] - mean) @ basis.T
    out = {f"aef_pc{i + 1:02d}": pcs[:, i] for i in range(basis.shape[0])}
    out["aef_len"] = n
    return pd.DataFrame(out, index=hrus.index)


def build_features(
    hrus: pd.DataFrame,
    *,
    variant: str,
    physical_path: str | Path | None = None,
    stats: FeatureSet | None = None,
    aef_path: str | Path | None = None,
) -> FeatureSet:
    """Assemble the (N, F) matrix.  Pass a previous :class:`FeatureSet` as
    ``stats`` to reuse its z-scoring and embedding projection (checkpoint
    evaluation)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}")

    cols: list[np.ndarray] = []
    names: list[str] = []
    if variant == "physical":
        for c in CONTINUOUS_STATICS:
            cols.append(hrus[c].to_numpy(np.float64))
            names.append(c)
        if physical_path is None:
            raise ValueError(f"{variant} variant needs physical_path "
                             "(sacsma.io.soilveg_path)")
        phys = physical_features(load_physical(hrus, physical_path))
        for c in PHYSICAL_FEATURES:
            cols.append(phys[c].to_numpy(np.float64))
            names.append(c)
    a_mean = a_basis = None
    if variant == "aef_u":
        if aef_path is None:
            raise ValueError(f"{variant} variant needs aef_path (features.aef_store)")
        if stats is not None:
            a_mean, a_basis = np.asarray(stats.aef_mean), np.asarray(stats.aef_basis)
        else:
            a_mean, a_basis = aef_pca(aef_path)
        emb = aef_features(hrus, aef_path, a_mean, a_basis)
        for c in emb.columns:
            cols.append(emb[c].to_numpy(np.float64))
            names.append(c)

    x = np.stack(cols, axis=1)
    if stats is not None:
        mean, std = stats.mean, stats.std
    else:
        mean = x.mean(axis=0)
        std = x.std(axis=0).clip(min=1e-9)
        if variant == "aef_u":
            std = aef_u_std(std)          # in std, so evaluation repeats it
    x = (x - mean) / std

    return FeatureSet(x=x.astype(np.float32), names=names, mean=mean, std=std,
                      variant=variant, aef_mean=a_mean, aef_basis=a_basis)
