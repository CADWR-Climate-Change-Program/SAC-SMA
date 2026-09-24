"""Per-HRU features for the parameter network: physical statics + climate indices.

Two variants (the study's ablation):

* ``static`` — physical statics only: elev, lat, lon, flowlen (z-scored; the
  flow length is optional, ``DplConfig.flowlen_feature``) plus one-hot
  ``soil_class`` / ``veg_class`` (the same information class the GA's soil/veg
  zonal regionalization used);
* ``climate`` — statics PLUS forcing-derived climatology, computed per HRU
  from its grid cell over a configurable ``(product, window)``.

Two embedding variants replace EVERY other input (statics, climate indices,
soil/veg, the one-hots), so their parameter field is climate-frozen:

* ``aef`` — AlphaEarth Foundations satellite embeddings (Google/DeepMind,
  ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``, CC-BY 4.0; the 2017-2025 mean 64-d
  vector per region cell, ``data/region/aef/aef_cell_mean.npz``): the first
  :data:`AEF_PCS` principal components of the unit DIRECTIONS plus the mean
  vector's LENGTH (how consistent the land surface is across the cell and the
  years).  The PCA is fit once on all 4410 region cells — no training data
  involved — and stored in the :class:`FeatureSet`, so an evaluation projects
  exactly as the training did;
* ``aef_random`` — its control: :data:`AEF_PCS` + 1 standard-normal values per
  cell, seeded by the cell key.  They carry the cell's identity and nothing
  about its land surface, so a gain the ``aef`` arm shares with this control is
  cell identity, not embedding content.

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

import zlib

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

#: the AlphaEarth store under the data directory (dataprep/gee_aef_region.py)
AEF_STORE = Path("region") / "aef" / "aef_cell_mean.npz"
AEF_PCS = 16                       # principal components of the unit directions
AEF_VARIANTS = ("aef", "aef_random")
VARIANTS = ("static", "climate", "physical", "physical_climate") + AEF_VARIANTS


@dataclass
class FeatureSet:
    """Feature matrix + everything needed to rebuild it identically."""

    x: np.ndarray                  # (N, F) float32, z-scored continuous cols
    names: list[str]
    mean: np.ndarray               # (F,) normalization stats (0/1 for one-hots)
    std: np.ndarray
    soil_categories: list[int]
    veg_categories: list[int]
    variant: str                   # one of VARIANTS
    climate_window: tuple[str, str] | None
    climate_product: str | None
    fourier_k: int = 0             # spatial Fourier order (0 = off)
    #: (lat_min, lat_max, lon_min, lon_max) normalization extent for the
    #: Fourier terms (None when fourier_k == 0)
    coord_bounds: tuple[float, float, float, float] | None = None
    #: the continuous statics the matrix opens with (a checkpoint without the
    #: field, from before ``flowlen_feature``, carries the full tuple)
    statics: tuple[str, ...] = CONTINUOUS_STATICS
    #: ``aef``: the region-cell PCA of the unit embedding directions — mean
    #: (64,) and components (AEF_PCS, 64)
    aef_mean: np.ndarray | None = None
    aef_basis: np.ndarray | None = None
    #: ``aef_random``: the seed of the per-cell random vectors
    aef_seed: int = 0


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
    return Path(data_dir) / AEF_STORE


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


def random_features(hrus: pd.DataFrame, seed: int, dims: int) -> pd.DataFrame:
    """The ``aef_random`` control: ``dims`` standard-normal values per HRU row,
    drawn from a generator seeded by ``(seed, crc32(key))`` — the same cell gets
    the same vector in every domain and caller, whatever the row order."""
    keys = hrus["key"].astype(str).to_numpy()
    per_key = {k: np.random.default_rng([seed, zlib.crc32(k.encode())]).standard_normal(dims)
               for k in dict.fromkeys(keys)}
    v = np.stack([per_key[k] for k in keys])
    return pd.DataFrame(v, columns=[f"rand{i + 1:02d}" for i in range(dims)],
                        index=hrus.index)


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
    statics: tuple[str, ...] | None = None,
    aef_path: str | Path | None = None,
    aef_seed: int = 0,
) -> FeatureSet:
    """Assemble the (N, F) matrix.  Pass a previous :class:`FeatureSet` as
    ``stats`` to reuse its categories, z-scoring, Fourier extent, statics and
    embedding projection (checkpoint evaluation).  ``statics`` (default
    :data:`CONTINUOUS_STATICS`) picks the continuous statics of a new set, e.g.
    without ``flowlen``; the embedding variants take none."""
    if variant not in VARIANTS:
        raise ValueError(f"variant {variant!r}")
    embed = variant in AEF_VARIANTS

    if stats is not None:
        st = tuple(getattr(stats, "statics", CONTINUOUS_STATICS))
    else:
        st = tuple(statics) if statics is not None else CONTINUOUS_STATICS
    if embed:
        st = ()                                   # the embeddings replace everything
    cols: list[np.ndarray] = []
    names: list[str] = []
    for c in st:
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
    a_mean = a_basis = None
    a_seed = stats.aef_seed if stats is not None else aef_seed
    if variant == "aef":
        if aef_path is None:
            raise ValueError("aef variant needs aef_path (features.aef_store)")
        if stats is not None:
            a_mean, a_basis = np.asarray(stats.aef_mean), np.asarray(stats.aef_basis)
        else:
            a_mean, a_basis = aef_pca(aef_path)
        emb = aef_features(hrus, aef_path, a_mean, a_basis)
    elif variant == "aef_random":
        emb = random_features(hrus, a_seed, AEF_PCS + 1)
    if embed:
        for c in emb.columns:
            cols.append(emb[c].to_numpy(np.float64))
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
    # REPLACE these with continuous soil/veg/terrain columns (added above), the
    # embedding variants with the embeddings.
    if variant in ("physical", "physical_climate") or embed:
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
                      fourier_k=fk, coord_bounds=bounds, statics=st,
                      aef_mean=a_mean, aef_basis=a_basis, aef_seed=a_seed)
