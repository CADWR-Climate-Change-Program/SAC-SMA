"""Where every file under ``data/`` lives.

This is the one module that knows the layout of the data store.  Everything else in the
package, and every build script under ``data/``, asks it for a path, so the layout can
change here and nowhere else.  It uses the standard library only: the build scripts load it
by file location, without importing the package.

The store has three parts (see ``data/README.md``):

* ``inputs/``     what a run reads to produce flow: forcing, the grid and its static
  attributes, the modeling domains, the CalSim3 geometry and mappings;
* ``targets/``    flow records a model is fitted to (all from outside this repository);
* ``reference/``  series that results are only compared with (all from outside).

Every function takes the store root (``data_dir``, default ``"data"``) and returns a
:class:`pathlib.Path`; none of them touches the file system.
"""

from __future__ import annotations

from pathlib import Path

CDEC15 = "15cdec"
CDEC15_GRID = "15cdec_grid"
CALLITE = ("9unimp", "11obs", "12rim")
MULTIFAMILY = "multifamily"
#: domains whose units sit on the 1/16-degree grid and read the shared forcing stores
GRID_DOMAINS = (CDEC15_GRID, *CALLITE, MULTIFAMILY)
DOMAINS = (CDEC15, CDEC15_GRID, *CALLITE, MULTIFAMILY)

# ----------------------------------------------------------------------------- the layout
#: folder of each part, relative to the store root
FORCING = "region/forcing"
GRID = "region"
CALSIM3_INPUTS = "calsim"
T_CDEC = "cdec_fnf"
T_CALLITE = "calsim"
T_DWR = "dwr_unimpaired"
T_USGS = "usgs"
T_CALSIM3 = "calsim"
R_MATLAB = None            # the MATLAB simulations sit in the domain folders
R_VIC = "calsim"
R_BCM = "region/bcm"
R_ET = "region/et_obs"
R_SWE = "region/swe_obs"
R_DWR_SWAT = "dwr_unimpaired"


def _domain(domain: str) -> tuple[str, str]:
    """(folder, file-name suffix) of a modeling domain."""
    if domain == CDEC15:
        return "cdec15", ""
    if domain == CDEC15_GRID:
        return "cdec15_grid", ""
    if domain in CALLITE:
        return "calsim", f"_{domain}"
    if domain == MULTIFAMILY:
        return "multifamily", ""
    raise ValueError(f"unknown domain {domain!r} (expected one of {DOMAINS})")


def _root(data_dir) -> Path:
    return Path("data" if data_dir is None else data_dir)


# ------------------------------------------------------------------------ inputs: forcing
def forcing_dir(data_dir="data") -> Path:
    """Folder of the 1/16-degree forcing stores (one NetCDF per product)."""
    return _root(data_dir) / FORCING


def forcing(data_dir="data", domain: str = CDEC15, product: str = "historical_livneh_unsplit") -> Path:
    """Forcing store of a domain: the shared grid store, or the dense store of the
    15cdec HRU points."""
    if domain in GRID_DOMAINS:
        return forcing_dir(data_dir) / f"{product}.nc"
    folder, sfx = _domain(domain)
    return _root(data_dir) / folder / "forcing" / f"{product}{sfx}.nc"


def wgen_scenario_key(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "wgen_product_a_scenarios.csv"


def prcp_x10_artifacts(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "prcp_x10_artifacts.csv"


# --------------------------------------------------------------------------- inputs: grid
def grid_cells(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "grid_cells.csv"


def soilveg(data_dir="data", domain: str = CDEC15) -> Path:
    """Continuous soil / vegetation / terrain attributes: per HRU for a calibration domain,
    per grid cell (all 4,410) for the multifamily domain."""
    if domain == MULTIFAMILY:
        return _root(data_dir) / GRID / "soilveg_continuous.csv"
    folder, sfx = _domain(domain)
    return _root(data_dir) / folder / f"soilveg_continuous{sfx}.csv"


def lai_climatology(data_dir="data", domain: str = CDEC15) -> Path:
    """46-value 8-day leaf-area climatology, companion of :func:`soilveg`."""
    if domain == MULTIFAMILY:
        return _root(data_dir) / GRID / "lai_climatology.csv"
    folder, sfx = _domain(domain)
    return _root(data_dir) / folder / f"lai_climatology{sfx}.csv"


def alphaearth(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "aef" / "aef_cell_mean.npz"


# ------------------------------------------------------------------------ inputs: domains
def domain_dir(data_dir="data", domain: str = CDEC15) -> Path:
    return _root(data_dir) / _domain(domain)[0]


def _domain_file(data_dir, domain: str, stem: str) -> Path:
    folder, sfx = _domain(domain)
    return _root(data_dir) / folder / f"{stem}{sfx}.csv"


def hruinfo(data_dir="data", domain: str = CDEC15) -> Path:
    return _domain_file(data_dir, domain, "hruinfo")


def ga_optimum(data_dir="data", domain: str = CDEC15) -> Path:
    return _domain_file(data_dir, domain, "ga_optimum")


def basin_area(data_dir="data", domain: str = CDEC15) -> Path:
    return _domain_file(data_dir, domain, "basin_area")


def basin_tminmax(data_dir="data") -> Path:
    """Watershed-mean daily Tmin / Tmax of the 15 CDEC watersheds."""
    return _root(data_dir) / _domain(CDEC15)[0] / "basin_tminmax_livneh.csv"


def footprints_15cdec(data_dir="data") -> Path:
    return _root(data_dir) / _domain(CDEC15)[0] / "gis" / "SACSMA_15CDEC.geojson"


def entities(data_dir="data") -> Path:
    return _root(data_dir) / _domain(MULTIFAMILY)[0] / "entities.csv"


def entity_cells(data_dir="data") -> Path:
    return _root(data_dir) / _domain(MULTIFAMILY)[0] / "entity_cells.csv"


def flowlens(data_dir="data") -> Path:
    return _root(data_dir) / _domain(MULTIFAMILY)[0] / "flowlens.csv"


# ------------------------------------------------- inputs: CalSim3 geometry and mappings
def calsim3_gpkg(data_dir="data", name: str = "calsim3.gpkg") -> Path:
    return _root(data_dir) / CALSIM3_INPUTS / "gis" / name


def crosswalk(data_dir="data") -> Path:
    return _root(data_dir) / CALSIM3_INPUTS / "calsim_crosswalk.csv"


def calsim_basin_area(data_dir="data", domain: str = "11obs") -> Path:
    """A set's watershed areas on the CalSim3 catchments."""
    return _root(data_dir) / CALSIM3_INPUTS / f"basin_area_{domain}_calsim.csv"


def screened_footprint(data_dir="data", domain: str = "11obs") -> Path:
    return _root(data_dir) / CALSIM3_INPUTS / f"screened_footprint_{domain}.csv"


def tier1_sets(data_dir="data") -> Path:
    return _root(data_dir) / CALSIM3_INPUTS / "tier1_sets.csv"


# -------------------------------------------------------------------------------- targets
def gage_15cdec(data_dir="data") -> Path:
    """Daily CDEC full natural flow of the 15 CDEC watersheds."""
    return _root(data_dir) / "cdec15" / "gage.csv"


def callite_calib(data_dir="data", domain: str = "11obs") -> Path:
    return _root(data_dir) / T_CALLITE / f"calib_{domain}_monthly.csv"


def callite_fnf(data_dir="data", domain: str = "11obs") -> Path:
    return _root(data_dir) / T_CALLITE / f"fnf_{domain}_monthly.csv"


def cdec_fnf(data_dir="data", name: str = "fnf_daily_mm.csv") -> Path:
    """A file of the CDEC daily full-natural-flow store: ``stations.csv``,
    ``fnf_daily.csv`` (cfs), ``fnf_daily_mm.csv``, ``fnf_daily_mask.csv``."""
    return _root(data_dir) / T_CDEC / name


def dwr_unimpaired(data_dir="data", name: str = "uf_monthly_mm.csv") -> Path:
    """A file of the DWR unimpaired-flow store: ``uf_monthly.csv`` (TAF),
    ``uf_monthly_mm.csv``, ``uf_locations.csv``, ``uf_gauges.csv``."""
    return _root(data_dir) / T_DWR / name


def usgs_flow(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "flow_daily.nc"


def usgs_gauges(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "gauges.csv"


def usgs_watersheds(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "gis" / "usgs_watersheds.gpkg"


def calsim3_targets(data_dir="data", name: str = "calsim3_inflow_monthly.csv") -> Path:
    """A file of the CalSim3 series store: ``calsim3_inflow_monthly.csv`` (TAF),
    ``calsim3_inflow_monthly_mm.csv``, ``arc_obs_mask.csv``,
    ``calsim_unimpaired_monthly.csv``, ``arc_hierarchy.csv``, ``calsim3_arc_derivation.csv``."""
    return _root(data_dir) / T_CALSIM3 / name


# ------------------------------------------------------------------------------ reference
def simflow(data_dir="data", domain: str = CDEC15) -> Path:
    """The archived MATLAB simulation of a calibration domain (the parity baseline)."""
    return _domain_file(data_dir, domain, "simflow")


def vic_routed(data_dir="data", product: str | None = None) -> Path:
    sfx = f"_{product}" if product else ""
    return _root(data_dir) / R_VIC / f"vic_routed_monthly{sfx}.csv"


def vic_gridinfo(data_dir="data", node: str = "I_SHSTA", variant: str = "") -> Path:
    return _root(data_dir) / R_VIC / f"vic_gridinfo_{node}{variant}.csv"


def bcm(data_dir="data", name: str = "bcm_s01_catchments_monthly.csv") -> Path:
    return _root(data_dir) / R_BCM / name


def et_obs(data_dir="data", name: str = "") -> Path:
    return _root(data_dir) / R_ET / name if name else _root(data_dir) / R_ET


def swe_obs(data_dir="data", name: str = "") -> Path:
    return _root(data_dir) / R_SWE / name if name else _root(data_dir) / R_SWE


def dwr_swat(data_dir="data", name: str = "swat_monthly.csv") -> Path:
    return _root(data_dir) / R_DWR_SWAT / name


# ---------------------------------------------------------------------------------- misc
def relative(path, data_dir="data") -> str:
    """``path`` relative to the store root, with forward slashes (how the entity registry
    names an observation store)."""
    return Path(path).relative_to(_root(data_dir)).as_posix()
