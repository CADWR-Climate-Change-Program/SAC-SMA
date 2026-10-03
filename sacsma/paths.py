"""Where every file under ``data/`` and ``artifacts/`` lives.

This is the one module that knows the layout of the data store and of the output tree.
Everything else in the package, and every build script under ``data/``, asks it for a path,
so a layout can change here and nowhere else.  It uses the standard library only: the build
scripts load it by file location, without importing the package.

The store has three parts (see ``data/README.md``):

* ``inputs/``     what a run reads to produce flow: forcing, the grid and its static
  attributes, the modeling domains, the CalSim3 geometry and mappings;
* ``targets/``    flow records a model is fitted to (all from outside this repository);
* ``reference/``  series that results are only compared with (all from outside).

Nothing in the package writes under ``targets/`` or ``reference/``: those folders change only
through the ingest script that sits in them.

Every function takes the store root (``data_dir``, default ``"data"``) and returns a
:class:`pathlib.Path`; none of them touches the file system.

The output tree is described at its own section below.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

CDEC15 = "15cdec"
CDEC15_GRID = "15cdec_grid"
CALLITE = ("9unimp", "11obs", "12rim")
MULTIFAMILY = "multifamily"
#: domains whose units sit on the 1/16-degree grid and read the shared forcing stores
GRID_DOMAINS = (CDEC15_GRID, *CALLITE, MULTIFAMILY)
DOMAINS = (CDEC15, CDEC15_GRID, *CALLITE, MULTIFAMILY)

# ----------------------------------------------------------------------------- the layout
#: folder of each part, relative to the store root
FORCING = "inputs/forcing"
GRID = "inputs/grid"
DOMAIN_ROOT = "inputs/domains"
CALSIM3_INPUTS = "inputs/calsim3"
T_CDEC = "targets/cdec"
T_CALLITE = "targets/callite"
T_DWR = "targets/dwr_unimpaired"
T_USGS = "targets/usgs"
T_CALSIM3 = "targets/calsim3"
R_MATLAB = "reference/matlab"
R_VIC = "reference/vic"
R_BCM = "reference/bcm"
R_ET = "reference/et"
R_SWE = "reference/swe"
R_DWR_SWAT = "reference/dwr_swat"


def _root(data_dir) -> Path:
    return Path("data" if data_dir is None else data_dir)


def _check(domain: str) -> str:
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r} (expected one of {DOMAINS})")
    return domain


# ------------------------------------------------------------------------ inputs: forcing
def forcing_dir(data_dir="data") -> Path:
    """Folder of the forcing stores (one NetCDF per product)."""
    return _root(data_dir) / FORCING


def forcing(data_dir="data", domain: str = CDEC15, product: str = "historical_livneh_unsplit") -> Path:
    """Forcing store of a domain: the shared 1/16-degree store of the product, or, for the
    fine 15cdec domain, the dense store of its off-grid HRU points."""
    if _check(domain) in GRID_DOMAINS:
        return forcing_dir(data_dir) / f"{product}.nc"
    return forcing_dir(data_dir) / f"{product}_{domain}_hru.nc"


def wgen_scenario_key(data_dir="data") -> Path:
    return _root(data_dir) / FORCING / "wgen_product_a_scenarios.csv"


def prcp_x10_artifacts(data_dir="data") -> Path:
    return _root(data_dir) / FORCING / "prcp_x10_artifacts.csv"


# --------------------------------------------------------------------------- inputs: grid
def grid_dir(data_dir="data") -> Path:
    """Folder of the 1/16-degree grid: the cell list and the per-cell static attributes."""
    return _root(data_dir) / GRID


def grid_cells(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "grid_cells.csv"


def soilveg(data_dir="data", domain: str = CDEC15) -> Path:
    """Continuous soil / vegetation / terrain attributes: per HRU for a calibration domain,
    per grid cell (all 4,410) for the multifamily domain."""
    if domain == MULTIFAMILY:
        return _root(data_dir) / GRID / "soilveg_continuous.csv"
    return domain_dir(data_dir, domain) / "soilveg_continuous.csv"


def lai_climatology(data_dir="data", domain: str = CDEC15) -> Path:
    """46-value 8-day leaf-area climatology, companion of :func:`soilveg`."""
    if domain == MULTIFAMILY:
        return _root(data_dir) / GRID / "lai_climatology.csv"
    return domain_dir(data_dir, domain) / "lai_climatology.csv"


def raster_sample(data_dir="data", stem: str = "soilveg_continuous") -> Path:
    """Every grid cell sampled from the raw rasters (``<stem>_raster.csv``; not tracked: the
    tracked per-cell tables are built from it)."""
    return _root(data_dir) / GRID / f"{stem}_raster.csv"


def alphaearth(data_dir="data") -> Path:
    return _root(data_dir) / GRID / "aef_cell_mean.npz"


def raw_gis(data_dir="data") -> Path:
    """Default stage of the raw rasters the grid attributes are sampled from (not tracked;
    usually kept on another drive, see ``data/local_paths.example.toml``)."""
    return _root(data_dir) / GRID / "raw_gis"


# ------------------------------------------------------------------------ inputs: domains
def domain_dir(data_dir="data", domain: str = CDEC15) -> Path:
    return _root(data_dir) / DOMAIN_ROOT / _check(domain)


def hruinfo(data_dir="data", domain: str = CDEC15) -> Path:
    return domain_dir(data_dir, domain) / "hruinfo.csv"


def basin_area(data_dir="data", domain: str = CDEC15) -> Path:
    return domain_dir(data_dir, domain) / "basin_area.csv"


def basin_tminmax(data_dir="data") -> Path:
    """Watershed-mean daily Tmin / Tmax of the 15 CDEC watersheds."""
    return domain_dir(data_dir, CDEC15) / "basin_tminmax_livneh.csv"


def footprints_15cdec(data_dir="data") -> Path:
    return domain_dir(data_dir, CDEC15) / "SACSMA_15CDEC.geojson"


def entities(data_dir="data") -> Path:
    return domain_dir(data_dir, MULTIFAMILY) / "entities.csv"


def entity_cells(data_dir="data") -> Path:
    return domain_dir(data_dir, MULTIFAMILY) / "entity_cells.csv"


def flowlens(data_dir="data") -> Path:
    return domain_dir(data_dir, MULTIFAMILY) / "flowlens.csv"


# ------------------------------------------------- inputs: CalSim3 geometry and mappings
def calsim3_gpkg(data_dir="data", name: str = "calsim3.gpkg") -> Path:
    return _root(data_dir) / CALSIM3_INPUTS / name


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
    """Daily CDEC full natural flow of the 15 CDEC watersheds (the calibration target)."""
    return _root(data_dir) / T_CDEC / "gage_15cdec.csv"


def cdec_fnf(data_dir="data", name: str = "fnf_daily_mm.csv") -> Path:
    """A file of the CDEC daily full-natural-flow store: ``stations.csv``,
    ``fnf_daily.csv`` (cfs), ``fnf_daily_mm.csv``, ``fnf_daily_mask.csv``."""
    return _root(data_dir) / T_CDEC / name


def callite_calib(data_dir="data", domain: str = "11obs") -> Path:
    return _root(data_dir) / T_CALLITE / f"calib_{domain}_monthly.csv"


def callite_fnf(data_dir="data", domain: str = "11obs") -> Path:
    return _root(data_dir) / T_CALLITE / f"fnf_{domain}_monthly.csv"


def dwr_unimpaired(data_dir="data", name: str = "uf_monthly_mm.csv") -> Path:
    """A file of the DWR unimpaired-flow store: ``uf_monthly.csv`` (TAF),
    ``uf_monthly_mm.csv``, ``uf_locations.csv``, ``uf_gauges.csv``."""
    return _root(data_dir) / T_DWR / name


def usgs_flow(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "flow_daily.nc"


def usgs_gauges(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "gauges.csv"


def usgs_watersheds(data_dir="data") -> Path:
    return _root(data_dir) / T_USGS / "usgs_watersheds.gpkg"


def calsim3_targets(data_dir="data", name: str = "calsim3_inflow_monthly.csv") -> Path:
    """A file of the CalSim3 series store: ``calsim3_inflow_monthly.csv`` (TAF),
    ``calsim3_inflow_monthly_mm.csv``, ``arc_obs_mask.csv``,
    ``calsim_unimpaired_monthly.csv``, ``arc_hierarchy.csv``, ``calsim3_arc_derivation.csv``."""
    return _root(data_dir) / T_CALSIM3 / name


# ------------------------------------------------------------------------------ reference
def simflow(data_dir="data", domain: str = CDEC15) -> Path:
    """The archived MATLAB simulation of a calibration domain (the parity baseline)."""
    return _root(data_dir) / R_MATLAB / f"simflow_{_check(domain)}.csv"


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


# ============================================================================ artifacts/
# The output tree (see ``artifacts/README.md``) has four parts:
#
# * ``product/``  what is delivered, one folder per application: ``calsim3`` (the rim-inflow
#   product of the learned model), ``callite`` and ``15cdec`` (the calibrated models), each
#   with one folder per forcing;
# * ``models/``   every model: the archived calibrations (``callite/<set>``, ``15cdec``,
#   ``15cdec_grid``) and what training made (``dpl/``), with the parameter tables;
# * ``results/``  what a command redraws: the calibrated sets (``callite/<set>``, ``15cdec``)
#   and their comparisons (``calsim3``, ``vic_bcm``, ``footprints``, ``forcing``), and the
#   learned runs (``dpl/``);
# * ``_local/``   not tracked: caches, scratch, the large outputs of a run, runs not adopted.
#
# A learned-parameter run has one folder per role (:func:`run_roles`), each under the same
# name: ``models/dpl/<group>/<run>``, ``results/dpl/<group>/<run>`` and
# ``_local/runs/dpl/<group>/<run>``, where ``<group>`` is ``15cdec`` (both 15-CDEC domains) or
# ``multifamily``.  Every function takes the output root (``artifacts_dir``, default
# ``"artifacts"``) and touches no file, except :func:`run_roles` and :func:`run_file`, which
# look for a run's model folder.

#: the folder that groups the three CalLite calibration sets in models/, results/, product/
CALLITE_DIR = "callite"
#: the calibrated model's result folders: one per set, then the comparisons
CALIBRATED = (CDEC15, *CALLITE, "calsim3", "vic_bcm", "footprints", "forcing")
#: the delivered products, one folder each under product/
PRODUCTS = ("calsim3", CALLITE_DIR, CDEC15)
#: the ``sacsma dpl study`` result folders
DPL_STUDIES = ("climatology", "hybrids", "adaptive", "forcing")
#: the roles of a learned-parameter run
ROLES = ("models", "results", "local")


class RunDirs(NamedTuple):
    """The folders of one learned-parameter run, by role."""

    model: Path
    results: Path
    local: Path


def _art(artifacts_dir) -> Path:
    return Path("artifacts" if artifacts_dir is None else artifacts_dir)


def _group(domain: str) -> str:
    """The folder of a domain's runs: the two 15-CDEC domains share one."""
    return MULTIFAMILY if _check(domain) == MULTIFAMILY else CDEC15


def _app(name: str) -> Path:
    """A calibration set's folder under an application: ``callite/<set>`` for the CalLite
    sets, the name itself otherwise."""
    return Path(CALLITE_DIR, name) if name in CALLITE else Path(name)


def calibrated_model(artifacts_dir="artifacts", domain: str = CDEC15) -> Path:
    """The archived GA calibration of a domain: ``models/callite/<set>``, ``models/15cdec``,
    ``models/15cdec_grid`` (``ga_optimum.csv``)."""
    if _check(domain) == MULTIFAMILY:
        raise ValueError("the multifamily domain has no calibration")
    return _art(artifacts_dir) / "models" / _app(domain)


def ga_optimum(artifacts_dir="artifacts", domain: str = CDEC15) -> Path:
    """The GA optimum of a domain: 31 parameters per modeling unit."""
    return calibrated_model(artifacts_dir, domain) / "ga_optimum.csv"


def calibrated(artifacts_dir="artifacts", name: str = CDEC15) -> Path:
    """Results of the calibrated model: one set's diagnostics (``results/15cdec``,
    ``results/callite/<set>``), the comparison with CalSim3 (``calsim3``), with VIC and BCM
    (``vic_bcm``), the footprint and HRU-attribute maps (``footprints``), the forcing
    comparison (``forcing``)."""
    if name not in CALIBRATED:
        raise ValueError(f"unknown result folder {name!r} (expected one of {CALIBRATED})")
    return _art(artifacts_dir) / "results" / _app(name)


def forcing_run(artifacts_dir="artifacts", product: str = "wgen_product_a") -> Path:
    """Daily flow of the calibrated sets under another forcing product
    (``sim_daily_<domain>.csv``), written and read by the forcing comparison."""
    return calibrated(artifacts_dir, "forcing") / product


def dpl_run(artifacts_dir="artifacts", run: str = "noah", domain: str = CDEC15,
            role: str = "models") -> Path:
    """One role's folder of a learned-parameter run: ``models``, ``results`` or ``local``."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r} (expected one of {ROLES})")
    sub = Path("dpl") / _group(domain) / run
    return _art(artifacts_dir) / ("_local/runs" if role == "local" else role) / sub


def dpl_metrics(artifacts_dir="artifacts", run: str = "noah", domain: str = CDEC15) -> Path:
    """The score table of a learned-parameter run."""
    return dpl_run(artifacts_dir, run, domain, role="results") / "metrics.csv"


def run_roles(run_dir, artifacts_dir="artifacts") -> RunDirs:
    """The three folders of the run that ``run_dir`` names: any one of them, or a folder
    inside one (a seed of an ensemble, ``hybrid/seed0``, gives the seed's folder in each
    role).  A run kept only under ``_local/`` (no model folder) and a folder outside the
    layout hold all three roles themselves."""
    p = Path(run_dir)
    root = _art(artifacts_dir).resolve()
    try:
        parts = p.resolve().relative_to(root).parts
    except ValueError:
        return RunDirs(p, p, p)
    in_local = parts[:2] == ("_local", "runs")
    if in_local:
        parts = parts[2:]
    elif parts[:1] in (("models",), ("results",)):
        parts = parts[1:]
    else:
        return RunDirs(p, p, p)
    if len(parts) < 3 or parts[0] != "dpl" or parts[1] not in (CDEC15, MULTIFAMILY):
        return RunDirs(p, p, p)
    group, run, rest = parts[1], parts[2], parts[3:]
    if in_local and not dpl_run(artifacts_dir, run, group).is_dir():   # a run kept in _local/ only
        return RunDirs(p, p, p)
    return RunDirs(*(dpl_run(artifacts_dir, run, group, role).joinpath(*rest) for role in ROLES))


def run_file(recorded, name: str, artifacts_dir="artifacts") -> Path | None:
    """A file of a learned-parameter run that a checkpoint recorded when it was trained (the
    hybrids record their physics run's tables): the path as recorded while its folder exists,
    else the file ``name`` of the run named by that folder, in the folder of its role (the
    daily simulation ``sim_daily.csv`` in the results, the parameter tables in the model
    folder).  None stays None; a path that names no run is returned as it is."""
    if recorded is None or str(recorded) == "":
        return None
    p = Path(recorded)
    if p.parent.is_dir():
        return p
    role = "results" if name.startswith("sim_daily") else "models"
    for group in (CDEC15, MULTIFAMILY):
        if dpl_run(artifacts_dir, p.parent.name, group).is_dir():
            return dpl_run(artifacts_dir, p.parent.name, group, role) / name
    return p


def dpl_study(artifacts_dir="artifacts", name: str = "climatology") -> Path:
    """Results of one ``sacsma dpl study``."""
    if name not in DPL_STUDIES:
        raise ValueError(f"unknown study {name!r} (expected one of {DPL_STUDIES})")
    return _art(artifacts_dir) / "results" / "dpl" / CDEC15 / "studies" / name


def dpl_benchmark(artifacts_dir="artifacts") -> Path:
    """The differentiable model against the reference model (``sacsma dpl benchmark``)."""
    return _art(artifacts_dir) / "results" / "dpl" / CDEC15 / "benchmark"


def product(artifacts_dir="artifacts", forcing: str | None = None, app: str = "calsim3") -> Path:
    """One product: ``calsim3`` (the rim-inflow product: the share model and its fit records,
    or one forcing's series with the tier-2 pass it was applied to), ``callite`` or
    ``15cdec`` (the calibrated models' series), or its folder for one forcing."""
    if app not in PRODUCTS:
        raise ValueError(f"unknown product {app!r} (expected one of {PRODUCTS})")
    root = _art(artifacts_dir) / "product" / app
    return root / forcing if forcing else root


def local(artifacts_dir="artifacts", name: str = "") -> Path:
    """A folder under the untracked part: ``cache/<what>``, ``testing/<run>``, ``eval``,
    ``product/calsim3/<forcing>`` (the large files of a product pass)."""
    root = _art(artifacts_dir) / "_local"
    return root / name if name else root


# ---------------------------------------------------------------------------------- misc
def relative(path, data_dir="data") -> str:
    """``path`` relative to the store root, with forward slashes (how the entity registry
    names an observation store)."""
    return Path(path).relative_to(_root(data_dir)).as_posix()
