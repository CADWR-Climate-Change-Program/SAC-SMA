# dataprep/ — statewide auxiliary-data program

Goal: every processed auxiliary layer the **noah_ft fine-tune** needs, at
**statewide coverage** (any California basin), **in-repo** (git-LFS), in the
most compact processed form. Decision (2026-07-16): *statewide or nothing* —
no interim domain-subset packaging; the existing 2074-cell stores stay on
`D:\sacsma-data` until their statewide replacements land here.

Unlike the retired one-off ingest scripts (git history, see
`data/INVENTORY.md`), the scripts here are **living tools**: they define how
coverage is extended and are kept working.

## The statewide grid

`build_statewide_grid.py` → `data/statewide/grid_cells.csv` (**done**):
13,786 cells of the 1/16° Livneh grid over California (from the WGEN
NonDetrend Unsplit Statewide store), keys `<lat>_<lon>` (5-decimal, the
`sacsma.io` convention). Verified a superset of every existing domain
(15cdec_grid 2074/2074, 9unimp 414/414). Every ingest below targets this list.

## Layer roadmap

| layer | script | source | statewide size (est.) | status |
|---|---|---|---|---|
| grid definition | `build_statewide_grid.py` | WGEN statewide dir (local) | 0.5 MB CSV | **done** |
| ET obs: terraclimate/fldas/era5land | `gee_obs_statewide.py` | GEE (user-run export) | ~60 MB npz | script ready — run `--verify` first, then `--products all` |
| SWE obs: daymet/terraclimate/fldas/era5land | `gee_obs_statewide.py` | GEE (user-run export) | ~80 MB npz | script ready (same run) |
| ET obs: gleam, fluxcom | TODO `local_obs_statewide.py` | `D:\sacsma-data\{gleam,fluxcom}` raw (local, 3 GB) | ~40 MB npz | ingest to rewrite; verify gate = reproduce the 2074-cell npz |
| LAI climatology + veg_frac | TODO | `D:\sacsma-data\raw_gis\lai` (local) | ~25 MB | ingest to rewrite |
| soilveg physical features | TODO | `raw_gis/{polaris,landfire,dem}` (local, 90 GB) | ~3 MB CSV | ingest to rewrite |
| per-cell Tmin/Tmax | TODO | WGEN statewide dir (local ASCII) | ~1–2 GB nc (LFS) | converter to write |
| daily forcing master | TODO | OneDrive `calsim3-stochastic-input-generation` (mount needed) | ~1.5–3 GB nc (LFS) | source mount + subset tooling |

Verification rule for EVERY statewide ingest: re-run restricted to the 2074
`15cdec_grid` cells and reproduce the existing `D:\sacsma-data` npz (rel RMS
< 1e-3) before the statewide output is trusted — the GEE script has this
built in (`--verify`).

## GEE export runbook (user-run)

```
pip install earthengine-api          # in the sacsma env
earthengine authenticate
python dataprep/gee_obs_statewide.py --verify        # must PASS all 7
python dataprep/gee_obs_statewide.py --products all  # statewide burn (hours)
```

Outputs land in `data/statewide/{et_obs,swe_obs}/*.npz` (LFS via
`data/**/*.npz`). After all 9 obs products exist statewide,
`sacsma/dpl/data.py` `ET_DIR`/`SWE_DIR` get repointed to the in-repo store
(env overrides kept) and the `D:\sacsma-data` dependency is retired.

## End state

A clone + LFS pull can fine-tune noah_ft on any California basin given only a
basin delineation + a gage/FNF target: forcing + tminmax subset from the
statewide masters, obs losses read the statewide stores, LAI/soilveg features
cover the full grid.
