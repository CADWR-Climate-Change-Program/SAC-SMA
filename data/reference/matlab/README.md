# data/reference/matlab: the archived MATLAB simulations

The simulated daily flow of the original MATLAB SAC-SMA model of Sungwook Wi and Scott
Steinschneider (Cornell / UMass Amherst; California DWR watershed studies), one file per
calibration domain. Role: reference. These files are the parity baseline: the Python model must
reproduce them, and that agreement is the regression baseline for every change.

## Files

| File | What | Source |
|---|---|---|
| `simflow_15cdec.csv` | `[date, basin, flow]`, mm/day, daily 1915–2018, the 15 CDEC watersheds (15 MB) | the archived MATLAB study materials |
| `simflow_9unimp.csv`, `simflow_11obs.csv`, `simflow_12rim.csv` | The same table for the 32 CalLite watersheds (10–12 MB each) | the archived MATLAB study materials |

## How it is built

Delivered; cannot be rebuilt from a clone. The files were converted once from the archived
MATLAB study materials. The one-time ingest scripts are in git history (commit `ad89558` and
earlier).

## Checks

`sacsma verify parity` runs one watershed per domain and requires KGE > 0.9999 and a largest
daily difference < 0.1 mm/day against these files. Measured:

| Domain | Watershed | KGE | Largest daily difference (mm/day) |
|---|---|---|---|
| `15cdec` | BND | 0.999997 | 0.0009 |
| `9unimp` | CacheCreek | 0.999989 | 0.0394 |
| `11obs` | SHA | 0.999952 | 0.0278 |
| `12rim` | SHAST | 0.999908 | 0.0629 |

## Know before using

- These are model simulations, not observations. The observed targets are under `data/targets/`.
- `flow` is area-normalized (mm/day). `basin` holds each domain's own watershed codes (for
  example `BND`, `CacheCreek`, `SHA`, `SHAST`).
- The baseline is the default run: historical Livneh-unsplit forcing and the reference cold
  start (no `--spinup-years`).
- Run the parity check after any change that touches the model or the path the data takes into
  it. The physics modules are not edited without it.
- The three CalLite domains differ more than `15cdec` because the shared forcing store carries
  full float32 precision, where the forcing first stored for those domains was rounded to
  3 decimals.
- The MATLAB monthly simulation printed in the calibration logs is a separate table,
  `data/targets/callite/calib_<domain>_monthly.csv` (see the
  [CalLite targets](../../targets/callite/README.md)).

## Read by

`sacsma.io.load_reference` (path from `sacsma.paths.simflow`), used by `sacsma verify parity`
(`sacsma.verify`), `sacsma plots` (`sacsma.cdec15.plots`, `sacsma.calsim.plots`) and
`sacsma calsim --forcing-compare` (`sacsma.calsim.forcing_compare`).
