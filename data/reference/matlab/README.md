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
| `forcing_9unimp.nc`, `forcing_11obs.nc`, `forcing_12rim.nc` | The forcing each CalLite simulation ran on: `prcp` (mm/day) and `tavg` (°C) by grid cell (`key`) and day, 1915–2018, on a 0.01 grid (56, 252 and 228 MB) | the per-set stores the repository read until 2026-07-16, restored from its history |

## How it is built

Delivered; cannot be rebuilt from a clone. The `simflow` files were converted once from the
archived MATLAB study materials; no script in the repository writes them.

## Checks

`sacsma verify parity` runs every watershed of the four domains on the forcing its MATLAB
simulation ran on (`sacsma.model.load_matlab_forcing`: the `forcing` files here for the CalLite
sets, the shared store for `15cdec`) and requires KGE > 0.9999 and a largest daily difference
< 0.1 mm/day. The largest per domain:

| Domain | Watershed | KGE | Largest daily difference (mm/day) |
|---|---|---|---|
| `15cdec` | NHG | 0.999908 | 0.0193 |
| `9unimp` | BearRiver | 1.000000 | 0.0000 |
| `11obs` | TNL | 0.999970 | 0.0743 |
| `12rim` | FOL_I | 1.000000 | 0.0013 |

TNL is off on 58 storm days from 1941; TRINI, the same watershed in `12rim`, is exact. The cause
is not known.

## Know before using

- These are model simulations, not observations. The observed targets are under `data/targets/`.
- `flow` is area-normalized (mm/day). `basin` holds each domain's own watershed codes (for
  example `BND`, `CacheCreek`, `SHA`, `SHAST`).
- The baseline is the reference cold start on historical Livneh-unsplit forcing, as the MATLAB
  simulation read it.
- Run the parity check after any change that touches the model or the path the data takes into
  it. The physics modules are not edited without it.
- Every run and product reads the shared forcing store, not the `forcing` files here. Its `tavg`
  is the mean of 2-decimal `tmax` and `tmin` (a 0.005 grid) and its `prcp` has 3 decimals, within
  0.0055 of these files. Two thresholds of the frozen Snow-17 turn that into whole storm days:
  the CalLite optima put the rain/snow threshold `PXTEMP` at 0, so a wet day at exactly 0.00 °C
  here is snow and at 0.005 °C there is rain, and rain on snow starts above 6 mm a day.
  On the shared forcing store the CalLite watersheds differ from MATLAB by up to 0.61 mm/day
  on a day (MokelumneRiver), 1.6 mm in a month and 0.007 % in long-term volume. `15cdec` keeps its own store.
- The MATLAB monthly simulation printed in the calibration logs is a separate table,
  `data/targets/callite/calib_<domain>_monthly.csv` (see the
  [CalLite targets](../../targets/callite/README.md)).

## Read by

`sacsma.io.load_reference` (path from `sacsma.paths.simflow`), used by `sacsma verify parity`
(`sacsma.verify`), `sacsma plots` (`sacsma.cdec15.plots`, `sacsma.calsim.plots`) and
`sacsma calsim --forcing-compare` (`sacsma.calsim.forcing_compare`). The `forcing` files:
`sacsma.model.load_matlab_forcing` (path from `sacsma.paths.matlab_forcing`), for `sacsma verify
parity` and the parity figures of `sacsma plots`.
