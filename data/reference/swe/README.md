# data/reference/swe: snow water equivalent reference products on the 1/16° grid

Four snow water equivalent (SWE) products from Google Earth Engine, each reduced to the 4,410
cells of the grid (`data/inputs/grid/grid_cells.csv`). Role: reference. No code in the `sacsma`
package reads them. Like the ET products, they are kept only for comparison (see
[Conventions](../../../docs/conventions.md)).

## Files

| File | What | Source |
|---|---|---|
| `daymet_swe_gee_cell_monthly.npz` | Daymet V4 `swe` (`NASA/ORNL/DAYMET_V4`), monthly mean of the daily band, reduced at 1000 m | exported 2026-07-16 |
| `terraclimate_swe_gee_cell_monthly.npz` | TerraClimate `swe` (`IDAHO_EPSCOR/TERRACLIMATE`), 4638 m; an end-of-month snapshot | exported 2026-07-16 |
| `fldas_swe_gee_cell_monthly.npz` | FLDAS Noah `SWE_inst` (`NASA/FLDAS/NOAH01/C/GL/M/V001`), 11132 m; NaN ≤0.05 % (FLDAS land mask) | exported 2026-07-16 |
| `era5land_swe_gee_cell_monthly.npz` | ERA5-Land `snow_depth_water_equivalent` (`ECMWF/ERA5_LAND/MONTHLY_AGGR`, m × 1000), 11132 m | exported 2026-07-16 |

All files are in LFS and cover all 4,410 cells × 1988-01 to 2018-12. Each holds the arrays
`keys` (cell keys), `dates` (month start), `swe` (cells × months, float32, mm), `lat`, `lon`
and `meta` (export date, collection and band, scale).

## How it is built

By `../et/gee_obs_region.py`: one Earth Engine script writes both folders. Environment,
commands and requirements are in the [ET README](../et/README.md).

## Checks

- The files are their own specification: the mean over the cell rectangle at each asset's
  native scale. Against the earlier snapshot the drift on 15-watershed monthly climatologies is
  SWE rel RMS 4.1 %, with the snowy-basin mask unchanged.
- Against BCM snowpack `pck` (1988–2018, regional monthly series), r at lag 0 is 0.923
  (`fldas`), 0.912 (`era5land`), 0.896 (`terraclimate`) and 0.561 (`daymet`). See the
  [BCM README](../bcm/README.md).

## Know before using

- Caution: Daymet's SWE climatology peaks in April and never melts out (August 9.5 mm,
  September 6.7 mm, against about 0 for BCM and ERA5-Land). BCM, ERA5-Land and FLDAS peak in
  February and melt out by August. Daymet is not a usable referee for snowpack timing, and its
  better correlation with BCM at lag −1 (0.673) is an artifact of that seasonality.
- TerraClimate is an end-of-month snapshot. The other three are monthly means of the state.
- Daymet must be reduced at its 1 km native scale: at 11132 m the rel RMS error was 0.33.
- Regional mean SWE differs widely by product: `fldas` 12.3 mm, `terraclimate` 18.6 mm,
  `daymet` 19.0 mm, `era5land` 26.9 mm. BCM at 270 m gives 37.8 mm, above every product: it
  resolves high-elevation snow that an 11-km product smooths away.
- Earth Engine assets drift. A new export will not reproduce these files exactly.

## Read by

No module of the `sacsma` package. `sacsma.paths.swe_obs` names the folder for the build script.
