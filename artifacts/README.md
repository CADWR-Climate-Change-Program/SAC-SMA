# `artifacts/`: simulated outputs and diagnostic figures

Generated outputs, organized by application: `cdec15/` (the 15-CDEC diagnostics) and `calsim/` (the CalLite domains `9unimp`/`11obs`/`12rim`, the cross-compare `compare/`, and the alternate-forcing runs `wgen_product_a/` + `historical_lto/` with their `forcing_compare/`). The committed copies are the published results of the current data and model. Regenerate with:

```bash
sacsma plots --domain 15cdec              # -> artifacts/cdec15/
sacsma plots --domain 11obs               # -> artifacts/calsim/11obs/  (or 9unimp / 12rim)
sacsma calsim                             # -> artifacts/calsim/compare/
python -m sacsma.calsim.forcing_compare   # -> artifacts/calsim/forcing_compare/
```

Throughout, "simulated" is `sacsma.model.run_basin` from the archived GA optimum, "observed" is the daily CDEC gage (`cdec15`) or the domain's monthly FNF (`calsim`), and "reference" (parity) is the MATLAB `simflow` tables. The full method and conventions are in the [documentation report](https://cadwr-climate-change-program.github.io/SAC-SMA/) and `data/INVENTORY.md`; the model guardrails are in `CLAUDE.md`. The dPL outputs have their own manifest in [`dpl/RUNS.md`](dpl/RUNS.md).

## Per-domain diagnostics (`cdec15/`, `calsim/<domain>/`)

| File | What |
|------|------|
| `figures/<BASIN>_diagnostics.png` | Per-basin 3-panel: sim-vs-observed time series with separate calibration and validation skill, plus mean-monthly regimes for both periods. Daily vs the CDEC gage for `cdec15`, monthly vs observed FNF for the CalLite domains. |
| `figures/skill_summary.png` | KGE and percent bias (cal vs val) across the domain's watersheds, ordered north→south. KGE on a full 0–1 scale; pbias on a fixed ±75% scale shared across all four calibration sets, so bars are comparable set to set. |
| `figures/parity_vs_matlab.png` | Python vs the original MATLAB `simflow`: the exact-match proof. |
| `metrics_<domain>.csv` | Per-basin cal/val KGE, NSE, pbias, r, and mean flow. |

A parallel `*_calsim3` variant of the diagnostics and metrics scores the anchor run against CalSim3's own unimpaired FNF (TAF/month) rather than the observed-FNF target, on the same cal/val windows. It is non-destructive (the FNF-target files are untouched) and skips `12rim`, which is not in the cross-compare.

## Alternate-forcing runs (`calsim/wgen_product_a/`, `calsim/historical_lto/`)

Every CalLite watershed forward-simulated with an alternate forcing product (`sacsma run <basin> --domain <d> --forcing <product>`), same GA optima and model. Each directory holds `flow_daily_<domain>.csv`: long `[date, basin, flow]` daily mm/day, in the `simflow` format.

- **`wgen_product_a`** changes temperature only (detrended to 1991–2020): a consistent ~2–4% runoff-volume loss (median −3%), precipitation unchanged.
- **`historical_lto`** is a different precipitation realization (the "split" Livneh lineage, through 2021), so its differences run end-to-end.

`calsim/forcing_compare/` (`python -m sacsma.calsim.forcing_compare`) renders each product against the Livneh baseline on the 1915–2018 overlap, on both SAC-SMA and VIC, everything split at 1950 (where the precipitation differences concentrate). Its figures carry the per-watershed and cross-model volume and regime effects; notably, VIC's volume response to the temperature detrending is much weaker than SAC-SMA's; it mostly shifts snowmelt earlier instead.

## CalSim cross-compare (`calsim/compare/`)

`sacsma calsim` cross-compares each SAC-SMA calibration set (`15cdec`, `9unimp`, `11obs`, kept separate) and VIC against CalSim3's historical inflow at the CalSim inflow nodes (`data/calsim/gis/calsim3.gpkg`). It reports two views.

**Per-catchment.** Each CalSim3 node scored individually (`calset_metrics.csv`, with each set's honest HRU coverage `cov_frac`). Median KGE ≈ 0.67 (VIC 0.66).

**Basin-level "anchor".** Each basin's `run_basin` total, on the canonical CalSim catchment area, vs the faithful CalSim3 reference for that basin (`anchor_metrics.csv`, `anchor_monthly.csv`). Median KGE 0.92 (15cdec), 0.95 (11obs), 0.93 (9unimp); aggregation cancels the per-catchment noise. Where the basin is a rim system, the reference is the single FLOW-UNIMPAIRED whole-watershed series (the only correct target for systems like Sac @ Bend Bridge, whose valley-floor accretion a sub-arc sum misses); otherwise it is the sum of the basin's CalSim3 INFLOW sub-arcs. The choice is recorded per row as `ref_kind`.

Maps and figures show skill at the **main-basin level**: every sub-area polygon is coloured by its watershed's anchor score, never its own sub-arc score, and per-sub-arc numbers stay in the CSVs. Key outputs:

| File | What |
|------|------|
| `figures/anchor_skill_{kge,nse,pbias}.png` | Per-basin SAC-SMA vs VIC dumbbells, north→south; 15cdec folded on after a dashed divider. |
| `figures/calsim_sacsma_map_{nse,kge,pbias}.png` (+ `calsim_vic_map_*`, `calsim_sacsma_minus_vic_*`) | Basin-level skill maps for SAC-SMA, VIC, and their difference. |
| `figures/anchor_hydrographs.png`, `main_river_climatology.png` | The 8 CA main river indices vs the CalSim3 FLOW-UNIMPAIRED reference. |
| `subarc_qmap_<set>.csv`, `subarc_validation_metrics.csv` | Per-sub-arc QMAP bias-correction (train/test), lifting median sub-arc KGE from ≈0.67 to 0.75–0.86. |
| `target_vs_calsim3.csv` | How far each set's calibration target itself sits from CalSim3, the bias floor a perfect-fit model inherits. |
| `vic_precip_split_vs_unsplit.{csv,md,png}` | How much the unsplit precip basis differs from the older split product per basin (overwhelmingly pre-1950). |

Footprint screening (`catchments.SCREENED_BASINS` = SHA, BND, SNS, ChowchillaRiver) trims the four basins whose HRU footprint materially over-reaches its CalSim3 catchment; every other basin keeps its full calibrated footprint. The everything-unscreened parallel and its delta are in `anchor_*_full.csv` and `anchor_screened_vs_full.csv`. The footprint-method maps (`figures/{shasta,sns,chowchilla,tnl,fresno}_footprint_panels.png`) and the HRU attribute maps (`figures/hru_{veg,soil,kpet}_*.png`) are single-basin and input illustrations, not part of the basin-level scoring.

The engine is `sacsma.calsim.catchments`. The full method (anchor reference, screening, QMAP, figure style) is documented in the [report](https://cadwr-climate-change-program.github.io/SAC-SMA/) and `CLAUDE.md`.
