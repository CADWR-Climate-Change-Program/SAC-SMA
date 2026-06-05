# `artifacts/` — simulated outputs & diagnostic figures

Generated, **not** version-controlled (everything here except this README is
gitignored). Outputs are organized per **run** under `artifacts/<run>/`
(default run name `15cdec`). Regenerate with:

```bash
python -m sacsma.plots                 # -> artifacts/15cdec/
python -m sacsma.plots --run myrun     # -> artifacts/myrun/
```

## Contents of `artifacts/<run>/`

| Path | What |
|------|------|
| `figures/<BASIN>_diagnostics.png` | Per-basin 4-panel: sim-vs-gage time series with **separate calibration & validation skill** (dashed split + shaded validation), daily scatter (1:1), mean-monthly regime, flow-duration curve. |
| `figures/skill_summary.png`       | KGE/NSE (calibration vs validation) and percent bias across all 15 basins. |
| `figures/parity_vs_matlab.png`    | Python port vs. MATLAB reference `simflow` — exact-match proof (worst-case overlay, pooled 1:1 scatter, max daily \|Δ\| per basin). |
| `metrics_15cdec.csv`              | Per-basin calibration/validation KGE, NSE, pbias, r, and mean flow (mm/day + cfs). |

"Simulated" = `sacsma.model.run_basin` (the faithful port from the GA optimum).
"Observed/calibrated" = the gage full-natural-flow target in
`data/reference/gage_15cdec.parquet`. "Reference" (parity) = the MATLAB
`simflow` in `data/reference/simflow_15cdec.parquet`. Flows are mm/day; areas in
`data/reference/basin_area_15cdec.parquet` convert to cfs.
