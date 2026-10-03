# `artifacts/product/`: the CalSim3 rim-inflow product

Monthly flow, in TAF, on the 196 CalSim3 rim arcs: the points where runoff from a mountain
watershed enters the CalSim3 network. It is the flow of the current model, dPL-CalSim, with the
share model dividing each multi-arc system's flow among its arcs. Method and scores:
[CalSim3 rim inflows](../../docs/calsim3_rim_inflows.md); state of the runs:
[Runs](../../docs/runs.md).

One folder per forcing, named as `--forcing` names it. Each series is one continuous run of the
model over the whole forcing record (spin-up on WY1916–1925, then October 1915 to December
2018); the series holds its complete water years.

| Folder | Forcing | Water years |
|---|---|---|
| `historical_livneh_unsplit/` | Livneh, unsplit precipitation (the training forcing) | 1916–2018 |
| `wgen_product_a/` | WGEN Product A scenario 1: the same precipitation, temperature detrended to a 1991–2020 baseline | 1916–2018 |
| `wgen_product_a_s12/` | WGEN Product A scenario 12: scenario 1 + 2 °C, extreme-tail rate 7 %/°C, wet-day mean unchanged | 1916–2018 |

The forcings are described in [`data/inputs/forcing/`](../../data/inputs/forcing/README.md).

| File | What |
|---|---|
| `<forcing>/rim_inflow_monthly.csv` | `arc, month, taf, dpl_taf, kind`. `kind` is `share` (an arc of one of the eleven multi-arc systems, divided by the share model), `single-arc system` or `non-anchor` (the model's own flow); `taf = dpl_taf` except on the share arcs. |
| `<forcing>/product_metrics.csv` | Historical forcing only: per arc, monthly KGE against CalSim3 of the model's arc flow and of the product, on the held-out water years (`holdout`, WY1976–85), on the other water years of WY1950–2015 (`train`) and on WY1922–1949 (`early`). |
| `<forcing>/tier2/` | The tier-2 pass the share model was applied to (`tier2_monthly.csv`, `tier2_components_monthly.csv`) and its record (`tier2_run_info.json`: forcing, envelope, checkpoint), so that `sacsma verify product` can repeat the series. `ref_taf` in `tier2_monthly.csv` is the historical CalSim3 series whatever the forcing. |
| `share_model.pt` | The share model and everything applying it needs (arc and system order, input scaling, per-arc volume ratios). |
| `share_selection.csv` | The out-of-fold candidates of the fit: the model's own shares and each penalty weight μ, with the median and p10 arc KGE on training water years and the response misses at the training climate points. |
| `response_gate.csv` | Per validation climate point, the product's miss against the model's own arcs in volume change (`V`, %) and April–July share change (`AJ`, pp), with the full gate (median ≤ 1, p90 ≤ 3) and the half gate. |
| `product_info.json` | The fit: the run, the settings, its water years, the climate points and the passes it used (relative to the run's local folder), and the scores on the held-out water years by kind of arc. |

```bash
# the fit, on the run's passes on the training forcing (WY1950-2018 envelope): the base tier 2
# with runoff parts, and the same at the 11 training and 5 validation climate points
# (sacsma.dpl.calsim.product.scenario_spec() lists them; split them to fit the GPU, 8 per pass
# at --batch-window 512 on 8 GB)
sacsma dpl calsim tier2 <run> --components parts --trace-python <python of sacsma-gis>
sacsma dpl calsim tier2 <run> --components parts --scenarios t1=1:1,t3=3:1,...
sacsma dpl calsim product fit <run>                # CPU, about 30 min; --mu 0.03 skips the selection

# the series of one forcing: the whole record, then the share model
sacsma dpl calsim tier2 <run> --components parts --forcing <forcing> --start 1915-10-01 \
    --extension-cells artifacts/_local/runs/dpl/multifamily/<run>/tier2/tier2_extension_cells.csv \
    --out artifacts/_local/product/<forcing>/tier2 --no-maps         # GPU, about 1.5 h
sacsma dpl calsim product apply --tier2 artifacts/_local/product/<forcing>/tier2 --forcing <forcing>
sacsma verify product                              # each tracked series repeats
```

After a refit, `apply` remakes the series of every forcing.
