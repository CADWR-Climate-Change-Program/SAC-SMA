# `artifacts/results/benchmark/`: three models against CDEC full natural flow

Three models at 12 CDEC full-natural-flow (FNF) watersheds of the learned-parameter registry
(its 17 CDEC sites less the four Tulare basins and New Hogan), scored against the observed CDEC FNF, monthly,
WY1991–2018. Every model runs on the same climate, WGEN Product A scenario 1. WY1991–2018 is
where the WGEN temperature detrending is small.

```bash
sacsma benchmark          # -> artifacts/results/benchmark/; about 2 min on CPU (the BCM fit), 10 s once cached
```

Code: `sacsma.benchmark` (`flows` for the observation, dPL-CalSim and the CalSim3 VIC,
`gridded` and `bcm_routing` for BCM, `report` for the scores and figures).

## The models

| Model | What | Forcing | Sites |
|---|---|---|---|
| dPL-CalSim | The trained field of the run `noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_w2ft15r10_aef` on its 12 `cdec_*` entities, on the CPU engine with the run's cycle spin-up | WGEN Product A s1 | 12 |
| BCM | BCM v8 Scenario 1 `run` and `rch`, routed to the gauge by the USGS monthly routing equations ([BCM README](../../../data/reference/bcm/README.md#monthly-routing)) with parameters fitted here on WY1991–2018 (below), routed from 1915-10 | BCM Scenario 1 (the same sequence) | 12 |
| VIC-CalSim3 | The routed monthly VIC of the CalSim3 pipeline at the site's CalSim3 node; at YRS the sum of the 16 arc series above Smartsville | WGEN Product A s1 | 12 |

Volumes. The two depth models (dPL-CalSim, BCM) are turned into volume with the site's
registry area, the area the observed daily depth was made with. VIC-CalSim3 is the TAF of its
CalSim3 node as delivered.

BCM's footprint. Each site's cells of the multifamily registry
(`entity_cells.csv`), weighted by `overlap_mi2`, renormalised over cells with a value each
month. The SHA and BND footprints stop south of the endorheic Goose Lake block, so nothing is
screened.

## How it is scored

- Observed: CDEC's monthly full natural flow (sensor 65, `fnf_monthly.csv`) at the site's own
  station, for 10 sites. CDEC publishes CLE and NML only at a station downstream (TNL and SNS)
  ([CDEC README](../../../data/targets/cdec/README.md)). CLE is scored on TNL's monthly series
  (Trinity River at Lewiston, 3.8 % more area), scaled to CLE's volume: times 0.986, CLE's
  daily volume over TNL's on the months both cover (r 0.997). NML is scored on the monthly sum
  of its daily record, a month counting only when all its days are valid: SNS parts from NML's
  record with the season, by up to 68 % in a month.
- New Hogan (NHG), a registry site outside the Tulare basin, is left out: CDEC publishes no
  monthly series for it, and its daily record has 95 complete months in the window, none in
  October.
- At each site every model present is scored on the same months: the months where the
  observation and all those models have a value.
- Metrics: KGE, NSE, percent bias (positive when the model is high), r, and the seasonal
  mismatch (the fraction of the annual flow in the wrong month, `sacsma.metrics.seasonal_mismatch`).

## Scores, WY1991–2018

KGE / percent bias. The months column is the number scored.

| Site | Months | dPL-CalSim | BCM | VIC-CalSim3 |
|---|---|---|---|---|
| SHA | 336 | 0.94 / 0 % | 0.85 / +1 % | 0.66 / +11 % |
| CLE | 336 | 0.94 / +1 % | 0.80 / −1 % | 0.85 / −7 % |
| BND | 336 | 0.96 / −3 % | 0.91 / +1 % | 0.70 / +6 % |
| ORO | 336 | 0.96 / +3 % | 0.81 / −1 % | 0.67 / +16 % |
| FOL | 336 | 0.97 / 0 % | 0.90 / +1 % | 0.78 / +6 % |
| YRS | 336 | 0.93 / −2 % | 0.89 / +1 % | 0.80 / +8 % |
| CSN | 336 | 0.96 / −1 % | 0.91 / +1 % | 0.48 / +39 % |
| MKM | 336 | 0.85 / −11 % | 0.83 / +1 % | 0.82 / 0 % |
| NML | 217 | 0.96 / +1 % | 0.91 / 0 % | 0.56 / +19 % |
| TLG | 336 | 0.91 / −5 % | 0.93 / +1 % | 0.86 / −3 % |
| MRC | 336 | 0.96 / +2 % | 0.91 / +1 % | 0.81 / +9 % |
| MIL | 336 | 0.92 / −2 % | 0.95 / 0 % | 0.83 / −3 % |

Pooled over the 12 sites (`summary.csv`). "Volume bias" is the bias of the summed mean
volumes; "Highest KGE" counts the sites where the model scores best.

| Model | Median KGE | Mean KGE | Median NSE | Median bias | Mean \|bias\| | Volume bias | Median seasonal mismatch | Highest KGE |
|---|---|---|---|---|---|---|---|---|
| dPL-CalSim | 0.949 | 0.939 | 0.940 | −0.6 % | 2.5 % | −1.1 % | 0.045 | 10 |
| BCM | 0.906 | 0.884 | 0.821 | +1.0 % | 0.8 % | +0.6 % | 0.076 | 2 |
| VIC-CalSim3 | 0.791 | 0.735 | 0.793 | +6.9 % | 10.6 % | +7.8 % | 0.109 | 0 |

What the table shows:

- dPL-CalSim has the highest KGE at 10 of the 12 sites, is within about 5 % in volume
  everywhere but MKM (−11 %), and has the smallest seasonal mismatch. It was
  trained on these records (see below).
- BCM, with its routing fitted here on the same records, has the second median KGE, is within
  about 1 % in volume at every site (the fit holds it there), and has the highest KGE at TLG
  (0.93 against 0.91) and MIL (0.95 against 0.92).
- VIC-CalSim3 has the lowest median KGE (0.79) and runs high at most sites, most at CSN
  (+39 %), NML (+19 %) and ORO (+16 %).

![Monthly KGE and percent bias by site and model](figures/skill.png)

## BCM: the routing fitted here

BCM has no channel routing: its `run` and `rch` are water generated in a month. The USGS routes
them through three monthly reservoirs fitted per basin; its parameters were fitted on
PRISM-forced BCM, and on the WGEN-forced Scenario 1 they carry a volume error of up to +15 %
(the yield of the two runs differs by −3 to +19 % on the same basins). The benchmark therefore
fits the same equations itself, per site, on Scenario 1 (`sacsma.benchmark.gridded.bcm_fit`):

- Input: Scenario 1 `run` and `rch` on the site's footprint times its registry area, routed
  from 1915-10.
- Target: the observed monthly volume of the site over the whole scoring window, WY1991–2018,
  the years dPL-CalSim was trained on, so the two fitted models are scored on the same footing.
  A site needs 60 observed months; NML, the fewest, has 217.
- Objective: KGE, with the volume bias held within 1 % (the sheets maximised NSE within the
  same tolerance; the NSE optimum damps the monthly variance).
- Bounds and lags: as the refit of the Kings sheet ([BCM README](../../../data/reference/bcm/README.md#monthly-routing)),
  causal lags only. Differential evolution with a fixed seed; the fit is cached and a rerun
  gives the same parameters (`bcm_routing_fit.csv`).

The fit is in-sample, as dPL-CalSim's training is. Fitted on either half of the window
(WY1991–2004, WY2005–2018) and scored on the other, the same routing has a median KGE of 0.899
and a volume bias within 2.2 % at every site, against 0.906 fitted on the whole window: the
in-sample advantage is small. That check was made once with the same equations, bounds and
optimiser; it is not part of `sacsma benchmark`.

The routing against the USGS sheets as delivered (the Kings by its refit) and against
`run + rch` unrouted, on the scored months (`bcm_routing_effect.csv`):

| Site | KGE fitted | KGE USGS sheet | KGE unrouted | Bias fitted | Bias USGS sheet | Bias unrouted |
|---|---|---|---|---|---|---|
| SHA | 0.854 | 0.823 | 0.385 | +1.2 % | +9.1 % | −1.0 % |
| CLE | 0.800 | – | 0.698 | −1.0 % | – | +0.7 % |
| BND | 0.907 | – | 0.523 | +1.1 % | – | +0.1 % |
| ORO | 0.814 | 0.839 | 0.609 | −1.0 % | +0.6 % | +3.7 % |
| FOL | 0.905 | 0.893 | 0.831 | +1.0 % | 0.0 % | +9.9 % |
| YRS | 0.894 | 0.843 | 0.874 | +1.0 % | −4.8 % | +2.4 % |
| CSN | 0.908 | 0.836 | 0.377 | +1.0 % | +9.0 % | +43.7 % |
| MKM | 0.831 | 0.820 | 0.810 | +1.0 % | +5.1 % | +1.6 % |
| NML | 0.913 | 0.696 | 0.690 | +0.1 % | +15.2 % | +13.9 % |
| TLG | 0.928 | 0.849 | 0.905 | +0.5 % | +11.1 % | +3.2 % |
| MRC | 0.910 | 0.846 | 0.773 | +1.0 % | +12.4 % | +15.0 % |
| MIL | 0.950 | 0.836 | 0.926 | 0.0 % | +9.6 % | −2.7 % |

- The fitted routing scores above the USGS sheets at 9 of the 10 sites with a sheet (median KGE
  0.91 against 0.84) and cuts the median volume bias from +9 % to +1 %. The sheet does better
  at ORO, whose sheet reads the next month's recharge.
- Routing lifts NSE above the unrouted series and lowers the seasonal mismatch at all 12 sites.
- Several parameters end on a bound (`bcm_routing_fit.csv`). At six sites SurfaceExp is at its
  upper bound, so the surface store releases nearly all it holds each month; at four (FOL, YRS,
  CSN, MKM) DeepExp and the antecedent storage are at their lower bounds, which turns the deep
  flow off. The antecedent storage and DeepExp trade against each other, since the deep flow is
  the storage to the power DeepExp.

## Files

| File | What |
|---|---|
| `monthly.csv` | `[date, site, model, flow_taf]`: every month of WY1991–2018 with a value, TAF/month, month-end dates; `model` is `obs`, `dpl`, `bcm` or `vic_calsim3` |
| `metrics.csv` | Per site and model: `n_months`, `wy_start`, `wy_end`, `kge`, `nse`, `pbias`, `r`, `seas_mismatch`, `mean_sim_taf`, `mean_obs_taf` |
| `summary.csv` | Per model, pooled over the 12 sites: median and mean KGE, median NSE, median, mean absolute and volume bias, median r and seasonal mismatch, `n_kge_best` (the sites where it has the highest KGE) |
| `sites.csv` | Per site: CDEC record, registry area, footprint size, VIC node area, the observed months of the BCM fit (`bcm_fit_months`) and the USGS sheet compared with, the models scored, the observed months |
| `bcm_routing_fit.csv` | The fitted BCM routing, per site: the parameters `route` reads, `area_m2`, `n_months` (observed months of the fit) and the fit's `kge`, `r2`, `nse`, `pbias` (the last three the sheets' statistics; `pbias` positive when low) |
| `bcm_routing_effect.csv` | BCM per site on the scored months: the fitted routing (`_fit`), the USGS sheets (`_workbook`) and `run + rch` unrouted (`_unrouted`) |
| `figures/skill.png` | KGE (0 to 1) and percent bias (±75 %) per site and model, north to south |
| `figures/hydrographs.png` | Monthly flow of every model and the observation, one panel per site |
| `figures/climatology.png` | Mean flow per calendar month on the months scored, one panel per site |

The dPL-CalSim run and the BCM fit are cached in `_local/cache/benchmark/`. The dPL run is keyed
by the parameters and the engine's code but not by the forcing's contents: clear the folder
after a forcing store changes. The BCM fit is keyed by its inputs and code.

## Know before using

- Caution: two of the three models were fitted to these records over this window, so their
  scores here are in-sample.
  - dPL-CalSim was trained on these daily series, from 1986–1999 (by site) to 2018. Its
    held-out decade (WY1976–85) is outside this window. It was trained on the daily records,
    which run 3 to 6 % below CDEC's monthly series at YRS, CSN, MRC and MKM, so its volume
    there sits 3 to 6 points lower against the monthly series than against the daily one.
  - BCM's routing was fitted here to the same observations over the same window; fitted on
    one half and scored on the other, its median KGE is 0.90 (above). The BCM water balance
    itself was not fitted.
  - VIC-CalSim3: how it was calibrated is not documented in the repository.
- BCM's relation to the observations is not constant. Scenario 1's `run + rch` against the
  observed volume moves by 5 to 10 % from decade to decade within WY1991–2018, and runs 10 to
  15 % lower in WY1961–1990 at SHA and MIL, where the WGEN detrending warms the record. The
  fit holds the volume of the whole window, not of each decade: by decade the routed volume
  is −12 to +9 % off the observed (MKM −12 % in WY2001–2010, MIL +9 %), a median spread of 10
  points between the decades of a site.
- NML is scored on complete months of its daily record, which drops dry-season months with a
  negative or missing day: it keeps 217 of 336. CLE's
  daily record would keep 233, mostly missing July to November, so CLE is scored on TNL's
  monthly series instead (above).
- VIC-CalSim3 is the flow of its CalSim3 node, whose catchment can differ from the site: BND
  +2.1 %, MRC −2.3 %, MIL −2.2 % (`sites.csv`). It is not rescaled. At YRS the basin-level
  series `8RI_SMART` includes Deer Creek below the gauge, so the site sums the 16 arc series
  above Smartsville instead (1,129 mi²).
- Regenerated tables can differ from the tracked ones in the last digits.
