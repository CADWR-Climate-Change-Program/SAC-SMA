# Reproduce

From a clone to the results, in the order the work depends on. Each section says what it
needs, what it writes, and about how long it takes. The last section lists what a clone cannot
rebuild.

## 1. Environments

There are three. Only the first is needed to run the models.

| Environment | File | Used for |
|---|---|---|
| `sacsma` | [`environment.yml`](../environment.yml) | The reference model, the calibrated sets, the learned-parameter code (PyTorch with CUDA), and most build scripts under `data/`. |
| `sacsma-gis` | [`environment-gis.yml`](../environment-gis.yml) | The five scripts that read rasters or pull from Earth Engine: `sample_gis.py`, `reitz_et.py`, `build_flowlens.py`, `gee_obs_region.py`, `gee_aef_region.py`. |
| `neuralhyd` | the neuralhyd-ca repository | `usgs_flows.py` only: it reads that repository's zarr store. |

```bash
git lfs install                          # once per machine, before the clone
mamba env create -f environment.yml
mamba activate sacsma
pip install -e .
mamba env create -f environment-gis.yml  # only for the five raster / Earth Engine scripts
```

The raster stack is kept out of `sacsma` on purpose. The two environments carry different GDAL
builds, and one process must not load both. Tier 2 is the one place where the model needs the
raster environment: it runs the flow-length tracer as a separate process, with the interpreter
you name by `--trace-python`.

Activate an environment before using it. Calling its `python` by path, without activation, can
miss the environment's own libraries on Windows.

## 2. Data

Everything a model run reads is tracked: `git lfs pull` brings 6.1 GB, of which 5.7 GB is
forcing. No run needs a file outside the repository. [`data/README.md`](../data/README.md)
lists every folder; each folder's `README.md` is its provenance record.

The build scripts under `data/` do need inputs from outside the repository (raw rasters,
weather-generator releases, a report PDF). Name where they sit on your machine in
`data/local_paths.toml`, copied from `data/local_paths.example.toml`. Rebuilding data is
section 7; it is not needed to reproduce results.

## 3. Check the installation

```bash
sacsma verify --quick    # imports, commands, links, layout, the BCM routing port: no model run
sacsma verify            # adds parity, the learned step and the products
```

`parity` runs one watershed per calibration set and compares with the archived MATLAB
simulation: KGE above 0.9999 and a largest daily difference below 0.1 mm/day. `product`
re-applies the share model to the base tier-2 pass and compares with the tracked rim-inflow
product; it is skipped, with a note, when that pass is not on disk (it is produced in
section 6). The whole check takes a few minutes on CPU.

## 4. The reference model and the calibrated sets

```bash
sacsma run ALL                         # the 15 CDEC watersheds, daily
sacsma run ALL --domain 11obs          # likewise 9unimp, 12rim
sacsma plots --domain 15cdec           # calibration and validation diagnostics -> artifacts/results/15cdec/
sacsma plots --domain 11obs            # -> artifacts/results/callite/11obs/
sacsma calsim                          # comparison with CalSim3 and VIC -> artifacts/results/calsim3/
sacsma benchmark                       # three models vs CDEC FNF -> artifacts/results/benchmark/
sacsma product callite                 # the CalLite inflow files, three forcings -> artifacts/product/callite/
sacsma product 15cdec                  # the daily flow of the 15 watersheds -> artifacts/product/15cdec/
```

All of it runs on CPU in minutes. `sacsma benchmark` also runs dPL-CalSim (section 6) on the
CPU, so it needs PyTorch. It caches both of its runs under `artifacts/_local/cache/benchmark/`,
keyed by the parameters and the engine's code but not by the forcing: clear that folder after a
forcing store changes. [`artifacts/results/README.md`](../artifacts/results/README.md) lists every file
these commands write. Regenerated tables can differ from the tracked ones in the last digits,
and the row order of `monthly_calsets.csv` is not fixed; compare before committing.

## 5. The ladder and the hybrids on the 15 CDEC watersheds

The runs of [Learned parameters](learned_parameters.md) are tracked with their selected
checkpoints, so scoring needs no training:

```bash
sacsma dpl evaluate artifacts/models/dpl/15cdec/5_px/checkpoints/best.pt   # likewise 1_hru to 6_aef
sacsma dpl fidelity        # the learned numerics vs the reference model -> artifacts/results/dpl/fidelity/
```

`evaluate` scores a rung at the 15 outlets, daily, on WY1989–2003 (training) and WY2004–18
(test), without the days of the CDEC mask.

**Train.** [The ladder](runs.md#the-ladder) trains every rung from scratch on one recipe: the
defaults of `sacsma dpl train` (among them the gradient through two water years, 120 epochs and
the `15cdec_grid` GA median as the starting field) and the flags in `R` below. Each rung adds
its own change. A checkpoint stores its full configuration; `sacsma dpl train --help` lists
every option. The commands write to local folders, so the tracked runs stay as they are.

```bash
R="--cpu-select --diagnostics --obs-mask data/targets/cdec/fnf_daily_mask.csv \
    --log-space lzsk,lzpk,MFMAX,MFMIN --param-bounds lzsk=0.001:0.5 --logit-penalty 0.01 --patience 20"
CDEC15=cdec_BND,cdec_FOL,cdec_ISB,cdec_MIL,cdec_MKM,cdec_MRC,cdec_NHG,cdec_NML,cdec_ORO,cdec_PNF,cdec_SCC,cdec_SHA,cdec_TLG,cdec_TRM,cdec_YRS
MF="--domain multifamily --basins $CDEC15 --train-window 1988-10-01:2003-09-30"
NOAH="--et noah --noah-pet priestley_taylor"
PX="--learn-pxtemp --pxtemp-tau 1.0"
L=artifacts/_local/runs/dpl
sacsma dpl train physical $R --domain 15cdec --et sac --sac-pet hamon --out $L/1_hru
sacsma dpl train physical $R $MF --et sac --sac-pet hamon --out $L/2_grid
sacsma dpl train physical $R $MF --et sac --sac-pet priestley_taylor --out $L/3_pt
sacsma dpl train physical $R $MF $NOAH --out $L/4_noah
sacsma dpl train physical $R $MF $NOAH $PX --out $L/5_px
sacsma dpl train aef_u $R $MF $NOAH $PX --out $L/6_aef
sacsma dpl evaluate $L/5_px/checkpoints/best.pt   # a retrained rung is scored in its own folder
```

`1_hru` trains on the HRUs of the `15cdec` domain. Rungs 2 to 6 run on the `multifamily` domain
(1/16° grid cells and the CalSim3 catchment outlines) restricted to the 15 CDEC entities, and
train on WY1989–2003. Each rung takes about 3 to 6 hours on an 8 GB laptop GPU (the `epoch_s`
column of its `train_log.csv`). Train one run at a time: the daily graph takes most of the memory.

**The hybrids** are LSTM models on the 15 watersheds, three seeds each, with the defaults of
`sacsma dpl hybrid`. `hybrid` takes the daily flow of `5_px` as an input; `hybrid_dt` is a
15-epoch fine-tune of `hybrid` that pulls its response to changed precipitation and temperature
toward that of `5_px` (the last epoch kept); `lstm` has no physics input. Each seed takes a few
minutes on the GPU.

```bash
for k in 0 1 2; do
  sacsma dpl hybrid --physics 5_px --seed $k --out $L/hybrid/seed$k
  sacsma dpl hybrid --init-from $L/hybrid/seed$k/checkpoints/best.pt --response-lambda 0.18 \
      --epochs 15 --lr 1e-4 --warmup-epochs 1 --out $L/hybrid_dt/seed$k
  sacsma dpl hybrid --seed $k --out $L/lstm/seed$k
done
```

`--physics` and `--statics-from` (default `5_px`) take a run name for a tracked run, or the
checkpoint of a retrained one. Each command scores its own seed. The tracked score of a hybrid
model is that of its three seeds' mean flow (`metrics.csv` in
`artifacts/results/dpl/15cdec/<model>/`), written by a function rather than a command:

```bash
python -c "from sacsma.dpl.hybrid.evaluate import score_ensemble; score_ensemble('artifacts/models/dpl/15cdec/hybrid')"   # likewise hybrid_dt, lstm
```

**The studies** read the tracked runs and write to `artifacts/results/dpl/studies/<name>/`. The
hybrid reconstructions run on the GPU (`--device cpu` otherwise).

```bash
sacsma dpl study climatology       # each watershed's mean-monthly regime, one figure per ladder step
sacsma dpl study hybrids --regen   # the hybrids' skill and response surfaces
sacsma dpl study forcing           # the sensitivity to the forcing product
```

## 6. dPL-CalSim and the rim-inflow product

`<run>` below is any folder of dPL-CalSim: `artifacts/models/dpl/multifamily/7_calsim/` (the
checkpoints), `artifacts/results/dpl/multifamily/7_calsim/` (the scores) or
`artifacts/_local/runs/dpl/multifamily/7_calsim/` (the large local files); every command finds
the other two. The tracked folders hold the selected checkpoint and the scores, and
`artifacts/product/calsim3/` the product, so steps can be entered anywhere.

**Train** (about 20 hours on an 8 GB laptop GPU: 120 epochs of about 10 minutes, from the
`epoch_s` column of its `train_log.csv`). dPL-CalSim is rung 7 of the ladder: the recipe of
section 5 on 159 entities of every family (69 USGS gauges, 17 CDEC records, 9 unimpaired-flow
subbasins, 64 CalSim3 arcs) and every year but WY1976–85, with the unimpaired-flow targets back
to WY1950, the families weighted by footprint area and a fixed loss scale per family.

```bash
sacsma dpl train aef_u $R --domain multifamily --holdout-wy 1976-1985 $NOAH $PX \
    --uf-train-start 1949-10-01 --calsim-arcs train_default --mt-select-weight area \
    --mt-family-weight usgs=0.2366,cdec=0.5508,uf=0.1226,calsim=0.0900 --mt-share-norm all \
    --mt-loss-ref usgs=0.6374,cdec=0.3511,uf=0.1140,calsim=0.3548 --mt-loss-ref-power 0.5 \
    --out $L/7_calsim
```

A retrained run will not match the tracked one digit for digit: GPU arithmetic is not
reproducible across drivers. Two epochs of the same command on the same machine do repeat.

**Score** (CPU; the trained field runs over the whole record on the engine in double precision,
a few minutes). The commands read the held-out years only with `--score-holdout`: without it
`tier1` and `atlas` stop, and `tier2` writes its flows without scores.

```bash
sacsma dpl evaluate <run>/checkpoints/best.pt --score-holdout   # metrics.csv, metrics_holdout.csv, sim_daily.npz
sacsma dpl calsim tier1 <run> --score-holdout             # the 20 locations -> tier1/
sacsma dpl calsim tier2 <run> --components parts --score-holdout \
    --trace-python <python of the sacsma-gis environment> # all 196 arcs -> tier2/
```

Tier 2 needs the HydroSHEDS direction and accumulation tiles (default `tmp/hydrosheds`;
`build_flowlens.py` downloads them). Without `--trace-python` the arcs outside every trained
footprint get straight-line flow lengths instead of traced ones, and the output says so in its
`flowlen_method` column.

The scores go to the run's results folder, the monthly series, maps and figures behind them
to its local folder.

**The share model** is fitted on the base pass and on tier-2 passes at changed climates, one
pass each (into `tier2_scenarios/` of the local folder, `<local>`):

```bash
sacsma dpl calsim tier2 <run> --components parts \
    --extension-cells <local>/tier2/tier2_extension_cells.csv \
    --scenarios p85=0:0.85,p95=0:0.95,p105=0:1.05,p115=0:1.15,t1=1:1,t25=2.5:1,t3=3:1,t4=4:1,t1p85=1:0.85,t1p115=1:1.15,t25p95=2.5:0.95,t25p105=2.5:1.05,t3p85=3:0.85,t3p115=3:1.15,t4p85=4:0.85,t4p115=4:1.15
sacsma dpl calsim product fit <run> --score-holdout       # -> artifacts/product/calsim3/
sacsma dpl calsim atlas <run> --score-holdout             # the validation atlas (HTML)
```

The fit runs on CPU and repeats exactly at the same thread count. The scenario passes stay
local.

**The product** is one series per forcing: a tier-2 pass over the whole forcing record
(spin-up on its first ten water years, then October 1915 to December 2018, into
`artifacts/_local/product/calsim3/<forcing>/tier2`), with the share model applied over its
complete water years:

```bash
for f in historical_livneh_unsplit wgen_product_a wgen_product_a_s12; do
  sacsma dpl calsim product apply --forcing $f --score-holdout   # the pass, then the share model
done                                                      # -> artifacts/product/calsim3/<forcing>/
sacsma verify product                                     # each tracked series repeats
```

The pass each series was made from is kept beside it (`artifacts/product/calsim3/<forcing>/tier2/`), so
that `sacsma verify product` can repeat it. After a refit, `apply` remakes every series. The
files each command writes are listed in the READMEs under [`artifacts/`](../artifacts/README.md).

## 7. Rebuilding data

Not needed for any result above. Each folder's README gives the commands, the inputs and the
checks. What depends on what:

1. **Forcing** (`data/inputs/forcing/`). `wgen_forcing.py --build-master` packs the raw
   per-cell files into a local master; `build_region_forcing.py` writes the three historical
   stores from it and from the release folder; `wgen_product_a_scenarios.py` and
   `aorc_region.py` add the scenario and AORC stores. All need sources outside the repository.
2. **Grid** (`data/inputs/grid/`). `build_region_grid.py` writes the cell list from the domain
   tables and the CalSim3 polygons. `download_gis.py` stages the raw rasters and
   `sample_gis.py region` samples every cell (`sacsma-gis`); `build_region_statics.py` then
   consolidates the per-cell attributes. `gee_aef_region.py` builds the embeddings.
3. **Targets.** `cdec_fnf.py`, `dwr_unimpaired.py --pdf <report>`, `usgs_flows.py` ingest their
   sources. Then `build_uf_depth.py`, then `build_fnf_depth.py` (it checks against the first).
4. **The registry** (`data/inputs/domains/multifamily/`), in this order:
   `build_entities.py` (once without the flag if there is no registry yet),
   `data/targets/calsim3/build_calsim_arcs.py`, `build_entities.py --calsim-arcs`,
   `build_entity_cells.py --calsim-arcs`, `build_flowlens.py --calsim-arcs` (`sacsma-gis`).

These builders reproduce their tracked tables byte for byte from the repository alone, and are
the check that a change to the data code or layout did no harm: `build_entities.py`,
`build_entity_cells.py`, `build_calsim_arcs.py`, `build_uf_depth.py`, `build_fnf_depth.py`.
`build_region_statics.py` does the same once the raster samples of `sample_gis.py region` are
on disk; they are not tracked.

## 8. What a clone cannot rebuild

| What | Why |
|---|---|
| The calibration domains and their GA optima (`artifacts/models/`), the monthly CalLite targets, the 15-CDEC daily gage file, the MATLAB simulations | Delivered with the original study archive. The one-time ingest scripts are in git history only. |
| The VIC series and the CalSim3 inflow and unimpaired series | Extracted from model output that is not in the repository. |
| `prcp_x10_artifacts.csv` | Most of it was derived against forcing stores that are not in the repository, and the rest by a later comparison. It is frozen. |
| The tracked `historical_livneh_unsplit.nc`, bit for bit | It was built, then patched in place when that table grew. A rebuild from the raw source should give the same values, but the tracked file is the one every result was made with. |
| The hand-kept tables | They are the source: the crosswalk, the tier-1 sets, the arc derivation table, the unimpaired-flow pour points, the daily mask. |
| Forcing, grid attributes, embeddings, ET and SWE products, BCM, USGS flows, DWR unimpaired flows | Rebuildable only with their external source: a release folder, the raw rasters, an Earth Engine project, a report PDF, or a sibling repository. |
| The BCM routing tables (`bcm_routing_params.csv`, `bcm_routing_check.csv`) | Extracted from the USGS workbook `CalBasins_v8_DWR_FNF_PRISM19.xlsx`, kept beside them but not tracked. `bcm_routing_params.py --check` checks them without it. |
| The exact tracked checkpoints | Retraining gives a different run in the last digits (section 6). |
| The runs that were not adopted | Kept locally under `artifacts/_local/runs/`, not tracked. [Runs](runs.md#tried-and-not-adopted) says what they showed. |
