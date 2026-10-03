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
sacsma verify --quick    # imports, commands, links: no model run
sacsma verify            # adds parity and the product check
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
sacsma product callite                 # the CalLite inflow files, three forcings -> artifacts/product/callite/
sacsma product 15cdec                  # the daily flow of the 15 watersheds -> artifacts/product/15cdec/
```

All of it runs on CPU in minutes. `sacsma calsim --parallel` uses all cores for the model
runs; results are unchanged. [`artifacts/results/README.md`](../artifacts/results/README.md) lists every file
these commands write. Regenerated tables can differ from the tracked ones in the last digits,
and the row order of `monthly_calsets.csv` is not fixed; compare before committing.

## 5. Learned parameters on the 15 CDEC watersheds

The runs of [Learned parameters](learned_parameters.md) are tracked with their selected
checkpoints, so scoring needs no training:

```bash
sacsma dpl evaluate artifacts/models/dpl/15cdec/noah/checkpoints/best.pt
sacsma dpl hybrid --help               # the hybrid and LSTM runs
sacsma dpl study --help                # the studies -> artifacts/results/dpl/15cdec/studies/
```

A checkpoint stores its full configuration; `sacsma dpl train --help` lists every option, and
[Runs](runs.md) names the variant of each run. Training one
of these runs takes hours on an 8 GB GPU. Train one run at a time: the daily graph takes most
of the memory.

## 6. dPL-CalSim and the rim-inflow product

`<run>` below is any folder of dPL-CalSim: `artifacts/models/dpl/multifamily/noah_cdec_uf_usgs_cs64_ho7685_ufx_areaw_all_kref05_sacx_carry_px_aef/`
(the checkpoints), `artifacts/results/dpl/multifamily/...` (the scores) or
`artifacts/_local/runs/dpl/multifamily/...` (the large local files); every command finds the
other two. The tracked folders hold the selected checkpoint and the scores, and
`artifacts/product/calsim3/` the product, so steps can be entered anywhere.

**Train** (about 21 hours on an 8 GB GPU: 120 epochs of about 10 minutes):

```bash
sacsma dpl train aef --domain multifamily --et noah --noah-pet priestley_taylor --canopy-lite \
    --patience 10 --warmup-epochs 4 --chunk-grid water_year --no-flowlen-feature \
    --nograd-window 256 --train-graph-segments 2 --seed 0 --noah-sac-exchanges \
    --tbptt-carry relative --diagnostics --dead-chunk-nograd --learn-pxtemp --pxtemp-tau 1.0 \
    --obs-mask data/targets/cdec/fnf_daily_mask.csv --holdout-wy 1976-1985 \
    --uf-train-start 1949-10-01 --calsim-arcs train_default --mt-select-weight area \
    --mt-family-weight usgs=0.2366,cdec=0.5508,uf=0.1226,calsim=0.0900 --mt-share-norm all \
    --mt-loss-ref usgs=0.7388,cdec=0.4202,uf=0.1140,calsim=0.3549 --mt-loss-ref-power 0.5 \
    --epochs 120 --out <run>
```

A retrained run will not match the tracked one digit for digit: GPU arithmetic is not
reproducible across drivers. Two epochs of the same command on the same machine do repeat.

**Score** (GPU; the evaluation streams the whole record in double precision, about 45 minutes):

```bash
sacsma dpl evaluate <run>/checkpoints/best.pt             # metrics.csv, metrics_holdout.csv, sim_daily.npz
sacsma dpl calsim tier1 <run>                             # the 20 locations -> tier1/
sacsma dpl calsim tier2 <run> --components parts \
    --trace-python <python of the sacsma-gis environment> # all 196 arcs -> tier2/
```

Tier 2 needs the HydroSHEDS direction and accumulation tiles (default `tmp/hydrosheds`;
`build_flowlens.py` downloads them). Without `--trace-python` the arcs outside every trained
footprint get straight-line flow lengths instead of traced ones, and the output says so in its
`flowlen_method` column.

The scores go to the run's results folder, the monthly series, maps and figures behind them
to its local folder.

**The share model** is fitted on the base pass and on tier-2 passes at changed climates,
which are one batched run (into `tier2_scenarios/` of the local folder, `<local>`):

```bash
sacsma dpl calsim tier2 <run> --components parts \
    --extension-cells <local>/tier2/tier2_extension_cells.csv \
    --scenarios p85=0:0.85,p95=0:0.95,p105=0:1.05,p115=0:1.15,t1=1:1,t25=2.5:1,t3=3:1,t4=4:1,t1p85=1:0.85,t1p115=1:1.15,t25p95=2.5:0.95,t25p105=2.5:1.05,t3p85=3:0.85,t3p115=3:1.15,t4p85=4:0.85,t4p115=4:1.15
sacsma dpl calsim product fit <run>                       # -> artifacts/product/calsim3/
sacsma dpl calsim atlas <run>                             # the validation atlas (HTML)
```

The fit runs on CPU and repeats exactly at the same thread count. The scenario passes stay
local.

**The product** is one series per forcing: a tier-2 pass over the whole forcing record
(spin-up on its first ten water years, then October 1915 to December 2018), with the share
model applied over its complete water years:

```bash
for f in historical_livneh_unsplit wgen_product_a wgen_product_a_s12; do
  sacsma dpl calsim tier2 <run> --components parts --forcing $f --start 1915-10-01 \
      --extension-cells <local>/tier2/tier2_extension_cells.csv \
      --out artifacts/_local/product/calsim3/$f/tier2 --no-maps     # GPU, one at a time
  sacsma dpl calsim product apply --tier2 artifacts/_local/product/calsim3/$f/tier2 --forcing $f
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
| `prcp_x10_artifacts.csv` | Most of it was derived against stores that were since retired, and the rest by a later comparison. It is frozen. |
| The tracked `historical_livneh_unsplit.nc`, bit for bit | It was built, then patched in place when that table grew. A rebuild from the raw source should give the same values, but the tracked file is the one every result was made with. |
| The hand-kept tables | They are the source: the crosswalk, the tier-1 sets, the arc derivation table, the unimpaired-flow pour points, the daily mask. |
| Forcing, grid attributes, embeddings, ET and SWE products, BCM, USGS flows, DWR unimpaired flows | Rebuildable only with their external source: a release folder, the raw rasters, an Earth Engine project, a report PDF, or a sibling repository. |
| The exact tracked checkpoints | Retraining gives a different run in the last digits (section 6). |
| The runs that were not adopted | Kept locally under `artifacts/_local/runs/`, not tracked. [Runs](runs.md#tried-and-not-adopted) says what they showed. |
