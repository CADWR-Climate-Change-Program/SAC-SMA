# PART I — CURRENT SAC-SMA IMPLEMENTATION AND EVALUATION

## Introduction and purpose

The goal of the CalSim stochastic hydrology effort is a complete-domain,
best-available hydrology, meaning simulated natural streamflow for all CalSim inflow
arcs (roughly 200 sub-arc catchments), suitable for driving stochastic and
climate-perturbed CalSim simulations. Two process-based modeling lineages currently
cover this domain. The first is the VIC model used in CalSim3 development. The second
is the set of distributed SAC-SMA models developed by Wi and Steinschneider (Cornell
University) for DWR's watershed climate studies {cite:p}`wi2022,wimemo`.

Prior comparison of the two lineages against CalSim3's own historical inflows
motivated the present program. Across all ~200 arcs the VIC benchmark attains a median
monthly NSE near 0.72, while at the major "anchor" basins (Shasta through Millerton)
SAC-SMA attains mean monthly KGE of 0.87–0.92 versus 0.62–0.77 for VIC, with lower
volume bias and better seasonality (§5.2). On the other hand, VIC carries a more physically
informed potential evapotranspiration formulation (derived from Penman–Monteith),
whereas the current SAC-SMA uses Hamon PET, which is driven by temperature alone and
has no radiation term. SAC-SMA offers stronger calibrated skill; VIC offers ET physics
that are easier to defend under a changing climate. Closing that gap from the SAC-SMA
side is the agenda for both parts of this document.

Part I documents the current SAC-SMA implementation, covering model structure (§2),
parameterization and genetic-algorithm calibration (§3), the four calibration domains
(§4), a performance review including the cross-comparison against VIC and CalSim3
(§5), and two forcing-sensitivity analyses relevant to stochastic applications, namely
temperature detrending (§6) and the split-versus-unsplit precipitation lineage (§7).
Part II then describes the differentiable reimplementation and the path toward a
retrained, ET-upgraded, hybrid model for the full CalSim domain.

## Model description

### Model chain

The model is a spatially distributed implementation of the Sacramento Soil Moisture
Accounting model {cite:p}`burnash1973,burnash1995` with four coupled per-HRU components,
run at a daily step and area-weighted to the watershed outlet:

1. **Hamon potential evapotranspiration** {cite:p}`hamon1961`, scaled by a calibrated
   proportionality coefficient $K_{pet}$, with daylength from the CBM model
   {cite:p}`forsythe1995`;
2. **Snow-17** snow accumulation and ablation {cite:p}`anderson1973,anderson2006`, using a
   hard rain/snow split at the temperature threshold `PXTEMP`;
3. **SAC-SMA** soil-moisture accounting (16 parameters, six storages, the five-component
   evapotranspiration cascade E1–E5);
4. **Lohmann routing** {cite:p}`lohmann1996,lohmann1998`, a gamma-distribution hillslope unit
   hydrograph followed by a linearized Saint-Venant channel Green's function along the
   flow path to the outlet.

Snowpack outflow (rain plus melt) is the effective precipitation input to SAC-SMA.
SAC-SMA's surface and baseflow channel inflows are routed separately (baseflow bypasses
the hillslope unit hydrograph) and summed, and basin discharge is the area-weighted sum
of routed HRU runoff. Full governing equations for all four components, exactly as
implemented, are given in Appendix A.

### Spatial discretization (HRUs)

Following {cite:t}`wi2022`, each watershed is discretized into hydrologic response units (HRUs)
defined by intersecting the ~6-km climate grid with the STATSGO soil classes
{cite:p}`millerwhite1998` present in each grid cell, so that every HRU is a (grid cell × soil
class) polygon. Elevation is assigned from the SRTM 90-m DEM {cite:p}`jarvis2008` (HRU mean)
and vegetation class from the 1-km AVHRR global land cover of {cite:t}`hansen2010` (HRU
majority). The 15-CDEC domain comprises 7,891 HRUs drawing forcing from 6,033 distinct
grid cells.

Because snow processes are strongly elevation-dependent, HRU temperatures are downscaled
from the grid cell using monthly lapse rates derived from the MODIS MOD11A1 land-surface
temperature product {cite:p}`wan2014` over 2001–2019 (one set of monthly rates for all
watersheds, ranging from −4.4 °C/km in November to −6.5 °C/km in March), applied to the
elevation difference between the HRU and its parent grid cell {cite:p}`wimemo,immerzeel2014`.
This within-cell orographic downscaling becomes relevant again in Part II, where the
native-grid dPL domain must give it up.

### Forcing data

The default historical forcing (`historical_livneh_unsplit`) is daily, 1/16°
(~6 km), 1915–2018. It has two components:

- **Precipitation** from the extreme-preserving ("unsplit") gridded dataset of
  {cite:t}`pierce2021`, which follows the Livneh gridding method while omitting the time
  adjustment that splits gauge accumulations across days, an adjustment shown to mute
  daily extremes. Twenty-four physically implausible summer values (> 254 mm/day, an
  order of magnitude above nearby GHCN-Daily gauges) were rescaled downward by a factor
  of 10 {cite:p}`wimemo`.
- **Temperature** (Tmin/Tmax) from {cite:t}`livneh2013`, extended through 2018 with PRISM
  {cite:p}`prism2014` and bias-corrected to monthly PRISM over the full period.

All 1/16°-grid calibration domains read one unified store per forcing product,
covering the full CalSim3 footprint at 4,410 grid cells; daily mean temperature is
derived as (Tmax + Tmin)/2, and a small set of misplaced-decimal precipitation spikes
is corrected against an audit table carried alongside the store. The 15-CDEC HRU cloud
keeps its own dense product at the original off-grid HRU points.

Two alternate products exist for the CalSim calibration domains (not for the 15-CDEC
HRU cloud, whose points are off the 1/16° grid). They are the levers for the
sensitivity analyses of §6 and §7.

- **WGEN Product A** (`wgen_product_a`) carries precipitation identical to the unsplit
  baseline, with temperature detrended to a 1991–2020 reference. This is the
  historical-parallel sequence of the CalSim3 stochastic-input pipeline. The detrending
  warms the early record most (about +0.4 °C in the 1910s, tapering to near zero by the
  2010s).
- **Historical LTO** (`historical_lto`) is the observed-climate product of the CalSim3
  long-term-operations study, carrying the pre-correction "split" Livneh precipitation
  lineage. This is a different precipitation realization (daily correlation
  0.83–0.93 against unsplit; annual totals within ±7%), daily 1915–2021.

### Python port and numerical parity

The production implementation is a Python/NumPy/Numba port of the original MATLAB
codes. The port reproduces the archived MATLAB simulations exactly. For every
calibration domain, simulated flow matches the archived `simflow` reference with
KGE ≈ 1.0, a maximum daily difference of about 0.0009 mm/day for the 15-CDEC set, and
exactly 0.0 for the CalLite domains. Several idiosyncrasies of the original code are
preserved deliberately (variable sub-stepping, the quadratic ADIMP runoff ratio,
specific storage clamps, and the riparian-ET channel adjustments; see Appendix A)
because this bit-level fidelity is the regression baseline for all subsequent work. The
physics modules are frozen, and any change to the model or data path is verified
against `simflow` parity.

## Parameterization and GA calibration

### Parameter set

Each HRU carries the full 31-parameter SAC-SMA structure, comprising $K_{pet}$ (1),
soil-moisture accounting (16), Snow-17 (10), and Lohmann routing (4). Three parameters
are fixed by convention (`side` = 0, `SCF` = 1, `PXTEMP` = 0 °C); the remainder are
calibrated within the feasible ranges reproduced in Appendix A (Table A.1) {cite:p}`wimemo`.

### Parsimonious regionalization

Instead of per-HRU calibration, parameters are tied to landscape
classes so that the effective number of free parameters stays small and the model
regionalizes to ungauged locations {cite:p}`wimemo`. In the archived calibration file,
soil-moisture-accounting parameters vary by STATSGO soil class (13 classes across the
domain); $K_{pet}$ likewise takes a single value per soil class (11 distinct values
across the domain); and one set of snow and routing parameters is shared across all 15
watersheds.

This yields approximately 204 free parameters for the entire 15-watershed domain
(about 14 per watershed). HRUs with the same soil class anywhere in the domain share
parameters, which is what allows parameters calibrated on gauged watersheds to be
transferred to ungauged creeks and interior sub-basins.

### Genetic-algorithm calibration

Two calibration designs cover the four domains.

**Pooled daily calibration (CDEC15).** A single real-coded genetic algorithm
{cite:p}`wang1991` run (initial population 1,000, evolved 100 generations, so 100,000 model
evaluations) maximizes the *mean* KGE {cite:p}`gupta2009` across all 15 daily FNF series
simultaneously, following the pooled approach of {cite:t}`wi2015`. Calibration uses WY1989–2003
(BND uses WY2000–2003 due to its short FNF record); validation is WY2004–2018. Pooling
sacrifices some per-basin fit in exchange for robustness and regionalization, since the
same parameter-to-landscape mapping must work everywhere at once.

**Per-watershed monthly calibration (CalLite sets).** Each of the 32 CalLite watersheds
(§4) gets its own GA calibration, with daily simulation aggregated to monthly before
comparison to the monthly target. Calibration uses WY1952 onward (through 2010 for
Unimpaired9, 2013 for Observed11, and 2003 for Rim12), and validation uses the earlier
record, roughly WY1922–1951 {cite:p}`wimemo`. Per-watershed calibration fits each basin
closely in-sample, at the cost of losing cross-basin parameter coherence.

The archived optima from these MATLAB-era calibrations are the parameter sets the
Python port runs today. They are the "current implementation" evaluated in §5 and the
GA baseline for Part II.

### Regionalization evidence

The original study provides direct out-of-sample evidence that the pooled,
landscape-tied parameterization transfers {cite:p}`wimemo`. At upstream reservoirs never used
in calibration, SAC-SMA matched independent HEC-HMS reconstructions with NSE 0.75
(Hetch Hetchy), 0.76 (Cherry/Eleanor), and 0.62 (Spicer Meadow). Of 19 ungauged or
nearly ungauged small creeks parameterized purely by soil/vegetation transfer, the two
with usable records scored KGE above 0.8 (Big Dry Creek) and 0.66 (South Duck Creek).
This precedent matters for Part II's nested-training proposal, which formalizes
this kind of interior-point transfer.

## Calibration domains

Four calibration sets cover the CalSim domain (Table 1). They differ in target
variable, timestep, and calibration design, and they play different roles downstream.

| Set | N | Target | Timestep | Calibration | Role |
|---|---|---|---|---|---|
| **CDEC15** (`15cdec`) | 15 | CDEC full natural flow | daily | pooled GA, WY1989–2003 | daily-capable backbone; regionalization donor |
| **Rim12** (`12rim`) | 12 | reservoir-inflow series (impaired) | monthly | per-watershed GA, WY1952–2003 | CalLite main water inputs |
| **Observed11** (`11obs`) | 11 | unimpaired gauge FNF | monthly | per-watershed GA, WY1952–2013 | water-year-type gauges; anchor set |
| **Unimpaired9** (`9unimp`) | 9 | unimpaired creek FNF | monthly | per-watershed GA, WY1952–2010 | rain-driven valley creeks; anchor set |

*Table 1. The four calibration sets covering the CalSim domain.*

**CDEC15** comprises the major Sierra and Cascade reservoir watersheds (SHA, BND, ORO,
YRS, FOL, MKM, NHG, NML, TLG, MRC, MIL, PNF, TRM, SCC, ISB, north to south), with
published drainage areas from 363 mi² (NHG) to 8,900 mi² (BND).

**Rim12 and Observed11 largely share outlets** (e.g. SHAST↔SHA, TRINI↔TNL, OROVI↔FTO).
The Rim12 series are the impaired reservoir-inflow inputs CalLite actually ingests,
while the Observed11 series are unimpaired full-natural-flow estimates at (nearly) the
same points, used for water-year-type classification. Separate models are calibrated
because the two targets embody different flow definitions {cite:p}`wimemo`. **Unimpaired9**
covers the rain-dominated interior creeks (Bear, Cache, Calaveras, Chowchilla,
Cosumnes, Fresno, Mokelumne, Putah, Stony) that Rim12 does not account for.

### Mapping to CalSim3 and the anchor convention

A hand-maintained crosswalk maps each of the ~203 CalSim3 inflow arcs to its rim
system, its VIC benchmark basin, and the corresponding basin in each SAC-SMA set. The
cross-comparison of §5 scores each calibration set against CalSim3's own historical
inflows under a **basin-anchor** convention.

- For rim systems, the reference is the single whole-watershed **FLOW-UNIMPAIRED**
  series rather than a sum of INFLOW sub-arcs, because the sub-arc sum misses
  valley-floor accretion (for Sacramento at Bend Bridge the difference is ~12% of
  volume).
- For creeks and secondary basins, the reference is the sum of the basin's CalSim3
  INFLOW sub-arcs.
- Four basins (SHA, BND, SNS, Chowchilla) are footprint-screened before scoring
  because their calibrated HRU footprints materially over-reach the CalSim3 catchment.
  SHA and BND carry the endorheic Goose Lake block (~1,000 mi² that never reaches the
  gauge; the VIC benchmark applies its own `no_gooselake` correction for the same
  reason), while SNS and Chowchilla carry a delineation over-reach. All other basins
  run their full calibrated footprint, with basin volume placed on the canonical
  CalSim3 catchment area.
- Rim12 is excluded from the cross-comparison (impaired-target series with no
  crosswalk column). The Observed11 and Unimpaired9 sets form the official anchor
  basis, and CDEC15 is scored on its own parallel track.

## Performance review

This section reports skill on three distinct bases, in order. §5.1 scores each set
against its own calibration target on its own calibration and validation windows.
§5.2 scores every set against CalSim3 on the shared WY1950–2019 anchor basis,
alongside VIC. §5.3 scores every set on a single validation window common to all of
them, the pre-1950 months. §5.4 closes with the consistency of the calibration targets
themselves.

### Skill against each set's own calibration target

Table 2 gives mean performance against each set's own target, on the windows defined
in §3.3. Means are used throughout the summary tables so that outlier basins register
in the set-level numbers. CDEC15 is scored on daily flows; the CalLite sets are scored
on monthly flows. Note that the two designs put their held-out years on opposite ends
of the record. CDEC15 validates on the later record (WY2004–2018), while the CalLite
sets validate on the earlier record (roughly WY1922–1951, before their calibration
windows begin).

| Set | Cal window | Val window | Cal KGE | Val KGE | Cal \|pbias\| | Val \|pbias\| |
|---|---|---|---|---|---|---|
| CDEC15 (daily) | WY1989–2003 | WY2004–2018 | 0.80 | 0.77 | 8.7% | 11.7% |
| Unimpaired9 (monthly) | WY1952–2010 | WY1922–1951 | 0.96 | 0.84 | 0.6% | 10.4% |
| Observed11 (monthly) | WY1952–2013 | WY1922–1951 | 0.96 | 0.82 | 0.8% | 13.1% |
| Rim12 (monthly) | WY1952–2003 | WY1922–1951 | 0.91 | 0.79 | 3.3% | 12.8% |

*Table 2. Mean per-basin skill against each set's own calibration target, on each set's own windows. CDEC15 scored daily; CalLite sets scored monthly. \|pbias\| is the mean of per-basin absolute volume bias, so opposite-signed basin errors do not cancel.*

Two Observed11 basins depart from the standard windows. BLB's record begins in 1995
(calibration only, so its validation columns average over the remaining 10 basins),
and SHA's unimpaired series begins in 1987 (calibration window 1987–2013). The CDEC15
means carry the weight of its few hard basins; the median basin scores higher (0.85
cal, 0.81 val).

The per-watershed CalLite calibrations fit closely in-sample (KGE 0.95–0.98, volume
bias near zero), as expected for independently fitted basins. The pooled CDEC15
calibration is lower in-sample but loses little skill out of sample, consistent with a
regionalized rather than overfitted parameterization.

Standout basins, consistent with the original study {cite:p}`wimemo`:

- **SCC (Tule River)** is the weakest case of the pooled daily set, with cal KGE 0.46,
  val KGE −0.18, and +36%/+68% volume bias. The simulation is too flashy against the
  FNF target, and the original study found this basin sensitive to the choice of
  gridded climate product.
- **SHA and BND** carry a persistent −16 to −18% low bias against their gage-FNF
  targets in the pooled set (a structural rim-system bias explored further below).
- **TNL (Trinity)** is the Observed11 outlier (val KGE 0.42, +35% bias), and **TRINI**
  its Rim12 counterpart (val 0.50, +31%). Trinity reappears throughout this document
  as the one basin where precipitation-lineage choices matter materially (§7).
- **StonyCreek** (val 0.62, +28%) and **FresnoRiver** (val 0.66, +16%) are the
  Unimpaired9 weak spots.

### Cross-comparison: SAC-SMA vs VIC against CalSim3

The cross-comparison scores every set on one shared basis, monthly flows against the
CalSim3 anchor references over WY1950–2019 (Table 3; Figures 1 and 2). CDEC15's daily
output is aggregated to monthly here, so all three sets and VIC are directly
comparable.

| Set | KGE | VIC KGE | \|pbias\| | VIC \|pbias\| | Seas. mism. | VIC seas. mism. |
|---|---|---|---|---|---|---|
| Observed11 | 0.91 | 0.77 | 4.2% | 8.5% | 3.3% | 7.6% |
| Unimpaired9 | 0.92 | 0.62 | 4.9% | 27.3% | 5.6% | 7.7% |
| CDEC15 | 0.87 | 0.77 | 7.6% | 10.0% | 8.2% | 7.1% |

*Table 3. Mean basin-level skill against the CalSim3 anchor references, WY1950–2019, monthly. VIC columns are means over the same basins as each set; \|pbias\| is the mean absolute per-basin volume bias, and seasonal mismatch is the percentage of annual volume placed in the wrong month.*

The seasonal-mismatch metric is the percentage of annual volume placed in the wrong
month (the total-variation distance between normalized mean-monthly regimes; Appendix
A.5). It separates timing errors from volume errors in a way KGE alone does not. The
anchor sets misplace 3–6% of annual volume seasonally against VIC's 8%; on the CDEC15
basins the pooled set misplaces 8%, slightly more than VIC on the same basins (7%),
which reflects its reservoir-oriented pooled calibration.

```{figure} ../artifacts/calsim/compare/figures/anchor_skill_kge.png
:width: 6.2in

Figure 1. Basin-level monthly KGE versus the CalSim3 anchor (FLOW-UNIMPAIRED for rim systems, INFLOW sub-arc sum otherwise), WY1950–2019. SAC-SMA (circles, by calibration set) versus VIC (diamonds); the dashed divider separates the officially scored Observed11/Unimpaired9 anchor sets from the CDEC15 parallel.
```

SAC-SMA outperforms VIC at nearly every basin, by +0.15 mean KGE on the rim set and
+0.30 on the creeks. The creek-set gap is wide because the margin is largest where a
complete CalSim hydrology is hardest, in the small southern creeks, where VIC runs
wet. At Fresno River VIC carries a +95% volume bias (KGE −0.18) against SAC-SMA's +11%
(0.87); Cosumnes is +35% versus +1.5%, Chowchilla +41% versus −4%, and Calaveras +31%
versus +5%. Rim basins are closer (at Shasta, SAC-SMA 0.94 and VIC 0.78). The hardest
rim basin for both models is Trinity (SAC-SMA 0.77, VIC 0.57).

```{figure} ../artifacts/calsim/compare/figures/calsim_sacsma_minus_vic_kge.png
:width: 5.4in

Figure 2. SAC-SMA minus VIC basin-level KGE against the CalSim3 anchor (blue = SAC-SMA better). Every sub-area polygon is colored by its parent basin's anchor score.
```

Four method-level findings follow.

- **Basin-level aggregation stabilizes the comparison.** At individual CalSim3 nodes
  the per-catchment median KGE is about 0.67 for the SAC sets (VIC 0.66), because
  sub-arc-scale noise (delineation mismatches, extrapolated small nodes) dominates;
  aggregation to basin anchors cancels most of it. A per-arc quantile-mapping
  correction with mass-balance rescaling (trained WY1922–1971, scored on held-out
  WY1972–2018) lifts the sub-arc median KGE to 0.76–0.86 depending on set, with volume
  bias magnitudes of 4–11%. That correction is the working route from basin-credible
  models to arc-level hydrology.
- **The footprint screening affects only the four screened basins.** Screening moves
  Shasta's anchor bias from −8.9% to +0.1% (the Goose Lake cut) and Chowchilla's from
  −14.3% to −4.3%; every other basin is unchanged.
- **CDEC15's rim bias is real but specific.** Scored against CalSim3, the pooled set
  under-runs the two big Sacramento rim systems by −24% (SHA) and −22% (BND). This is
  the reservoir-calibrated pooled optimum's known weakness, and the reason CDEC15 is
  kept off the official anchor basis. Its Tulare-basin coverage (PNF, TRM, SCC, ISB),
  which no CalLite set provides, is unaffected.
- **Skill is stable in time.** In every 30-year rolling window since 1922, SAC-SMA's
  anchor-set KGE stays within 0.86–0.96 (volume bias within ±8%) and exceeds VIC in
  the same window throughout.

### A common validation window: pre-1950 monthly skill

The per-set numbers of §5.1 are not comparable across sets, because each set defines
its own windows, target, and timestep. The pre-1950 months provide a common basis.
They lie outside every calibration window (CDEC15 calibrates on WY1989–2003; the
CalLite sets calibrate on WY1952 onward), and all sets are scored monthly against the
same CalSim3 anchor references. Table 4 gives the anchor skill re-scored on the
pre-/post-WY1950 split.

| Set / model | KGE (pre) | \|pbias\| (pre) | Seas. (pre) | KGE (post) | \|pbias\| (post) | Seas. (post) |
|---|---|---|---|---|---|---|
| Observed11 | 0.84 | 10.7% | 5.1% | 0.92 | 4.0% | 3.3% |
| VIC, same basins | 0.70 | 17.1% | 7.3% | 0.78 | 8.0% | 7.9% |
| Unimpaired9 | 0.83 | 12.6% | 7.6% | 0.93 | 4.2% | 5.6% |
| VIC, same basins | 0.57 | 31.2% | 9.7% | 0.63 | 26.8% | 7.2% |
| CDEC15 | 0.82 | 12.9% | 9.8% | 0.88 | 7.0% | 7.7% |
| VIC, same basins | 0.75 | 15.8% | 7.3% | 0.77 | 10.3% | 7.2% |

*Table 4. Mean basin-level skill against the CalSim3 anchors, split at WY1950. All values are means over each set's basins (\|pbias\| is the mean absolute per-basin volume bias; Seas. is the seasonally misplaced percentage of annual volume); pre-1950 months lie outside every set's calibration window.*

Read as a common out-of-sample test, all three SAC-SMA sets hold up before 1950 (mean
KGE 0.82–0.84, mean absolute volume bias of 11–13%), and all three degrade somewhat
relative to their post-1950 skill. Part of that pre-1950 degradation is a forcing
signal rather than a model signal, as §7 shows. VIC degrades further on the same
months, with pre-1950 mean absolute volume biases of 16–31%. This pre-1950 monthly
window is also the ancestor of the validation design proposed for the retrained model
in Part II, which holds out WY1950–1987 monthly CalSim FNF entirely.

### Consistency of the calibration targets with CalSim3

Scoring every set against CalSim3 assumes the calibration targets themselves agree
with CalSim3's FNF. The repository quantifies that agreement directly
(`target_vs_calsim3.csv`). Each basin's observed-FNF target, placed on the canonical
CalSim3 catchment area, is scored against CalSim3's own flow. Most basins agree within
a few percent with correlation near 1.0, meaning the observed FNF targets and CalSim3
are in practice the same series.

One apparent exception deserves explanation, because it shaped the scoring
conventions. The Observed11 FNF targets at SHA and BND initially read about +12% high
against CalSim3. On investigation this turned out to be an area-accounting artifact
rather than any real inflation. The published drainage area for the SHA gauge (7,470
mi²) includes the endorheic Goose Lake block, while CalSim3's SHSTA catchment (6,588
mi²) does not, and dividing the same FNF volume by the two different areas produces
the apparent +12%. Placed on the CalSim area, the SHA target agrees with CalSim3 to
−1.1% and BND to +2.1%. The handling follows from that diagnosis. The calibration
targets were never adjusted; the model's over-reaching HRU footprint (which was
calibrated against the published-area target) is screened at scoring time (§4.1); and
the basin volume is placed on the canonical CalSim3 area.

A few genuine residuals remain and are documented rather than corrected. Three basins
carry the same area artifact in the opposite direction (Chowchilla −10.2%, SNS −9.3%,
YRS +8.5%, each traceable to the published-versus-CalSim area ratio), and a handful
carry a real product difference between the historical FNF record and CalSim3 (Cache
−11.0%, Bear −8.4%, Fresno +7.2%, among others). These offsets are the bias floor a
perfectly fitted model would inherit from its target. They are left visible in the
anchor scores because re-normalizing the target depths would corrupt the per-basin
calibration diagnostics that share the same tables.

## Warming sensitivity versus VIC (WGEN Product A)

Because the stochastic pipeline runs on detrended temperature, the response of each
hydrologic model to temperature detrending is itself a model property worth measuring.
The WGEN Product A forcing isolates it, since precipitation is identical to
the baseline and temperature is detrended to 1991–2020 (the early record warmed by up
to about 0.4 °C). Both models were re-run under both forcings with parameters
unchanged, and differences are reported as Detrended − Baseline on the shared
1915–2018 record, split at WY1950.

**SAC-SMA responds with a volume loss.** Warming the early record raises Hamon PET and
snow-season ET, and the model loses 2–4% of long-term runoff volume (median near −3%
per domain), concentrated where the detrending is largest. Per-basin losses run −9 to
−2.7% before 1950, tapering to about −2% after. Daily flow correlation stays at or
above 0.999, so the response is almost entirely in volume rather than timing.

**VIC responds with a timing shift.** VIC's aggregate volume response is only −2.5 to
+0.7% pre-1950; it mostly shifts snowmelt earlier instead (Figure 3).

```{figure} ../artifacts/calsim/forcing_compare/figures/agg_wgen_annual_diff.png
:width: 6.2in

Figure 3. Aggregate water-year runoff volume difference (Detrended − Baseline, 5-year rolling) for SAC-SMA and VIC over the disjoint CalLite watersheds. SAC-SMA loses volume where the detrending warms most; VIC's volume response is much smaller.
```

The contrast quantifies the liability of a temperature-only PET. Any temperature
adjustment feeds directly into ET and volume, whereas an energy-budget model mostly
re-times the season. This is a first-order consideration for stochastic and
climate-perturbed applications, and it motivates the PET and ET upgrades of Part II.
Per-watershed and per-period breakdowns, and the monthly regime versions, are in
Appendix B.4.

## Split versus unsplit precipitation (Historical LTO)

The CalSim3 LTO study's observed climate carries the pre-correction "split" Livneh
precipitation lineage, whereas the baseline here (and the VIC benchmark's historical
run) uses the {cite:t}`pierce2021` unsplit product. Re-running SAC-SMA under the LTO forcing
(and comparing the VIC parallels) measures how much this lineage choice matters.

**The difference is concentrated before 1950.** The median absolute water-year volume
difference is about 11% in 1915–1949 versus 1.5–3.2% in 1950–2018. Directionally, the
unsplit product is wetter in the southern and Sierra basins and drier in the northern
Sacramento systems before 1950 (spread up to ±21%), collapsing to within ±2% after
1950 everywhere except **Trinity (+8%)**. Trinity is the one basin where the lineage
choice remains consequential in the modern era, and it is also the weakest anchor
basin in §5.

**On the full record, anchor skill differs little between the two forcings** (median
anchor KGE 0.934 unsplit versus 0.926 split), but the split forcing redistributes
skill. It recovers the weakest pre-1950 cases (Trinity pre-1950 KGE 0.40 → 0.83; Cache
0.63 → 0.78; Stony 0.73 → 0.87) while degrading several basins that were already in
good agreement (Cosumnes 0.97 → 0.87, Calaveras 0.92 → 0.81, Yuba 0.90 → 0.85). The
practical reading is that pre-1950 disagreements with CalSim3 at a handful of basins
owe as much to the precipitation lineage as to the model, and conclusions drawn from
pre-1950 skill should carry that caveat (Figure 4). This also tempers the cross-set
comparison of §5.3, since some of every set's pre-1950 degradation is inherited from
the forcing.

```{figure} ../artifacts/calsim/forcing_compare/figures/split_unsplit_skill_boxplot.png
:width: 6.2in

Figure 4. Anchor-basin skill under the unsplit baseline (grey) versus the split-lineage LTO forcing (orange), pooled over the 20 anchor basins, by period.
```

Volume-by-period, annual-difference, and monthly-regime figures for both models, plus
the SAC/VIC aggregate comparison, are collected in Appendix B.5.
