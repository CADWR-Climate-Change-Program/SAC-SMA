# CalSim3 FNF basis & the corrected (GIS-screened) anchor footprint

**Status: methods note.** Documents how the SAC-SMA CalLite
calibration sets (`11obs`, `9unimp`) relate to CalSim3's own unimpaired full-natural-flow
(FNF), and the **canonical corrected footprint** — a GIS-screened anchor that scores each
basin on its true CalSim catchment instead of the full (often over-reaching) HRU set.

**As of 2026-07-07 the corrected footprint IS the official anchor basis**:
`anchor_metrics.csv`/`anchor_monthly.csv`, every anchor-skill figure (incl. the pre/post-1950
splits and rolling skill), the maps, and the sub-arc QMAP mass-balance target all run on it —
consistent with the per-catchment sub-arcs and VIC's `no_gooselake` substitution. The original
**full-HRU-footprint** view is kept as the parallel artifact (`anchor_metrics_full.csv` +
`anchor_monthly_full.csv`, `compare.make_anchor_full`). 15cdec keeps its full footprint (its
HRUs are off the 1/16° grid; it is not an `ANCHOR_SET`). The `fnf_<domain>_monthly.csv`
calibration basis and the fnf-target diagnostics are untouched by the anchor basis.

Consolidates and **corrects** the earlier SHA/BND-only draft (its deep-dive is preserved in
§6–§8). **One earlier conclusion is corrected here** (§3): the "`fnf_11obs` reads +12% high vs
CalSim3 for SHA/BND" finding was an *area-basis artifact* (published vs CalSim drainage area),
not a real bias in the calibration target.

**Follow-up (2026-07-06):** `target_vs_calsim3.csv` now carries an `area_pub` column and a
per-basin `class` (§3) that separates *area-basis artifacts* from *real product offsets*. Three
anchor basins (**ChowchillaRiver, SNS, YRS**) turn out to be the same kind of artifact as the
SHA/BND +12% — their target depth was normalized on the **published** area but scored on the
CalSim catchment area. These are **documented, not re-normalized**: editing the shared
`obs_mm` depth would inject a spurious offset (= the area ratio) into the depth-based fnf
calibration `skill_summary` without changing real model skill, so the honest floor is left
visible and merely labelled (see §3).

---

## 0. TL;DR

1. **The CalSim3 unimpaired FNF is already the anchor reference.** The cross-compare
   (`_anchor_set_taf`) scores `11obs`/`9unimp` against `calsim_unimpaired_monthly.csv`
   (`FLOW-UNIMPAIRED` per rim system) for anchor basins, else the summed `I_` INFLOW. That
   file is **bit-identical** (pbias +0.00%, r=1.0000, all 10 systems) to the CalSim3
   stochastic-pipeline's `cs3_val`, extended to 1920–2021.
2. **The calibration target itself ≈ CalSim3** on the canonical CalSim area — within a few %
   for most basins (`target_vs_calsim3.csv`). The old +12% SHA/BND number came from dividing
   the observed FNF depth by the larger *published* drainage area (SHA 7470 mi²) instead of
   the CalSim catchment area (6588 mi²).
3. **The full HRU footprint over-reaches the true CalSim catchment** for a few basins (SHA
   +12.4% area, SNS/BND +6%, ChowchillaRiver/FresnoRiver +13–18%). On the full footprint the
   out-of-catchment (low-yield) HRUs dilute the basin mean depth.
4. **Screening to the true catchment** (`screened_footprint`) fixes the diluted basins —
   SHA −8.9%→+0.1%, SNS −7.8%→−0.8%, ChowchillaRiver −14.3%→−4.3% — and for BND and
   FresnoRiver it *removes a compensating low-yield dilution to expose an honest
   over-prediction* (BND −1.9%→+4.8%, FresnoRiver +11.3%→+29.4%), consistent with the +36%
   tributary bias documented in §8. Everywhere else the change is <1 pp.
5. **The screened footprint is the official anchor basis (2026-07-07)** — every anchor
   product (metrics CSVs, skill dumbbells, period splits, rolling skill, maps, the QMAP
   mass-balance target) runs on it; the full footprint survives as the parallel
   `anchor_*_full.csv` view (§5).

---

## 1. The three CalSim3-adjacent datasets (don't confuse them)

| Dataset | File | What it is | Used by |
|---|---|---|---|
| **Unimpaired FNF** | `calsim_unimpaired_monthly.csv` | CalSim3 `FLOW-UNIMPAIRED` per rim **system** (SHAS, SRBB, OROV, YUBA, FOLS, ST, TU, ME, SJ, TRIN, WH); the whole-watershed unimpaired flow **incl. valley-floor accretion** | anchor reference for rim basins |
| **INFLOW sub-nodes** | `calsim3_inflow_monthly.csv` | CalSim3 `INFLOW` per `I_<node>` (219 arcs); the sum of a system's sub-arcs **excludes** ungauged valley accretion | per-catchment view; anchor reference for secondary basins (sum) |
| **Pipeline `cs3_val`** | (sibling repo `calsim_qmap_validation_TS.csv`) | the CalSim3-baseline historical input the stochastic pipeline validates against, 1971–2018 — **identical** to `calsim_unimpaired_monthly` for `UNIMP_<sys>` | cross-check only |

The distinction that matters: for Sac R @ **Bend Bridge**, the unimpaired `UNIMP_SRBB`
exceeds the sum of its `I_` sub-arcs by the ungauged valley-floor accretion (~80 TAF/mo,
§8) — so the anchor uses the unimpaired series, not the sub-arc sum, for rim systems.

---

## 2. CDEC ground-truth (the observed FNF is real, and matches CalSim3)

Pulled live from CDEC (2026-07-06): Shasta = station **SIS**, Bend Bridge = **SBB**
(sensor 8 = daily FNF, sensor 65 = monthly FNF; merge on a normalized month period — CDEC's
`OBS DATE` mixes month-start/end day stamps). CalSim3's own official input matches CDEC's
live FNF closely (`I_SHSTA`/`UNIMP_SRBB` vs CDEC sensor 65: +0.5% / +4.8%), and our
`data/cdec15/gage.csv` is an essentially exact copy of CDEC's daily feed (SHA +0.1%, BND
0.0%). So CDEC ≈ CalSim3 ≈ our gage — there is no accounting gap between these.

**Where the downloaded CDEC FNF lives:** `data/cdec15/gage.csv` — the daily observed CDEC
full-natural-flow (15 basins, WY1987–2019), converted from raw cfs to **mm/day** over
`data/cdec15/basin_area.csv` (the CDEC/study catchment areas — SHA 6665, BND 8900 mi², which is
why it matches CDEC, not the published 7470/9978). It is the 15cdec calibration target
(`cdec15.load_gage`). The raw per-basin `FNF_<CODE>_cfs.txt` downloads were consumed by the
retired ingest (`dataprep.build_gage`) and live in git history (checkpoint `ad89558`).

---

## 3. The calibration TARGET vs CalSim3 (`target_vs_calsim3.csv`) — corrects the old +12%

The CalLite sets are GA-calibrated to an observed monthly FNF **depth** (`fnf_<domain>_monthly`
`obs_mm`). Expressed on the **same canonical CalSim catchment area** the cross-compare uses for
the model, that target volume is close to CalSim3's unimpaired FNF for most basins.

`r ≈ 1.000` almost everywhere: the observed FNF target *is* essentially CalSim3's own series
in shape — the only question is the volume scale, which depends on the area the depth is
multiplied by. `area_implied` (the area at which target volume = CalSim3) reveals which area the
`obs_mm` depth was actually normalized on. Comparing it to `area_gis` (CalSim catchment, what the
cross-compare multiplies by) and `area_pub` (published drainage area) classifies every basin:

| `class` | meaning | basins (target pbias vs CalSim3) |
|---|---|---|
| **consistent** | `area_implied` ≈ `area_gis` (<3%): target already on the CalSim area | SHA −1.1, BND +2.1, FTO −0.1, AMF −1.1, MRC −0.4, SJF −2.2, TLG −0.7, TNL −0.8 (11obs); Cosumnes +0.6, Mokelumne −1.4, Putah +2.1 (9unimp) |
| **area_artifact** | `area_implied` ≈ `area_pub`, **not** `area_gis`: target normalized on the *published* area but scored on the CalSim catchment | **ChowchillaRiver −10.2** (implied 261.6 ≈ pub 264.5), **SNS −9.3** (996 ≈ 980), **YRS +8.5** (1100 ≈ 1110) |
| **product_offset** | `area_implied` off **both** areas: a genuine historical-FNF-vs-CalSim3 volume difference no area choice fixes | CacheCreek −11.0, BearRiver −8.4, StonyCreek −4.8, CalaverasRiver +4.8, FresnoRiver +7.2 (9unimp); BLB −7.8 (11obs) |

- **SHA** is `consistent`: `area_implied` 6665 ≈ CalSim 6588 mi² (published 7470). On the CalSim
  area the target reads −1.1%; on the **published** area it reads +12.0%. **The old note's "+12%
  target bias" was this published-vs-CalSim area mismatch, not a real discrepancy in the FNF
  product.** BND is the same (8900 ≈ 9084, published 9978 → +2.1%).
- **ChowchillaRiver, SNS, YRS** are the *inverse* case — `area_implied` lands on the **published**
  area, so scoring their target on the (smaller/larger) CalSim area produces the −10/−9/+8% floor.
  This is an area-accounting artifact, identical in kind to SHA/BND's +12%.

**Why these are documented, not corrected.** Re-normalizing an `area_artifact` target's `obs_mm`
depth (× `area_pub`/`area_gis`) would zero its `target_vs_calsim3` bias, but `obs_mm` is also the
target of the **depth-based** fnf calibration diagnostic (`_make_calib_monthly`, no area) — where
the full-footprint model was GA-calibrated to the *original* depth. Rescaling the depth there just
injects the area ratio as a spurious skill offset (Chowchilla +0.0%→−11.2%, SNS +1.1%→−6.8%,
YRS +0.1%→+7.7% CAL pbias) without changing real model skill. So the `class` column **labels** the
artifact and the honest floor is left visible (repo policy), rather than editing the shared depth.
The corrected-*footprint* anchor (§4–§5) is the sanctioned way to put the model on the true
CalSim catchment; for Chowchilla it already recovers −14.3%→−4.3% (§5) without touching any target.

`target pbias` is the floor a *perfect-fit* model inherits from its target alone — so the
`product_offset` basins carry a real target-vs-CalSim3 offset independent of model skill, while the
`area_artifact` basins' floor is purely the published-vs-CalSim area basis.

---

## 4. The corrected footprint (`screened_footprint_<domain>.csv`)

`catchments.screened_footprint(domain)` keeps, per basin, only the HRUs whose grid-cell
footprint overlaps the basin's **true CalSim GIS catchment** (`derive_basin_nodes` arcs + the
`<SYS>_VAL` valley-accretion node, on the merged layer `CalSim3_Merged`), weighted by that
**overlap area** instead of the domain's own `area_weight`. It is deterministic from tracked
data (`calsim3.gpkg` + `calsim_crosswalk.csv`) — the same overlap the per-sub-arc cross-compare
already uses. How much each basin's full footprint over-reaches the CalSim catchment:

| set | basin | full HRU area | CalSim area | over-reach | HRUs kept |
|---|---|---|---|---|---|
| 11obs | SHA | 7406.7 | 6588.5 | **+12.4%** | 491/564 |
| 11obs | BND | 9668.9 | 9083.7 | +6.4% | 671/744 |
| 11obs | SNS | 960.1 | 903.7 | +6.2% | 87/95 |
| 9unimp | FresnoRiver | 276.8 | 235.6 | **+17.5%** | 29/32 |
| 9unimp | ChowchillaRiver | 264.5 | 234.8 | +12.6% | 30/33 |
| (all others) | | | | within ±6% | most/all kept |

---

## 5. Full → screened anchor (`anchor_screened_vs_full.csv`)

Same basins, same CalSim3 unimpaired-FNF reference, full period; only the footprint (and its
weights) change. Since the 2026-07-07 flip, `pbias_screened` reproduces the **official**
`anchor_metrics.csv` and `pbias_full` the parallel `anchor_metrics_full.csv`. The table/figure
also carry the **VIC benchmark** (`pbias_vic`/`kge_vic`, purple diamond) on the same months and
reference — VIC solves the same footprint problem by series substitution (`no_gooselake` at
SHA/BND), so the fair pairing is **VIC vs the screened SAC**, not the full.

| set | basin | pbias full → screened | KGE full → screened | VIC pbias / KGE | read |
|---|---|---|---|---|---|
| 11obs | **SHA** | −8.9 → **+0.1** | 0.869 → 0.939 | +2.4 / 0.781 | dilution removed — clear fix; screened SAC beats VIC |
| 11obs | **SNS** | −7.8 → **−0.8** | 0.878 → 0.958 | +18.2 / 0.717 | same, clear fix |
| 9unimp | **ChowchillaRiver** | −14.3 → **−4.3** | 0.790 → 0.921 | +41.1 / 0.547 | same, clear fix |
| 11obs | **BND** | −1.9 → **+4.8** | 0.958 → 0.887 | **−2.1** / 0.820 | honest: compensating dilution removed (§8); VIC closer on volume, SAC better KGE |
| 9unimp | **FresnoRiver** | +11.3 → **+29.4** | 0.870 → 0.674 | +95.0 / −0.181 | honest: exposes real over-prediction (VIC far worse) |
| 11obs | YRS | +6.3 → +4.5 | 0.898 → 0.921 | +6.4 / 0.802 | small improvement |
| (all others) | | Δ < 1 pp | ~unchanged | | negligible footprint mismatch |

*(pbias/KGE per the committed `anchor_screened_vs_full.csv`; an earlier draft of this table
carried slightly different values from the investigation-phase run.)*

**Interpretation.** Screening removes out-of-catchment low-yield HRUs, which *raises* the basin
mean depth. Where the full footprint's dilution was masking a low bias (SHA/SNS/Chowchilla),
the screened number improves. Where the full footprint's dilution happened to *cancel* a real
positive bias (BND, FresnoRiver), removing it exposes the honest over-prediction — consistent
with the repo policy of leaving honest depth biases visible. The BND/FresnoRiver "worsening" is
therefore the more defensible number, not a regression. Against VIC, the screened SAC wins
clearly on the 9unimp creeks (VIC runs +9 to +95% high there) and on SHA; BND is the one anchor
where VIC's volume is closer (−2.1% vs +4.8%), though SAC keeps the better KGE (0.887 vs 0.820).

---

## 6. SHA/BND deep-dive — the 15cdec pooled-calibration shortfall

The investigation began with the 15cdec-vs-`fnf_11obs` monthly cross-check
(`make_cdec15_fnf_check`), where SHA/BND ran −24%/−26% — far worse than the other matched
basins. That decomposes as **15cdec pooled-GA model bias × benchmark/area effects**:

- 15cdec's *own* model, scored monthly against its own (CDEC-sourced) gage over its original
  cal/val split, runs **−15.3%/−16.7% (SHA)** and **−16.5%/−18.3% (BND)** — a genuine,
  basin-specific shortfall of the pooled GA optimum (the other 13 basins run −5% to +10%).
- The remaining factor was attributed in the draft to a "+11% gage-vs-FNF benchmark gap"; §3
  now shows most of that was the published-vs-CalSim **area** basis, not a real FNF discrepancy.

11obs's *per-watershed* GA model, on the correct screened footprint, matches CalSim3's true
input to +0.3% (SHA) / +4.5% (BND) — so for SHA essentially the whole 15cdec–11obs gap is the
pooled-vs-per-watershed calibration choice, once area and footprint are handled correctly.

---

## 7. Goose Lake / Modoc plateau (why SHA/BND over-reach)

15cdec's HRUs for SHA/BND stop at ~lat 41.75 N; 11obs's extend to 42.41 N (Goose Lake / Modoc
plateau, CA/OR border) — arid, endorheic, **low-yield relative to area** (consistent with the
`VIC no_gooselake` substitution already used for these two basins). Those are exactly the HRUs
the GIS screening drops (SHA 73 of 564, BND 73 of 744). 15cdec's HRU points are **not** on the
1/16° Livneh grid, so a direct HRU-key intersection with 11obs is impossible — the screening
uses the CalSim GIS catchment polygon instead.

---

## 8. BND residual, decomposed by tributary

BND's screened +4.5% is not diffuse — it concentrates in the 7 intervening rim tributaries
(Battle, Cow, Clear, Cottonwood, Whiskeytown, …), each scored against its own CalSim3 `I_<node>`
actual input:

| node | area (mi²) | KGE | pbias |
|---|---|---|---|
| BCN010 | 73.5 | 0.475 | +50.1% |
| BTL006 (Battle Cr) | 360.4 | −0.166 | +53.9% |
| CLR011 (Clear Cr) | 28.3 | 0.760 | −9.6% |
| COW014 (Cow Cr) | 384.1 | 0.749 | +20.7% |
| CWD018 (Cottonwood) | 386.5 | 0.722 | +20.9% |
| SCW008 (S. Cow Cr) | 364.8 | 0.364 | +63.0% |
| WKYTN (Whiskeytown) | 189.8 | 0.807 | +18.0% |

**Area-weighted mean +36.5%** — every tributary except tiny CLR011 runs high, three badly
(KGE ≤ 0.48). These small, flashy, rain-dominated foothill streams (vs snow-fed Shasta) are the
source of BND's honest positive bias. The valley-floor node `SRBB_VAL` (682 mi²) carries no
CalSim3 series; a closure estimate (`UNIMP_SRBB − I_SHSTA − Σtribs`) implies a consistently
positive accretion (mean 80.4 TAF/mo, 3% of months negative) — a real ungauged process.

---

## Files

- `data/calsim/screened_footprint_{11obs,9unimp}.csv` — the corrected footprint (`[basin, key,
  overlap_area_mi2]`), from `catchments.screened_footprint(..., write=True)`.
- `artifacts/calsim/compare/anchor_metrics.csv` + `anchor_monthly.csv` — the **official** anchor,
  on the screened footprint (since 2026-07-07); feeds every skill figure, the period splits,
  rolling skill, maps, and the sub-arc QMAP mass-balance target.
- `artifacts/calsim/compare/anchor_metrics_full.csv` + `anchor_monthly_full.csv` — the
  full-HRU-footprint anchor, kept as the **parallel** view (`compare.make_anchor_full`).
- `artifacts/calsim/compare/anchor_screened_vs_full.csv` (+ `figures/anchor_screened_vs_full.png`)
  — the full→screened comparison + VIC benchmark (§5).
- `artifacts/calsim/compare/target_vs_calsim3.csv` — the calibration-target-vs-CalSim3 table (§3),
  with `area_gis`/`area_implied`/`area_pub` and a per-basin `class`
  (`consistent`/`area_artifact`/`product_offset`).
- **Unchanged by the anchor basis:** `fnf_<domain>_monthly.csv` (the calibration basis), the
  fnf-target diagnostics under `artifacts/calsim/<domain>/`, all physics.
