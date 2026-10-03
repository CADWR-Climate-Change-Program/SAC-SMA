# data/targets/dwr_unimpaired: DWR's published Central Valley unimpaired flows

Monthly unimpaired flow (TAF) of the 24 Central Valley subbasins (UF 1 to 24), WY1922 to
2014, transcribed from Appendix B of DWR Bay-Delta Office, *Estimates of Natural and
Unimpaired Flows for the Central Valley of California: WY 1922-2014* (DRAFT, March 2016), the
fifth edition of the series that began as *Central Valley Natural Flow Data* (1980).
Unimpaired flow is measured flow adjusted to remove upstream storage, diversion, import and
export. Role: target. The `uf_monthly` family of the learned-parameter (dPL) models is fitted
to these series, and they are the source of the `9unimp` and `11obs` calibration targets of
[`data/targets/callite/`](../callite/README.md).

At the rim, the report's natural flow and its unimpaired flow are the same quantity: the
natural-flow estimate departs from unimpaired only on the valley floor (natural Delta inflow
21,533 against unimpaired 29,003 TAF/yr, Table 5-2). The report's SWAT rim simulation
(Appendix C) was calibrated to these series; it and the six Delta totals of Appendix B are in
[`data/reference/dwr_swat/`](../../reference/dwr_swat/README.md).

## Files

| File | What | Source |
|------|------|--------|
| `uf_monthly.csv` | `date, uf, flow_taf`: unimpaired flow, 24 subbasins x 1116 month-end dates (1921-10-31 to 2014-09-30) = 26784 rows. 0.5 MB | `dwr_unimpaired.py` (Tables B-1 to B-24) |
| `uf_locations.csv` | `uf, table, name, cdec_id, basin_11obs, basin_9unimp, n_arcs, arcs, area_mi2_calsim, has_swat, swat_scale_appendix_d, swat_partial, note`: each subbasin's calibration basin, CalSim3 arc set and arc-sum area, and the flags of its SWAT table | `dwr_unimpaired.py`, then corrected by hand (see below) |
| `uf_gauges.csv` | `uf, lat, lon, gauge_source, area_mi2_swat, area_source`: one pour point for each of the 18 arc-mapped subbasins (UF 7 has none by construction) and, for 16 of them, the area the report gives | Hand-maintained, never generated |
| `uf_monthly_mm.csv` | `uf, date, depth_mm`: `uf_monthly.csv` as mm/month over each subbasin's CalSim3 arc-sum area (`area_mi2_calsim`), the 18 arc-mapped subbasins, full record = 20088 rows | `build_uf_depth.py` |
| `verification/` | The checks of `uf_locations.csv` against independent sources (below) | `check_uf_locations.py` |

`verification/` holds `findings.md` (the flags and notes, per subbasin), `report_table.csv`
(the numbers behind every check: areas, volumes, outlets), `uf_outlets.csv` (the USGS gauge,
and the CDEC coordinates where a station exists, that identifies each subbasin's outlet in the
checks: site, name, coordinates, published drainage area; an output of the checks, not an input
of any model), `figures/uf_NN.png` and `figures/uf_dissolved_overview.png` (one map per
subbasin with its member arcs and pinned outlet, and all subbasins dissolved onto one
overview), and `web_cache.json` and `nldi_bend_basin.json` (the cached NWIS, CDEC and NLDI
responses, so the tables regenerate offline; delete them to fetch again). The script also
writes `uf_dissolved.gpkg` for GIS viewing, which is not tracked.

## How it is built

The report has no machine-readable release: `dwr_unimpaired.py` parses the text layer of the
PDF, which is not redistributed here (pass it with `--pdf`). Environment `sacsma`, which
includes `pypdf`.

    python data/targets/dwr_unimpaired/dwr_unimpaired.py --pdf <report.pdf>
    python data/targets/dwr_unimpaired/dwr_unimpaired.py --verify
    python data/targets/dwr_unimpaired/build_uf_depth.py
    python data/targets/dwr_unimpaired/check_uf_locations.py

- `dwr_unimpaired.py` writes `uf_monthly.csv` and `uf_locations.csv` here, and
  `swat_monthly.csv` and `delta_monthly.csv` to `data/reference/dwr_swat/`.
- caution: the tracked `uf_locations.csv` holds hand corrections the script does not write:
  the whole arc set, area and note of UF 7, and the note of UF 21. A run with `--pdf`
  rewrites the table without them; restore those cells from version control.
- `build_uf_depth.py` needs only tracked files and reproduces its table byte for byte. Run it
  before `data/targets/cdec/build_fnf_depth.py`, which reads its output for a check.
- `check_uf_locations.py` writes its findings, table and maps to `verification/`. Its web
  responses (NWIS, CDEC, NLDI) are cached there, so it reruns offline and reproduces its
  tables byte for byte. It needs geopandas (environment `sacsma` or `sacsma-gis`).

## Checks

- Ingest gates of `dwr_unimpaired.py`: every table complete for the 93 water years, and the
  report's own identities hold within rounding: B-25 = sum of UF 1 to 11, B-26 = sum of UF 12
  to 15, B-27 = sum of UF 16 to 24, B-28 = B-25 + B-26 + B-27.
- `--verify` scores each series against `data/targets/callite/fnf_<domain>_monthly.csv`:
  r = 1.000000 on 16 of the 18 mapped basins. The residual is area normalisation only, and
  the mean ratio recovers the area DWR normalised by: UF 6 / BND 8900 mi2 (Sacramento R.
  above Bend Bridge), UF 8 / FTO 3607 mi2 (Feather R. at Oroville), UF 11 / AMF 1885 mi2
  (American R. at Fair Oaks). Those are the official gauge drainage areas, so the
  identification does not rest on the name match. UF 3 and UF 4 are not exact (below).
- `build_uf_depth.py` gates: 18 subbasins, no duplicate, no missing or negative depth,
  WY1985 to 2014 complete (360 months each).
- `check_uf_locations.py` tests `uf_locations.csv` against sources that did not produce it:
  the arc-summed CalSim3 inflow against the published volumes over WY1950 to 1984, the NWIS
  name, coordinates and drainage area of each outlet gauge (and CDEC coordinates), and the
  dissolved geometry. Result: 3 flags, 4 notes. The flags are volume closures: UF 3 +16.3 %
  (monthly r 0.877), UF 6 -11.3 % (r 0.998), UF 10 +12.2 % (r 0.997). A flag is a check
  outside tolerance, not automatically an error in the table; a note records an expected or
  structural condition.
- Paynes Creek: `I_PYN001` overlaps the NLDI-delineated Bend Bridge watershed (USGS 11377100)
  by 0.4 % of its area, against 99.9 % for a true member, so the creek joins below the gauge
  and is rightly left out of UF 6.

## Know before using

- Mapping to CalSim3. 17 subbasins resolve to a calibration basin and its arcs through
  `data/inputs/calsim3/calsim_crosswalk.csv` (`sacsma.calsim.catchments.derive_basin_nodes`),
  so basin nesting applies: UF 6 (Bend Bridge) picks up Shasta's `I_SHSTA`, plus the
  series-less valley-accretion node `I_SRBB_VAL`. UF 4 carries two basins, BLB and StonyCreek.
  UF 7 has no calibration basin; its eight arcs follow the report's p. 3-5 definition (Joint
  Depletion Study Areas 6, 7, 8, 9 and 14). That makes 18 arc-mapped subbasins.
- The other six (UF 1, 5, 12, 17, 23, 24: the two valley floors, three minor-stream groups,
  Tulare Lake Basin outflow) have no arcs. They cover real CalSim3 catchments, but no
  authoritative subbasin-to-arc assignment exists, and none was invented. Neither Shasta
  (SHA) nor Trinity (TNL) has a table of its own: Shasta is inside UF 6, and the Trinity is
  not a Central Valley subbasin.
- caution: UF 4 is not BLB. The `11obs` BLB target spans 1994 to 2014 and correlates only
  loosely with UF 4 (annual r 0.07 to 0.997, ratio 0.63 to 6.2): it is a gauged Black Butte
  reservoir-inflow record. UF 4 transcribes to StonyCreek (`9unimp`) exactly: same watershed,
  same arcs, different record.
- UF 3 / CacheCreek matches exactly in 88 of 89 water years; only WY2010 departs. The
  published UF 3 series is the routed outflow below Clear Lake and Indian Valley, while its
  four arcs are the inflows CalSim3 routes through those lakes, hence its volume flag.
- UF 6 and UF 7 include valley-floor accretion that no rim arc carries. UF 7's arc sum is
  1327.8 against 1380.1 TAF/yr unimpaired over WY1950 to 1984 (-3.8 %, r 0.9988).
- UF 21 is the Fresno River gauged near Daulton, below Hidden Dam. `I_HNSLY` alone, the
  inflow to the dam, undercounts it by 21.7 %; with the three below-dam catchments
  (`I_FRS046`, `I_DBC024`, `I_COT033`) the set closes to -1.7 %, r 0.9935 against CalSim3
  over WY1950 to 1984.
- UF 14 leaves out `I_PARDE` (the increment between the Mokelumne Hill gauge and Pardee Dam):
  the published series matches the gauge footprint (544 mi2) exactly. UF 11 includes
  `I_RUB002`, which has a series but no polygon of its own.
- Published defect, stored as printed: monthly values and the printed annual total are
  rounded independently, so months need not sum to the total (in Appendix B, 1085 of 2790
  rows differ by up to 3 TAF). Only the months are stored. The defect of Table B-29, a Delta
  total, is in [that series' README](../../reference/dwr_swat/README.md).
- `uf_monthly_mm.csv` uses the CalSim3 arc-sum areas as its depth basis; the Appendix A SWAT
  areas (`area_mi2_swat`) are the alternative. Dates are month-end stamps. The table is keyed
  by UF number; training windows live in the entity registry, not here.
- `uf_gauges.csv`: 14 of the 16 areas are the Appendix A SWAT model areas (Tables A-1, A-2);
  UF 8 and UF 15 draw on the Chapter 3 text, as their `area_source` cells state; UF 6 and UF 7
  have no usable single model. `gauge_source` is `cdec_stations:<id>` (copied from
  `data/targets/cdec/stations.csv` and checked against it when the registry is built) or
  `USGS NWIS <gid>` (the NWIS site coordinates). The pour point is where the arc set drains
  (the dam or gauge the registry takes as outlet), so it can sit several km from the station
  the subbasin is named after (up to about 14 km, UF 16).
- `has_swat`, `swat_scale_appendix_d` and `swat_partial` describe the SWAT tables; that
  README explains them.

## Read by

- `uf_monthly_mm.csv`: `sacsma.dpl.multi_timescale` and `sacsma.dpl.evaluate_multi_timescale`
  (the `uf_monthly` entities, through the registry's `obs_store` column).
- `uf_monthly.csv`: `sacsma.dpl.calsim.arcs` (closure anchors) and `sacsma.dpl.calsim.tier1`.
- All but `uf_monthly_mm.csv`: `data/inputs/domains/multifamily/build_entities.py`.
- `uf_locations.csv`: also `data/targets/calsim3/build_calsim_arcs.py` and
  `data/targets/cdec/build_fnf_depth.py`.
