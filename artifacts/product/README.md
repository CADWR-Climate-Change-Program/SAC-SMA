# `artifacts/product/`: what is delivered

One folder per application, and in each one folder per forcing, named as `--forcing` names it.
The forcings are described in [`data/inputs/forcing/`](../../data/inputs/forcing/README.md).

| Folder | What | Model | Forcings | Written by |
|---|---|---|---|---|
| [`calsim3/`](calsim3/README.md) | Monthly flow, in TAF, on the 196 CalSim3 rim arcs, WY1916–2018 | dPL-CalSim and the share model | `historical_livneh_unsplit`, `wgen_product_a`, `wgen_product_a_s12` | `sacsma dpl calsim product` |
| [`callite/`](callite/README.md) | The three CalLite inflow files (9, 11 and 12 watersheds), monthly, October 1921 to September 2018 | the archived GA calibrations of `9unimp`, `11obs`, `12rim` | the same three | `sacsma product callite` |
| [`15cdec/`](15cdec/README.md) | Daily flow of the 15 CDEC watersheds, in mm and cfs, October 1921 to September 2018 | the archived GA calibration of `15cdec` | `historical_livneh_unsplit` | `sacsma product 15cdec` |

`sacsma verify product` writes every series again and compares it with the tracked one.
