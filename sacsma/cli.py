"""Command-line interface: ``sacsma run | plots | calsim``.

``run`` forward-simulates a watershed from its archived GA optimum; ``plots``
writes the per-watershed calibration/validation diagnostics for a domain; and
``calsim`` runs the CalSim3-vs-VIC-vs-SAC-SMA cross-compare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import calsim as _calsim_pkg
from . import cdec15 as _cdec15_pkg
from .io import DEFAULT_FORCING, forcing_path, load_hru_table

#: selectable modeling domains (calibration sets): the 15-CDEC application + CalLite sets.
DOMAINS = [_cdec15_pkg.DOMAIN, *_calsim_pkg.DOMAINS]


def _run(args: argparse.Namespace) -> int:
    from .model import load_domain_forcing, run_basin

    domain = args.domain
    if args.basin.upper() == "ALL":
        # basin codes are domain-specific; read them from the HRU table
        basins = sorted(load_hru_table(args.data_dir, domain=domain)["basin"].unique())
    elif domain == _cdec15_pkg.DOMAIN:
        basins = [args.basin.upper()]
    else:
        basins = [args.basin]  # CalLite basin codes are case-sensitive (CamelCase / mixed)

    # For multi-basin native runs, read the ~900 MB/var forcing store ONCE and
    # reuse it across every basin instead of re-reading it per basin.
    product = args.forcing or DEFAULT_FORCING
    forcing = None
    if (
        len(basins) > 1
        and args.data_dir is not None
        and forcing_path(args.data_dir, domain, product).exists()
    ):
        print(f"loading domain forcing once for all basins ({product})...", flush=True)
        forcing = load_domain_forcing(args.data_dir, domain=domain, start=args.start,
                                      end=args.end, product=product)

    for basin in basins:
        df = run_basin(
            basin,
            data_dir=args.data_dir,
            domain=domain,
            start=args.start,
            end=args.end,
            progress=args.progress,
            forcing=forcing,
            parallel=args.parallel,
            product=product,
            spinup_years=args.spinup_years,
        )
        if args.out:
            out = Path(args.out)
            if len(basins) > 1:
                out = out.with_name(f"{out.stem}_{basin}{out.suffix or '.csv'}")
            df.to_csv(out, index=False)
            print(f"{basin}: wrote {len(df)} days -> {out}")
        else:
            print(f"{basin}: {len(df)} days, mean flow {df['flow'].mean():.4f} mm/day")
    return 0


def _plots(args: argparse.Namespace) -> int:
    if args.domain == _cdec15_pkg.DOMAIN:
        from .cdec15 import plots as p

        p.make_all(basins=args.basins, data_dir=args.data_dir,
                   artifacts_dir=args.artifacts_dir, run=args.run or "cdec15")
        if args.fnf_check:
            from .calsim.plots import make_cdec15_fnf_check

            make_cdec15_fnf_check(basins=args.basins, data_dir=args.data_dir,
                                  artifacts_dir=args.artifacts_dir, run=args.run or "cdec15")
    else:
        from .calsim import plots as p

        p.make_all(domain=args.domain, basins=args.basins, data_dir=args.data_dir,
                   artifacts_dir=args.artifacts_dir, run=args.run)
    return 0


def _calsim(args: argparse.Namespace) -> int:
    from .calsim.compare import DEFAULT_CALSETS, make_all

    sets = tuple(args.sets) if args.sets else DEFAULT_CALSETS
    make_all(args.data_dir, args.artifacts_dir, args.run, sets,
             covered_frac=getattr(args, "covered_frac", None), parallel=args.parallel)
    return 0


def _dpl_benchmark(args: argparse.Namespace) -> int:
    from .dpl.evaluate import fidelity_benchmark

    fidelity_benchmark(args.data_dir, args.out, configs=args.configs,
                       device=args.device, chunk_days=args.chunk_days)
    return 0


def _dpl_train(args: argparse.Namespace) -> int:
    from .dpl.config import DplConfig
    from .dpl.train import train

    cfg = DplConfig(
        n_inc=args.n_inc, perc_mode=args.perc_mode,
        fracp_floor=args.fracp_floor, dtype=args.dtype, device=args.device,
        loss=args.loss, log_loss_lambda=args.log_lambda,
        var_loss_lambda=args.var_lambda, bias_loss_lambda=args.bias_lambda,
        et_loss_lambda=args.et_loss_lambda,
        et_level_lambda=args.et_level_lambda,
        swe_loss_lambda=args.swe_loss_lambda,
        shape_sigma_floor=args.shape_sigma_floor,
        et_anchor_band=args.et_anchor_band,
        et_products=args.et_products,
        init_from=args.init_from,
        lr=args.lr,
        lr_warmup_epochs=args.warmup_epochs, n_epochs=args.epochs,
        spinup_refresh_every=args.spinup_refresh,
        spinup_start=args.spinup_start, patience=args.patience,
        hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        grouped_heads=args.grouped_heads, fourier_k=args.fourier_k,
        gnn_k=args.gnn_k,
        spatial_reg_lambda=args.spatial_reg_lambda,
        spatial_reg_k=args.spatial_reg_k,
        spatial_reg_attr_scale=args.spatial_reg_attr_scale,
        adaptive_loss=args.adaptive_loss, adaptive_loss_beta=args.adaptive_beta,
        seasonal_params=(tuple(args.seasonal.split(",")) if args.seasonal else ()),
        seasonal_amp=args.seasonal_amp,
        seasonal_amp_frac=args.seasonal_amp_frac,
        et_mode=args.et, noah_pet=args.noah_pet, sac_pet=args.sac_pet,
        pt_snow_albedo=args.pt_snow_albedo,
        pt_dewpoint_depression=args.pt_dewpoint_depression,
        canopy_lite=args.canopy_lite,
        calsim_footprint=args.calsim_footprint,
        dynamic_params=(tuple(args.dynamic_params.split(","))
                        if args.dynamic_params else ()),
        dynamic_amp=args.dynamic_amp, dynamic_window=args.dynamic_window,
        seed=args.seed, use_cuda_graphs=not args.no_graphs,
    )
    train(args.variant, data_dir=args.data_dir, out_dir=args.out, cfg=cfg,
          resume=args.resume, domain=args.domain)
    return 0


def _dpl_evaluate(args: argparse.Namespace) -> int:
    from .dpl.evaluate import evaluate_checkpoint

    evaluate_checkpoint(args.checkpoint, data_dir=args.data_dir,
                        out_dir=args.out, parallel=not args.serial,
                        temp_delta=args.temp_delta)
    return 0


def _dpl_hybrid(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .dpl.hybrid.evaluate import compare_all, score_hybrid
    from .dpl.hybrid.train import HybridConfig, train_hybrid

    out = args.out or "artifacts/dpl/testing/hybrid"
    physics = (None if str(args.physics).lower() in ("", "none", "ga")
               else args.physics)
    # --response-grid: build the 5 corner (Δprecip, ΔT) anchors and ensure the
    # cached physics teacher sim under each (fast frozen noah-lite response on the
    # torch baseline; the SAME physics the response-surface sweep uses).
    response_anchors: tuple = ()
    if args.response_grid:
        from .dpl.dtdp_response import physics_daily
        from .dpl.evaluate import (corner_anchors, grid_anchors,
                                   teacher_cache_path)
        if args.response_dps and args.response_dts:
            anchor_pts = grid_anchors(
                [float(x) for x in args.response_dps.split(",")],
                [float(x) for x in args.response_dts.split(",")])
        else:
            anchor_pts = corner_anchors(args.response_dp, args.response_dt)
        ancs = []
        for dp, dt in anchor_pts:
            physics_daily(dp, dt, data_dir=args.data_dir)   # ensure cached teacher
            ancs.append({"dp": dp, "dt": dt, "lambda": args.response_lambda,
                         "sim_cache": str(teacher_cache_path(dp, dt))})
        response_anchors = tuple(ancs)
    cfg = HybridConfig(
        use_statics=args.statics, n_epochs=args.epochs,
        hidden=args.hidden, dropout=args.dropout, lr=args.lr,
        batch_size=args.batch_size, device=args.device, seed=args.seed,
        input_noise=args.input_noise,
        use_doy=not args.no_doy, use_pet=args.pet_input,
        temp_lambda=args.temp_lambda, temp_delta=args.temp_delta,
        temp_sim_cache=args.temp_sim_cache, response_anchors=response_anchors,
        physics_domain=args.physics_domain, pet_source=args.sac_pet,
        pt_snow_albedo=args.pt_snow_albedo,
        pt_dewpoint_depression=args.pt_dewpoint_depression,
        physics_et_scheme=args.physics_et, canopy_csv=args.canopy_params)
    train_hybrid(cfg, data_dir=args.data_dir, out_dir=out,
                 physics_csv=physics, sim_cache=args.sim_cache)
    score_hybrid(Path(out) / "checkpoints" / "best.pt",
                 data_dir=args.data_dir, out_dir=out)
    if args.compare:
        compare_all(Path(out).parent)
    return 0


def _dpl_climatology(args: argparse.Namespace) -> int:
    from .dpl.climatology import make_cdec15_climatology

    make_cdec15_climatology(data_dir=args.data_dir, out_dir=args.out,
                            device=args.device)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sacsma", description="Distributed SAC-SMA for CA watersheds")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="forward-simulate a basin (or ALL) for a domain")
    run.add_argument("basin", help="basin ID (e.g. BND, or BearRiver for 9unimp) or ALL")
    run.add_argument(
        "--domain", default=_cdec15_pkg.DOMAIN, choices=DOMAINS,
        help="calibration set / forcing store (default: 15cdec)",
    )
    run.add_argument(
        "--data-dir", default="data",
        help="organized data/ store to read (default: data)",
    )
    run.add_argument("--forcing", default=None, metavar="PRODUCT",
                     help="forcing product (store filename stem), e.g. wgen_product_a "
                          "for the WGEN historical-parallel sequence (CalSim domains "
                          "only; default: the historical Livneh-unsplit store)")
    run.add_argument("--start", default=None, help="start date YYYY-MM-DD")
    run.add_argument("--end", default=None, help="end date YYYY-MM-DD")
    run.add_argument("--out", default=None, help="output CSV path")
    run.add_argument("--progress", action="store_true", help="print HRU progress")
    run.add_argument("--parallel", action="store_true",
                     help="fan HRUs across cores (Numba prange); ~8x faster, matches "
                          "the serial result to floating tolerance")
    run.add_argument("--spinup-years", type=int, default=None, metavar="N",
                     help="prepend an N-year climatological average-year burn-in "
                          "so the run starts from an equilibrated state at any "
                          "--start (default: none = reference cold start)")
    run.set_defaults(func=_run)

    pl = sub.add_parser("plots", help="per-watershed cal/val diagnostic figures for a domain")
    pl.add_argument("--domain", default=_cdec15_pkg.DOMAIN, choices=DOMAINS,
                    help="calibration set (default: 15cdec)")
    pl.add_argument("--basins", nargs="*", default=None,
                    help="subset of watershed codes (default: all)")
    pl.add_argument("--data-dir", default="data", help="data store")
    pl.add_argument("--artifacts-dir", default="artifacts", help="output root")
    pl.add_argument("--run", default=None,
                    help="run name (default: cdec15 -> artifacts/cdec15/, or the domain "
                         "-> artifacts/calsim/<domain>/)")
    pl.add_argument("--fnf-check", action="store_true",
                    help="15cdec only: also score the same basins MONTHLY against CalSim3's "
                         "unimpaired FNF (longer independent validation window) -> extra "
                         "*_diagnostics_calsim3.png figures + metrics_15cdec_calsim3.csv")
    pl.set_defaults(func=_plots)

    cs = sub.add_parser(
        "calsim",
        help="cross-compare CalSim3 (actual) vs VIC vs multi-set SAC-SMA "
             "-> artifacts/calsim/<run>/",
    )
    cs.add_argument("--data-dir", default="data", help="organized data/ store")
    cs.add_argument("--artifacts-dir", default="artifacts", help="output root")
    cs.add_argument("--run", default="compare", help="run name -> artifacts/calsim/<run>/")
    cs.add_argument("--sets", nargs="+", default=None,
                    help="SAC-SMA calibration sets to score separately vs CalSim3 "
                         "(default: 15cdec 9unimp 11obs)")
    cs.add_argument("--covered-frac", type=float, default=None,
                    help="informational 'covered'/'partial' status label only "
                         "(default: catchments.COVERED_FRAC); inclusion is crosswalk-driven")
    cs.add_argument("--parallel", action="store_true",
                    help="fan the SAC-SMA model runs across cores (Numba prange); "
                         "results unchanged, ~8x faster on the model-run phase")
    cs.set_defaults(func=_calsim)

    dpl = sub.add_parser(
        "dpl",
        help="differentiable-parameter-learning variant (torch; 15cdec application)",
    )
    dpl_sub = dpl.add_subparsers(dest="dpl_command", required=True)
    bm = dpl_sub.add_parser(
        "benchmark",
        help="fidelity benchmark: archived GA params through the torch forward "
             "vs the frozen reference -> artifacts/dpl/fidelity/",
    )
    bm.add_argument("--data-dir", default="data", help="organized data/ store")
    bm.add_argument("--out", default="artifacts/dpl/fidelity", help="output dir")
    bm.add_argument("--configs", nargs="+", default=None,
                    help="subset of named numerics configs (default: all; see "
                         "dpl.evaluate.FIDELITY_CONFIGS)")
    bm.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device (default: cuda; GPU is asserted)")
    bm.add_argument("--chunk-days", type=int, default=4096,
                    help="streaming chunk length in days (memory knob)")
    bm.set_defaults(func=_dpl_benchmark)

    tr = dpl_sub.add_parser(
        "train",
        help="train a feature variant (spinup + water-year TBPTT; GPU asserted) "
             "-> artifacts/dpl/<variant>/",
    )
    tr.add_argument("variant",
                    choices=["static", "climate", "physical", "physical_climate"],
                    help="feature ablation arm (physical = continuous "
                         "soil/veg/terrain/LAI in place of one-hot soil/veg; "
                         "physical_climate = physical + the 4 climate indices, "
                         "so the learned params ADAPT under a perturbed climate)")
    tr.add_argument("--data-dir", default="data", help="organized data/ store")
    tr.add_argument("--domain", default="15cdec",
                    choices=["15cdec", "15cdec_grid"],
                    help="training domain: 15cdec HRU cloud (7891) or the native "
                         "1/16-deg Livneh grid (2074 cells); baked into the "
                         "checkpoint so evaluate scores the same domain")
    tr.add_argument("--et", default="sac", choices=["sac", "noah"],
                    help="ET scheme: sac = frozen Hamon PET (scorable via "
                         "run_basin); noah = Noah canopy-resistance ET (NEW "
                         "physics, needs per-cell tmin/tmax = 15cdec_grid, "
                         "scored via the torch pipeline)")
    tr.add_argument("--noah-pet", default="hamon",
                    choices=["hamon", "priestley_taylor"],
                    help="Noah potential-ET source: hamon = temperature-only "
                         "(low ET ceiling); priestley_taylor = energy-based from "
                         "Bristow-Campbell net radiation (lifts the ceiling)")
    tr.add_argument("--sac-pet", default="hamon",
                    choices=["hamon", "priestley_taylor"],
                    help="PET source for the PLAIN SAC ET (et=sac): priestley_taylor "
                         "drives the frozen SAC ET with energy-based PET, no Noah "
                         "canopy module")
    tr.add_argument("--pt-snow-albedo", type=float, default=0.0, metavar="ALBEDO",
                    help="raise the Priestley-Taylor albedo toward this value over "
                         "snow (Snow-17 SWE-driven; ~0.5-0.7 bright snow); 0 = fixed "
                         "0.23 (any PT PET: sac-pet OR noah-pet = priestley_taylor)")
    tr.add_argument("--pt-dewpoint-depression", type=float, default=0.0, metavar="DEGC",
                    help="max dewpoint depression (degC) below Tmin in arid air for "
                         "the PT net-longwave term, scaled by diurnal range; 0 = "
                         "Tdew=Tmin (any PT PET: sac-pet OR noah-pet = priestley_taylor)")
    tr.add_argument("--canopy-lite", action="store_true",
                    help="minimal identifiable Noah ET: AET=beta(soil moisture)*PET "
                         "with ONE learned exponent (soil_chi); drops the Jarvis "
                         "resistance, froot, redist_k and the separate canopy trunk "
                         "(needs --et noah; --noah-pet still selects the potential)")
    tr.add_argument("--calsim-footprint", action="store_true",
                    help="re-foot basin aggregation onto the CalSim3 catchments "
                         "(overlap weights) to correct the coarse-grid footprint "
                         "over-reach; the 4 Tulare/Kern basins keep full footprint")
    tr.add_argument("--dynamic-params", default="",
                    help="comma list of params made climate-state-dependent "
                         "(Kpet | canopy params e.g. soil_chi); '' = static")
    tr.add_argument("--dynamic-window", type=int, default=365,
                    help="trailing-precip window (days) for the wetness state index")
    tr.add_argument("--dynamic-amp", type=float, default=0.5,
                    help="tanh cap on the state-response coeff |b|")
    tr.add_argument("--out", default=None,
                    help="output dir (default: artifacts/dpl/<variant>)")
    tr.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device (default: cuda; GPU is asserted)")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--n-inc", type=int, default=10,
                    help="fixed SAC-SMA substep count (fidelity-gate choice: "
                         "ref-ninc10, obs-KGE delta <= 0.0102 all basins)")
    tr.add_argument("--perc-mode", default="reference",
                    choices=["reference", "implicit", "tanh"])
    tr.add_argument("--fracp-floor", type=float, default=1e-3,
                    help="LZ fill-fraction denominator floor: bounds the one "
                         "unbounded division's backward; engages only above "
                         "99.9%% LZ saturation")
    tr.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    tr.add_argument("--loss", default="nnse", choices=["nnse", "mse"])
    tr.add_argument("--log-lambda", type=float, default=0.15,
                    help="low-flow log-space loss weight (0 disables)")
    tr.add_argument("--var-lambda", type=float, default=1.0,
                    help="per-chunk variance-matching weight (std ratio - 1)^2; "
                         "counters squared-error variance damping (0 disables)")
    tr.add_argument("--bias-lambda", type=float, default=0.0,
                    help="per-chunk bias penalty (mean ratio - 1)^2; the KGE beta "
                         "term the MSE/NNSE loss lacks (0 disables)")
    tr.add_argument("--et-loss-lambda", type=float, default=0.0,
                    help="ET seasonal-SHAPE loss weight: inverse-variance pull of "
                         "the model's NORMALIZED monthly ET cycle to the 5-product "
                         "consensus shape — level-blind (0 disables)")
    tr.add_argument("--et-level-lambda", type=float, default=0.0,
                    help="ET volume envelope hinge weight: zero inside the product "
                         "min-max total, quadratic outside (catches arid basins "
                         "above every product; 0 disables)")
    tr.add_argument("--swe-loss-lambda", type=float, default=0.0,
                    help="SWE seasonal-SHAPE loss weight: normalized accumulation/"
                         "melt-cycle pull to the 4-product consensus (snow basins "
                         "only, no level term; 0 disables)")
    tr.add_argument("--shape-sigma-floor", type=float, default=0.1,
                    help="absolute floor on the normalized-cycle ensemble sigma "
                         "(hedges correlated-products over-confidence; default 0.1)")
    tr.add_argument("--et-anchor-band", type=float, default=0.0,
                    help="re-target the ET level hinge to the WATER-BALANCE "
                         "anchor: per-basin annual ET = cal mean(P) - mean(Q_obs) "
                         "over gage days, +/- this fractional band (needs "
                         "--et-level-lambda > 0; 0 = product min-max envelope)")
    tr.add_argument("--et-products", default="",
                    help="comma list restricting the ET obs target to named "
                         "products (e.g. fluxcom = the single-product steering "
                         "arm; one product requires --et-anchor-band > 0). "
                         "Empty = all 5 (consensus)")
    tr.add_argument("--init-from", default="",
                    help="warm-start checkpoint (e.g. a baseline best.pt): net "
                         "weights load strict=False so fresh zero-init heads "
                         "(e.g. --seasonal) start EXACTLY at the donor's field; "
                         "fresh optimizer/scheduler — pair with a low --lr for "
                         "the fine-tune regime")
    tr.add_argument("--fourier-k", type=int, default=0,
                    help="net-v2: spatial Fourier feature order (4k extra "
                         "features; low-frequency regional fields; 0 = off)")
    tr.add_argument("--grouped-heads", action="store_true",
                    help="net-v2: separate output heads per physics group "
                         "(PET/SMA/snow/routing)")
    tr.add_argument("--gnn-k", type=int, default=0,
                    help="net-v2: learned spatial smoother — one weighted-mean "
                         "message-passing round over within-basin geographic "
                         "k-NN neighborhoods (zero-init mixing = exact v1 at "
                         "init; 0 = off)")
    tr.add_argument("--spatial-reg-lambda", type=float, default=0.0,
                    help="attribute-weighted geographic smoothness penalty on "
                         "the per-HRU parameter field (0 = off); small-sample "
                         "complexity brake, does NOT anchor to the GA optimum")
    tr.add_argument("--spatial-reg-k", type=int, default=8,
                    help="geographic k-NN neighbours per HRU for the spatial reg")
    tr.add_argument("--spatial-reg-attr-scale", type=float, default=1.0,
                    help="attr-distance decay of the spatial-reg edge weights "
                         "exp(-scale * attr_dist / median); higher = only very "
                         "attribute-similar neighbours are tied")
    tr.add_argument("--adaptive-loss", action="store_true",
                    help="Rahman-ALF per-basin loss weights ∝ (1-cal_KGE)^beta "
                         "(reweight toward the worst-fitting basins each eval)")
    tr.add_argument("--adaptive-beta", type=float, default=1.0,
                    help="exponent on (1 - cal_KGE) for the adaptive weights")
    tr.add_argument("--seasonal", nargs="?", const="Kpet,uzk,lzpk,lzsk", default=None,
                    metavar="P1,P2,...",
                    help="give these params a day-of-year harmonic shape (bare flag "
                         "= Kpet,uzk,lzpk,lzsk); the net emits 2 zero-init coeffs each "
                         "so the field is exactly static at init")
    tr.add_argument("--seasonal-amp", type=float, default=0.18,
                    help="tanh cap on the harmonic coeffs |a_sin|,|a_cos| (additive "
                         "param units); hard-bounds the day-of-year swing so it "
                         "cannot diverge (0.18 ~ +/-25%% of Kpet~1)")
    tr.add_argument("--seasonal-amp-frac", type=float, default=0.10,
                    help="PER-PARAM harmonic cap as a fraction of each seasonal "
                         "param's bound range (supersedes --seasonal-amp): each "
                         "param gets a comparable RELATIVE day-of-year swing, so a "
                         "mixed set (Kpet + melt factors) is balanced (0.10 -> Kpet "
                         "+/-0.21, MFMAX/MFMIN +/-0.50, MBASE +/-0.50)")
    tr.add_argument("--hidden", type=int, default=64, help="trunk width")
    tr.add_argument("--embed", type=int, default=32, help="embedding width")
    tr.add_argument("--dropout", type=float, default=0.1,
                    help="encoder dropout (0 = deterministic parameter map)")
    tr.add_argument("--warmup-epochs", type=int, default=3,
                    help="linear LR warmup epochs (protects the GA-prior init)")
    tr.add_argument("--patience", type=int, default=10,
                    help="early-stop after this many stale cal-KGE selections "
                         "(lower = stop sooner at plateau; selection cadence is "
                         "every 2 epochs)")
    tr.add_argument("--spinup-refresh", type=int, default=1,
                    help="re-run the no-grad spinup every k epochs (k=2 "
                         "reuses one-epoch-stale state on odd epochs — same "
                         "staleness order as within-epoch TBPTT drift; "
                         "selection evals always respin fresh)")
    tr.add_argument("--spinup-start", default="1978-10-01",
                    help="no-grad spinup cold-start date (default 10 water "
                         "years before the WY1989 cal window: spans the "
                         "record-wet WY1982-83, which resets the LZ "
                         "tension-store memory exactly — measured KGE vs "
                         "the full prefix 1.000000; clamped to the record "
                         "start, so pass 1915-01-01 for the exact frozen "
                         "full-prefix convention)")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--no-graphs", action="store_true",
                    help="disable CUDA-graph capture (eager; much slower)")
    tr.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/last.pt")
    tr.set_defaults(func=_dpl_train)

    ev = dpl_sub.add_parser(
        "evaluate",
        help="checkpoint -> params_dpl.csv -> FROZEN-model cal/val metrics + "
             "figures (all reported dPL skill comes from this path)",
    )
    ev.add_argument("checkpoint", help="path to checkpoints/best.pt")
    ev.add_argument("--data-dir", default="data", help="organized data/ store")
    ev.add_argument("--out", default=None,
                    help="output dir (default: the checkpoint's own run dir "
                         "for the standard <run>/checkpoints/*.pt layout, "
                         "else artifacts/dpl/<variant>)")
    ev.add_argument("--serial", action="store_true",
                    help="disable the parallel (numba prange) frozen model")
    ev.add_argument("--temp-delta", type=float, default=0.0,
                    help="add a uniform delta (degC) to tavg/tmin/tmax and dump "
                         "the perturbed daily sim (torch path only; label gets a "
                         "_dT suffix, gage metrics/figures are skipped) — the "
                         "TEACHER for the hybrid temperature-consistency loss")
    ev.set_defaults(func=_dpl_evaluate)

    hy = dpl_sub.add_parser(
        "hybrid",
        help="train + score the hybrid SAC-SMA x LSTM (physics sim as an input "
             "channel) on the 15cdec daily basis -> artifacts/dpl/testing/hybrid/ "
             "(local scratch; the canonical ensemble lives at artifacts/dpl/hybrid)",
    )
    hy.add_argument("--physics", required=True,
                    help="frozen SAC-SMA parameter table for the physics baseline "
                         "(REQUIRED, no default); 'GA' or '' -> archived GA optimum. "
                         "Must match --physics-domain (validated at load: fine "
                         "15cdec ~6033 keys, 15cdec_grid ~2074).")
    hy.add_argument("--physics-domain", default="15cdec",
                    choices=["15cdec", "15cdec_grid"],
                    help="HRU resolution of the frozen sim + forcing features "
                         "(default: 15cdec fine; 15cdec_grid for the pt/noah "
                         "grid exports)")
    hy.add_argument("--sac-pet", default="hamon",
                    choices=["hamon", "priestley_taylor"],
                    help="PET source of the frozen sim -- MUST match the --physics "
                         "export (priestley_taylor for pt/noah)")
    hy.add_argument("--pt-snow-albedo", type=float, default=0.0,
                    help="PT snow-cover albedo refinement, match the export "
                         "(pt = 0.6; needs --sac-pet priestley_taylor)")
    hy.add_argument("--pt-dewpoint-depression", type=float, default=0.0,
                    help="PT arid dewpoint-depression refinement, match the export "
                         "(pt = 2.0; needs --sac-pet priestley_taylor)")
    hy.add_argument("--physics-et", default="sac",
                    choices=["sac", "noah_lite"],
                    help="frozen-sim ET scheme: sac = the E1-E5 cascade; noah_lite "
                         "= the Noah-lite external ET (noah; forces PT, "
                         "needs --canopy-params)")
    hy.add_argument("--canopy-params", default="",
                    help="params_canopy.csv (soil_chi) for --physics-et noah_lite")
    hy.add_argument("--sim-cache", default=None,
                    help="cache path for the frozen 15-basin daily sim (default: a "
                         "physics-tagged file in the run's parent dir; delete it if "
                         "you change the --physics export or PT knobs)")
    hy.add_argument("--statics", action="store_true",
                    help="add per-basin static features (elev/flowlen/precip/snow)")
    hy.add_argument("--no-doy", action="store_true",
                    help="drop the sin/cos day-of-year LSTM inputs (the sim "
                         "channel already carries the calendar; an explicit doy "
                         "enables calendar-keyed mean corrections that inject "
                         "val-period volume bias)")
    hy.add_argument("--pet-input", action="store_true",
                    help="add the raw PT potential (basin-average, alb 0/dew 0 "
                         "— the noah energy demand, recomputed from forcing) as "
                         "an LSTM input channel: a physics-shaped temperature "
                         "pathway")
    hy.add_argument("--temp-lambda", type=float, default=0.0,
                    help="temperature-consistency loss weight: pull the hybrid's "
                         "daily warming response Q(T+dT)-Q(T) toward the physics "
                         "response (0 disables; needs --temp-sim-cache)")
    hy.add_argument("--temp-delta", type=float, default=2.0,
                    help="the perturbation (degC) baked into --temp-sim-cache "
                         "(must match the --temp-delta the teacher was dumped with)")
    hy.add_argument("--temp-sim-cache", default="",
                    help="teacher daily-sim CSV: the SAME physics as --sim-cache "
                         "re-run under +temp_delta (`sacsma dpl evaluate <physics "
                         "ckpt> --temp-delta <dT>`)")
    hy.add_argument("--response-grid", action="store_true",
                    help="multi-anchor dp/dt response-consistency loss: anchor the "
                         "hybrid's response to physics at the 5 corners of "
                         "{−dp,0,+dp}×{0,+dt} (precip + warming + joint). Ensures "
                         "the cached noah teacher sim under each corner via "
                         "--noah-ckpt")
    hy.add_argument("--response-lambda", type=float, default=0.1,
                    help="per-anchor weight for --response-grid (total = 5×; "
                         "start ~0.1, screen if pooled cal-KGE drops)")
    hy.add_argument("--response-dp", type=float, default=0.10,
                    help="precip perturbation fraction for --response-grid "
                         "(0.10 = ±10%%)")
    hy.add_argument("--response-dt", type=float, default=3.0,
                    help="warming perturbation (degC) for --response-grid (+3)")
    hy.add_argument("--response-dps", default="",
                    help="comma-separated Δprecip fractions for a full anchor GRID "
                         "(with --response-dts) — a wider/interior anchor set that "
                         "overrides the 5-corner default, e.g. '-0.2,-0.1,0,0.1,0.2'")
    hy.add_argument("--response-dts", default="",
                    help="comma-separated ΔT (degC) for the full anchor grid "
                         "(with --response-dps), e.g. '0,2,4' (origin dropped)")
    hy.add_argument("--data-dir", default="data", help="organized data/ store")
    hy.add_argument("--out", default=None,
                    help="output dir (default: artifacts/dpl/testing/hybrid)")
    hy.add_argument("--epochs", type=int, default=60)
    hy.add_argument("--hidden", type=int, default=128, help="LSTM hidden size")
    hy.add_argument("--dropout", type=float, default=0.15)
    hy.add_argument("--input-noise", type=float, default=0.1,
                    help="gaussian input-jitter regularizer std (0 disables)")
    hy.add_argument("--lr", type=float, default=4e-4)
    hy.add_argument("--batch-size", type=int, default=512)
    hy.add_argument("--device", default="cuda", help="cuda | cpu")
    hy.add_argument("--seed", type=int, default=0)
    hy.add_argument("--compare", action="store_true",
                    help="also write the GA/dPL/hybrid comparison + dumbbell")
    hy.set_defaults(func=_dpl_hybrid)

    cl = dpl_sub.add_parser(
        "climatology",
        help="per-watershed mean-monthly TAF regime (GA + dPL + hybrids) vs the "
             "observed CalSim3 FNF, as a 5-step ablation + all-series metric bars "
             "-> artifacts/dpl/figures/cdec15_climatology_*.png",
    )
    cl.add_argument("--data-dir", default="data", help="organized data/ store")
    cl.add_argument("--out", default="artifacts/dpl",
                    help="output root (figures -> <out>/figures/)")
    cl.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device for the hybrid reconstructions")
    cl.set_defaults(func=_dpl_climatology)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
