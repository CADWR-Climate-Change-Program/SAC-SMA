"""The ``sacsma dpl`` commands: benchmark, train, evaluate, hybrid, and the two tool groups
``calsim`` (the CalSim3 rim-arc validation and the rim-inflow product) and ``study`` (the
15-CDEC study figures).

:func:`register` only defines arguments, so ``sacsma --help`` works without torch; every handler
imports what it runs when it is called.
"""

from __future__ import annotations

import argparse

#: ``sacsma dpl calsim <tool>``: the tool's module (under :mod:`sacsma.dpl.calsim`) and what it does.
#: Each tool keeps its own argument parser; the arguments after the tool name go to it unchanged.
CALSIM_TOOLS: dict[str, tuple[str, str]] = {
    "tier1": ("tier1", "score a run against CalSim3 at the anchors (set sums) -> <run>/tier1/"),
    "tier2": ("tier2", "simulate and score every rim arc -> <run>/tier2/ (or --scenarios)"),
    "atlas": ("atlas", "the validation atlas of a run (HTML) -> <run>/atlas/"),
    "windows": ("windows", "the trimmed validation window of each set -> data/calsim/tier1_sets.csv"),
    "compare": ("compare", "two runs side by side on tier 1"),
    "product": ("product", "the CalSim3 rim-inflow product: fit <run> | apply <run> --tier2 DIR"),
}

#: ``sacsma dpl study <name>``: module (under :mod:`sacsma.dpl.studies`), function, what it writes.
STUDIES: dict[str, tuple[str, str, str]] = {
    "climatology": ("climatology", "make_cdec15_climatology",
                    "per-watershed mean-monthly TAF regime (GA + dPL + hybrids) vs the observed "
                    "CalSim3 FNF, as a 5-step ablation + all-series metric bars "
                    "-> artifacts/dpl/figures/climatology_*.png"),
    "response": ("dtdp_response", "make_dtdp_response",
                 "(dp, dT) response surfaces of one hybrid ensemble vs the physics"),
    "adaptive": ("adaptive_physics", "make_adaptive_physics_surfaces",
                 "physics-only (dp, dT) surfaces: climate-frozen vs climate-adaptive noah "
                 "-> artifacts/dpl/figures/noah_climate_adaptive*"),
    "hybrids": ("hybrids", "make_hybrids",
                "the hybrid family (hybrid, hybrid_dt, lstm) response surfaces and skill summary "
                "-> artifacts/dpl/figures/hybrid*"),
    "forcing": ("forcing_sensitivity", "make_forcing_sensitivity",
                "forcing-product sensitivity of the dPL chain "
                "-> artifacts/dpl/figures/forcing_sensitivity_*.png"),
}


def _dpl_benchmark(args: argparse.Namespace) -> int:
    from .evaluate import fidelity_benchmark

    fidelity_benchmark(args.data_dir, args.out, configs=args.configs,
                       device=args.device, chunk_days=args.chunk_days)
    return 0


def _param_box(spec: str | None) -> dict[str, tuple[float, float]]:
    """``"riva=0:0,lzsk=0.01:0.5"`` -> ``{"riva": (0.0, 0.0), "lzsk": (0.01, 0.5)}``."""
    box: dict[str, tuple[float, float]] = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        name, _, rng = item.partition("=")
        lo, sep, hi = rng.partition(":")
        if not sep:
            raise SystemExit(f"--param-box {item!r}: expected name=lo:hi")
        box[name.strip()] = (float(lo), float(hi))
    return box


def _obs_mask(path: str | None) -> tuple[str, ...]:
    """Hand-edited mask CSV (``entity_id,date,...``) -> ``("entity_id|YYYY-MM-DD", ...)``."""
    if not path:
        return ()
    import pandas as pd

    m = pd.read_csv(path, dtype=str, comment="#")
    if not {"entity_id", "date"} <= set(m.columns):
        raise SystemExit(f"--obs-mask {path}: needs entity_id and date columns")
    return tuple(f"{e.strip()}|{pd.Timestamp(d).date().isoformat()}"
                 for e, d in zip(m["entity_id"], m["date"], strict=True))


def _holdout_wy(text: str) -> tuple[int, ...]:
    """``--holdout-wy 1976-1985`` (or a single water year ``1983``) -> ``(1976, 1985)``."""
    if not text:
        return ()
    parts = text.split("-")
    try:
        a, b = (int(parts[0]), int(parts[-1])) if len(parts) <= 2 else (None, None)
    except ValueError:
        a = None
    if a is None:
        raise SystemExit(f"--holdout-wy {text!r}: expected FIRST-LAST water years, e.g. "
                         "1976-1985")
    return (a, b)


def _dpl_train(args: argparse.Namespace) -> int:
    from .config import DplConfig
    from .train import train

    cfg = DplConfig(
        n_inc=args.n_inc, perc_mode=args.perc_mode,
        fracp_floor=args.fracp_floor, dtype=args.dtype, device=args.device,
        loss=args.loss, log_loss_lambda=args.log_lambda,
        var_loss_lambda=args.var_lambda, bias_loss_lambda=args.bias_lambda,
        timing_loss_lambda=args.timing_lambda,
        peak_loss_lambda=args.peak_lambda, peak_loss_frac=args.peak_frac,
        shape_min_days=args.shape_min_days, timing_vol_gate=args.timing_vol_gate,
        obs_mask=_obs_mask(args.obs_mask),
        holdout_wy=_holdout_wy(args.holdout_wy),
        uf_train_start=args.uf_train_start, calsim_arcs=args.calsim_arcs,
        pxtemp_learn=args.learn_pxtemp,
        pxtemp_box=tuple(float(v) for v in args.pxtemp_box.split(":")),
        pxtemp_tau=args.pxtemp_tau,
        var_gate_frac=args.var_gate_frac, var_huber_cap=args.var_huber_cap,
        init_from=args.init_from, init_gate=args.init_gate,
        lr=args.lr,
        lr_warmup_epochs=args.warmup_epochs, n_epochs=args.epochs,
        spinup_refresh_every=args.spinup_refresh,
        spinup_start=args.spinup_start, patience=args.patience,
        spinup_mode=args.spinup_mode, spinup_years=args.spinup_years,
        min_stop_epoch=args.min_stop_epoch,
        dead_chunk_nograd=args.dead_chunk_nograd, diagnostics=args.diagnostics,
        tbptt_carry=args.tbptt_carry,
        tbptt_window_years=args.tbptt_window_years,
        graph_recompute_days=args.graph_recompute_days,
        hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        grouped_heads=args.grouped_heads, fourier_k=args.fourier_k,
        flowlen_feature=not args.no_flowlen_feature,
        param_box=_param_box(args.param_box),
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
        noah_sac_exchanges=args.noah_sac_exchanges,
        calsim_footprint=args.calsim_footprint,
        dynamic_params=(tuple(args.dynamic_params.split(","))
                        if args.dynamic_params else ()),
        dynamic_amp=args.dynamic_amp, dynamic_window=args.dynamic_window,
        mt_family_weight=args.mt_family_weight,
        mt_share_norm=args.mt_share_norm, mt_select_weight=args.mt_select_weight,
        mt_loss_ref=args.mt_loss_ref, mt_loss_ref_power=args.mt_loss_ref_power,
        train_chunk_days=args.train_chunk_days, chunk_grid=args.chunk_grid,
        nograd_window=args.nograd_window,
        train_graph_segments=args.train_graph_segments,
        dedup_cells=args.dedup_cells,
        seed=args.seed, use_cuda_graphs=not args.no_graphs,
    )
    train(args.variant, data_dir=args.data_dir, out_dir=args.out, cfg=cfg,
          resume=args.resume, domain=args.domain,
          basins=(tuple(args.basins.split(",")) if args.basins else None))
    return 0


def _dpl_evaluate(args: argparse.Namespace) -> int:
    import torch

    from .evaluate import evaluate_checkpoint

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("domain") == "multifamily":
        # multi-timescale checkpoints score per entity at native timescales
        from .evaluate_multi_timescale import evaluate_checkpoint_mt

        if args.temp_delta:
            raise ValueError("--temp-delta is not wired for the "
                             "multi-timescale domain")
        evaluate_checkpoint_mt(args.checkpoint, data_dir=args.data_dir,
                               out_dir=args.out,
                               hydrographs=args.hydrographs,
                               spinup=args.spinup, dedup_cells=args.dedup_cells)
        return 0
    if args.dedup_cells:
        raise ValueError("--dedup-cells is wired for multi-timescale checkpoints only")
    evaluate_checkpoint(args.checkpoint, data_dir=args.data_dir,
                        out_dir=args.out, parallel=not args.serial,
                        temp_delta=args.temp_delta)
    return 0


def _dpl_hybrid(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .hybrid.evaluate import compare_all, score_hybrid
    from .hybrid.train import HybridConfig, train_hybrid

    out = args.out or "artifacts/dpl/_local/testing/hybrid"
    physics = (None if str(args.physics).lower() in ("", "none", "ga")
               else args.physics)
    # --response-grid: build the 5 corner (Δprecip, ΔT) anchors and ensure the
    # cached physics teacher sim under each (fast frozen noah-lite response on the
    # torch baseline; the SAME physics the response-surface sweep uses).
    response_anchors: tuple = ()
    if args.response_grid:
        from .studies.dtdp_response import physics_daily
        from .evaluate import (corner_anchors, grid_anchors,
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


def _run_tool(tool: str, tool_args: list[str]) -> int:
    import importlib

    mod = importlib.import_module(f".calsim.{CALSIM_TOOLS[tool][0]}", __package__)
    mod.main(tool_args, prog=f"sacsma dpl calsim {tool}")
    return 0


def _dpl_calsim(args: argparse.Namespace) -> int:
    return _run_tool(args.tool, args.tool_args)


def dispatch_tool(argv: list[str]) -> int | None:
    """``dpl calsim <tool> ...``: hand everything after the tool name to the tool's own parser
    (argparse cannot forward a remainder that starts with an option).  None = not a tool call."""
    if len(argv) >= 3 and argv[:2] == ["dpl", "calsim"] and argv[2] in CALSIM_TOOLS:
        return _run_tool(argv[2], argv[3:])
    return None


def _dpl_study(args: argparse.Namespace) -> int:
    import importlib

    module, func, _ = STUDIES[args.study]
    kw = {"device": args.device} if args.study != "adaptive" else {}
    if args.study in ("response", "adaptive", "hybrids"):
        kw["regen"] = args.regen
    getattr(importlib.import_module(f".studies.{module}", __package__), func)(
        data_dir=args.data_dir, out_dir=args.out, **kw)
    return 0


def register(sub) -> None:
    """Add the ``dpl`` command and its sub-commands to the top-level ``sacsma`` parser."""
    dpl = sub.add_parser(
        "dpl",
        help="differentiable parameter learning (torch): train, evaluate, the CalSim3 rim-arc "
             "tools and the study figures",
    )
    dpl_sub = dpl.add_subparsers(dest="dpl_command", required=True)
    bm = dpl_sub.add_parser(
        "benchmark",
        help="fidelity benchmark: archived GA params through the torch forward "
             "vs the frozen reference -> artifacts/dpl/noah/fidelity/",
    )
    bm.add_argument("--data-dir", default="data", help="organized data/ store")
    bm.add_argument("--out", default="artifacts/dpl/noah/fidelity", help="output dir")
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
                    choices=["static", "climate", "physical", "physical_climate",
                             "aef", "aef64", "aef_random"],
                    help="feature ablation arm (physical = continuous "
                         "soil/veg/terrain/LAI in place of one-hot soil/veg; "
                         "physical_climate = physical + the 4 climate indices, "
                         "so the learned params ADAPT under a perturbed climate; "
                         "aef = AlphaEarth embeddings only (16 PCs of the unit "
                         "directions + the vector length, region grid only); "
                         "aef64 = all 64 unit-direction coordinates + the length, "
                         "no PCA; aef_random = its control, 17 random values "
                         "per cell)")
    tr.add_argument("--data-dir", default="data", help="organized data/ store")
    tr.add_argument("--domain", default="15cdec",
                    choices=["15cdec", "15cdec_grid", "multifamily"],
                    help="training domain: 15cdec HRU cloud (7891), the native "
                         "1/16-deg Livneh grid (2074 cells), or the "
                         "multi-timescale training entities (the registry in "
                         "data/multifamily, restrict with --basins; "
                         "daily + monthly targets on the registry envelope; "
                         "physical variants only); baked into the "
                         "checkpoint so evaluate scores the same domain")
    tr.add_argument("--basins", default="",
                    help="comma list restricting training to these basin/"
                         "entity ids (subset runs, e.g. debug slices or timing tests); "
                         "'' = the full domain")
    tr.add_argument("--mt-family-weight", default="none",
                    help="multi-timescale family weighting: none = every "
                         "valid daily entity weighs equally and the monthly "
                         "term adds with coefficient 1 (baseline); equal = "
                         "shares 1:1:1 over the families the run trains "
                         "(selection = mean of the family means); or numeric "
                         "shares 'usgs=0.27,cdec=0.54,uf=0.19' (+ calsim= with "
                         "--calsim-arcs; renormalized "
                         "over the families present, entities equal within a "
                         "family; selection uses the same share-weighted family "
                         "mean) (multifamily domain only)")
    tr.add_argument("--mt-share-norm", default="present", choices=["present", "all"],
                    help="with --mt-family-weight shares/equal: 'present' divides each "
                         "chunk's loss by the weight of the entities it scores (a chunk "
                         "holding one family gives it the whole term); 'all' divides by "
                         "every entity's weight, so the shares hold summed over chunks "
                         "whose families train in different eras")
    tr.add_argument("--mt-select-weight", default="",
                    help="checkpoint-SELECTION shares of a numeric --mt-family-weight "
                         "run, when they must differ from the loss shares: 'area' = the "
                         "footprint-area shares of the run's families (registry "
                         "area_mi2 summed over its entities), or numeric shares in the "
                         "--mt-family-weight syntax. Default '' = the loss shares")
    tr.add_argument("--mt-loss-ref", default="",
                    help="FROZEN per-family loss scale (needs --mt-share-norm all): "
                         "'usgs=0.7368,cdec=0.4202,uf=0.0980' = each family's per-entity "
                         "chunk loss (sum l_f / sum c_f) at a reference state; each "
                         "family term is multiplied by Lbar / L_ref_f^p (Lbar = the "
                         "share-weighted reference level^p; p = --mt-loss-ref-power), "
                         "frozen for the run.  Selection unchanged. Default '' = off")
    tr.add_argument("--mt-loss-ref-power", type=float, default=None,
                    help="exponent p of --mt-loss-ref, REQUIRED with it: kappa_f = "
                         "Lbar / L_ref_f^p (1 = equal loss mass at the reference; 0.5 = "
                         "matches the families' optimizer-step shares at trained states)")
    tr.add_argument("--et", default="sac", choices=["sac", "noah"],
                    help="ET scheme: sac = frozen Hamon PET (scorable via "
                         "run_basin); noah = Noah canopy-resistance ET (NEW "
                         "physics, needs per-cell tmin/tmax = 15cdec_grid or "
                         "multifamily, scored via the torch pipeline)")
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
    tr.add_argument("--noah-sac-exchanges", action="store_true",
                    help="Noah ET replaces only the SAC E1-E3 withdrawals: keep the "
                         "upper free->tension rebalance, the lower free->tension "
                         "resupply (rserv) and the ADIMP ET(5) of the reference ET "
                         "block (off: the external-ET path skips all three; needs --canopy-lite)")
    tr.add_argument("--calsim-footprint", action="store_true",
                    help="re-foot basin aggregation onto the CalSim3 catchments "
                         "(overlap weights) to correct the coarse-grid footprint "
                         "over-reach; the 4 Tulare/Kern basins keep full footprint "
                         "(15cdec domains only: no effect on multifamily, whose "
                         "entity weights are already footprint overlaps)")
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
                    help="per-chunk variance-matching weight on alpha = std ratio: "
                         "(alpha-1)^2 up to |alpha-1| = --var-huber-cap, linear "
                         "beyond, skipped for a basin-chunk under --var-gate-frac "
                         "of the basin's record variance; counters squared-error "
                         "variance damping (0 disables)")
    tr.add_argument("--var-gate-frac", type=float, default=1e-3,
                    help="share of the basin's record variance a chunk must carry "
                         "for the variance term to apply (0 = only the 1e-8 floor)")
    tr.add_argument("--var-huber-cap", type=float, default=1.0,
                    help="|alpha-1| beyond which the variance term grows linearly; "
                         "<= 0 = quadratic throughout (with --var-gate-frac 0 the "
                         "loss of the pre-2026-09 canonical runs)")
    tr.add_argument("--bias-lambda", type=float, default=0.0,
                    help="per-chunk bias penalty (mean ratio - 1)^2; the KGE beta "
                         "term the MSE/NNSE loss lacks (0 disables)")
    tr.add_argument("--timing-lambda", type=float, default=0.0,
                    help="per-chunk summer-RECESSION timing penalty: mean squared "
                         "difference of the normalized cumulative flow over 1 Jul - "
                         "30 Sep, sim vs obs (recession shape; volume-blind; no pull on "
                         "winter or the flood peaks; 0 disables)")
    tr.add_argument("--peak-lambda", type=float, default=0.0,
                    help="per-chunk flood-peak penalty: the mean of the top --peak-frac "
                         "valid days, sim vs obs, each sorted on its own (flow-duration "
                         "high segment), their difference over the basin's record mean "
                         "of that statistic (flood years carry it; Huber-capped; 0 "
                         "disables)")
    tr.add_argument("--peak-frac", type=float, default=0.02,
                    help="share of a chunk's valid days the peak term averages "
                         "(0.02 = 7 days a year)")
    tr.add_argument("--shape-min-days", type=int, default=300,
                    help="the timing and peak terms score a basin-chunk only with at "
                         "least this many valid observed days (whole water years)")
    tr.add_argument("--timing-vol-gate", type=float, default=0.05,
                    help="the timing term skips a basin-chunk whose observed Jul-Sep mean "
                         "flow is under this share of the basin's record mean flow")
    tr.add_argument("--init-from", default="",
                    help="warm-start checkpoint (e.g. a baseline best.pt): net "
                         "weights load strict=False so fresh zero-init heads "
                         "(e.g. --seasonal) start EXACTLY at the donor's field, "
                         "and the donor's feature standardization is reused "
                         "(exact start even on a different --basins subset); "
                         "fresh optimizer/scheduler — pair with a low --lr for "
                         "the fine-tune regime")
    tr.add_argument("--init-gate", default="warn", choices=["warn", "abort"],
                    help="what to do when the epoch-0 selection of a warm "
                         "start does not reproduce the donor's sel cal KGE "
                         "(|d| > 1e-3): warn and continue, or abort the run "
                         "(unattended fine-tunes)")
    tr.add_argument("--fourier-k", type=int, default=0,
                    help="net-v2: spatial Fourier feature order (4k extra "
                         "features; low-frequency regional fields; 0 = off)")
    tr.add_argument("--no-flowlen-feature", action="store_true",
                    help="leave the cell's flow length to its basin outlet out of "
                         "the parameter net's inputs (the routing still uses it): "
                         "on the multifamily domain a cell in several nested "
                         "entities then keeps one parameter set")
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
    tr.add_argument("--learn-pxtemp", action="store_true",
                    help="learn the Snow-17 rain/snow threshold PXTEMP per cell (else "
                         "the fixed 0 degC): zero-init head, hard split forward, "
                         "straight-through sigmoid-surrogate gradient")
    tr.add_argument("--pxtemp-box", default="-1:3", metavar="LO:HI",
                    help="learned PXTEMP range in degC (must contain 0)")
    tr.add_argument("--pxtemp-tau", type=float, default=1.0,
                    help="width (degC) of the sigmoid surrogate that carries the "
                         "PXTEMP gradient; the forward split stays hard")
    tr.add_argument("--obs-mask", default=None, metavar="CSV",
                    help="hand-edited CSV (entity_id,date,...) of daily observations "
                         "to mask out of training and scoring, e.g. "
                         "data/cdec_fnf/fnf_daily_mask.csv; the checkpoint carries the "
                         "list")
    tr.add_argument("--holdout-wy", default="", metavar="FIRST-LAST",
                    help="water years held out of EVERY family (multifamily), e.g. "
                         "1976-1985: their targets are blanked after the registry n_obs "
                         "audit and --obs-mask, before the NNSE normalizers, so neither "
                         "the loss, the normalizers nor selection read them; the "
                         "checkpoint carries it. '' = off")
    tr.add_argument("--uf-train-start", default="", metavar="YYYY-MM-01",
                    help="back-extend the uf_monthly training targets to this month "
                         "start (multifamily), e.g. 1949-10-01: each uf entity also "
                         "trains from it to its registry train_start (minus "
                         "--holdout-wy); the registry windows are unchanged. '' = off")
    tr.add_argument("--calsim-arcs", default="none", choices=["none", "train_default"],
                    help="train_default: append the train_default CalSim3 rim arcs of "
                         "data/calsim/arc_hierarchy.csv (tier A) as the calsim_monthly "
                         "family (cs_<ARC>, hierarchy order) after the other entities "
                         "(multifamily; share runs name it 'calsim=' in "
                         "--mt-family-weight and --mt-loss-ref)")
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
    tr.add_argument("--min-stop-epoch", type=int, default=0,
                    help="arm early stopping only from this epoch on (a stale "
                         "streak before it never stops the run)")
    tr.add_argument("--tbptt-carry", choices=("absolute", "relative", "flux"),
                    default="absolute",
                    help="state carried into each TBPTT chunk: 'absolute' (detached "
                         "SAC contents), 'relative' (same values, but the backward "
                         "holds each store's relative saturation fixed, so a larger "
                         "capacity is not seen as free deficit at every chunk start) "
                         "or 'flux' (relative, plus the lower-zone free stores' "
                         "carried drainage flux held fixed, so a faster store carries "
                         "less water into the next year); segmented/eager chunk path")
    tr.add_argument("--tbptt-window-years", type=int, choices=(1, 2, 3), default=1,
                    help="TBPTT window in water years (water_year grid): n > 1 runs each "
                         "live chunk's previous n - 1 water years as a gradient-carrying "
                         "burn-in before it, the loss on its own year only, so the "
                         "gradient sees the water carried into the scored year; the "
                         "state still advances one year per step (~n x compute and "
                         "activation memory; segmented/eager chunk path)")
    tr.add_argument("--graph-recompute-days", type=int, default=0,
                    help="> 0: activation recompute — one captured graph of this many "
                         "days replayed over every training window, the backward "
                         "re-running each segment from its stored start state, so "
                         "graph memory stays one segment's for any window length "
                         "(e.g. 73 with --tbptt-window-years 2/3); ~1 extra forward "
                         "per segment; 0 = off")
    tr.add_argument("--dead-chunk-nograd", action="store_true",
                    help="run chunks with no scoreable observation (a "
                         "26-entity multifamily run's WY1950-84) forward-only, "
                         "without autograd; the carried state and every later "
                         "chunk are unchanged (segmented/eager chunk path)")
    tr.add_argument("--diagnostics", action="store_true",
                    help="write chunk_log.csv (per chunk: loss by family, "
                         "pre-clip gradient norm), eval_terms.csv (selection "
                         "epochs: eval-mode loss by family and term) and "
                         "per-epoch net snapshots with an EMA shadow under "
                         "checkpoints/snapshots/; numerics unchanged")
    tr.add_argument("--param-box", default=None,
                    help="narrow learned parameters' bounds, name=lo:hi[,...] "
                         "in physical units inside the GA box; lo == hi pins "
                         "the parameter (riva=0:0 turns riparian ET off)")
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
    tr.add_argument("--spinup-mode", default="window", choices=["window", "cycle"],
                    help="window = spin up over --spinup-start .. the cal window (the "
                         "multifamily domain: the ten water years before it); cycle = "
                         "timing-independent: loop the window's own first "
                         "--spinup-years years from the cold start 20 times, then "
                         "from the previous state until a pass moves no basin's "
                         "annual flow by more than 0.1%% (reads nothing before the "
                         "window)")
    tr.add_argument("--spinup-years", type=int, default=10,
                    help="length of the looped block for --spinup-mode cycle")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--train-chunk-days", type=int, default=366,
                    help="TBPTT chunk length in days (366 = one water year; "
                         "shorter chunks cut the backward's VRAM peak at the "
                         "cost of a shorter gradient horizon)")
    tr.add_argument("--chunk-grid", default="fixed", choices=["fixed", "water_year"],
                    help="fixed = --train-chunk-days each from the window start "
                         "(the boundary drifts and splits a calendar month most "
                         "years, which the monthly-flow term then skips); "
                         "water_year = one chunk per water year, 1 Oct to 1 Oct, "
                         "every month whole (the graphs capture 365 days; a leap "
                         "year's last day continues eagerly)")
    tr.add_argument("--nograd-window", type=int, default=512,
                    help="CUDA-graph replay window (days) for the no-grad "
                         "spinup/selection streams; numerics-neutral (256 "
                         "on drivers that fault on very large graphs)")
    tr.add_argument("--train-graph-segments", type=int, default=1,
                    help="split the train-chunk CUDA graph into N consecutive "
                         "segment graphs with autograd across them (same TBPTT "
                         "gradient as the single graph up to float32 summation "
                         "order); 2 keeps 366-day chunks under the graph-size "
                         "limit of drivers that fault on whole-year captures")
    tr.add_argument("--dedup-cells", action="store_true",
                    help="cell-deduplicated forward: the parameter net and the per-cell "
                         "physics (PET, Snow-17, ET, SAC-SMA, states, PXTEMP) run once per "
                         "distinct grid cell, routing and aggregation per (entity, cell) "
                         "row; refused when rows of a cell differ in their net features "
                         "(e.g. the flow-length feature) or with --gnn-k; with dropout a "
                         "shared cell draws one mask instead of one per row")
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
    ev.add_argument("--hydrographs", default="review",
                    choices=["review", "all", "none"],
                    help="multi-timescale checkpoints only: which per-entity "
                         "diagnostics figures to draw (review = the cdec_daily "
                         "+ uf_monthly set; all adds the 69 USGS entities)")
    ev.add_argument("--spinup", default="cycle", choices=["cycle", "window"],
                    help="multi-timescale checkpoints only: state at the envelope "
                         "start — cycle = loop its first ten water years 20 times "
                         "from the cold start (timing-independent, default); window "
                         "= the legacy ten water years before it")
    ev.add_argument("--dedup-cells", action="store_true",
                    help="multi-timescale checkpoints only: run the per-cell physics "
                         "once per distinct grid cell (routing per (entity, cell) row); "
                         "same flows to float round-off, less compute")
    ev.set_defaults(func=_dpl_evaluate)

    hy = dpl_sub.add_parser(
        "hybrid",
        help="train + score the hybrid SAC-SMA x LSTM (physics sim as an input "
             "channel) on the 15cdec daily basis -> artifacts/dpl/_local/testing/hybrid/ "
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
                         "the cached noah teacher sim under each corner")
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
                    help="output dir (default: artifacts/dpl/_local/testing/hybrid)")
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
                    help="also write the GA/dPL/hybrid comparison table")
    hy.set_defaults(func=_dpl_hybrid)

    cs = dpl_sub.add_parser(
        "calsim",
        help="the dPL on the CalSim3 rim arcs: tier 1, tier 2, atlas, windows, compare, product",
        description="Tools of sacsma.dpl.calsim.  Everything after the tool name goes to the "
                    "tool; `sacsma dpl calsim <tool> --help` lists its arguments.",
    )
    cs_sub = cs.add_subparsers(dest="tool", required=True)
    for name, (_, text) in CALSIM_TOOLS.items():
        tool = cs_sub.add_parser(name, help=text, add_help=False)
        tool.add_argument("tool_args", nargs=argparse.REMAINDER)
        tool.set_defaults(func=_dpl_calsim)

    st = dpl_sub.add_parser(
        "study",
        help="figures and tables of the 15-CDEC dPL study (regimes, response surfaces, forcing)",
    )
    st_sub = st.add_subparsers(dest="study", required=True)
    for name, (_, _, text) in STUDIES.items():
        sp = st_sub.add_parser(name, help=text)
        sp.add_argument("--data-dir", default="data", help="organized data/ store")
        sp.add_argument("--out", default="artifacts/dpl",
                        help="output root (figures -> <out>/figures/)")
        if name != "adaptive":
            sp.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                            help="torch device for the hybrid reconstructions")
        if name in ("response", "adaptive", "hybrids"):
            sp.add_argument("--regen", action="store_true",
                            help="recompute the metrics table instead of reloading it")
        sp.set_defaults(func=_dpl_study)
