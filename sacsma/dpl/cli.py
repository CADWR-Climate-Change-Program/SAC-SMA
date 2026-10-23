"""The ``sacsma dpl`` commands: fidelity, train, evaluate, score, record, hybrid, and the two tool
groups ``calsim`` (the CalSim3 rim-arc validation and the rim-inflow product) and ``study`` (the
15-CDEC study figures).

:func:`register` only defines arguments, so ``sacsma --help`` works without torch; every handler
imports what it runs when it is called.
"""

from __future__ import annotations

import argparse

#: ``sacsma dpl calsim <tool>``: the tool's module (under :mod:`sacsma.dpl.calsim`) and what it does.
#: Each tool keeps its own argument parser; the arguments after the tool name go to it unchanged.
CALSIM_TOOLS: dict[str, tuple[str, str]] = {
    "tier1": ("tier1", "score a run against CalSim3 at the anchors (set sums) -> the run's tier1/"),
    "tier2": ("tier2", "simulate and score every rim arc -> the run's tier2/ (or --scenarios)"),
    "atlas": ("atlas", "the validation atlas of a run (HTML) -> the run's atlas/"),
    "product": ("product", "the CalSim3 rim-inflow product: fit <run> | apply --tier2 DIR --forcing NAME"),
}

#: ``sacsma dpl study <name>``: module (under :mod:`sacsma.dpl.studies`), function, what it writes.
STUDIES: dict[str, tuple[str, str, str]] = {
    "climatology": ("climatology", "make_cdec15_climatology",
                    "per-watershed mean-monthly TAF regime vs the observed CalSim3 FNF, one "
                    "figure per ladder step (8: GA -> 1_hru ... 7_aef, then the hybrids on "
                    "5_px) + all-series metric bars"),
    "hybrids": ("hybrids", "make_hybrids",
                "the hybrid family (hybrid, hybrid_dt, lstm) response surfaces and skill summary"),
    "forcing": ("forcing_sensitivity", "make_forcing_sensitivity",
                "forcing-product sensitivity of the dPL chain"),
    "ladder": ("ladder", "make_ladder",
               "skill and climate response (+3 degC, precipitation +-10 %%) of every rung and "
               "LSTM model (CPU)"),
}


def _dpl_fidelity(args: argparse.Namespace) -> int:
    from .. import paths
    from .evaluate import fidelity_benchmark

    fidelity_benchmark(args.data_dir, args.out if args.out is not None else paths.dpl_fidelity())
    return 0


def _param_bounds(spec: str | None) -> dict[str, tuple[float, float]]:
    """``"lzsk=0.001:0.5,uzk=0.05:0.9"`` -> ``{"lzsk": (0.001, 0.5), "uzk": (0.05, 0.9)}``."""
    bounds: dict[str, tuple[float, float]] = {}
    for item in (spec or "").split(","):
        if not item.strip():
            continue
        name, _, rng = item.partition("=")
        lo, sep, hi = rng.partition(":")
        if not sep:
            raise SystemExit(f"--param-bounds {item!r}: expected name=lo:hi")
        bounds[name.strip()] = (float(lo), float(hi))
    return bounds


def _obs_mask(path: str | None) -> tuple[str, ...]:
    """Mask CSV(s) (``entity_id,date,...``; several joined by commas) ->
    ``("entity_id|YYYY-MM-DD", ...)``, a day named twice kept once."""
    if not path:
        return ()
    import pandas as pd

    out: dict[str, None] = {}
    for p in path.split(","):
        m = pd.read_csv(p, dtype=str, comment="#")
        if not {"entity_id", "date"} <= set(m.columns):
            raise SystemExit(f"--obs-mask {p}: needs entity_id and date columns")
        for e, d in zip(m["entity_id"], m["date"], strict=True):
            out[f"{e.strip()}|{pd.Timestamp(d).date().isoformat()}"] = None
    return tuple(out)


def _cdec_monthly_mask(path: str | None) -> tuple[str, ...]:
    """Mask CSV(s) (``entity_id,month,...``, month YYYY-MM; several joined by commas) ->
    ``("cdec_<SITE>|YYYY-MM", ...)``, a month named twice kept once."""
    if not path:
        return ()
    import pandas as pd

    out: dict[str, None] = {}
    for p in path.split(","):
        m = pd.read_csv(p, dtype=str, comment="#")
        if not {"entity_id", "month"} <= set(m.columns):
            raise SystemExit(f"--cdec-monthly-mask {p}: needs entity_id and month columns")
        for e, mo in zip(m["entity_id"], m["month"], strict=True):
            out[f"{e.strip()}|{pd.Period(mo, freq='M')}"] = None
    return tuple(out)


def _target_mask(path: str | None) -> tuple[str, ...]:
    """Mask CSV(s) (``entity_id`` with ``date``, ``month`` or ``start``/``end`` columns;
    several joined by commas) -> ``target_mask`` entries, each entity's consecutive days
    (months) joined into ``"entity_id|first..last"`` ranges."""
    if not path:
        return ()
    import pandas as pd

    days: dict[str, set] = {}
    mons: dict[str, set] = {}
    for p in path.split(","):
        m = pd.read_csv(p, dtype=str, comment="#")
        if "entity_id" not in m.columns or not ({"date"} <= set(m.columns)
                                                or {"month"} <= set(m.columns)
                                                or {"start", "end"} <= set(m.columns)):
            raise SystemExit(f"--target-mask {p}: needs entity_id and date, month or "
                             "start,end columns")
        if "start" in m.columns:
            spans = zip(m["entity_id"], m["start"], m["end"], strict=True)
        else:
            col = m["date"] if "date" in m.columns else m["month"]
            spans = zip(m["entity_id"], col, col, strict=True)
        for e, a, b in spans:
            e, a, b = e.strip(), a.strip(), b.strip()
            if len(a) == 10:
                days.setdefault(e, set()).update(pd.date_range(a, b, freq="D"))
            else:
                mons.setdefault(e, set()).update(pd.period_range(a, b, freq="M"))
    out = []
    for store, step, fmt in ((days, pd.Timedelta(days=1), lambda d: d.date().isoformat()),
                             (mons, 1, str)):
        for e in sorted(store):
            xs = sorted(store[e])
            a = prev = xs[0]
            for x in xs[1:] + [None]:
                if x is not None and x == prev + step:
                    prev = x
                    continue
                out.append(f"{e}|{fmt(a)}" + (f"..{fmt(prev)}" if prev != a else ""))
                if x is not None:
                    a = prev = x
    return tuple(out)


def _target_series(spec: str | None, data_dir: str) -> tuple[str, ...]:
    """``--target-series ID=CSV[,ID=CSV]`` -> ``target_series`` entries, each CSV's path
    relative to the data directory (the trainer and the evaluator read it under theirs)."""
    from pathlib import Path

    root = Path(data_dir).resolve()
    out = []
    for v in (spec or "").split(","):
        if not v:
            continue
        eid, _, p = v.partition("=")
        try:
            rel = Path(p).resolve().relative_to(root)
        except ValueError:
            raise ValueError(f"--target-series {v!r}: the CSV must lie under the data "
                             f"directory ({data_dir})") from None
        out.append(f"{eid}={rel.as_posix()}")
    return tuple(out)


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
        n_inc=args.n_inc, fracp_floor=args.fracp_floor, dtype=args.dtype, device=args.device,
        log_loss_lambda=args.log_lambda, var_loss_lambda=args.var_lambda,
        var_huber_cap=args.var_huber_cap, shape_min_days=args.shape_min_days,
        obs_mask=_obs_mask(args.obs_mask),
        cdec_monthly=args.cdec_monthly,
        cdec_monthly_mask=_cdec_monthly_mask(args.cdec_monthly_mask),
        target_mask=_target_mask(args.target_mask),
        target_scale=tuple(v for v in (args.target_scale or "").split(",") if v),
        target_series=_target_series(args.target_series, args.data_dir),
        dry_season_lambda=args.dry_season_lambda, dry_season_eps=args.dry_season_eps,
        deep_store=args.deep_store, deep_lr_mult=args.deep_lr_mult,
        init_from=args.init_from,
        holdout_wy=_holdout_wy(args.holdout_wy),
        uf_train_start=args.uf_train_start, calsim_arcs=args.calsim_arcs,
        train_window=tuple(args.train_window.split(":")) if args.train_window else (),
        select_cpu=args.cpu_select,
        pxtemp_learn=args.learn_pxtemp,
        pxtemp_box=tuple(float(v) for v in args.pxtemp_box.split(":")),
        pxtemp_tau=args.pxtemp_tau,
        lr=args.lr, grad_clip=args.grad_clip,
        lr_warmup_epochs=args.warmup_epochs, n_epochs=args.epochs,
        spinup_start=args.spinup_start, patience=args.patience,
        diagnostics=args.diagnostics,
        graph_recompute_days=args.graph_recompute_days,
        hidden=args.hidden, embed=args.embed, dropout=args.dropout,
        log_space_params=tuple(p.strip() for p in args.log_space.split(",") if p.strip()),
        param_bounds=_param_bounds(args.param_bounds),
        logit_penalty=args.logit_penalty,
        et_mode=args.et, noah_pet=args.noah_pet, sac_pet=args.sac_pet,
        mt_family_weight=args.mt_family_weight,
        mt_share_norm=args.mt_share_norm, mt_select_weight=args.mt_select_weight,
        mt_loss_ref=args.mt_loss_ref, mt_loss_ref_power=args.mt_loss_ref_power,
        nograd_window=args.nograd_window,
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
    # a multifamily run on a train window (the 15-CDEC rungs) is scored like a 15-CDEC run
    if ck.get("domain") == "multifamily" and not (ck.get("cfg") or {}).get("train_window"):
        # multi-timescale checkpoints score per entity at native timescales
        from .evaluate_multi_timescale import evaluate_checkpoint_mt

        evaluate_checkpoint_mt(args.checkpoint, data_dir=args.data_dir,
                               out_dir=args.out,
                               hydrographs=args.hydrographs,
                               spinup=args.spinup, score_holdout=args.score_holdout)
        return 0
    evaluate_checkpoint(args.checkpoint, data_dir=args.data_dir, out_dir=args.out)
    return 0


def _dpl_score(args: argparse.Namespace) -> int:
    import pandas as pd

    from .. import paths
    from .evaluate_multi_timescale import score_wy

    entities = None
    if args.entities:
        reg = pd.read_csv(paths.entities(args.data_dir), usecols=["entity_id", "family"])
        entities = tuple(e for tok in args.entities.split(",") for e in (
            reg.loc[reg["family"] == tok, "entity_id"] if (reg["family"] == tok).any()
            else [tok]))
    score_wy(args.checkpoint, _holdout_wy(args.wy), data_dir=args.data_dir,
             entities=entities, out_dir=args.out)
    return 0


def _dpl_record(args: argparse.Namespace) -> int:
    from .record import write_record

    write_record(args.run, data_dir=args.data_dir, out=args.out)
    return 0


def _dpl_hybrid(args: argparse.Namespace) -> int:
    import dataclasses

    import torch

    from .. import paths
    from .hybrid.evaluate import score_hybrid
    from .hybrid.train import RESPONSE_ANCHORS, HybridConfig, train_hybrid

    out = args.out or paths.local(name="testing/hybrid")
    fields = {f.name for f in dataclasses.fields(HybridConfig)}
    if args.init_from:
        # a fine-tune: the base's recipe, then the options given here
        base = torch.load(args.init_from, map_location="cpu", weights_only=False)["cfg"]
        cfg = {k: v for k, v in base.items() if k in fields}
        cfg.update(init_from=str(args.init_from), keep_last=True)
    else:
        cfg = dict(physics=args.physics, use_sim=bool(args.physics),
                   statics_from=args.statics_from)
    for k in ("n_epochs", "hidden", "lr", "warmup_epochs", "lr_min", "patience",
              "batch_size", "seed", "device"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if args.response_lambda > 0:
        cfg["response_anchors"] = tuple({"dp": dp, "dt": dt, "lambda": args.response_lambda}
                                        for dp, dt in RESPONSE_ANCHORS)
    train_hybrid(HybridConfig(**cfg), data_dir=args.data_dir, out_dir=out)
    run = paths.run_roles(out)
    score_hybrid(run.model / "checkpoints" / "best.pt",
                 data_dir=args.data_dir, out_dir=run.local)
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
    kw = {} if args.study == "ladder" else {"device": args.device}
    if args.study == "hybrids":
        kw["regen"] = args.regen
    if args.out is not None:
        kw["out_dir"] = args.out
    getattr(importlib.import_module(f".studies.{module}", __package__), func)(
        data_dir=args.data_dir, **kw)
    return 0


def register(sub) -> None:
    """Add the ``dpl`` command and its sub-commands to the top-level ``sacsma`` parser."""
    from .config import DplConfig

    dpl = sub.add_parser(
        "dpl",
        help="differentiable parameter learning (torch): train, evaluate, the CalSim3 rim-arc "
             "tools and the study figures",
    )
    dpl_sub = dpl.add_subparsers(dest="dpl_command", required=True)
    fi = dpl_sub.add_parser(
        "fidelity",
        help="the fidelity check: the archived GA optimum through the learned numerics "
             "(n_inc 1-20) vs the reference model, on the CPU engine "
             "-> artifacts/results/dpl/fidelity/",
    )
    fi.add_argument("--data-dir", default="data", help="organized data/ store")
    fi.add_argument("--out", default=None,
                    help="output dir (default: artifacts/results/dpl/fidelity)")
    fi.set_defaults(func=_dpl_fidelity)

    tr = dpl_sub.add_parser(
        "train",
        help="train a feature variant (spinup + water-year TBPTT; GPU asserted) "
             "-> artifacts/models/dpl/<domain>/<variant>/",
    )
    tr.add_argument("variant", choices=["physical", "aef_u"],
                    help="the network's inputs: physical = elev, lat, lon, the flow length "
                         "and the continuous soil/veg/terrain/LAI features; aef_u = "
                         "AlphaEarth embeddings only (16 PCs of the unit directions, "
                         "keeping their own spread, + the vector length; region grid only)")
    tr.add_argument("--data-dir", default="data", help="organized data/ store")
    tr.add_argument("--domain", default="15cdec",
                    choices=["15cdec", "15cdec_grid", "multifamily"],
                    help="training domain: 15cdec HRU cloud (7891), the native "
                         "1/16-deg Livneh grid (2074 cells), or the "
                         "multi-timescale training entities (the registry in "
                         "data/inputs/domains/multifamily, restrict with --basins; "
                         "daily + monthly targets on the registry envelope); baked into "
                         "the checkpoint so evaluate scores the same domain")
    tr.add_argument("--basins", default="",
                    help="comma list restricting training to these basin/"
                         "entity ids (the 15-CDEC rungs on the multifamily domain, or a "
                         "subset for a test); '' = the full domain")
    tr.add_argument("--mt-family-weight", default="none",
                    help="multi-timescale family weighting: none = every "
                         "valid daily entity weighs equally and the monthly "
                         "term adds with coefficient 1 (baseline); or numeric "
                         "shares 'usgs=0.27,cdec=0.54,uf=0.19' (+ calsim= with "
                         "--calsim-arcs; renormalized "
                         "over the families present, entities equal within a "
                         "family; selection uses the same share-weighted family "
                         "mean) (multifamily domain only)")
    tr.add_argument("--mt-share-norm", default="present", choices=["present", "all"],
                    help="with --mt-family-weight shares: 'present' divides each "
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
                         "'usgs=a,cdec=b,uf=c,calsim=d' = each family's per-entity "
                         "chunk loss (sum l_f / sum c_f) at a reference state, measured "
                         "under this run's loss and targets (re-measure when either "
                         "changes); each family term is multiplied by Lbar / L_ref_f^p "
                         "(Lbar = the share-weighted reference level^p; p = "
                         "--mt-loss-ref-power), frozen for the run.  Selection unchanged. "
                         "Default '' = off")
    tr.add_argument("--mt-loss-ref-power", type=float, default=None,
                    help="exponent p of --mt-loss-ref, REQUIRED with it: kappa_f = "
                         "Lbar / L_ref_f^p (1 = equal loss mass at the reference; 0.5 = "
                         "matches the families' optimizer-step shares at trained states)")
    tr.add_argument("--et", default="sac", choices=["sac", "noah"],
                    help="ET: sac = the SAC-SMA ET cascade; noah = the Noah-lite ET "
                         "(bare soil + canopy on the observed green fraction, one learned "
                         "exponent soil_chi, in place of the SAC E1-E3 withdrawals; needs "
                         "the observed veg/LAI tables = 15cdec_grid or multifamily)")
    tr.add_argument("--noah-pet", default="hamon",
                    choices=["hamon", "priestley_taylor"],
                    help="the potential ET of the Noah-lite ET: hamon = temperature-only; "
                         "priestley_taylor = energy-based from Bristow-Campbell net "
                         "radiation (needs per-cell tmin/tmax)")
    tr.add_argument("--sac-pet", default="hamon",
                    choices=["hamon", "priestley_taylor"],
                    help="the PET of the SAC-SMA ET (--et sac): hamon or priestley_taylor")
    tr.add_argument("--out", default=None,
                    help="output dir (default: artifacts/models/dpl/<domain>/<variant>)")
    tr.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="torch device (default: cuda; GPU is asserted)")
    tr.add_argument("--epochs", type=int, default=DplConfig.n_epochs)
    tr.add_argument("--n-inc", type=int, default=DplConfig.n_inc,
                    help="fixed SAC-SMA substep count a day (sacsma dpl fidelity)")
    tr.add_argument("--fracp-floor", type=float, default=DplConfig.fracp_floor,
                    help="LZ fill-fraction denominator floor: bounds the one "
                         "unbounded division's backward; engages only above "
                         "99.9%% LZ saturation")
    tr.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    tr.add_argument("--log-lambda", type=float, default=0.15,
                    help="low-flow log-space loss weight (0 disables)")
    tr.add_argument("--var-lambda", type=float, default=1.0,
                    help="variance-matching weight: (std sim - std obs) over the basin's "
                         "record std (the NNSE's normalizer), per chunk; counters "
                         "squared-error variance damping (0 disables)")
    tr.add_argument("--var-huber-cap", type=float, default=1.0,
                    help="the variance term's difference beyond which it grows linearly; "
                         "<= 0 = quadratic throughout")
    tr.add_argument("--shape-min-days", type=int, default=300,
                    help="the variance term scores a basin-chunk only with at least this "
                         "many valid observed days (whole water years)")
    tr.add_argument("--learn-pxtemp", action="store_true",
                    help="learn the Snow-17 rain/snow threshold PXTEMP per cell (else "
                         "the fixed 0 degC): zero-init head, hard split forward, "
                         "straight-through sigmoid-surrogate gradient")
    tr.add_argument("--pxtemp-box", default="-1:3", metavar="LO:HI",
                    help="learned PXTEMP range in degC (must contain 0)")
    tr.add_argument("--pxtemp-tau", type=float, default=1.0,
                    help="width (degC) of the sigmoid surrogate that carries the "
                         "PXTEMP gradient; the forward split stays hard")
    tr.add_argument("--obs-mask", default=None, metavar="CSV[,CSV]",
                    help="hand-edited CSV (entity_id,date,...) of daily observations "
                         "to mask out of training and scoring, e.g. "
                         "data/targets/cdec/fnf_daily_mask.csv; the checkpoint carries the "
                         "list")
    tr.add_argument("--cdec-monthly", default="", metavar="YYYY-MM-01",
                    help="train CDEC's monthly full natural flow as a monthly companion of the "
                         "cdec_daily entities from this month (multifamily), e.g. 1949-10-01; "
                         "its own share key 'cdecm' in --mt-family-weight and --mt-loss-ref. "
                         "'' = off")
    tr.add_argument("--cdec-monthly-mask", default=None, metavar="CSV[,CSV]",
                    help="months (entity_id,month YYYY-MM) left out of the companion")
    tr.add_argument("--target-mask", default=None, metavar="CSV[,CSV]",
                    help="targets of any family left out after the normalizers: CSVs with "
                         "entity_id and date, month or start,end columns (multifamily)")
    tr.add_argument("--target-scale", default=None, metavar="ID=F[,ID=F]",
                    help="per-entity factors on the target as read (a depth-basis fix)")
    tr.add_argument("--target-series", default=None, metavar="ID=CSV[,ID=CSV]",
                    help="a daily entity's replacement record (date,value_mm), a CSV under "
                         "the data directory (the checkpoint keeps its path relative to it)")
    tr.add_argument("--dry-season-lambda", type=float, default=0.0,
                    help="weight of the per-year Jul-Sep log-volume term (multifamily); 0 = off")
    tr.add_argument("--dry-season-eps", type=float, default=DplConfig.dry_season_eps,
                    help="its offset, as a share of each entity's median Jul-Sep volume")
    tr.add_argument("--deep-store", default="none", choices=["none", "cascade"],
                    help="'cascade' = a learned share of the supplemental drainage through a "
                         "slow store of learned residence (2-10 years), outside the lower-zone "
                         "deficit")
    tr.add_argument("--deep-lr-mult", type=float, default=1.0,
                    help="the deep store head's learning rate as a multiple of --lr")
    tr.add_argument("--init-from", default="", metavar="CKPT",
                    help="warm start (a fine-tune) from this checkpoint's net: same variant, "
                         "domain, entities, parameter mapping and input scaling; fresh "
                         "optimizer and schedule")
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
    tr.add_argument("--train-window", default="", metavar="FIRST:LAST",
                    help="train on this window of whole calendar months only (multifamily), "
                         "e.g. 1988-10-01:2003-09-30 (it starts on 1 Oct): targets outside "
                         "it are blanked after "
                         "the n_obs audit, and the chunks start at it (the spinup runs "
                         "ahead of it). '' = the envelope")
    tr.add_argument("--cpu-select", action="store_true",
                    help="score the selection every epoch on the CPU engine in a separate "
                         "process while the GPU trains (--patience then counts epochs)")
    tr.add_argument("--calsim-arcs", default="none", choices=["none", "train_default"],
                    help="train_default: append the train_default CalSim3 rim arcs of "
                         "data/targets/calsim3/arc_hierarchy.csv (tier A) as the calsim_monthly "
                         "family (cs_<ARC>, hierarchy order) after the other entities "
                         "(multifamily; share runs name it 'calsim=' in "
                         "--mt-family-weight and --mt-loss-ref)")
    tr.add_argument("--hidden", type=int, default=64, help="trunk width")
    tr.add_argument("--embed", type=int, default=32, help="embedding width")
    tr.add_argument("--dropout", type=float, default=DplConfig.dropout,
                    help="encoder dropout (0 = deterministic parameter map)")
    tr.add_argument("--warmup-epochs", type=int, default=DplConfig.lr_warmup_epochs,
                    help="linear LR warmup epochs (protects the GA-prior init)")
    tr.add_argument("--patience", type=int, default=10,
                    help="early-stop once more than this many selections in a row are "
                         "stale (every epoch with --cpu-select, else every --eval-every)")
    tr.add_argument("--graph-recompute-days", type=int,
                    default=DplConfig.graph_recompute_days,
                    help="activation recompute — one captured graph of this many days "
                         "replayed over every training window, the backward re-running "
                         "each segment from its stored start state, so graph memory "
                         "stays one segment's for the two-year window; ~1 extra forward "
                         "per segment; 0 = the training windows run eagerly")
    tr.add_argument("--diagnostics", action="store_true",
                    help="write chunk_log.csv (per chunk: loss by family, "
                         "pre-clip gradient norm), eval_terms.csv (selection "
                         "epochs: eval-mode loss by family and term) and "
                         "per-epoch net snapshots with an EMA shadow under "
                         "checkpoints/snapshots/; numerics unchanged")
    tr.add_argument("--log-space", default="", metavar="NAMES",
                    help="map these learned parameters in log space too (comma list, e.g. "
                         "lzsk,lzpk,MFMAX,MFMIN); the checkpoint carries the mapping")
    tr.add_argument("--param-bounds", default=None, metavar="NAME=LO:HI[,...]",
                    help="replace learned parameters' GA box (it may widen it), e.g. "
                         "lzsk=0.001:0.5; the checkpoint carries it")
    tr.add_argument("--logit-penalty", type=float, default=0.0,
                    help="weight of the soft penalty on head pre-activations beyond |6| "
                         "(mean squared excess, in each chunk's backward; 0 = off)")
    tr.add_argument("--spinup-start", default="1978-10-01",
                    help="no-grad spinup cold-start date (default 10 water "
                         "years before the WY1989 cal window: spans the "
                         "record-wet WY1982-83, which resets the LZ "
                         "tension-store memory exactly — measured KGE vs "
                         "the full prefix 1.000000; clamped to the record "
                         "start, so pass 1915-01-01 for the exact frozen "
                         "full-prefix convention; on the multifamily domain a "
                         "date inside the envelope is moved to 10 water years "
                         "before it)")
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--grad-clip", type=float, default=DplConfig.grad_clip,
                    help="max total gradient norm of a chunk's optimizer step (one step "
                         "per chunk: a chunk clipped from norm g enters with weight clip / g)")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--nograd-window", type=int, default=DplConfig.nograd_window,
                    help="CUDA-graph replay window (days) for the no-grad "
                         "spinup/selection streams; numerics-neutral (256 "
                         "on drivers that fault on very large graphs)")
    tr.add_argument("--no-graphs", action="store_true",
                    help="disable CUDA-graph capture (eager; much slower)")
    tr.add_argument("--resume", action="store_true",
                    help="continue from checkpoints/last.pt")
    tr.set_defaults(func=_dpl_train)

    ev = dpl_sub.add_parser(
        "evaluate",
        help="checkpoint -> parameter tables, then the field as trained on the CPU engine "
             "-> metrics, daily series and figures (a 15-CDEC run, or a multifamily run on a "
             "train window: the 15 outlets on WY1989-2003 and WY2004-18, the obs mask out)",
    )
    ev.add_argument("checkpoint", help="path to checkpoints/best.pt")
    ev.add_argument("--data-dir", default="data", help="organized data/ store")
    ev.add_argument("--out", default=None,
                    help="the run to write (any of its folders; default: the run of "
                         "the checkpoint): parameter tables go to its model folder, "
                         "scores and figures to its results folder")
    ev.add_argument("--hydrographs", default="review",
                    choices=["review", "all", "none"],
                    help="multi-timescale checkpoints only: which per-entity "
                         "diagnostics figures to draw (review = the cdec_daily "
                         "+ uf_monthly set; all adds the 69 USGS entities)")
    ev.add_argument("--spinup", default="cycle", choices=["cycle", "window"],
                    help="multi-timescale checkpoints only: state at the envelope "
                         "start — cycle = loop its first ten water years 20 times "
                         "from the cold start (timing-independent, default); window "
                         "= the ten water years before it")
    ev.add_argument("--score-holdout", action="store_true",
                    help="multi-timescale checkpoints with held-out water years only: score "
                         "them too (metrics_holdout.csv, medians printed); off = not read")
    ev.set_defaults(func=_dpl_evaluate)

    sc = dpl_sub.add_parser(
        "score",
        help="a cell-domain checkpoint's network (not a 15cdec HRU run) on multifamily "
             "entities over given water years (cycle spinup, the evaluator's targets) -> "
             "metrics_wy<A>-<B>.csv",
    )
    sc.add_argument("checkpoint", help="path to a checkpoint")
    sc.add_argument("--wy", required=True, metavar="FIRST-LAST", help="e.g. 1976-1985")
    sc.add_argument("--entities", default="",
                    help="comma-separated registry ids and/or family names (usgs_daily, "
                         "cdec_daily, uf_monthly, calsim_monthly); default: every registry "
                         "entity")
    sc.add_argument("--data-dir", default="data", help="organized data/ store")
    sc.add_argument("--out", default=None,
                    help="the run to write (any of its folders; default: the run of the "
                         "checkpoint)")
    sc.set_defaults(func=_dpl_score)

    rc = dpl_sub.add_parser(
        "record",
        help="what a multifamily run trained on, stage by stage (the run and its init_from "
             "donors): every entity's targets and every exclusion with its reason -> the "
             "run's results training_record/",
    )
    rc.add_argument("run", help="a run name, any of the run's folders, or a checkpoint")
    rc.add_argument("--data-dir", default="data", help="organized data/ store")
    rc.add_argument("--out", default=None,
                    help="output dir (default: training_record/ of the run's results folder)")
    rc.set_defaults(func=_dpl_record)

    hy = dpl_sub.add_parser(
        "hybrid",
        help="train + score one LSTM hybrid on the 15 CDEC watersheds (the ladder recipe: "
             "precip, Tmin, Tmax and the physics run's daily flow; with --statics-from that "
             "run's physical inputs and four climate indices) -> artifacts/_local/testing/hybrid/",
    )
    hy.add_argument("--init-from", default="",
                    help="a fine-tune of this hybrid checkpoint: its recipe and network, the "
                         "options given here on top, the last epoch kept (hybrid_dt: "
                         "--init-from <hybrid best.pt> --response-lambda 0.18 --epochs 15 "
                         "--lr 1e-4 --warmup-epochs 1)")
    hy.add_argument("--physics", default="",
                    help="the learned run whose daily flow, as trained, is the physics "
                         "channel: a run name or a checkpoint path; empty: a pure LSTM")
    hy.add_argument("--statics-from", default="5_px",
                    help="a physical-variant dPL run (name or checkpoint) whose inputs and the "
                         "dPL's four climate indices are the statics (default: 5_px)")
    hy.add_argument("--response-lambda", type=float, default=0.0,
                    help="weight of the response-consistency loss at the 14 (dprecip, dT) "
                         "anchors (dp 0, +-10, +-20 %%, dT 0, 2, 4 degC): pull the hybrid's "
                         "response toward the physics run's (0 = off)")
    hy.add_argument("--data-dir", default="data", help="organized data/ store")
    hy.add_argument("--out", default=None,
                    help="output dir (default: artifacts/_local/testing/hybrid)")
    # unset: the recipe's value (HybridConfig, or the base's with --init-from)
    hy.add_argument("--epochs", dest="n_epochs", type=int, default=None)
    hy.add_argument("--hidden", type=int, default=None, help="LSTM hidden size")
    hy.add_argument("--lr", type=float, default=None)
    hy.add_argument("--warmup-epochs", type=int, default=None)
    hy.add_argument("--lr-min", type=float, default=None)
    hy.add_argument("--patience", type=int, default=None)
    hy.add_argument("--batch-size", type=int, default=None)
    hy.add_argument("--device", default=None, help="cuda | cpu")
    hy.add_argument("--seed", type=int, default=None)
    hy.set_defaults(func=_dpl_hybrid)

    cs = dpl_sub.add_parser(
        "calsim",
        help="the dPL on the CalSim3 rim arcs: tier 1, tier 2, atlas, product",
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
        help="figures and tables of the 15-CDEC dPL study (regimes, response surfaces, forcing, "
             "the ladder)",
    )
    st_sub = st.add_subparsers(dest="study", required=True)
    for name, (_, _, text) in STUDIES.items():
        sp = st_sub.add_parser(name, help=text)
        sp.add_argument("--data-dir", default="data", help="organized data/ store")
        sp.add_argument("--out", default=None,
                        help="output folder (default: "
                             f"artifacts/results/dpl/studies/{name}/)")
        if name != "ladder":
            sp.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                            help="torch device for the hybrid reconstructions")
        if name == "hybrids":
            sp.add_argument("--regen", action="store_true",
                            help="recompute the metrics table instead of reloading it")
        sp.set_defaults(func=_dpl_study)
