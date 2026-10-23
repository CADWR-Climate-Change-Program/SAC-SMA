"""What a multifamily run trained on, stage by stage (``sacsma dpl record``).

A run's stages are its kept checkpoint and the chain of ``init_from`` donors behind it, first
stage first (dPL-CalSim: the warm start in ``provenance/warm_start.pt``, then the fine-tune).  Each
stage's targets are read with the trainer's own loader (:func:`load_entity_obs` with
:func:`entity_obs_kwargs` of the stage's configuration), so every count is what that stage's loss
saw.  A checkpoint keeps its mask entries but not why each was made: the reasons come from the
hand-kept tables of :func:`sacsma.paths.target_screens`.  Every unit a stage masks must be in
them, or the record stops (a table edited since the stage trained); ``stages.csv`` says whether
the tables hold exactly the stage's entries.

Writes to ``training_record/`` of the run's results folder:

* ``stages.csv``: one row per stage, with the checkpoint, its donor, the epochs, the learning
  rate, the epoch kept and its selection score, and the settings that choose the targets.
* ``families.csv``: per stage and family, the entities in the run and with a target, the units
  trained, the July-September units among them, and the CDEC monthly record.
* ``entities.csv``: per stage and entity, the registry window, then every count in the loader's
  order (``n_registry - n_series_removed - n_obs_mask - n_holdout + n_uf_extension -
  n_outside_window - n_screened = n_trained``), the first and last unit trained, the target's
  scale or replacement series, and the CDEC monthly record (available, screened, trained).
* ``exclusions.csv``: per stage, entity, stream (``target`` or ``cdec_monthly``) and reason, the
  units removed, with the table that gives the reason (a unit two rows name carries both).

Usage::

    sacsma dpl record <run> [--data-dir data] [--out DIR]
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .. import paths
from .config import config_from_checkpoint
from .multi_timescale import (
    ENVELOPE_END,
    ENVELOPE_START,
    _mask_ranges,
    entity_obs_kwargs,
    load_entity_obs,
)

#: the record's folder in the run's results folder, and its tables
RECORD_DIR = "training_record"
TABLES = ("stages.csv", "families.csv", "entities.csv", "exclusions.csv")
#: the settings that choose a stage's targets, recorded per stage in ``stages.csv``
TARGET_KEYS = ("holdout_wy", "uf_train_start", "train_window", "calsim_arcs", "cdec_monthly",
               "target_scale", "target_series", "dry_season_lambda", "deep_store")
#: the two mask settings whose reasons come from :func:`sacsma.paths.target_screens`
STREAMS = {"target_mask": "target", "cdec_monthly_mask": "cdec_monthly"}


def stage_chain(checkpoint) -> list[tuple[Path, dict]]:
    """``(path, checkpoint)`` of each stage, first stage first: ``checkpoint`` and the chain of
    its ``init_from`` donors."""
    chain = [(Path(checkpoint), torch.load(checkpoint, map_location="cpu", weights_only=False))]
    while src := chain[0][1]["cfg"].get("init_from"):
        p = Path(src)
        if not p.exists():
            raise FileNotFoundError(f"{chain[0][0]}: its init_from {src} does not exist")
        if any(p.resolve() == q.resolve() for q, _ in chain):
            raise ValueError(f"init_from chain loops at {src}")
        chain.insert(0, (p, torch.load(p, map_location="cpu", weights_only=False)))
    return chain


def _table_units(tables) -> dict[tuple[str, object], list[tuple[str, str]]]:
    """``(entity, day or month)`` -> ``(table, reason)`` of each table row that names it."""
    out = defaultdict(list)
    for p in tables:
        m = pd.read_csv(p, dtype=str, comment="#")
        why = m["rule"] if "rule" in m.columns else m["reason"]
        if "start" in m.columns:
            first, last = m["start"], m["end"]
        else:
            first = last = m["date"] if "date" in m.columns else m["month"]
        src = f"{p.parent.name}/{p.name}"
        for e, a, b, w in zip(m["entity_id"], first, last, why, strict=True):
            a, b = a.strip(), b.strip()
            rng = pd.date_range(a, b, freq="D") if len(a) == 10 else pd.period_range(a, b, freq="M")
            for u in rng:
                out[(e.strip(), u)].append((src, w.strip()))
    return out


def _why(rows: list[tuple[str, str]]) -> tuple[str, str]:
    """The tables and the reasons of one unit, each joined by `` + ``."""
    return (" + ".join(dict.fromkeys(s for s, _ in rows)), " + ".join(w for _, w in rows))


def _entry_units(entries) -> set:
    """``target_mask`` / ``cdec_monthly_mask`` entries -> the ``(entity, day or month)``
    units they name."""
    out = set()
    for eid, a, b, by_day in _mask_ranges(tuple(entries)):
        rng = pd.date_range(a, b, freq="D") if by_day else pd.period_range(a, b, freq="M")
        out.update((eid, u) for u in rng)
    return out


def _span(idx, finite) -> tuple[str, str]:
    k = np.flatnonzero(finite)
    if not len(k):
        return "", ""
    fmt = (lambda u: u.date().isoformat()) if isinstance(idx, pd.DatetimeIndex) else str
    return fmt(idx[k[0]]), fmt(idx[k[-1]])


def _jul_sep(idx, finite) -> int:
    return int((finite & np.isin(np.asarray(idx.month), (7, 8, 9))).sum())


def stage_record(ck: dict, data_dir, reasons: dict) -> tuple[list[dict], dict, dict]:
    """One stage: its entity rows, its exclusions ``{(entity, stream, reason): units}`` and how its
    mask entries compare with the tables ``{setting: "identical" | "tables hold N more units"}``."""
    cfg = config_from_checkpoint(ck)
    basins = list(ck["basins"])
    kw = entity_obs_kwargs(cfg)
    match = {}
    for key, stream in STREAMS.items():
        if kw[key]:
            have, table = _entry_units(kw[key]), set(reasons[stream])
            if not have <= table:
                raise ValueError(f"{key}: {len(have - table)} masked units are in no table of "
                                 "paths.target_screens (a table edited since the stage trained)")
            extra = len(table - have)
            match[key] = "identical" if not extra else f"tables hold {extra} more units"
    days = pd.date_range(ENVELOPE_START, ENVELOPE_END, freq="D")
    months = pd.period_range(days[0].to_period("M"), days[-1].to_period("M"), freq="M")
    dom = SimpleNamespace(basins=basins, dates=days, device="cpu", dtype=torch.float64)
    obs = load_entity_obs(dom, data_dir, **kw)
    raw = load_entity_obs(dom, data_dir, **{**kw, "target_mask": (), "cdec_monthly_mask": ()})
    reg = pd.read_csv(paths.entities(data_dir), dtype={"site_id": str}).set_index("entity_id")
    n_obs_mask = defaultdict(int)
    for s in kw["obs_mask"]:
        n_obs_mask[s.partition("|")[0]] += 1
    scale = dict(s.split("=", 1) for s in cfg.target_scale)
    series = dict(s.split("=", 1) for s in cfg.target_series)
    pos = {i: (obs.obs_daily, raw.obs_daily, j, days) for j, i in enumerate(obs.daily_rows)}
    pos.update({i: (obs.obs_monthly, raw.obs_monthly, j, months)
                for j, i in enumerate(obs.monthly_rows)})
    cm = {i: j for j, i in enumerate(obs.cm_rows)}
    rows, excl = [], defaultdict(int)
    for i, eid in enumerate(basins):
        r = reg.loc[eid]
        after, before, j, idx = pos[i]
        fa, fb = np.isfinite(after[j].numpy()), np.isfinite(before[j].numpy())
        for k in np.flatnonzero(fb & ~fa):
            excl[(eid, "target", *_why(reasons["target"][(eid, idx[k])]))] += 1
        row = {"entity": eid, "family": r["family"], "site": r["site_id"], "name": r["name"],
               "unit": "day" if idx is days else "month",
               "registry_start": r["train_start"], "registry_end": r["train_end"],
               "n_registry": int(r["n_obs"]),
               "n_series_removed": int(obs.n_masked[i]) - n_obs_mask[eid],
               "n_obs_mask": n_obs_mask[eid], "n_holdout": int(obs.n_holdout[i]),
               "n_uf_extension": int(obs.n_ext[i]), "n_outside_window": int(obs.n_out[i]),
               "n_screened": int(obs.n_screen[i]), "n_trained": int(fa.sum()),
               "n_trained_jul_sep": _jul_sep(idx, fa),
               "first_trained": _span(idx, fa)[0], "last_trained": _span(idx, fa)[1],
               "target_scale": scale.get(eid, ""), "target_series": series.get(eid, "")}
        if (row["n_registry"] - obs.n_masked[i] - row["n_holdout"] + row["n_uf_extension"]
                - row["n_outside_window"] - row["n_screened"]) != row["n_trained"]:
            raise AssertionError(f"{eid}: the counts do not add up to the units trained")
        if i in cm:
            ca = np.isfinite(obs.obs_cm[cm[i]].numpy())
            cb = np.isfinite(raw.obs_cm[cm[i]].numpy())
            for k in np.flatnonzero(cb & ~ca):
                excl[(eid, "cdec_monthly", *_why(reasons["cdec_monthly"][(eid, months[k])]))] += 1
            row.update({"cm_available": int(cb.sum()), "cm_screened": int((cb & ~ca).sum()),
                        "cm_trained": int(ca.sum()), "cm_trained_jul_sep": _jul_sep(months, ca),
                        "cm_first": _span(months, ca)[0], "cm_last": _span(months, ca)[1]})
        rows.append(row)
    return rows, excl, match


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_record(run, data_dir="data", out=None) -> Path:
    """The record of ``run`` (a run name, any of its folders, or a checkpoint) -> ``out``
    (default ``training_record/`` of the run's results folder)."""
    ckpt = paths.dpl_checkpoint(run=str(run), domain=paths.MULTIFAMILY)
    chain = stage_chain(ckpt)
    if any(ck.get("domain") != paths.MULTIFAMILY for _, ck in chain):
        raise ValueError(f"{ckpt}: the record reads multifamily runs only")
    out = Path(out) if out else paths.run_roles(ckpt.parents[1]).results / RECORD_DIR
    screens = paths.target_screens(data_dir)
    reasons = {stream: _table_units(screens[key]) for key, stream in STREAMS.items()}
    stages, entities, exclusions = [], [], []
    for n, (p, ck) in enumerate(chain, 1):
        cfg = config_from_checkpoint(ck)
        rows, excl, match = stage_record(ck, data_dir, reasons)
        stages.append({
            "stage": n, "checkpoint": _rel(p),
            "started_from": _rel(Path(cfg.init_from)) if cfg.init_from else "scratch",
            "variant": ck.get("variant"), "epochs": cfg.n_epochs, "epoch_kept": ck.get("epoch"),
            "selection_score": ck.get("best_kge"), "lr": cfg.lr, "entities": len(rows),
            "obs_mask_entries": len(cfg.obs_mask), "target_mask_entries": len(cfg.target_mask),
            "cdec_monthly_mask_entries": len(cfg.cdec_monthly_mask),
            "target_mask_vs_tables": match.get("target_mask", ""),
            "cdec_monthly_mask_vs_tables": match.get("cdec_monthly_mask", ""),
            **{k: ";".join(map(str, v)) if isinstance(v, tuple) else v
               for k, v in ((k, getattr(cfg, k)) for k in TARGET_KEYS)}})
        entities += [{"stage": n, **r} for r in rows]
        family = {r["entity"]: r["family"] for r in rows}
        exclusions += [{"stage": n, "entity": e, "family": family[e], "stream": s,
                        "source": src, "reason": why, "units": u}
                       for (e, s, src, why), u in sorted(excl.items())]
    E = pd.DataFrame(entities)
    E["trains"] = E.n_trained > 0
    fam = E.groupby(["stage", "family"], sort=False).agg(
        unit=("unit", "first"), entities=("entity", "size"), with_target=("trains", "sum"),
        trained=("n_trained", "sum"), trained_jul_sep=("n_trained_jul_sep", "sum"),
        screened=("n_screened", "sum"), holdout=("n_holdout", "sum"))
    if "cm_trained" in E:
        cm = E.groupby(["stage", "family"], sort=False)
        fam["cdec_monthly_entities"] = cm.cm_trained.count()
        fam["cdec_monthly_trained"] = cm.cm_trained.sum().astype(int)
        fam["cdec_monthly_jul_sep"] = cm.cm_trained_jul_sep.sum().astype(int)
        fam["cdec_monthly_screened"] = cm.cm_screened.sum().astype(int)
    out.mkdir(parents=True, exist_ok=True)
    tables = (pd.DataFrame(stages), fam.reset_index(), E.drop(columns="trains"),
              pd.DataFrame(exclusions))
    for name, t in zip(TABLES, tables, strict=True):
        t.to_csv(out / name, index=False)
    for (n, f), r in fam.iterrows():
        cm_txt = (f", CDEC monthly {r.cdec_monthly_trained:,} months at {r.cdec_monthly_entities}"
                  if "cdec_monthly_trained" in r and r.cdec_monthly_entities else "")
        print(f"stage {n} {f}: {r.with_target} of {r.entities} train, {r.trained:,} "
              f"{r.unit}s ({r.trained_jul_sep:,} in Jul-Sep){cm_txt}")
    print(f"record: {out}")
    return out
