"""S1 frozen-scale reference levels (design option 2a): per-entity, per-chunk training-loss
levels at H95's REFERENCE STATE (noah_cdec_uf_usgs_areaw best.pt, its eval-mode
sim_daily_mm.npz) with the shipped loss functions on the water-year grid — a copy of
resume/weighting/design_frozen_scale/levels_sim.py run THROUGH THE PATCHED LOADER
(sacsma.dpl.multi_timescale.load_entity_obs with holdout_wy / uf_train_start), so the
targets and NNSE normalizers are exactly the ones S1 trains on.

Four target definitions, each on its own EntityObs (usgs / cdec / uf from the areaw
entity sims; calsim from the areaw tier-2 arc stream, tier2/tier2_monthly.csv TAF ->
mm over the hierarchy SQ_MI with calsim_tier1.AF_PER_MM_MI2 — design (6) option (b)):
  old      : no holdout, no extension       -> must reproduce 0.7368 / 0.4202 / 0.0980
  ho       : holdout WY1976-85              -> must reproduce 0.7388 / 0.4202 / 0.0970
  ho+ufx   : holdout + uf from 1949-10-01   -> the S1 levels (uf over its extended window)
  calsim   : the 64 train_default arcs (holdout-masked; the arc mask already excludes it)
Per-entity level of family f = mean over scored (entity, chunk) of l_ik
== sum_k l_f,k / sum_k c_f,k under mt_share_norm='all'.
Writes build/levels_s1.json.  CPU only (no forward).  Run with build/py.sh (cwd = wt_merge).
Usage: levels_sim_s1.py [RUN_DIR]"""
import dataclasses
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import sacsma  # noqa: E402
from sacsma import cli  # noqa: E402
from sacsma.dpl.calsim_arcs import load_hierarchy  # noqa: E402
from sacsma.dpl.calsim_tier1 import AF_PER_MM_MI2  # noqa: E402
from sacsma.dpl.data import load_domain_tensors  # noqa: E402
from sacsma.dpl.loss import masked_basin_loss  # noqa: E402
from sacsma.dpl.multi_timescale import (load_entity_obs, monthly_chunk_target,  # noqa: E402
                                        monthly_nnse_loss)
from sacsma.dpl.train import _chunk_grid, _with_calsim_arcs  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
assert "wt_merge" in sacsma.__file__, sacsma.__file__          # never LIVE's sacsma
RUN = (sys.argv[1] if len(sys.argv) > 1 else
       "<repo>/artifacts/dpl/multifamily/"
       "noah_cdec_uf_usgs_areaw")
HOLDOUT, UF_START = (1976, 1985), "1949-10-01"
dt = torch.float64
mask = cli._obs_mask("data/cdec_fnf/fnf_daily_mask.csv")
basins = _with_calsim_arcs("data", None)                        # 95 base + 64 cs_ arcs
dom = load_domain_tensors("data", domain="multifamily", device="cpu", basins=basins)
print(f"sacsma from {os.path.dirname(sacsma.__file__)}; run {RUN}; {len(basins)} entities")

z = np.load(os.path.join(RUN, "sim_daily_mm.npz"))
ids = list(z["entity_id"])
sim_all = torch.as_tensor(z["sim_mm"].astype(np.float64))
# the arcs: areaw tier-2 monthly stream -> mm/month over SQ_MI
hier = load_hierarchy("data").set_index("arc")
t2 = pd.read_csv(os.path.join(RUN, "tier2", "tier2_monthly.csv"))
t2["month"] = pd.PeriodIndex(t2["month"], freq="M")


def levels(tag: str, eo, fams: tuple[str, ...]) -> dict:
    fam = np.array(eo.family)
    dates = z["date"]
    assert str(dates[0])[:10] == str(dom.dates[eo.t0].date()), (dates[0], dom.dates[eo.t0])
    assert len(dates) == eo.t1 - eo.t0
    grid = _chunk_grid(dom.dates, eo.t0, eo.t1, dom.n_time, mode="water_year", chunk=366)
    d_rows = [int(r) for r in eo.daily_rows if fam[r] in fams]
    d_idx = [int(np.flatnonzero(eo.daily_rows == r)[0]) for r in d_rows]
    m_rows = [int(r) for r in eo.monthly_rows if fam[r] in fams]
    m_idx = [int(np.flatnonzero(eo.monthly_rows == r)[0]) for r in m_rows]
    simd = sim_all[[ids.index(basins[r]) for r in d_rows]]
    months = pd.period_range(dom.dates[eo.t0], dom.dates[eo.t1 - 1], freq="M")
    # monthly sims on the envelope month grid: entity rows from the npz, arcs from tier 2
    simm = np.zeros((len(m_rows), len(months)))
    per = pd.DatetimeIndex(dates.astype(str)).to_period("M")
    col = (per.year * 12 + per.month - 1) - int(eo.month_code[0])
    for j, r in enumerate(m_rows):
        if fam[r] == "calsim_monthly":
            a = basins[r][3:]
            g = t2[t2["arc"] == a].set_index("month")["sim_taf"].reindex(months)
            simm[j] = g.to_numpy(np.float64) * 1000.0 / (float(hier.loc[a, "sq_mi"])
                                                         * AF_PER_MM_MI2)
        else:
            np.add.at(simm[j], col, z["sim_mm"][ids.index(basins[r])].astype(np.float64))
    obsd = eo.obs_daily.to(dt)[d_idx]
    vard = eo.var_daily.to(dt)[d_idx]
    obsm = eo.obs_monthly.to(dt)[m_idx]
    varm = eo.var_monthly.to(dt)[m_idx]
    nd, nm, K = len(d_rows), len(m_rows), len(grid)
    L = np.full((nd + nm, K), np.nan)
    KW = dict(kind="nnse", log_eps=0.01, var_gate_frac=1e-3, var_huber_cap=1.0,
              bias_lambda=0.0)
    for k, (c0, ce) in enumerate(grid):
        a, b = c0 - eo.t0, ce - eo.t0
        if nd:
            s, o = simd[:, a:b], obsd[:, a:b]
            ok = torch.isfinite(o).sum(1) >= 90
            for i in range(nd):
                if not bool(ok[i]):
                    continue
                w = torch.zeros(nd, dtype=dt)
                w[i] = 1.0
                L[i, k] = float(masked_basin_loss(s, o, vard, log_lambda=0.15, var_lambda=1.0,
                                                  weight=w, weight_total=1.0, **KW))
        if nm:
            _, cols, mk = monthly_chunk_target(dom.dates, c0, ce - c0, eo.t0, eo.t1,
                                               eo.month_code)
            tgt = obsm[:, torch.as_tensor(cols)]
            fin = torch.isfinite(tgt) & (torch.as_tensor(mk) > 0).unsqueeze(0)
            mtgt = torch.where(fin, tgt, torch.zeros_like(tgt))
            mfin = fin.to(dt)
            sm = torch.as_tensor(simm[:, cols])
            assert bool(torch.isfinite(sm)[fin].all()), f"chunk {k}: NaN sim on a scored month"
            sm = torch.nan_to_num(sm) * torch.as_tensor(mk > 0)
            for j in range(nm):
                if mfin[j].sum() >= 1:
                    L[nd + j, k] = float(monthly_nnse_loss(sm[j:j + 1], mtgt[j:j + 1],
                                                           mfin[j:j + 1], varm[j:j + 1],
                                                           n_total=1.0))
    efam = np.array([fam[r] for r in d_rows] + [fam[r] for r in m_rows])
    out = {}
    for f in fams:
        m = efam == f
        if m.any():
            x = L[m]
            out[f] = (float(np.nanmean(x)), int(np.isfinite(x).sum()))
    print(f"  [{tag}] " + ", ".join(f"{f.split('_')[0]} {v:.4f} (n={n})"
                                    for f, (v, n) in out.items()), flush=True)
    return out


base = tuple(b for b in basins if not b.startswith("cs_"))
dom95 = dataclasses.replace(dom, basins=base)
three = ("usgs_daily", "cdec_daily", "uf_monthly")
res = {}
res["old"] = levels("old: no holdout, no extension", load_entity_obs(
    dom95, "data", obs_mask=mask), three)
res["ho"] = levels("ho: holdout WY1976-85", load_entity_obs(
    dom95, "data", obs_mask=mask, holdout_wy=HOLDOUT), three)
eo = load_entity_obs(dom, "data", obs_mask=mask, holdout_wy=HOLDOUT, uf_train_start=UF_START)
res["s1"] = levels("s1: holdout + uf from 1949-10-01 + 64 arcs", eo, three + ("calsim_monthly",))
chk = {"old": (0.7368, 0.4202, 0.0980), "ho": (0.7388, 0.4202, 0.0970)}
for tag, want in chk.items():
    got = tuple(round(res[tag][f][0], 4) for f in three)
    print(f"  G3 {tag}: {got} vs {want} -> {'PASS' if got == want else 'FAIL'}")
lv = {f.split("_")[0]: res["s1"][f][0] for f in res["s1"]}
json.dump({"levels": lv, "n": {f.split("_")[0]: res["s1"][f][1] for f in res["s1"]},
           "old": {f.split("_")[0]: v for f, (v, _) in res["old"].items()},
           "ho": {f.split("_")[0]: v for f, (v, _) in res["ho"].items()},
           "run": RUN, "holdout_wy": HOLDOUT, "uf_train_start": UF_START},
          open(os.path.join(HERE, "build", "levels_s1.json"), "w"), indent=1)
print("S1 reference levels (4 d.p.): " + ", ".join(f"{f} {v:.4f}" for f, v in lv.items()))
