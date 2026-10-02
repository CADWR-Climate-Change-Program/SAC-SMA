"""Per-entity, per-chunk training-loss levels of a TRAINED state, from a run's eval-mode
sim_daily_mm.npz, with the CURRENT recipe loss (nnse + 0.15 log + 1.0 Huber var, gate 1e-3,
cap 1.0; monthly NNSE on fixed envelope variances) on the water-year chunk grid, exactly the
shipped loss functions.  Output: <tag>_entity_chunk.npz  (L[entity, chunk], plus the
nnse / log / var split for daily entities) and a family summary printed.
Per-entity level of family f  =  sum over (entity, chunk) scored of l_ik / number scored
== sum_k l_f,k / sum_k c_f,k under mt_share_norm='all' (share cancels).
Usage: levels_sim.py RUN_DIR TAG"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import torch  # noqa: E402

from sacsma import cli  # noqa: E402
from sacsma.dpl.data import load_domain_tensors  # noqa: E402
from sacsma.dpl.loss import masked_basin_loss  # noqa: E402
from sacsma.dpl.multi_timescale import (load_entity_obs, monthly_chunk_target,  # noqa: E402
                                        monthly_nnse_loss)
from sacsma.dpl.train import _chunk_grid  # noqa: E402

run_dir, tag = sys.argv[1], sys.argv[2]
OUT = os.path.dirname(os.path.abspath(__file__))
dt = torch.float64
dom = load_domain_tensors("data", domain="multifamily", device="cpu")
eo = load_entity_obs(dom, "data", obs_mask=cli._obs_mask("data/cdec_fnf/fnf_daily_mask.csv"))
fam = np.array(eo.family)
basins = list(dom.basins)
z = np.load(os.path.join(run_dir, "sim_daily_mm.npz"))
ids = list(z["entity_id"])
dates = z["date"]
assert str(dates[0])[:10] == str(dom.dates[eo.t0].date()), (dates[0], dom.dates[eo.t0])
assert len(dates) == eo.t1 - eo.t0
sim_all = torch.as_tensor(z["sim_mm"].astype(np.float64))
grid = _chunk_grid(dom.dates, eo.t0, eo.t1, dom.n_time, mode="water_year", chunk=366)

# daily entities present in this run
d_rows = [int(r) for r in eo.daily_rows if basins[r] in ids]
d_idx = [int(np.flatnonzero(eo.daily_rows == r)[0]) for r in d_rows]
m_rows = [int(r) for r in eo.monthly_rows if basins[r] in ids]
m_idx = [int(np.flatnonzero(eo.monthly_rows == r)[0]) for r in m_rows]
simd = sim_all[[ids.index(basins[r]) for r in d_rows]]
simm = sim_all[[ids.index(basins[r]) for r in m_rows]]
obsd = eo.obs_daily.to(dt)[d_idx]
vard = eo.var_daily.to(dt)[d_idx]
obsm = eo.obs_monthly.to(dt)[m_idx]
varm = eo.var_monthly.to(dt)[m_idx]
nd, nm, K = len(d_rows), len(m_rows), len(grid)
L = np.full((nd + nm, K), np.nan)
Ln = np.full((nd, K), np.nan)
Ll = np.full((nd, K), np.nan)
Lv = np.full((nd, K), np.nan)
KW = dict(kind="nnse", log_eps=0.01, var_gate_frac=1e-3, var_huber_cap=1.0, bias_lambda=0.0)
for k, (c0, ce) in enumerate(grid):
    a, b = c0 - eo.t0, ce - eo.t0
    s, o = simd[:, a:b], obsd[:, a:b]
    ok = torch.isfinite(o).sum(1) >= 90
    for i in range(nd):
        if not bool(ok[i]):
            continue
        w = torch.zeros(nd, dtype=dt)
        w[i] = 1.0
        f = lambda lg, vr: float(masked_basin_loss(s, o, vard, log_lambda=lg, var_lambda=vr,  # noqa: E731
                                                   weight=w, weight_total=1.0, **KW))
        full, n0, nl = f(0.15, 1.0), f(0.0, 0.0), f(0.15, 0.0)
        L[i, k], Ln[i, k], Ll[i, k], Lv[i, k] = full, n0, nl - n0, full - nl
    if nm:
        bucket, cols, mk = monthly_chunk_target(dom.dates, c0, ce - c0, eo.t0, eo.t1, eo.month_code)
        tgt = obsm[:, torch.as_tensor(cols)]
        fin = torch.isfinite(tgt) & (torch.as_tensor(mk) > 0).unsqueeze(0)
        mtgt = torch.where(fin, tgt, torch.zeros_like(tgt))
        mfin = fin.to(dt)
        sm = simm[:, a:b] @ torch.as_tensor(bucket, dtype=dt)
        for j in range(nm):
            if mfin[j].sum() >= 1:
                L[nd + j, k] = float(monthly_nnse_loss(sm[j:j + 1], mtgt[j:j + 1], mfin[j:j + 1],
                                                       varm[j:j + 1], n_total=1.0))
ent = [basins[r] for r in d_rows] + [basins[r] for r in m_rows]
efam = np.array([fam[r] for r in d_rows] + [fam[r] for r in m_rows])
np.savez(os.path.join(OUT, f"{tag}_entity_chunk.npz"), L=L, Ln=Ln, Ll=Ll, Lv=Lv,
         entity=np.array(ent), family=efam)
print(f"[{tag}] {run_dir}: {nd} daily + {nm} monthly entities, {K} chunks")
for f in ("usgs_daily", "cdec_daily", "uf_monthly"):
    m = efam == f
    if not m.any():
        continue
    x = L[m]
    msg = (f"  {f:11s} n={m.sum():2d} entity-chunks {np.isfinite(x).sum():5d}  per-entity level "
           f"{np.nanmean(x):.4f}  median {np.nanmedian(x):.4f}")
    if f != "uf_monthly":
        dm = m[:nd]
        msg += (f"  (nnse {np.nanmean(Ln[dm]):.4f} log {np.nanmean(Ll[dm]):.4f} "
                f"var {np.nanmean(Lv[dm]):.4f})")
    print(msg)
