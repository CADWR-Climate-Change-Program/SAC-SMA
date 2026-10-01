"""S1 nominal family shares, re-solved so the REALIZED coefficient shares equal the 4-family
footprint-area shares (user decision 1 + 2, 2026-09-29), on the ACTUAL S1 chunk grid:
95 base + 64 train_default arcs, holdout WY1976-85, uf back-extended from 1949-10-01,
through the PATCHED loader (load_entity_obs holdout_wy / uf_train_start).

Realized share = H95's logged definition (h95_gates.py (c), RUNS.md): per epoch, sum over
the stepped chunks of c_f / kappa_f, normalized.  Under mt_share_norm='all' the trainer's
c_f is share_f * n_f,k / N_f (daily: daily_scale * w_f / w_total with w = share/N per
entity, n = entities with >= 90 finite days in the chunk; monthly: scale * n_m / N_monthly_f,
n_m = entities with >= 1 scored complete month), so realized_f ∝ s_f * E_f with
E_f = sum_k n_f,k / N_f over the live (= stepped) chunks.  Solve s_f ∝ A_f / E_f.
Check: the same model on H95's grid (95 entities, no holdout, no extension, nominal
.216/.558/.226) must give H95's logged 0.234/0.546/0.220.
With build/levels_s1.json (levels_sim_s1.py) it prints kappa at p = 0.5 for the solved
nominals.  Writes build/shares_s1.json.  Run with build/py.sh (cwd = wt_merge)."""
import dataclasses
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import sacsma  # noqa: E402
from sacsma import cli  # noqa: E402
from sacsma.dpl.config import FAMILY_KEYS, DplConfig  # noqa: E402
from sacsma.dpl.data import load_domain_tensors  # noqa: E402
from sacsma.dpl.multi_timescale import load_entity_obs, monthly_chunk_target  # noqa: E402
from sacsma.dpl.train import _chunk_grid, _print_holdout, _with_calsim_arcs  # noqa: E402

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
assert "wt_merge" in sacsma.__file__, sacsma.__file__
HOLDOUT, UF_START = (1976, 1985), "1949-10-01"
FAMS = tuple(FAMILY_KEYS.values())
mask = cli._obs_mask("data/cdec_fnf/fnf_daily_mask.csv")
basins = _with_calsim_arcs("data", None)
dom = load_domain_tensors("data", domain="multifamily", device="cpu", basins=basins)
reg = pd.read_csv("data/multifamily/entities.csv").set_index("entity_id")


def presence(eo) -> tuple[dict, dict, int, list]:
    """Entity-chunks per family over the live chunks (the trainer's liveness rule),
    entity counts per family, the number of live chunks, the dead water years."""
    fam = np.array(eo.family)
    grid = _chunk_grid(dom.dates, eo.t0, eo.t1, dom.n_time, mode="water_year", chunk=366)
    dfam, mfam = fam[eo.daily_rows], fam[eo.monthly_rows]
    ec = {f: 0 for f in FAMS}
    n_live, dead = 0, []
    for c0, ce in grid:
        a, b = c0 - eo.t0, min(ce, eo.t1) - eo.t0
        ok = (torch.isfinite(eo.obs_daily[:, a:b]).sum(1) >= 90).numpy()
        _, cols, mk = monthly_chunk_target(dom.dates, c0, ce - c0, eo.t0, eo.t1, eo.month_code)
        fin = (torch.isfinite(eo.obs_monthly[:, torch.as_tensor(cols)]).numpy()
               & (mk > 0)[None, :])
        mok = fin.sum(1) >= 1
        if not (ok.any() or mok.any()):
            dead.append(dom.dates[c0].year + 1)
            continue
        n_live += 1
        for f in FAMS:
            ec[f] += int((ok & (dfam == f)).sum()) + int((mok & (mfam == f)).sum())
    n = {f: int((fam == f).sum()) for f in FAMS}
    return ec, n, n_live, dead


def realized(s: dict, e: dict) -> dict:
    t = {f: s[f] * e[f] for f in s}
    z = sum(t.values())
    return {f: v / z for f, v in t.items()}


# ---- model check on H95's grid -----------------------------------------------------------
base = tuple(b for b in basins if not b.startswith("cs_"))
eo95 = load_entity_obs(dataclasses.replace(dom, basins=base), "data", obs_mask=mask)
ec, n, nl, dead = presence(eo95)
E95 = {f: ec[f] / n[f] for f in FAMS if n[f]}
r95 = realized({"usgs_daily": .216, "cdec_daily": .558, "uf_monthly": .226}, E95)
print(f"H95 grid: {nl} live chunks, dead WY {dead}; entity-chunks/entity "
      + ", ".join(f"{f.split('_')[0]} {v:.2f}" for f, v in E95.items())
      + " | realized at .216/.558/.226: " + "/".join(f"{v:.3f}" for v in r95.values())
      + ("  -> PASS (H95 logged 0.234/0.546/0.220)"
         if tuple(round(v, 3) for v in r95.values()) == (0.234, 0.546, 0.220) else "  -> FAIL"))

# ---- the S1 grid -------------------------------------------------------------------------
eo = load_entity_obs(dom, "data", obs_mask=mask, holdout_wy=HOLDOUT, uf_train_start=UF_START)
_print_holdout(dom, eo, DplConfig(holdout_wy=HOLDOUT, uf_train_start=UF_START))
ec, n, nl, dead = presence(eo)
E = {f: ec[f] / n[f] for f in FAMS}
print(f"S1 grid: {len(basins)} entities ({', '.join(f'{f} {n[f]}' for f in FAMS)}), {nl} live "
      f"chunks, dead WY {dead}; entity-chunks/entity "
      + ", ".join(f"{f.split('_')[0]} {E[f]:.3f} ({ec[f]})" for f in FAMS))
# the 4-family footprint-area shares (sum of registry area_mi2 per family)
area = reg.loc[list(basins)].groupby("family")["area_mi2"].sum()
A = {f: float(area[f] / area.sum()) for f in FAMS}
print("area shares (target realized): " + ", ".join(
    f"{f.split('_')[0]} {v:.4f} ({area[f]:,.1f} mi2)" for f, v in A.items()))
raw = {f: A[f] / E[f] for f in FAMS}
z = sum(raw.values())
nom = {f: v / z for f, v in raw.items()}
nom4 = {f: round(v, 4) for f, v in nom.items()}
print("SOLVED nominal shares (exact): " + ", ".join(f"{f.split('_')[0]} {v:.6f}"
                                                   for f, v in nom.items()))
print("SOLVED nominal shares (4 d.p.): " + ", ".join(f"{f.split('_')[0]} {v:.4f}"
                                                    for f, v in nom4.items())
      + f"  (sum {sum(nom4.values()):.4f}; the trainer renormalizes)")
t4 = sum(nom4.values())
nom4n = {f: v / t4 for f, v in nom4.items()}
rz = realized(nom4n, E)
print("realized at the 4 d.p. nominals: " + ", ".join(
    f"{f.split('_')[0]} {v:.4f} (target {A[f]:.4f}, d {v - A[f]:+.5f})" for f, v in rz.items()))
r_area = realized(A, E)
print("(for reference) realized if the area shares were used as nominals: "
      + "/".join(f"{v:.4f}" for v in r_area.values()))
spec = ",".join(f"{k}={nom4[v]:.4f}" for k, v in FAMILY_KEYS.items())
print(f"--mt-family-weight {spec}")

out = {"nominal_exact": nom, "nominal_4dp": nom4, "realized_4dp": rz, "area_share": A,
       "entity_chunks_per_entity": E, "n_entities": n, "live_chunks": nl, "dead_wy": dead,
       "mt_family_weight": spec}
lp = os.path.join(HERE, "build", "levels_s1.json")
if os.path.exists(lp):
    lv = json.load(open(lp))["levels"]
    L = {FAMILY_KEYS[k]: v for k, v in lv.items()}
    L4 = {f: round(v, 4) for f, v in L.items()}
    pw = 0.5
    lbar = sum(nom4n[f] * L4[f] ** pw for f in FAMS)
    kap = {f: lbar / L4[f] ** pw for f in FAMS}
    ref = ",".join(f"{k}={L4[v]:.4f}" for k, v in FAMILY_KEYS.items())
    print(f"reference levels (levels_sim_s1.py, 4 d.p. as passed): {ref}")
    print(f"kappa (p = 0.5, Lbar {lbar:.4f}): " + ", ".join(
        f"{f.split('_')[0]} x {v:.4f}" for f, v in kap.items()))
    # loss-mass share at the reference state = realized coefficient x kappa x level
    lm = {f: rz[f] * kap[f] * L4[f] for f in FAMS}
    zz = sum(lm.values())
    print("predicted loss-mass shares at the reference state: " + ", ".join(
        f"{f.split('_')[0]} {v / zz:.3f}" for f, v in lm.items()))
    out.update(mt_loss_ref=ref, kappa=kap, lbar=lbar, levels=L4)
    print(f"--mt-loss-ref {ref} --mt-loss-ref-power 0.5")
json.dump(out, open(os.path.join(HERE, "build", "shares_s1.json"), "w"), indent=1)
