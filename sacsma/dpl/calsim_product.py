"""The CalSim3 rim-inflow product of a multifamily dPL run: monthly TAF on every rim arc.

Tier 2 (:mod:`sacsma.dpl.calsim_tier2`) gives the dPL's own flow on each of the 196 rim INFLOW
arcs.  The product keeps that flow as it is wherever the dPL is the best estimate of the
CalSim3 series, and re-divides it among the arcs where it is not:

* **System flow** — the sum of a closure group's arcs (``data/calsim/arc_hierarchy.csv``,
  ``closure_group``) — is the dPL's, uncorrected.  A closure group of one arc is a
  *single-arc system* and takes the dPL arc flow.
* **Non-anchor arcs** — rim arcs with no closure group, and the groups of
  :data:`NON_ANCHOR_GROUPS` — also take the dPL arc flow.
* **Share arcs** — the arcs of a closure group with two or more arcs — take the *share
  model*: a small network gives each arc's share of its system's flow in a month, and each
  arc's water-year volume is then closed to its share of the system's water-year volume.  The
  arcs of a system therefore sum to the dPL system flow over every water year; inside the year
  their sum moves by a few percent of the monthly flow.

The share model.  Per arc and month, ``q = (dpl + floor) * exp(f)``, with
``f = log 2 * tanh(MLP)`` bounded to a factor of two and zero at initialization (= the dPL's own
shares).  Inputs: the arc's log share of the system's dPL flow and its lags 1, 2, 3, 6 and 12
months, the four routed runoff-part fractions, the log system flow, the month as sin / cos, the
log arc area, and learned arc (4) and system (2) embeddings.  Monthly flow is
``q / sum_system(q) * system flow``; then each (arc, water year) is rescaled to
``V_system * k_a D_a / sum_b k_b D_b``, where ``D`` is the dPL arc water-year volume of the same
pass and ``k_a`` the arc's training-year ratio of CalSim3 to dPL volume.  The loss is the mean
over systems of the arcs' ``1 - NSE`` against the CalSim3 series (arcs equal inside a system)
plus ``mu`` times a climate-response penalty: at three of the training climate points per epoch,
the squared miss of the product's volume change (%) and April-July share change (pp) against
the dPL arc's own.  No temperature or precipitation enters the model; the response comes
through the dPL flows of each pass.

``fit`` needs the run's base tier-2 pass with runoff parts (``<run>/tier2``, written by
``calsim_tier2 --components parts``) and the same at the eleven training climate points of
:data:`TRAIN_POINTS` (``calsim_tier2 --components parts --scenarios ...`` writes them to
``<run>/tier2_scenarios/<name>``; :func:`scenario_spec` gives the argument).  It fits on water
years 1950-2015 without the run's held-out water years (``DplConfig.holdout_wy``, or
``--holdout-wy``), selects ``mu`` out of fold over seven blocks of training water years
(:func:`select`) unless ``--mu`` fixes it, and scores the product on the held-out years.  Where
the passes of :data:`VALIDATION_POINTS` exist, the response gate is read at them: per point the
median and 90th percentile over arcs of the absolute misses, full gate = median <= 1 and
p90 <= 3 pp on both.

Usage::

    python -m sacsma.dpl.calsim_product fit <run_dir> [--scenarios DIR ...] [--out DIR]
                                  [--data-dir data] [--mu MU] [--holdout-wy A-B] [--epochs N]
                                  [--threads N]
    python -m sacsma.dpl.calsim_product apply <run_dir> --tier2 DIR [--out CSV] [--data-dir data]

``fit`` writes to ``<run>/calsim_product``: ``share_model.pt`` (the network and everything
:func:`apply` needs), ``rim_inflow_monthly.csv`` (``arc, month, taf, dpl_taf, kind``: the product
on the base pass), ``product_metrics.csv`` (per arc: KGE of the dPL arc and of the product on
the held-out and on the training water years), ``share_selection.csv`` (the out-of-fold
candidates), ``response_gate.csv`` and ``product_info.json``.  ``apply`` runs the fitted model on
any other tier-2 pass with runoff parts (a climate scenario) and writes the product for it.

The fit is on the CPU and seeded; with the same thread count it reproduces to round-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..metrics import kge
from .calsim_tier1 import run_holdout_wy

#: closure groups whose arcs are NOT closed to a system: Whiskeytown (``I_WKYTN``) lies inside
#: the Bend Bridge anchor, which no arc is closed to
NON_ANCHOR_GROUPS = ("FU_WH",)
#: the training climate points, name -> (temperature delta degC, precipitation scale)
TRAIN_POINTS = {"t1": (1.0, 1.0), "t3": (3.0, 1.0), "t4": (4.0, 1.0), "p85": (0.0, 0.85),
                "p115": (0.0, 1.15), "t1p85": (1.0, 0.85), "t1p115": (1.0, 1.15),
                "t3p85": (3.0, 0.85), "t3p115": (3.0, 1.15), "t4p85": (4.0, 0.85),
                "t4p115": (4.0, 1.15)}
#: climate points no fit sees: the response gate is read at them
VALIDATION_POINTS = {"t25": (2.5, 1.0), "p95": (0.0, 0.95), "p105": (0.0, 1.05),
                     "t25p95": (2.5, 0.95), "t25p105": (2.5, 1.05)}
PRODUCT_WY = (1950, 2015)
MUS = (0.03, 0.1, 0.3)
EPOCHS = 300
N_BLOCKS = 7
PENALTY_POINTS = 3
CLIP = float(np.log(2.0))
LAGS = (1, 2, 3, 6, 12)
PARTS = ("quick", "inter", "supp", "prim")
FEATURES = (["msin", "mcos", "lshare"] + [f"lshare_l{k}" for k in LAGS]
            + [f"frac_{c}" for c in PARTS] + ["lsys", "larea"])
_APR_JUL = (4, 5, 6, 7)


def scenario_spec(points: dict[str, tuple[float, float]] | None = None) -> str:
    """The ``calsim_tier2 --scenarios`` argument for ``points`` (default: the training and the
    validation points)."""
    points = points or {**TRAIN_POINTS, **VALIDATION_POINTS}
    return ",".join(f"{k}={t:g}:{s:g}" for k, (t, s) in points.items())


# --------------------------------------------------------------------------------- units
def load_units(data_dir: str | Path = "data") -> pd.DataFrame:
    """The rim arcs with their role: ``[sq_mi, system, kind]`` indexed by arc, in the
    hierarchy table's order.  ``kind`` is ``share`` (an arc of a multi-arc system),
    ``single-arc system`` or ``non-anchor``; ``system`` is the closure group (NaN for a
    non-anchor arc)."""
    h = pd.read_csv(Path(data_dir) / "calsim" / "arc_hierarchy.csv")
    h = h[h.status == "rim_arc"].set_index("arc")
    n = h.closure_group.value_counts()
    anchored = h.closure_group.notna() & ~h.closure_group.isin(NON_ANCHOR_GROUPS)
    u = pd.DataFrame({"sq_mi": h.sq_mi, "system": h.closure_group.where(anchored)})
    size = u.system.map(n)
    u["kind"] = np.where(~anchored, "non-anchor", np.where(size > 1, "share", "single-arc system"))
    return u


def load_pass(tier2_dir: str | Path, arcs) -> pd.DataFrame:
    """One tier-2 pass as ``[arc, month, sim, ref, quick, inter, supp, prim, wy, moy]`` for
    ``arcs`` over :data:`PRODUCT_WY`, sorted by arc and month (flows in TAF/month)."""
    d = Path(tier2_dir)
    comp = d / "tier2_components_monthly.csv"
    if not comp.exists():
        raise FileNotFoundError(f"{comp}: the pass needs calsim_tier2 --components parts")
    m = pd.read_csv(d / "tier2_monthly.csv")
    c = pd.read_csv(comp)
    m = m[m.arc.isin(arcs)].merge(
        c[["arc", "month", "quick_taf", "interflow_taf", "supplemental_taf", "primary_taf"]],
        on=["arc", "month"])
    m = m.rename(columns={"sim_taf": "sim", "ref_taf": "ref", "quick_taf": "quick",
                          "interflow_taf": "inter", "supplemental_taf": "supp",
                          "primary_taf": "prim"})
    m["month"] = pd.PeriodIndex(m.month, freq="M")
    m["wy"] = [p.year + (p.month >= 10) for p in m.month]
    m["moy"] = [p.month for p in m.month]
    m = m[(m.wy >= PRODUCT_WY[0]) & (m.wy <= PRODUCT_WY[1])]
    return m.sort_values(["arc", "month"]).reset_index(drop=True)


def find_pass(name: str, run_dir: Path, scenario_dirs) -> Path | None:
    """The folder of climate point ``name`` (``base`` = ``<run>/tier2``), or None."""
    if name == "base":
        return run_dir / "tier2"
    for d in scenario_dirs:
        if (Path(d) / name / "tier2_monthly.csv").exists():
            return Path(d) / name
    return None


# --------------------------------------------------------------------------------- the model
def share_features(D: pd.DataFrame, eps: pd.Series) -> pd.DataFrame:
    """The share model's inputs for one pass (``D`` sorted by arc and month, with ``system``).
    Shares and the system flow are centred on the pass's own arc / system means."""
    f = pd.DataFrame(index=D.index)
    tot = D.groupby(["system", "month"]).sim.transform("sum")
    sh = np.log((D.sim + D.arc.map(eps)) / (tot + 1e-9))
    f["lshare"] = sh - sh.groupby(D.arc).transform("mean")
    g = f.lshare.groupby(D.arc)
    for k in LAGS:
        f[f"lshare_l{k}"] = g.shift(k)
    ptot = D[list(PARTS)].sum(axis=1) + 1e-9
    for c in PARTS:
        f[f"frac_{c}"] = D[c] / ptot
    lsys = np.log(tot + 1e-9)
    f["lsys"] = lsys - lsys.groupby(D.system).transform("mean")
    f["msin"], f["mcos"] = np.sin(2 * np.pi * D.moy / 12), np.cos(2 * np.pi * D.moy / 12)
    return f


class ShareNet(nn.Module):
    """``f = log 2 * tanh(MLP(inputs, arc embedding, system embedding))``, zero at init."""

    def __init__(self, n_features: int, n_arcs: int, n_systems: int):
        super().__init__()
        self.ea, self.es = nn.Embedding(n_arcs, 4), nn.Embedding(n_systems, 2)
        self.mlp = nn.Sequential(nn.Linear(n_features + 6, 64), nn.SiLU(), nn.Linear(64, 64),
                                 nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, arc_i, sys_i):
        h = torch.cat([x, self.ea(arc_i), self.es(sys_i)], 1)
        return CLIP * torch.tanh(self.mlp(h).squeeze(1))


def _seg(x, g, size=None):
    size = int(g.max()) + 1 if size is None else size
    return torch.zeros(size, dtype=x.dtype).index_add_(0, g, x)


class _Rows:
    """The row structure of one pass's share-arc frame: arc / system indices and the
    (system, month), (arc, water year) and (system, water year) groups."""

    def __init__(self, B: pd.DataFrame, arcs: list[str], systems: list[str], n_per_system):
        self.n_arcs = len(arcs)
        arc_id = {a: i for i, a in enumerate(arcs)}
        sys_id = {s: i for i, s in enumerate(systems)}
        self.arc_i = torch.tensor(B.arc.map(arc_id).to_numpy(), dtype=torch.long)
        self.sys_i = torch.tensor(B.system.map(sys_id).to_numpy(), dtype=torch.long)
        fac = lambda *c: torch.tensor(pd.MultiIndex.from_arrays(c).factorize()[0],  # noqa: E731
                                         dtype=torch.long)
        self.g_sm, self.g_aw = fac(B.system, B.month), fac(B.arc, B.wy)
        self.g_sw = fac(B.system, B.wy)
        self.nsys = torch.tensor(B.system.map(n_per_system).to_numpy(), dtype=torch.float64)
        self.aj = torch.tensor(B.moy.isin(_APR_JUL).to_numpy())

    def close(self, f, sim, anchor, kv, floor):
        """Arc flow from the network output ``f``: shares of the system-month flow ``anchor``,
        then each (arc, water year) closed to its ``k x dPL`` share of the system's volume."""
        q = (sim + floor) * torch.exp(f.double())
        Qm = q / _seg(q, self.g_sm)[self.g_sm] * anchor
        Vs = _seg(anchor / self.nsys, self.g_sw)[self.g_sw]
        kD = kv * sim
        Va = Vs * _seg(kD, self.g_aw)[self.g_aw] / _seg(kD, self.g_sw)[self.g_sw].clamp_min(1e-12)
        return Qm * Va / _seg(Qm, self.g_aw)[self.g_aw].clamp_min(1e-12)

    def sums(self, Q, sim, m):
        """Per arc over rows ``m``: (volume, Apr-Jul volume) of ``Q`` and of ``sim``."""
        md, n = m.double(), self.n_arcs
        return (_seg(Q * md, self.arc_i, n), _seg(Q * md * self.aj, self.arc_i, n),
                _seg(sim * md, self.arc_i, n), _seg(sim * md * self.aj, self.arc_i, n))


def _system_flow(D: pd.DataFrame) -> np.ndarray:
    """The dPL system flow of each row's (system, month): the sum of the system's arcs."""
    return D.groupby(["system", "month"]).sim.transform("sum").to_numpy()


def response_misses(dpl_b, dpl_p, prod_b, prod_p, unit, aj) -> pd.DataFrame:
    """Per unit, the product's miss against the dPL at one climate point: ``mV`` = difference of
    the volume changes (%), ``mAJ`` = difference of the April-July share changes (pp)."""
    d = pd.DataFrame({"u": unit, "tb": dpl_b, "tp": dpl_p, "cb": prod_b, "cp": prod_p, "aj": aj})
    rows = []
    for u, x in d.groupby("u"):
        if x.tb.sum() <= 0 or x.cb.sum() <= 0:
            continue
        a = x.aj.to_numpy()
        dVt = 100 * (x.tp.sum() / x.tb.sum() - 1)
        dVc = 100 * (x.cp.sum() / x.cb.sum() - 1)
        dAt = 100 * (x.tp[a].sum() / x.tp.sum() - x.tb[a].sum() / x.tb.sum())
        dAc = 100 * (x.cp[a].sum() / x.cp.sum() - x.cb[a].sum() / x.cb.sum())
        rows.append((u, dVc - dVt, dAc - dAt))
    return pd.DataFrame(rows, columns=["unit", "mV", "mAJ"])


def gate_table(R: pd.DataFrame) -> pd.DataFrame:
    """Per climate point: median and p90 over units of the absolute misses, with the full gate
    (median <= 1 and p90 <= 3 pp) and the half gate (half those) on each of dV and dAJ."""
    q = lambda v: v.abs().quantile(.9)   # noqa: E731
    g = R.groupby("pt", sort=False).agg(medV=("mV", lambda v: v.abs().median()), p90V=("mV", q),
                                        medAJ=("mAJ", lambda v: v.abs().median()), p90AJ=("mAJ", q))
    for nm, (a, b) in (("half", (.5, 1.5)), ("full", (1.0, 3.0))):
        g[f"V_{nm}"] = (g.medV <= a) & (g.p90V <= b)
        g[f"AJ_{nm}"] = (g.medAJ <= a) & (g.p90AJ <= b)
    return g


def _unit_kge(Q, ref, unit, mask) -> pd.Series:
    s = pd.DataFrame({"Q": Q[mask], "r": ref[mask], "u": unit[mask]})
    return s.groupby("u").apply(lambda d: kge(d.Q.to_numpy(), d.r.to_numpy()))


class _Fit:
    """The share arcs of one run at the base and the climate points, ready to fit."""

    def __init__(self, units: pd.DataFrame, passes: dict[str, Path], holdout_wy):
        hm = units[units.kind == "share"]
        self.arcs, self.systems = hm.index.tolist(), sorted(hm.system.unique())
        self.n = hm.system.value_counts()
        self.points = list(passes)
        runs = {p: load_pass(d, self.arcs) for p, d in passes.items()}
        B = runs["base"]
        B["system"] = B.arc.map(hm.system)
        n_ref = B.groupby(["system", "month"]).ref.transform("count")
        keep = (n_ref == B.system.map(self.n)).to_numpy()
        keys = B[["arc", "month"]].to_numpy()
        for p, r in runs.items():
            if len(r) != len(keys) or not (r[["arc", "month"]].to_numpy() == keys).all():
                raise ValueError(f"pass {p}: its arc-months differ from the base pass")
            r["system"] = r.arc.map(hm.system)
            runs[p] = r[keep].reset_index(drop=True)
        self.B = B = runs["base"]
        ho = holdout_wy or ()
        self.TR = ~((B.wy >= ho[0]) & (B.wy <= ho[1])).to_numpy() if ho else np.ones(len(B), bool)
        mean_tr = B[self.TR].groupby("arc").sim.mean()
        self.eps, self.floor_arc = mean_tr * 0.01, mean_tr * 1e-6
        self.floor = torch.tensor(B.arc.map(self.floor_arc).to_numpy(), dtype=torch.float64)
        self.larea = np.log(B.arc.map(hm.sq_mi)).to_numpy()
        self.Fx = {}
        for p in self.points:
            f = share_features(runs[p], self.eps)
            f["larea"] = self.larea
            self.Fx[p] = f[FEATURES].to_numpy(float)
        self.rows = _Rows(B, self.arcs, self.systems, self.n)
        self.ref = torch.tensor(B.ref.to_numpy(), dtype=torch.float64)
        self.sim_np = {p: runs[p].sim.to_numpy() for p in self.points}
        self.SIM = {p: torch.tensor(v, dtype=torch.float64) for p, v in self.sim_np.items()}
        self.ANC = {p: torch.tensor(_system_flow(runs[p]), dtype=torch.float64)
                    for p in self.points}
        self.arc_w = torch.tensor(1.0 / hm.system.map(self.n).reindex(self.arcs).to_numpy())
        self.train_points = [p for p in self.points if p in TRAIN_POINTS]

    def kvec(self, rows):
        """Per arc, the ratio of CalSim3 to dPL volume over ``rows`` (clipped to 0.1-10)."""
        g = self.B[rows].groupby("arc")
        return (g.ref.sum() / g.sim.sum()).clip(0.1, 10)

    def kv(self, k: pd.Series):
        return torch.tensor(self.B.arc.map(k).to_numpy(), dtype=torch.float64)

    def scaler(self, tr):
        return np.nanmean(self.Fx["base"][tr], 0), np.nanstd(self.Fx["base"][tr], 0) + 1e-9

    def xs(self, scaler):
        mu_x, sd_x = scaler
        return {p: torch.tensor(np.nan_to_num((self.Fx[p] - mu_x) / sd_x), dtype=torch.float32)
                for p in self.points}

    def predict(self, net, Xs, p, kv):
        r = self.rows
        f = torch.zeros(len(self.B)) if net is None else net(Xs[p], r.arc_i, r.sys_i)
        return r.close(f, self.SIM[p], self.ANC[p], kv, self.floor)

    def fit(self, Xs, tr, kv, mu: float, epochs: int = EPOCHS) -> ShareNet:
        r, n_arcs = self.rows, len(self.arcs)
        net = ShareNet(Xs["base"].shape[1], n_arcs, len(self.systems))
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
        trm = torch.tensor(tr)
        m = trm.double()
        mref = _seg(self.ref * m, r.arc_i, n_arcs) / _seg(m, r.arc_i, n_arcs).clamp_min(1)
        sst = _seg((self.ref - mref[r.arc_i]) ** 2 * m, r.arc_i, n_arcs).clamp_min(1e-9)
        pts = self.train_points
        for _ in range(epochs):
            opt.zero_grad()
            Qb = self.predict(net, Xs, "base", kv)
            loss = (self.arc_w * _seg((Qb - self.ref) ** 2 * m, r.arc_i, n_arcs) / sst).sum() \
                / len(self.systems)
            if mu > 0:
                cb, cba, tb, tba = r.sums(Qb, self.SIM["base"], trm)
                pen = 0.0
                sub = [pts[i] for i in torch.randperm(len(pts))[:PENALTY_POINTS].tolist()]
                for p in sub:
                    cp, cpa, tp, tpa = r.sums(self.predict(net, Xs, p, kv), self.SIM[p], trm)
                    mV = 100 * (cp / cb.clamp_min(1e-9) - tp / tb.clamp_min(1e-9))
                    mA = 100 * ((cpa / cp.clamp_min(1e-9) - cba / cb.clamp_min(1e-9))
                                - (tpa / tp.clamp_min(1e-9) - tba / tb.clamp_min(1e-9)))
                    pen = pen + ((mV ** 2 + mA ** 2) / 2).mean()
                loss = loss + mu * pen / len(sub)
            loss.backward()
            opt.step()
        return net

    def misses(self, Q: dict, pts, mask) -> pd.DataFrame:
        unit, aj = self.B.arc.to_numpy(), self.B.moy.isin(_APR_JUL).to_numpy()
        return pd.concat([response_misses(self.sim_np["base"][mask], self.sim_np[p][mask],
                                          Q["base"][mask], Q[p][mask], unit[mask], aj[mask]
                                          ).assign(pt=p) for p in pts])


def select(F: _Fit, mus=MUS, epochs: int = EPOCHS, n_blocks: int = N_BLOCKS, log=print):
    """Out-of-fold selection of ``mu`` over ``n_blocks`` contiguous blocks of training water
    years, at the base and the training climate points.  Rule: among the ``mu`` whose response
    misses meet the half gate on dAJ and the full gate on dV at every training point, the one
    with the highest median arc KGE (ties: the smaller ``mu``); else among those meeting the full
    gate on both; else None (the dPL's own shares inside the closure).  Returns
    ``(mu or None, candidates)``."""
    B, TR = F.B, F.TR
    blocks = np.array_split(sorted(B.wy[TR].unique()), n_blocks)
    pts = ["base"] + F.train_points
    recs = []
    for mu in (None, *mus):
        Q = {p: np.zeros(len(B)) for p in pts}
        for b in blocks:
            te = B.wy.isin(b).to_numpy()
            tr = TR & ~te
            kv = F.kv(F.kvec(tr))
            Xs = net = None
            if mu is not None:
                Xs = F.xs(F.scaler(tr))
                net = F.fit(Xs, tr, kv, mu, epochs)
            with torch.no_grad():
                for p in pts:
                    Q[p][te] = F.predict(net, Xs, p, kv).numpy()[te]
        ks = _unit_kge(Q["base"], B.ref.to_numpy(), B.arc.to_numpy(), TR)
        g = gate_table(F.misses(Q, F.train_points, TR))
        w = g[["medV", "p90V", "medAJ", "p90AJ"]].max()
        full = bool(g.V_full.all() and g.AJ_full.all())
        rec = dict(model="dPL shares" if mu is None else "share model", mu=mu, kge_med=ks.median(),
                   kge_p10=ks.quantile(.1), medV=w.medV, p90V=w.p90V, medAJ=w.medAJ, p90AJ=w.p90AJ,
                   aj_half_v_full=bool(g.AJ_half.all() and full), full=full)
        recs.append(rec)
        log(f"  {rec['model']:11s} mu {'-' if mu is None else mu:>4}  out-of-fold KGE median "
            f"{rec['kge_med']:.3f} p10 {rec['kge_p10']:.3f} | dV {w.medV:.2f}/{w.p90V:.2f} "
            f"dAJ {w.medAJ:.2f}/{w.p90AJ:.2f}{' *' if rec['aj_half_v_full'] else ''}")
    cand = pd.DataFrame(recs)
    net_rows = cand[cand.mu.notna()]
    for col in ("aj_half_v_full", "full"):
        ok = net_rows[net_rows[col]].sort_values(["kge_med", "mu"], ascending=[False, True])
        if len(ok):
            return float(ok.iloc[0].mu), cand.assign(rule=col)
    return None, cand.assign(rule="none passes: dPL shares")


# --------------------------------------------------------------------------------- product
def _assemble(units: pd.DataFrame, tier2_dir, share: pd.DataFrame) -> pd.DataFrame:
    """``[arc, month, taf, dpl_taf, kind]`` for every rim arc of a pass: the share arcs from
    ``share`` (``arc, month, taf``), every other arc its dPL flow."""
    m = pd.read_csv(Path(tier2_dir) / "tier2_monthly.csv", usecols=["arc", "month", "sim_taf"])
    m = m[m.arc.isin(units.index)].rename(columns={"sim_taf": "dpl_taf"})
    per = pd.PeriodIndex(m.month, freq="M")
    wy = per.year + (per.month >= 10)
    m = m[(wy >= PRODUCT_WY[0]) & (wy <= PRODUCT_WY[1])]
    m = m.merge(share.assign(month=share.month.astype(str)), on=["arc", "month"], how="left")
    m["kind"] = m.arc.map(units.kind)
    missing = m.taf.isna() & (m.kind == "share")
    if missing.any():
        raise ValueError(f"{int(missing.sum())} share arc-months have no share-model flow")
    m["taf"] = m.taf.where(m.kind == "share", m.dpl_taf)
    m = m[["arc", "month", "taf", "dpl_taf", "kind"]]
    return m.sort_values(["arc", "month"]).reset_index(drop=True)


def apply(model: dict, tier2_dir: str | Path, data_dir: str | Path = "data") -> pd.DataFrame:
    """The product for one tier-2 pass (with runoff parts) under a fitted share model
    (``share_model.pt`` as loaded by ``torch.load``): ``[arc, month, taf, dpl_taf, kind]``."""
    units = load_units(data_dir)
    hm = units[units.kind == "share"]
    arcs, systems = model["arcs"], model["systems"]
    if sorted(arcs) != sorted(hm.index):
        raise ValueError("the share arcs of data/calsim/arc_hierarchy.csv differ from the model's")
    D = load_pass(tier2_dir, arcs)
    D["system"] = D.arc.map(hm.system)
    as_s = lambda v: pd.Series(v, index=arcs)   # noqa: E731
    f = share_features(D, as_s(model["eps"]))
    f["larea"] = np.log(D.arc.map(hm.sq_mi)).to_numpy()
    x = np.nan_to_num((f[FEATURES].to_numpy(float) - model["x_mean"]) / model["x_std"])
    rows = _Rows(D, arcs, systems, hm.system.value_counts())
    t64 = lambda v: torch.tensor(np.asarray(v), dtype=torch.float64)   # noqa: E731
    with torch.no_grad():
        if model["state"] is None:
            out = torch.zeros(len(D))
        else:
            net = ShareNet(len(FEATURES), len(arcs), len(systems))
            net.load_state_dict(model["state"])
            out = net(torch.tensor(x, dtype=torch.float32), rows.arc_i, rows.sys_i)
        Q = rows.close(out, t64(D.sim.to_numpy()), t64(_system_flow(D)),
                       t64(D.arc.map(as_s(model["k"])).to_numpy()),
                       t64(D.arc.map(as_s(model["floor"])).to_numpy())).numpy()
    return _assemble(units, tier2_dir, pd.DataFrame({"arc": D.arc, "month": D.month, "taf": Q}))


def _summary(M: pd.DataFrame, col: str) -> pd.DataFrame:
    def row(d):
        v = d[col]
        return pd.Series(dict(n=int(v.notna().sum()), median=v.median(), mean=v.mean(),
                              p10=v.quantile(.1), below_0=int((v < 0).sum())))
    out = M.groupby("kind").apply(row)
    out.loc["all rim arcs"] = row(M)
    return out.astype({"n": int, "below_0": int})


def fit_product(run_dir: str | Path, scenario_dirs=None, out: str | Path | None = None,
                data_dir: str | Path = "data", mu: float | None = None, holdout_wy=None,
                epochs: int = EPOCHS, threads: int = 8, log=print) -> dict:
    """Fit the share model of a run and write its product (module docstring)."""
    run_dir = Path(run_dir)
    out = Path(out) if out else run_dir / "calsim_product"
    scenario_dirs = list(scenario_dirs or [run_dir / "tier2_scenarios"])
    holdout_wy = tuple(holdout_wy) if holdout_wy else run_holdout_wy(run_dir)
    passes = {p: find_pass(p, run_dir, scenario_dirs)
              for p in ["base", *TRAIN_POINTS, *VALIDATION_POINTS]}
    absent = [p for p in ["base", *TRAIN_POINTS] if passes[p] is None]
    if absent:
        raise FileNotFoundError(
            f"no tier-2 pass for the training points {absent}; run calsim_tier2 "
            f"--components parts --scenarios {scenario_spec(TRAIN_POINTS)}")
    passes = {p: d for p, d in passes.items() if d is not None}
    val_points = [p for p in VALIDATION_POINTS if p in passes]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    units = load_units(data_dir)
    F = _Fit(units, passes, holdout_wy)
    log(f"calsim_product: {run_dir.name}: {len(F.arcs)} share arcs in {len(F.systems)} systems, "
        f"{int((units.kind != 'share').sum())} arcs on the dPL flow; held-out water years "
        f"{holdout_wy or 'none'}; climate points {len(F.train_points)} training + "
        f"{len(val_points)} validation")
    cand = None
    if mu is None:
        log(f"out-of-fold selection of mu ({N_BLOCKS} blocks of training water years):")
        mu, cand = select(F, epochs=epochs, log=log)
        cand.to_csv(out / "share_selection.csv", index=False)
        log(f"  selected: {'the dPL shares (no network)' if mu is None else f'mu {mu:g}'}")
    k = F.kvec(F.TR)
    kv, scaler = F.kv(k), F.scaler(F.TR)
    Xs = F.xs(scaler)
    net = None if mu is None else F.fit(Xs, F.TR, kv, mu, epochs)
    with torch.no_grad():
        Q = {p: F.predict(net, Xs, p, kv).numpy() for p in F.points}
    model = dict(state=None if net is None else net.state_dict(), arcs=F.arcs, systems=F.systems,
                 x_mean=scaler[0], x_std=scaler[1], k=k.reindex(F.arcs).to_numpy(),
                 eps=F.eps.reindex(F.arcs).to_numpy(), floor=F.floor_arc.reindex(F.arcs).to_numpy(),
                 mu=mu, epochs=epochs, features=FEATURES, holdout_wy=holdout_wy,
                 train_points=F.train_points, run=run_dir.name)
    torch.save(model, out / "share_model.pt")
    # the product on the base pass, and its scores
    prod = _assemble(units, passes["base"], pd.DataFrame({"arc": F.B.arc, "month": F.B.month,
                                                           "taf": Q["base"]}))
    prod.to_csv(out / "rim_inflow_monthly.csv", index=False, float_format="%.8g")
    ref = pd.read_csv(passes["base"] / "tier2_monthly.csv", usecols=["arc", "month", "ref_taf"])
    S = prod.merge(ref, on=["arc", "month"]).dropna(subset=["ref_taf"])
    per = pd.PeriodIndex(S.month, freq="M")
    wy = np.asarray(per.year + (per.month >= 10))
    ho = (wy >= holdout_wy[0]) & (wy <= holdout_wy[1]) if holdout_wy else np.zeros(len(S), bool)
    unit, r = S.arc.to_numpy(), S.ref_taf.to_numpy()
    M = pd.DataFrame({"kind": units.kind.reindex(sorted(S.arc.unique()))})
    for tag, mask in (("holdout", ho), ("train", ~ho)):
        if mask.any():
            M[f"kge_dpl_{tag}"] = _unit_kge(S.dpl_taf.to_numpy(), r, unit, mask)
            M[f"kge_product_{tag}"] = _unit_kge(S.taf.to_numpy(), r, unit, mask)
    M.rename_axis("arc").to_csv(out / "product_metrics.csv")
    info = dict(run=run_dir.name, mu=mu, epochs=epochs, threads=threads, holdout_wy=holdout_wy,
                product_wy=PRODUCT_WY, train_points=TRAIN_POINTS,
                validation_points={p: VALIDATION_POINTS[p] for p in val_points},
                passes={p: str(d) for p, d in passes.items()}, n_arcs=int(prod.arc.nunique()))
    tag = "holdout" if holdout_wy else "train"
    for who in ("dpl", "product"):
        t = _summary(M, f"kge_{who}_{tag}")
        info[f"kge_{who}_{tag}"] = t.round(4).to_dict("index")
        log(f"monthly KGE vs CalSim3, {'held-out' if holdout_wy else 'all'} water years — "
            f"{'the dPL arcs' if who == 'dpl' else 'the product'}:\n"
            + t.to_string(float_format=lambda v: f"{v:.3f}"))
    if val_points:
        G = gate_table(F.misses(Q, val_points, np.ones(len(F.B), bool)))
        G.to_csv(out / "response_gate.csv")
        ok = bool(G.V_full.all() and G.AJ_full.all())
        info["validation_full_gate"] = ok
        log("response misses of the share arcs against the dPL at the validation points (pp):\n"
            + G[["medV", "p90V", "medAJ", "p90AJ"]].to_string(float_format=lambda v: f"{v:.2f}")
            + f"\nfull gate: {'PASS' if ok else 'FAIL'}")
    (out / "product_info.json").write_text(json.dumps(info, indent=1))
    log(f"calsim_product: wrote {out}")
    return dict(model=model, product=prod, metrics=M, Q=Q, frame=F.B, info=info)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="sacsma.dpl.calsim_product",
        description="The CalSim3 rim-inflow product of a multifamily dPL run")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit the share model and write the product of the base pass")
    f.add_argument("run", help="run folder (holds checkpoints/best.pt and tier2/)")
    f.add_argument("--scenarios", action="append", default=None, metavar="DIR",
                   help="folder of climate-point tier-2 passes, one subfolder per point "
                        "(repeatable; default <run>/tier2_scenarios)")
    f.add_argument("--out", default=None, help="output folder (default <run>/calsim_product)")
    f.add_argument("--data-dir", default="data")
    f.add_argument("--mu", type=float, default=None,
                   help="response-penalty weight; omit to select it out of fold")
    f.add_argument("--holdout-wy", default=None, metavar="A-B",
                   help="water years kept out of the fit (default: the checkpoint's holdout_wy)")
    f.add_argument("--epochs", type=int, default=EPOCHS)
    f.add_argument("--threads", type=int, default=8,
                   help="torch CPU threads (the fit reproduces at the same count)")
    g = sub.add_parser("apply", help="the product for another tier-2 pass (a climate scenario)")
    g.add_argument("run", help="run folder (holds calsim_product/share_model.pt)")
    g.add_argument("--tier2", required=True, help="the tier-2 pass (with --components parts)")
    g.add_argument("--out", default=None, help="CSV (default <tier2>/rim_inflow_monthly.csv)")
    g.add_argument("--data-dir", default="data")
    a = p.parse_args(argv)
    if a.cmd == "fit":
        ho = tuple(int(v) for v in a.holdout_wy.split("-")) if a.holdout_wy else None
        fit_product(a.run, a.scenarios, a.out, a.data_dir, a.mu, ho, a.epochs, a.threads)
        return
    model = torch.load(Path(a.run) / "calsim_product" / "share_model.pt", weights_only=False)
    out = Path(a.out) if a.out else Path(a.tier2) / "rim_inflow_monthly.csv"
    apply(model, a.tier2, a.data_dir).to_csv(out, index=False, float_format="%.8g")
    print(f"calsim_product: wrote {out}")


if __name__ == "__main__":
    main()
