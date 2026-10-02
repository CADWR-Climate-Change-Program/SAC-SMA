"""Model-state spinup for a simulated window: timing-independent cycling or the
legacy preceding window.

``cycle`` (timing-independent): the state at the window start is the fixed point of
the window's OWN first ``spinup_years`` years — that block is streamed again and
again, each pass starting from the previous pass's end state.  Nothing before the
window is read, so the rule is the same for the historical record, a
climate-perturbed copy of it and a stochastic trace, and the state does not depend
on where the window sits in the record.  A block of whole years loops seamlessly
(its last day is the calendar day before its first).

The loop starts from the frozen cold start (SMA ``[0, 0, 100, 100, 100, 0]``, snow,
canopy water and routing history empty — the state every run trains from) and runs
a FIXED number of passes, ``spinup_passes`` (default 20).  Fixed, because a stopping
rule misleads here: the change between passes understates the remaining distance to
the fixed point 10-20x, and any rule evaluated on some set of rows makes a cell's
state depend on which rows a caller watches.  A fixed count gives every cell the
same state in every caller and domain holding it.

Trained multifamily fields put the lower-zone tension capacity at its 5000 mm
ceiling, and under Noah-lite that store is drawn by ET only above a 5 % wilting
fraction, so it can take centuries to settle, filling in some cells and emptying in
others.  No start is near the fixed point everywhere: from full storages a riva ~
0.9 field ran twice its equilibrium block volume on the first pass and was still 10 %
off after 20 passes (its stores empty), while a riva = 0 field reached 1e-3 in 16
passes from full storages against ~65 from the cold start (its stores fill).  From
the cold start both were within 0.5 % in entity block volume after 20 passes, and
the riva = 0 field's WY1950-84 tier-1 scores matched the full-start ones (0.857 vs
0.858 mean KGE).  A few cells never settle from either start (lztwc 4900 mm apart
after 100 passes) while moving the entity flow little.  The largest annual flow
change of the last pass is reported as a diagnostic.  The trainer runs the full
count at its first spinup; each later spinup continues from the previous state for
at least ``spinup_warm_passes`` (2) passes and until a pass moves no basin's annual
flow by more than ``spinup_warm_tol`` (1e-3) — at most the full count — tracking the
fixed point as the parameters move.

``window`` (legacy): stream the ``DplConfig.spinup_start`` window ahead of the
start — on the multifamily envelope the ten water years before it.  It needs
forcing before the window, and trained fields whose storages fill over decades are
not at equilibrium after ten years (a riva = 0 multifamily field was +53 % in
WY1950 volume at UF13 against a 1915 spinup).

:func:`cycle_spinup` takes any ``stream(t0, t1, state) -> (rows, state)``, so the
trainer drives it with its CUDA-graph replays and the evaluators with
:func:`stream_rows` (eager).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch

from .config import DplConfig
from .data import DomainTensors
from .forward import PipelineState, initial_state, run_window

SPINUP_MODES = ("cycle", "window")


def block_end(dates: pd.DatetimeIndex, t0: int, years: int) -> int:
    """Record index ``years`` years after ``dates[t0]`` (the same calendar day),
    clipped to the record end — the exclusive end of the spinup block."""
    d0 = dates[t0]
    return min(int(dates.searchsorted(d0 + pd.DateOffset(years=years))), len(dates))


def window_start(dates: pd.DatetimeIndex, t0: int, spinup_start: str) -> int:
    """The legacy ``window`` spinup start for a window beginning at ``t0``:
    ``spinup_start`` when it lies before the window, else the ten years before
    it; clamped to the record start."""
    spin = int(dates.searchsorted(pd.Timestamp(spinup_start)))
    if spin >= t0:
        spin = int(dates.searchsorted(dates[t0] - pd.DateOffset(years=10)))
    return max(min(spin, t0), 0)


def year_index(dates: pd.DatetimeIndex, t0: int, t1: int) -> torch.Tensor:
    """Year-of-block index (0, 1, ...) of each day in ``[t0, t1)``, years counted
    from ``dates[t0]`` (the last one partial when the block is not whole years)."""
    d = dates[t0:t1]
    ann = pd.DatetimeIndex([dates[t0] + pd.DateOffset(years=k)
                            for k in range(int((d[-1] - d[0]).days // 365) + 2)])
    return torch.as_tensor(ann.searchsorted(d, side="right") - 1, dtype=torch.int64)


def annual_totals(rows: torch.Tensor, yidx: torch.Tensor) -> torch.Tensor:
    """``(R, T)`` daily rows -> ``(R, Y)`` totals per block year (float64)."""
    out = torch.zeros(rows.shape[0], int(yidx.max()) + 1, dtype=torch.float64,
                      device=rows.device)
    return out.index_add_(1, yidx.to(rows.device), rows.double())


def cycle_spinup(
    stream: Callable[[int, int, PipelineState], tuple[torch.Tensor | None, PipelineState]],
    t0: int, t1: int, state: PipelineState, passes: int, *,
    min_passes: int | None = None, until: float | None = None, floor: float = 1.0,
) -> tuple[PipelineState, int, float]:
    """Loop the block ``[t0, t1)`` from ``state``: ``passes`` times, or with
    ``until``, from ``min_passes`` on as soon as the last pass's change is below
    ``until`` (at most ``passes``).  ``stream`` returns ``(annual flow totals (R, Y)
    or None, end state)``.  The change is the last pass's largest annual change
    relative to the row's mean annual flow (at least ``floor`` mm; NaN with fewer
    than two passes or no rows).  Returns ``(state, passes run, change)``."""
    prev = None
    change = float("nan")
    for k in range(1, passes + 1):
        q, state = stream(t0, t1, state)
        if q is not None:
            if prev is not None:
                d = (q - prev).abs().max(dim=1).values / q.mean(dim=1).abs().clamp_min(floor)
                d = d[torch.isfinite(d)]
                change = float(d.max()) if d.numel() else float("nan")
            prev = q
        if until is not None and k >= (min_passes or 1) and change < until:
            return state, k, change
    return state, passes, change


def _pert_col(v, neutral: float):
    """A perturbation as applied to an ``(n_phys, T)`` forcing chunk: None when it is the
    scalar ``neutral`` value (no-op, the forcing untouched), the scalar itself otherwise,
    or a per-row tensor as an ``(n_phys, 1)`` column."""
    if isinstance(v, torch.Tensor):
        return v.reshape(-1, 1)
    return None if v == neutral else v


def _perturbed(temp_delta, precip_scale) -> bool:
    return (isinstance(temp_delta, torch.Tensor) or isinstance(precip_scale, torch.Tensor)
            or bool(temp_delta) or precip_scale != 1.0)


def cold_state(dom: DomainTensors, cfg: DplConfig,
               params: dict[str, torch.Tensor]) -> PipelineState:
    """The cycle spinup's start: the frozen cold start (SMA [0, 0, 100, 100, 100, 0],
    snow, canopy water and routing history empty)."""
    return initial_state(dom.n_phys, dom.device, dom.dtype, init_mode="reference",
                         params=params, et_mode=cfg.et_mode, n_rows=dom.n_hru)


def stream_rows(
    dom: DomainTensors, cfg: DplConfig, params: dict[str, torch.Tensor],
    uh: tuple[torch.Tensor, torch.Tensor],
    canopy: dict[str, torch.Tensor] | None,
    t0: int, t1: int, state: PipelineState,
    agg: Callable[[torch.Tensor], torch.Tensor] | None, *, window: int = 512,
    components: bool = False, temp_delta: float = 0.0, precip_scale: float = 1.0,
    parts: bool = False,
) -> tuple[torch.Tensor | None, PipelineState]:
    """Eager no-grad ``run_window`` over ``[t0, t1)`` in ``window``-day pieces;
    ``agg`` maps each piece's cell flow ``(N, T)`` to output rows ``(R, T)``
    (``None``: stream the state only).  Returns ``(rows (R, t1 - t0) or None, state)``.
    Under cell dedup (``dom.dedup``) ``params``/``canopy`` and the state's snow / SAC /
    canopy stores are per distinct cell; the flow handed to ``agg`` (with its
    components and parts) is per HRU row.

    ``components``: ``agg`` is called as ``agg(flow, fast, slow)`` with the routed
    fast/slow runoff components of ``run_window`` (``flow = fast + slow``; both net
    of the et4 channel-ET deduction).  ``parts``: the routed runoff parts
    ``(quick, interflow, supplemental, primary)`` of ``run_window(return_parts=True)``
    follow as further ``agg`` arguments (after ``fast, slow`` when ``components``),
    and the state carries their routing history (``state.hist_parts``).
    ``temp_delta`` (degC, added to tavg/tmin/tmax) and ``precip_scale`` (multiplies
    precip) are a uniform climate perturbation of the forcing, the scalar path of
    ``evaluate._noah_stream``; the defaults leave the forcing untouched.  Either may
    also be a ``(n_phys,)`` tensor, one value per physics row (the scenario batch of
    :func:`sacsma.dpl.calsim_tier2.stream_batch`: each row's forcing perturbed by its
    own scenario's value, the same arithmetic as the scalar path)."""
    out: list[torch.Tensor] = []
    pscale, tdelta = _pert_col(precip_scale, 1.0), _pert_col(temp_delta, 0.0)
    with torch.no_grad():
        t = t0
        while t < t1:
            te = min(t + window, t1)
            pr, ta, doy, leap = dom.chunk(t, te)
            tn, tx = dom.chunk_tmm(t, te)
            if pscale is not None:              # multiplicative precip perturbation
                pr = pr * pscale
            if tdelta is not None:              # uniform warming perturbation
                ta = ta + tdelta
                if tn is not None:
                    tn, tx = tn + tdelta, tx + tdelta
            res = run_window(
                pr, ta, doy, leap, dom.phys_lat_rad, dom.phys_elev, params, uh, state,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode, fracp_floor=cfg.fracp_floor,
                ninc_mode="fixed", et_mode=cfg.et_mode, canopy_params=canopy,
                tmin=tn, tmax=tx, veg_frac=dom.phys_veg_frac, lai=dom.chunk_lai(t, te),
                noah_pet=cfg.noah_pet, sac_pet=cfg.sac_pet,
                pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                canopy_lite=cfg.canopy_lite, sac_exchanges=cfg.noah_sac_exchanges, state_idx=dom.chunk_state(t, te),
                row_cell=dom.row_cell, return_components=components, return_parts=parts)
            flow, state = res[0], res[1]
            if agg is not None:
                extra = [c for grp in res[2:] for c in grp]   # [fast, slow] + [parts]
                out.append(agg(flow, *extra) if extra else agg(flow))
            t = te
    return (torch.cat(out, dim=1) if agg is not None else None), state


def describe(dates: pd.DatetimeIndex, t0: int, t1: int, passes: int, change: float) -> str:
    """One line on a cycle spinup."""
    return (f"cycle spinup over {dates[t0].date()}..{dates[t1 - 1].date()}: {passes} passes "
            f"from the cold start, last-pass annual flow change {change:.1e}")


def spin_state(
    dom: DomainTensors, cfg: DplConfig, params: dict[str, torch.Tensor],
    uh: tuple[torch.Tensor, torch.Tensor],
    canopy: dict[str, torch.Tensor] | None,
    t0: int, *, mode: str = "cycle",
    agg: Callable[[torch.Tensor], torch.Tensor] | None = None,
    temp_delta: float = 0.0, precip_scale: float = 1.0,
    parts: bool = False, window: int = 512,
) -> tuple[PipelineState, str]:
    """The state at record day ``t0`` under ``mode`` (eager): ``cycle`` from the cold
    start over the window's first ``cfg.spinup_years`` years, ``cfg.spinup_passes``
    times, with ``agg`` (cell flow -> rows) giving the rows of the convergence
    diagnostic; ``window`` from the frozen cold start (``cfg.init_mode``) over the
    legacy preceding window.  Returns the state and a one-line description.
    ``temp_delta`` / ``precip_scale`` perturb the spin-up forcing as in
    :func:`stream_rows` (defaults: none), so a perturbed pass spins up under its
    own climate.  ``parts`` streams the spin-up with the runoff parts on, so the
    returned state carries their routing history (``state.hist_parts``) for a
    following ``stream_rows(parts=True)``; the SAC/snow/canopy states and the
    diagnostic are unchanged.  Either perturbation may be a per-row tensor
    (:func:`stream_rows`); ``window`` is :func:`stream_rows`' chunk length (memory only)."""
    pert = {}
    if _perturbed(temp_delta, precip_scale):
        pert = dict(temp_delta=temp_delta, precip_scale=precip_scale)
    if window != 512:
        pert["window"] = window
    if parts:
        pert["parts"] = True
        if agg is not None:        # the diagnostic rows see the flow only
            agg = (lambda f0: (lambda f, *_: f0(f)))(agg)
    if mode == "window":
        s = window_start(dom.dates, t0, cfg.spinup_start)
        st = initial_state(dom.n_phys, dom.device, dom.dtype, init_mode=cfg.init_mode,
                           params=params, et_mode=cfg.et_mode, n_rows=dom.n_hru)
        _, st = stream_rows(dom, cfg, params, uh, canopy, s, t0, st, None, **pert)
        return st, f"window spinup from {dom.dates[s].date()}"
    if mode != "cycle":
        raise ValueError(f"spinup mode {mode!r} (one of {SPINUP_MODES})")
    te = block_end(dom.dates, t0, cfg.spinup_years)
    yidx = year_index(dom.dates, t0, te)

    def rows(a: int, b: int, s: PipelineState):
        r, s = stream_rows(dom, cfg, params, uh, canopy, a, b, s, agg, **pert)
        return (annual_totals(r, yidx) if r is not None else None), s

    st, _, change = cycle_spinup(rows, t0, te, cold_state(dom, cfg, params), cfg.spinup_passes)
    return st, describe(dom.dates, t0, te, cfg.spinup_passes, change)
