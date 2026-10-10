"""CUDA-graph capture of the day-stepped pipeline.

Eager execution of the physics is dispatch-bound: one Snow-17 + SAC-SMA day
step issues ~300 tiny elementwise kernels, so a full record is ~10M launches
with the GPU mostly idle between them.  Capturing a fixed-length window once
and replaying it turns each window into a single graph launch.

Two capture shapes (PyTorch stream-capture recipes):

* :class:`NoGradWindow` — forward-only, fixed ``window`` days.  Replayed to
  stream the long no-grad segments: the spinup each epoch (from
  ``cfg.spinup_start``) and the full-calibration selection forward.
  Parameters/UHs are static buffers, refreshed per epoch with ``set_params``.
  It threads the carried state through static ping-pong buffers
  (``in.copy_(out)`` after each replay); remainder days (segment length not a
  multiple of the window) run eagerly through the same
  :func:`sacsma.dpl.forward.run_window` — identical numerics, so graph on/off
  changes performance only.
* :class:`RecomputeTrainWindow` — the training window's physics, forward and
  backward: one segment graph (:class:`SegmentGraph`) replayed over a window of
  any length, the backward re-running each segment from its stored state; the
  net, the routing UH and the loss stay eager.
"""

from __future__ import annotations

import torch

from .config import DplConfig
from .data import DomainTensors, Window
from .forward import PipelineState, initial_state, run_window
from .physics.sma import SacState
from .physics.snow17 import Snow17State

_STATE_FIELDS = (
    ("snow", "w_i"), ("snow", "ati"), ("snow", "w_q"), ("snow", "deficit"),
    ("sac", "uztwc"), ("sac", "uzfwc"), ("sac", "lztwc"),
    ("sac", "lzfsc"), ("sac", "lzfpc"), ("sac", "adimc"),
    (None, "hist_surf"), (None, "hist_base"),
)
_N_STATE = len(_STATE_FIELDS)


def _state_tensors(st: PipelineState) -> list[torch.Tensor]:
    return [getattr(st if sub is None else getattr(st, sub), name)
            for sub, name in _STATE_FIELDS]


def _unflatten_state(ts) -> PipelineState:
    """Inverse of :func:`_state_tensors`."""
    return PipelineState(snow=Snow17State(*ts[0:4]), sac=SacState(*ts[4:10]),
                         hist_surf=ts[10], hist_base=ts[11])


def _clone_state(st: PipelineState) -> PipelineState:
    return _unflatten_state([x.clone() for x in _state_tensors(st)])


class _WindowBase:
    """Static forcing/state buffers + ping-pong of :class:`NoGradWindow`."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int):
        self.dom = dom
        self.length = length
        self.physics = cfg.physics()
        self.w = dom.window_buffers(length, self.physics)
        self.state_in = initial_state(dom.n_hru, dom.device, dom.dtype)
        self.state_out: PipelineState | None = None   # captured outputs

    def set_state(self, st: PipelineState) -> None:
        for buf, src in zip(_state_tensors(self.state_in), _state_tensors(st),
                            strict=True):
            buf.copy_(src)

    def get_state(self) -> PipelineState:
        return _clone_state(self.state_in)

    def _pingpong(self) -> None:
        assert self.state_out is not None
        # value copy only: no autograd history reaches the static state_in buffers
        with torch.no_grad():
            for i_buf, o_buf in zip(_state_tensors(self.state_in),
                                    _state_tensors(self.state_out), strict=True):
                i_buf.copy_(o_buf)


class NoGradWindow(_WindowBase):
    """Forward-only window graph: replay to stream long no-grad records."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dom, cfg, length)
        # every key the net emits (soil_chi and PXTEMP_tau included), refreshed per epoch
        self.params = {p: params[p].detach().clone() for p in params}
        self.uh = (uh[0].detach().clone(), uh[1].detach().clone())

        def _fwd():
            flow, st = run_window(
                self.w, dom.lat_rad, dom.elev, self.params, self.uh, self.state_in,
                self.physics, veg_frac=dom.veg_frac)
            return dom.W @ flow, st

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(2):
                _fwd()
        torch.cuda.current_stream().wait_stream(side)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.basin, self.state_out = _fwd()

    def set_params(self, params: dict[str, torch.Tensor],
                   uh: tuple[torch.Tensor, torch.Tensor]) -> None:
        for p in self.params:
            self.params[p].copy_(params[p].detach())
        self.uh[0].copy_(uh[0].detach())
        self.uh[1].copy_(uh[1].detach())

    def replay(self, w: Window) -> torch.Tensor:
        """One window; carries state; returns the static (B, length) basin flow
        (clone it before the next replay if collecting)."""
        self.w.copy_(w)
        self.graph.replay()
        self._pingpong()
        return self.basin


class SegmentGraph:
    """Autograd-composable graphed ``run_window`` over one ``length``-day segment, captured
    through ``torch.cuda.make_graphed_callables`` (a forward graph and a backward graph):
    the graph :class:`RecomputeTrainWindow` replays.

    Only the day-stepped pipeline is graphed.  ``net(x)``, ``routing_uh`` and the
    loss stay eager (they are tiny), so no static ``.grad`` buffers, no static
    target buffers and no graph-side RNG are involved.
    """

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor],
                 *, sample_c0: int | None = None):
        self.dom, self.cfg, self.length = dom, cfg, length
        self.physics = cfg.physics()
        self.param_keys = tuple(params.keys())

        # the static forcing buffer (never requires grad)
        self.buf = dom.window_buffers(length, self.physics)
        if sample_c0 is not None:          # realistic forcing for warmup/capture
            self._copy_forcing(sample_c0)

        st0 = initial_state(dom.n_hru, dom.device, dom.dtype)
        st_args = tuple(t.requires_grad_(True) for t in _state_tensors(st0))
        p_args = tuple(params[k].detach().clone().requires_grad_(True)
                       for k in self.param_keys)
        uh_args = (uh[0].detach().clone().requires_grad_(True),
                   uh[1].detach().clone().requires_grad_(True))
        self.fn = torch.cuda.make_graphed_callables(
            self._make_fn(self.buf), st_args + p_args + uh_args, allow_unused_input=True)

    def _copy_forcing(self, c0: int) -> None:
        self.buf.copy_(self.dom.window(c0, c0 + self.length))

    def _make_fn(self, buf: Window):
        dom, pk = self.dom, self.param_keys

        def fn(*args):
            st = _unflatten_state(args[:_N_STATE])
            params = dict(zip(pk, args[_N_STATE:_N_STATE + len(pk)], strict=True))
            uh = (args[_N_STATE + len(pk)], args[_N_STATE + len(pk) + 1])
            flow, st_out = run_window(buf, dom.lat_rad, dom.elev, params, uh, st,
                                      self.physics, veg_frac=dom.veg_frac)[:2]
            # graphed callables must not return their own inputs (static
            # buffers would alias); a pass-through state field gets a copy
            in_ptrs = {a.data_ptr() for a in args}
            outs = [flow] + list(_state_tensors(st_out))
            return tuple(o.clone() if o.data_ptr() in in_ptrs else o for o in outs)
        return fn


class _RecomputeFn(torch.autograd.Function):
    """``run_window`` over ``[c0, c0 + length)`` with activation recompute (see
    :class:`RecomputeTrainWindow`).  Inputs after ``(win, c0, length)``: the carried
    state tensors, then the parameter and routing-UH tensors.  Outputs: the flow (N, length)
    and the end-state tensors."""

    @staticmethod
    def forward(ctx, win, c0, length, *args):
        ns, S = _N_STATE, win.seg_days
        nseg, tail = divmod(length, S)
        rest = tuple(t.detach() for t in args[ns:])
        cur = tuple(t.detach() for t in args[:ns])
        bounds, flows = [], []
        with torch.no_grad():
            for j in range(nseg):
                # every graph output is a static buffer the next replay overwrites:
                # keep owned copies of the boundary states and the flow
                bounds.append(tuple(t.clone() for t in cur))
                win.seg._copy_forcing(c0 + j * S)
                out = win.seg.fn(*(cur + rest))
                flows.append(out[0].clone())
                cur = tuple(o.clone() for o in out[1:])
            if tail:
                bounds.append(tuple(t.clone() for t in cur))
                f_t, cur = win._eager(c0 + nseg * S, c0 + length, cur, rest)
                flows.append(f_t)
        ctx.win, ctx.c0, ctx.nseg, ctx.tail, ctx.bounds = win, c0, nseg, tail, bounds
        ctx.save_for_backward(*args[ns:])
        return (torch.cat(flows, dim=1),) + tuple(cur)

    @staticmethod
    def backward(ctx, g_flow, *g_end):
        win, S, ns = ctx.win, ctx.win.seg_days, _N_STATE
        rest = ctx.saved_tensors
        g_state = [g if g is not None else torch.zeros_like(b)
                   for g, b in zip(g_end, ctx.bounds[-1], strict=True)]
        acc: list = [None] * len(rest)
        # last piece first: the eager tail, then the whole segments in reverse
        pieces = ([(ctx.nseg * S, ctx.tail, ctx.nseg)] if ctx.tail else []) + \
            [(j * S, S, j) for j in reversed(range(ctx.nseg))]
        for d0, L, j in pieces:
            with torch.enable_grad():
                st_in = tuple(t.detach().requires_grad_(True) for t in ctx.bounds[j])
                r_in = tuple(t.detach().requires_grad_(t.requires_grad) for t in rest)
                if j == ctx.nseg:                       # the eager tail
                    f, st_out = win._eager(ctx.c0 + d0, ctx.c0 + d0 + L, st_in, r_in)
                else:
                    win.seg._copy_forcing(ctx.c0 + d0)
                    out = win.seg.fn(*(st_in + r_in))
                    f, st_out = out[0], tuple(out[1:])
                outs = (f,) + tuple(st_out)
                gouts = (g_flow[:, d0:d0 + L],) + tuple(g_state)
                keep = [i for i, o in enumerate(outs) if o.requires_grad]
                wrt = st_in + tuple(r for r in r_in if r.requires_grad)
                grads = torch.autograd.grad([outs[i] for i in keep], wrt,
                                            [gouts[i] for i in keep], allow_unused=True)
            # the graphed backward returns views of its static grad buffers, which
            # the next replay overwrites: take owned copies now
            g_state = [g.clone() if g is not None else torch.zeros_like(s)
                       for g, s in zip(grads[:ns], st_in, strict=True)]
            gi = iter(grads[ns:])
            for i, r in enumerate(r_in):
                if not r.requires_grad:
                    continue
                g = next(gi)
                if g is not None:
                    acc[i] = g.clone() if acc[i] is None else acc[i] + g
        return (None, None, None, *g_state, *acc)


class RecomputeTrainWindow:
    """Autograd-composable graphed ``run_window`` over a window of ANY length with
    activation recompute: ONE captured ``seg_days``-day graph (fwd + bwd) replayed
    over consecutive segments.

    The forward runs every segment without autograd and keeps only the
    segment-boundary states; the backward re-runs each segment with autograd from
    its stored start state, last segment first, chaining the state gradient back
    through the window (days past the last whole segment run eagerly, recomputed the
    same way).  The graph pool therefore holds one segment's activations whatever
    the window length — the two-water-year TBPTT window costs the memory of
    ``seg_days`` days — for one extra forward per segment.  Same gradient as the
    eager window up to float32 summation order.  Under ``torch.no_grad()`` it is a
    plain graphed forward (dead chunks)."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, seg_days: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor],
                 *, sample_c0: int | None = None):
        self.seg = SegmentGraph(dom, cfg, seg_days, params, uh, sample_c0=sample_c0)
        self.dom, self.seg_days, self.physics = dom, seg_days, self.seg.physics
        self.param_keys = self.seg.param_keys

    def _eager(self, a: int, b: int, st: tuple, rest: tuple):
        """Eager ``run_window`` over ``[a, b)`` from flat state/param tensors."""
        dom, npk = self.dom, len(self.param_keys)
        params = dict(zip(self.param_keys, rest[:npk], strict=True))
        flow, st_out = run_window(dom.window(a, b), dom.lat_rad, dom.elev, params,
                                  (rest[npk], rest[npk + 1]), _unflatten_state(st),
                                  self.physics, veg_frac=dom.veg_frac)[:2]
        return flow, tuple(_state_tensors(st_out))

    def forward(self, c0: int, length: int, params: dict[str, torch.Tensor],
                uh: tuple[torch.Tensor, torch.Tensor],
                state: PipelineState) -> tuple[torch.Tensor, PipelineState]:
        """(flow (N, length), end state) over ``[c0, c0 + length)``."""
        st = tuple(_state_tensors(state))
        rest = tuple(params[k] for k in self.param_keys) + (uh[0], uh[1])
        outs = _RecomputeFn.apply(self, c0, length, *st, *rest)
        return outs[0], _unflatten_state(outs[1:])
