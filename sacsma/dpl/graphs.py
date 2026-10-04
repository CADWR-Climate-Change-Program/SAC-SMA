"""CUDA-graph capture of the day-stepped pipeline.

Eager execution of the physics is dispatch-bound: one Snow-17 + SAC-SMA day
step issues ~300 tiny elementwise kernels, so a full record is ~10M launches
with the GPU mostly idle between them.  Capturing a fixed-length window once
and replaying it turns each window into a single graph launch.

Two capture shapes (both PyTorch stream-capture recipes):

* :class:`NoGradWindow` — forward-only, fixed ``window`` days.  Replayed to
  stream the long no-grad segments: the spinup each epoch (from
  ``cfg.spinup_start``) and the full-calibration selection forward.
  Parameters/UHs are static buffers, refreshed per epoch with ``set_params``.
* :class:`TrainChunk` — whole-iteration capture of
  ``net -> UH -> physics chunk -> basin aggregation -> loss -> backward`` for
  a fixed ``chunk`` length.  Gradients land in the network's static ``.grad``
  tensors; the trainer clips/steps/zeroes them eagerly per replay
  (``zero_grad(set_to_none=False)`` — deallocating grads would invalidate the
  graph).  Dropout RNG is graph-safe (fresh mask per replay).

Both thread carried state through static ping-pong buffers
(``in.copy_(out)`` after each replay), which also detaches chunk-to-chunk
state exactly as TBPTT requires.  Remainder days (segment length not a
multiple of the window) run eagerly through the same
:func:`sacsma.dpl.forward.run_window` — identical numerics, so graph on/off
changes performance only.  State moves between graph objects (the two chunk
lengths of a water-year grid) and eager chunks through ``get_state()`` /
``set_state()``: detached value copies, routing history included, so the
hand-off is length-agnostic.

Under cell dedup (``DplConfig.dedup_cells``, ``dom.dedup``) the forcing buffers,
parameters and snow/SAC states are sized to the distinct cells
(``dom.n_phys``) and the routing histories to the HRU rows (``dom.n_hru``); the
captured forward gathers each cell's runoff to its rows (``row_cell``) before the
per-row routing.  :class:`TrainChunk` then takes the per-cell net input
(``dom.phys_x(x)``).  These paths are exercised on CPU only through the eager
closures (the CUDA capture itself needs a GPU).
"""

from __future__ import annotations

import torch

from .config import DplConfig
from .data import CHUNK_MAXM, DomainTensors, Window
from .forward import PipelineState, initial_state, routing_uh, run_window
from .loss import jul_sep_window, masked_basin_loss
from .multi_timescale import monthly_nnse_loss
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
    """Static forcing/state buffers + ping-pong shared by both graphs."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int):
        # physics buffers per physics row (the distinct cells under cell dedup),
        # the routing histories per HRU row
        self.dom = dom
        self.length = length
        self.physics = cfg.physics()
        self.w = dom.window_buffers(length, self.physics)
        self.state_in = initial_state(dom.n_phys, dom.device, dom.dtype, n_rows=dom.n_hru)
        self.state_out: PipelineState | None = None   # captured outputs

    def set_state(self, st: PipelineState) -> None:
        for buf, src in zip(_state_tensors(self.state_in), _state_tensors(st),
                            strict=True):
            buf.copy_(src)

    def get_state(self) -> PipelineState:
        return _clone_state(self.state_in)

    def _pingpong(self) -> None:
        assert self.state_out is not None
        # value copy only: TrainChunk's state_out carries autograd history from
        # the capture pass; copying it WITH grad would attach that history to the
        # static state_in buffers, and an eager chunk or another graph object
        # started from get_state() (the multi-timescale tail, the other chunk
        # length of a water-year grid) would then backward into the freed
        # capture graph ("Trying to backward through the graph a second time").
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
                self.w, dom.phys_lat_rad, dom.phys_elev, self.params, self.uh, self.state_in,
                self.physics, veg_frac=dom.phys_veg_frac, row_cell=dom.row_cell)
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


class TrainChunk(_WindowBase):
    """Whole-iteration capture: forward + loss + backward for one TBPTT chunk."""

    def __init__(self, net: torch.nn.Module, dom: DomainTensors,
                 cfg: DplConfig, length: int, x: torch.Tensor,
                 obs_var: torch.Tensor, weight: torch.Tensor | None = None,
                 mt_rows: torch.Tensor | None = None,
                 mt_var: torch.Tensor | None = None,
                 daily_scale: float = 1.0, monthly_scale: float = 1.0,
                 shape_kw: dict | None = None,
                 weight_total: float | None = None, mt_total: float | None = None):
        super().__init__(dom, cfg, length)
        #: mt_share_norm="all": FIXED loss denominators (all daily entities'
        #: weight / the monthly entity count), None = per-chunk renormalization
        self.weight_total = weight_total
        self.mt_total = mt_total
        #: the peak / timing terms' settings and per-basin record constants
        #: (static tensors, loss.record_references), as the eager path passes them
        shape_kw = shape_kw or {}
        b = dom.W.shape[0]
        self.obs = torch.full((b, length), float("nan"),
                              device=dom.device, dtype=dom.dtype)
        self.x = x                      # static net input (constant; the per-cell
                                        # rows dom.phys_x(x) under cell dedup)
        self.obs_var = obs_var
        #: adaptive per-basin loss weights (static buffer, None = uniform); the
        #: SAME tensor is passed here and updated in place via set_weights, so
        #: the captured loss reads the current weights on every replay.
        self.weight = weight
        dev, dt = dom.device, dom.dtype
        #: monthly-flow term (multi-timescale domain): static per-chunk target
        #: buffers.  Init tgt=0/fin=0 -> the term is 0 at
        #: capture while its backward path IS captured (mask-zero pattern).
        self.mt = mt_rows is not None
        self.daily_scale = daily_scale
        self.monthly_scale = monthly_scale
        if self.mt:
            m = int(mt_rows.numel())
            self.mt_rows = mt_rows.detach().clone()
            self.mt_var = mt_var.detach().clone()
            self.mt_bucket = torch.zeros(length, CHUNK_MAXM, device=dev, dtype=dt)
            self.mt_tgt = torch.zeros(m, CHUNK_MAXM, device=dev, dtype=dt)
            self.mt_fin = torch.zeros(m, CHUNK_MAXM, device=dev, dtype=dt)

        def _step():
            params = net(self.x)
            uh = routing_uh(params, dom.flowlen, row_cell=dom.row_cell)
            flow, st = run_window(
                self.w, dom.phys_lat_rad, dom.phys_elev, params, uh, self.state_in,
                self.physics, veg_frac=dom.phys_veg_frac, row_cell=dom.row_cell)
            basin = dom.W @ flow
            loss = masked_basin_loss(basin, self.obs, self.obs_var,
                                     kind=cfg.loss,
                                     log_lambda=cfg.log_loss_lambda,
                                     log_eps=cfg.log_loss_eps,
                                     var_lambda=cfg.var_loss_lambda,
                                     var_gate_frac=cfg.var_gate_frac,
                                     var_huber_cap=cfg.var_huber_cap,
                                     bias_lambda=cfg.bias_loss_lambda,
                                     timing_lambda=cfg.timing_loss_lambda,
                                     peak_lambda=cfg.peak_loss_lambda, **shape_kw,
                                     timing_window=jul_sep_window(self.w.doy, self.w.leap),
                                     weight=self.weight, weight_total=self.weight_total)
            if self.mt:
                # multi-timescale monthly NNSE on the same simulated flow; the
                # scales only depart from 1.0 under family weighting
                sim_m = basin[self.mt_rows] @ self.mt_bucket
                loss = self.daily_scale * loss + self.monthly_scale * \
                    monthly_nnse_loss(sim_m, self.mt_tgt, self.mt_fin,
                                      self.mt_var, n_total=self.mt_total)
            return loss, st

        net.train()
        # side-stream warmup (the documented capture recipe) reuses the
        # parameters' AccumulateGrad nodes at capture time — the resulting
        # stream-mismatch warning is intentional here (equivalence verified
        # exact: graph loss/grads == eager to 0.0)
        try:
            torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        except AttributeError:
            pass
        # A first capture lets backward ALLOCATE the .grad tensors inside the
        # capture, so they are static.  A later capture (a second chunk length
        # under the water-year grid) must NOT drop them: that would re-point
        # p.grad at its own pool and the first graph's replays would write into
        # orphaned memory (silent zero gradients).  It keeps the existing
        # tensors and records an in-place accumulate into them instead — both
        # graphs then land in the same static .grad, which the trainer never
        # deallocates (zero_grad(set_to_none=False)).
        own = all(p.grad is None for p in net.parameters())

        def _drop_grads():
            if own:
                for p in net.parameters():
                    p.grad = None
            else:
                net.zero_grad(set_to_none=False)

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                loss, _ = _step()
                loss.backward()
                _drop_grads()
        torch.cuda.current_stream().wait_stream(side)

        _drop_grads()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.state_out = _step()
            self.loss.backward()
        # the capture pass itself ran backward: clear its values from the
        # (now static) .grad tensors — replays ACCUMULATE onto them
        net.zero_grad(set_to_none=False)

    def set_weights(self, w: torch.Tensor) -> None:
        """Refresh the adaptive per-basin loss weights in place (no-op if the
        graph was built without a weight buffer)."""
        if self.weight is not None:
            self.weight.copy_(w)

    def set_mt_target(self, bucket, tgt, fin) -> None:
        """Copy this chunk's monthly-flow target (day->month bucket, observed
        mm/month 0-filled, finite-slot mask) into the static buffers."""
        if self.mt:
            self.mt_bucket.copy_(bucket)
            self.mt_tgt.copy_(tgt)
            self.mt_fin.copy_(fin)

    def run(self, w: Window, obs, mt_target=None) -> float:
        """One chunk: replay forward+backward, carry state, return the loss.
        Gradients are left in the net's static ``.grad`` tensors.  ``mt_target``
        (bucket, tgt, fin) is copied in when the monthly-flow term is active."""
        self.w.copy_(w)
        self.obs.copy_(obs)
        if mt_target is not None:
            self.set_mt_target(*mt_target)
        self.graph.replay()
        self._pingpong()
        return float(self.loss.detach())


class SegmentedTrainWindow:
    """Autograd-composable graphed ``run_window`` for one TBPTT chunk, captured as
    ``segments`` consecutive sub-windows through ``torch.cuda.make_graphed_callables``
    (a forward graph and a backward graph per segment, one shared memory pool).

    Why it exists: some GPU/driver combinations (RTX A4500 under WDDM, driver
    596.86) fault when a captured graph is very large -- the whole-year fwd+bwd
    :class:`TrainChunk` exceeds that limit while half-year pieces do not.
    Splitting the *capture* does not split the *gradient*: the chunk loss is
    formed eagerly on the concatenated segment flows, and backward runs the
    segments' captured backward graphs in reverse, passing the state gradient
    from each segment into the previous one (chain rule through the carried
    state).  The optimisation problem is therefore the same 366-day TBPTT as the
    single-graph path; differences are float32 summation order only.

    Only the day-stepped pipeline is graphed.  ``net(x)``, ``routing_uh`` and the
    loss stay eager (they are tiny), so no static ``.grad`` buffers, no static
    target buffers and no graph-side RNG are involved.
    """

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int,
                 segments: int, params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor],
                 *, sample_c0: int | None = None):
        if segments < 1 or segments > length:
            raise ValueError(f"segments {segments} outside [1, {length}]")
        # physics buffers/states per physics row (distinct cells under cell dedup),
        # routing histories per HRU row
        self.dom, self.cfg, self.length, self.segments = dom, cfg, length, segments
        self.physics = cfg.physics()
        base, extra = divmod(length, segments)
        self.lens = [base + (1 if j < extra else 0) for j in range(segments)]
        self.offs = [sum(self.lens[:j]) for j in range(segments)]
        self.param_keys = tuple(params.keys())

        # per-segment static forcing buffers (never require grad)
        self.bufs = [dom.window_buffers(L, self.physics) for L in self.lens]
        if sample_c0 is not None:          # realistic forcing for warmup/capture
            for j in range(segments):
                self._copy_forcing(j, sample_c0)

        fns, sample_args = [], []
        for j in range(segments):
            fns.append(self._make_fn(self.bufs[j]))
            st0 = initial_state(dom.n_phys, dom.device, dom.dtype, n_rows=dom.n_hru)
            st_args = tuple(t.requires_grad_(True) for t in _state_tensors(st0))
            p_args = tuple(params[k].detach().clone().requires_grad_(True)
                           for k in self.param_keys)
            uh_args = (uh[0].detach().clone().requires_grad_(True),
                       uh[1].detach().clone().requires_grad_(True))
            sample_args.append(st_args + p_args + uh_args)
        self.fns = torch.cuda.make_graphed_callables(
            tuple(fns), tuple(sample_args), allow_unused_input=True)

    def _copy_forcing(self, j: int, c0: int) -> None:
        a = c0 + self.offs[j]
        self.bufs[j].copy_(self.dom.window(a, a + self.lens[j]))

    def _make_fn(self, buf: Window):
        dom, pk = self.dom, self.param_keys

        def fn(*args):
            st = _unflatten_state(args[:_N_STATE])
            params = dict(zip(pk, args[_N_STATE:_N_STATE + len(pk)], strict=True))
            uh = (args[_N_STATE + len(pk)], args[_N_STATE + len(pk) + 1])
            flow, st_out = run_window(buf, dom.phys_lat_rad, dom.phys_elev, params, uh, st,
                                      self.physics, veg_frac=dom.phys_veg_frac,
                                      row_cell=dom.row_cell)[:2]
            # graphed callables must not return their own inputs (static
            # buffers would alias); a pass-through state field gets a copy
            in_ptrs = {a.data_ptr() for a in args}
            outs = [flow] + list(_state_tensors(st_out))
            return tuple(o.clone() if o.data_ptr() in in_ptrs else o for o in outs)
        return fn

    def forward(self, c0: int, params: dict[str, torch.Tensor],
                uh: tuple[torch.Tensor, torch.Tensor],
                state: PipelineState) -> tuple[torch.Tensor, PipelineState]:
        """Graphed ``run_window`` over ``[c0, c0 + length)``; returns
        (flow (N, length), state) with autograd through every segment."""
        p_args = tuple(params[k] for k in self.param_keys)
        uh_args = (uh[0], uh[1])
        # segment 1 starts from a detached carried state (TBPTT boundary); the
        # graphs were captured with grad-requiring state inputs, so mark leaves.
        # A carried field that already requires grad (tbptt_carry="relative"/"flux":
        # c * cap/cap.detach(), a function of the parameters) passes as is, so
        # its gradient reaches the capacities.
        st = tuple(t if t.requires_grad else t.detach().requires_grad_(True)
                   for t in _state_tensors(state))
        flows = []
        for j in range(self.segments):
            self._copy_forcing(j, c0)
            out = self.fns[j](*(st + p_args + uh_args))
            flows.append(out[0])
            st = tuple(out[1:])
        return torch.cat(flows, dim=1), _unflatten_state(st)


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
                win.seg._copy_forcing(0, c0 + j * S)
                out = win.seg.fns[0](*(cur + rest))
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
                    win.seg._copy_forcing(0, ctx.c0 + d0)
                    out = win.seg.fns[0](*(st_in + r_in))
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
    the window length — a 2- or 3-water-year TBPTT window costs the memory of
    ``seg_days`` days — for one extra forward per segment.  Same gradient as
    :class:`SegmentedTrainWindow` up to float32 summation order.  Under
    ``torch.no_grad()`` it is a plain graphed forward (dead chunks)."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, seg_days: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor],
                 *, sample_c0: int | None = None):
        self.seg = SegmentedTrainWindow(dom, cfg, seg_days, 1, params, uh, sample_c0=sample_c0)
        self.dom, self.seg_days, self.physics = dom, seg_days, self.seg.physics
        self.param_keys = self.seg.param_keys

    def _eager(self, a: int, b: int, st: tuple, rest: tuple):
        """Eager ``run_window`` over ``[a, b)`` from flat state/param tensors."""
        dom, npk = self.dom, len(self.param_keys)
        params = dict(zip(self.param_keys, rest[:npk], strict=True))
        flow, st_out = run_window(dom.window(a, b), dom.phys_lat_rad, dom.phys_elev, params,
                                  (rest[npk], rest[npk + 1]), _unflatten_state(st),
                                  self.physics, veg_frac=dom.phys_veg_frac,
                                  row_cell=dom.row_cell)[:2]
        return flow, tuple(_state_tensors(st_out))

    def forward(self, c0: int, length: int, params: dict[str, torch.Tensor],
                uh: tuple[torch.Tensor, torch.Tensor],
                state: PipelineState) -> tuple[torch.Tensor, PipelineState]:
        """(flow (N, length), end state) over ``[c0, c0 + length)``."""
        st = tuple(_state_tensors(state))
        rest = tuple(params[k] for k in self.param_keys) + (uh[0], uh[1])
        outs = _RecomputeFn.apply(self, c0, length, *st, *rest)
        return outs[0], _unflatten_state(outs[1:])
