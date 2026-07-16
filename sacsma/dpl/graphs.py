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
changes performance only.
"""

from __future__ import annotations

import torch

from .config import DplConfig
from .data import ET_MAXM, DomainTensors
from .et_noah import NoahCanopyState
from .forward import PipelineState, routing_uh, run_window
from .loss import level_hinge_loss, masked_basin_loss, shape_pull_loss
from .routing import N_TAPS
from .sma import SacState
from .snow17 import Snow17State

_STATE_FIELDS = (
    ("snow", "w_i"), ("snow", "ati"), ("snow", "w_q"), ("snow", "deficit"),
    ("sac", "uztwc"), ("sac", "uzfwc"), ("sac", "lztwc"),
    ("sac", "lzfsc"), ("sac", "lzfpc"), ("sac", "adimc"),
    (None, "hist_surf"), (None, "hist_base"),
)


def _state_tensors(st: PipelineState) -> list[torch.Tensor]:
    ts = [getattr(st if sub is None else getattr(st, sub), name)
          for sub, name in _STATE_FIELDS]
    if st.canopy is not None:              # Noah ET: carry canopy storage wc too
        ts.append(st.canopy.wc)
    return ts


def _clone_state(st: PipelineState) -> PipelineState:
    t = [x.clone() for x in _state_tensors(st)]
    canopy = NoahCanopyState(wc=t[12]) if st.canopy is not None else None
    return PipelineState(snow=Snow17State(*t[0:4]), sac=SacState(*t[4:10]),
                         hist_surf=t[10], hist_base=t[11], canopy=canopy)


class _WindowBase:
    """Static forcing/state buffers + ping-pong shared by both graphs."""

    def __init__(self, dom: DomainTensors, length: int, et_mode: str = "sac",
                 need_tmm: bool = False):
        n, dev, dt = dom.n_hru, dom.device, dom.dtype
        self.dom = dom
        self.length = length
        self.et_mode = et_mode
        # per-cell tmin/tmax buffers are needed by the Noah ET AND by the plain
        # SAC ET when it runs on Priestley-Taylor PET (else the graph would fall
        # back to a synthetic diurnal range while the eager path uses the real
        # per-cell values -> graph != eager).
        self.need_tmm = need_tmm or et_mode == "noah"
        self.pr = torch.zeros(n, length, device=dev, dtype=dt)
        self.ta = torch.zeros(n, length, device=dev, dtype=dt)
        self.doy = torch.zeros(length, device=dev, dtype=dom.doy.dtype)
        self.leap = torch.zeros(length, device=dev, dtype=torch.bool)
        # Noah ET: canopy state so wc rides the ping-pong (else it silently
        # resets each window); LAI rides a per-chunk buffer (canopy structure).
        noah = et_mode == "noah"
        self.tmin = torch.zeros(n, length, device=dev, dtype=dt) if self.need_tmm else None
        self.tmax = torch.zeros(n, length, device=dev, dtype=dt) if self.need_tmm else None
        # observed seasonal LAI rides a per-chunk buffer like tmin/tmax; the
        # static veg_frac is read straight off ``dom`` (a per-cell constant).
        self.lai = torch.zeros(n, length, device=dev, dtype=dt) if noah else None
        # climate-state index for dynamic params — another per-chunk buffer,
        # present only when the domain carries a state field.
        self.state_idx = (torch.zeros(n, length, device=dev, dtype=dt)
                          if dom.state is not None else None)
        self.state_in = PipelineState(
            snow=Snow17State.zeros(n, dev, dt),
            sac=SacState.reference_init(n, dev, dt),
            hist_surf=torch.zeros(n, N_TAPS - 1, device=dev, dtype=dt),
            hist_base=torch.zeros(n, N_TAPS - 1, device=dev, dtype=dt),
            canopy=NoahCanopyState.zeros(n, dev, dt) if noah else None,
        )
        self.state_out: PipelineState | None = None   # captured outputs

    def set_state(self, st: PipelineState) -> None:
        for buf, src in zip(_state_tensors(self.state_in), _state_tensors(st),
                            strict=True):
            buf.copy_(src)

    def get_state(self) -> PipelineState:
        return _clone_state(self.state_in)

    def _copy_forcing(self, pr, ta, doy, leap, tmin=None, tmax=None,
                      lai=None, state_idx=None) -> None:
        self.pr.copy_(pr)
        self.ta.copy_(ta)
        self.doy.copy_(doy)
        self.leap.copy_(leap)
        if self.need_tmm:
            self.tmin.copy_(tmin)
            self.tmax.copy_(tmax)
        if self.et_mode == "noah":
            self.lai.copy_(lai)
        if self.state_idx is not None:
            self.state_idx.copy_(state_idx)

    def _pingpong(self) -> None:
        assert self.state_out is not None
        for i_buf, o_buf in zip(_state_tensors(self.state_in),
                                _state_tensors(self.state_out), strict=True):
            i_buf.copy_(o_buf)


class NoGradWindow(_WindowBase):
    """Forward-only window graph: replay to stream long no-grad records."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor],
                 canopy_params: dict[str, torch.Tensor] | None = None):
        super().__init__(dom, length, et_mode=cfg.et_mode,
                         need_tmm=cfg.sac_pet == "priestley_taylor")
        # copy ALL emitted keys (base PARAM_ORDER + any seasonal/dynamic coeffs)
        # so the no-grad spinup/selection graph applies the same time-varying
        # field as training — not just the static base.
        self.params = {p: params[p].detach().clone() for p in params}
        self.uh = (uh[0].detach().clone(), uh[1].detach().clone())
        self._cfg = cfg
        # Noah ET: static LEARNED canopy params (+ any _dyn coeffs), refreshed per
        # epoch via set_params (veg_frac/lai are observed, read from dom/buffer).
        self.canopy_params = (
            {p: canopy_params[p].detach().clone() for p in canopy_params}
            if cfg.et_mode == "noah" and canopy_params is not None else None)

        def _fwd():
            flow, st = run_window(
                self.pr, self.ta, self.doy, self.leap, dom.lat_rad, dom.elev,
                self.params, self.uh, self.state_in,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                et_mode=cfg.et_mode, canopy_params=self.canopy_params,
                tmin=self.tmin, tmax=self.tmax,
                veg_frac=dom.veg_frac, lai=self.lai, noah_pet=cfg.noah_pet,
                sac_pet=cfg.sac_pet, pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                canopy_lite=cfg.canopy_lite, state_idx=self.state_idx)
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
                   uh: tuple[torch.Tensor, torch.Tensor],
                   canopy_params: dict[str, torch.Tensor] | None = None) -> None:
        for p in self.params:
            self.params[p].copy_(params[p].detach())
        self.uh[0].copy_(uh[0].detach())
        self.uh[1].copy_(uh[1].detach())
        if self.canopy_params is not None and canopy_params is not None:
            for p in self.canopy_params:
                self.canopy_params[p].copy_(canopy_params[p].detach())

    def replay(self, pr, ta, doy, leap, tmin=None, tmax=None,
               lai=None, state_idx=None) -> torch.Tensor:
        """One window; carries state; returns the static (B, length) basin flow
        (clone it before the next replay if collecting)."""
        self._copy_forcing(pr, ta, doy, leap, tmin, tmax, lai, state_idx)
        self.graph.replay()
        self._pingpong()
        return self.basin


class TrainChunk(_WindowBase):
    """Whole-iteration capture: forward + loss + backward for one TBPTT chunk."""

    def __init__(self, net: torch.nn.Module, dom: DomainTensors,
                 cfg: DplConfig, length: int, x: torch.Tensor,
                 obs_var: torch.Tensor, weight: torch.Tensor | None = None,
                 swe_basin_w: torch.Tensor | None = None):
        super().__init__(dom, length, et_mode=cfg.et_mode,
                         need_tmm=cfg.sac_pet == "priestley_taylor")
        b = dom.W.shape[0]
        self.obs = torch.full((b, length), float("nan"),
                              device=dom.device, dtype=dom.dtype)
        self.x = x                      # static net input (constant)
        self.obs_var = obs_var
        #: adaptive per-basin loss weights (static buffer, None = uniform); the
        #: SAME tensor is passed here and updated in place via set_weights, so
        #: the captured loss reads the current weights on every replay.
        self.weight = weight
        #: ET/SWE auxiliary losses (opt-in): static per-chunk target buffers,
        #: copied in on every replay via set_et_target / set_swe_target.  Init so
        #: capture sees no NaN and every obs term is 0 at capture (sig=1, mask=0,
        #: lo=0/hi=1) while its tet/swe backward path IS captured.
        self.et = cfg.et_loss_lambda > 0.0 or cfg.et_level_lambda > 0.0
        self.et_shape_lambda = cfg.et_loss_lambda
        self.et_level_lambda = cfg.et_level_lambda
        self.swe = cfg.swe_loss_lambda > 0.0
        self.swe_lambda = cfg.swe_loss_lambda
        dev, dt = dom.device, dom.dtype
        if self.et:
            self.et_bucket = torch.zeros(length, ET_MAXM, device=dev, dtype=dt)
            self.et_mu = torch.zeros(b, ET_MAXM, device=dev, dtype=dt)
            self.et_sig = torch.ones(b, ET_MAXM, device=dev, dtype=dt)
            self.et_mask = torch.zeros(ET_MAXM, device=dev, dtype=dt)
            self.et_lo = torch.zeros(b, device=dev, dtype=dt)
            self.et_hi = torch.ones(b, device=dev, dtype=dt)
        if self.swe:
            if swe_basin_w is None:
                raise ValueError("swe_loss_lambda > 0 requires swe_basin_w")
            self.swe_bucket = torch.zeros(length, ET_MAXM, device=dev, dtype=dt)
            self.swe_mu = torch.zeros(b, ET_MAXM, device=dev, dtype=dt)
            self.swe_sig = torch.ones(b, ET_MAXM, device=dev, dtype=dt)
            self.swe_mask = torch.zeros(ET_MAXM, device=dev, dtype=dt)
            self.swe_w = swe_basin_w.detach().clone()   # (B,) static snow mask

        def _step():
            out = net(self.x)
            if cfg.et_mode == "noah":               # split off the canopy subdict
                cp = out.get("_canopy")             # (params.values() must be tensors)
                params = {k: v for k, v in out.items() if k != "_canopy"}
            else:
                cp, params = None, out
            uh = routing_uh(params, dom.flowlen)
            res = run_window(
                self.pr, self.ta, self.doy, self.leap, dom.lat_rad, dom.elev,
                params, uh, self.state_in,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode="fixed",
                et_mode=cfg.et_mode, canopy_params=cp,
                tmin=self.tmin, tmax=self.tmax,
                veg_frac=dom.veg_frac, lai=self.lai, noah_pet=cfg.noah_pet,
                sac_pet=cfg.sac_pet, pt_snow_albedo=cfg.pt_snow_albedo,
                pt_dewpoint_depression=cfg.pt_dewpoint_depression,
                canopy_lite=cfg.canopy_lite, state_idx=self.state_idx,
                return_tet=self.et, return_swe=self.swe)
            flow, st = (res[0], res[1])
            basin = dom.W @ flow
            loss = masked_basin_loss(basin, self.obs, self.obs_var,
                                     kind=cfg.loss,
                                     log_lambda=cfg.log_loss_lambda,
                                     log_eps=cfg.log_loss_eps,
                                     var_lambda=cfg.var_loss_lambda,
                                     bias_lambda=cfg.bias_loss_lambda,
                                     weight=self.weight)
            k = 2
            if self.et:
                et_monthly = (dom.W @ res[k]) @ self.et_bucket   # (B, MAXM) mm/mo
                k += 1
                if self.et_shape_lambda > 0.0:
                    loss = loss + self.et_shape_lambda * shape_pull_loss(
                        et_monthly, self.et_mu, self.et_sig, self.et_mask)
                if self.et_level_lambda > 0.0:
                    loss = loss + self.et_level_lambda * level_hinge_loss(
                        et_monthly, self.et_mask, self.et_lo, self.et_hi)
            if self.swe:
                # swe_bucket columns are pre-divided by day counts -> monthly MEAN
                swe_monthly = (dom.W @ res[k]) @ self.swe_bucket
                loss = loss + self.swe_lambda * shape_pull_loss(
                    swe_monthly, self.swe_mu, self.swe_sig, self.swe_mask,
                    basin_w=self.swe_w)
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
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                loss, _ = _step()
                loss.backward()
                for p in net.parameters():
                    p.grad = None
        torch.cuda.current_stream().wait_stream(side)

        # grads must be ALLOCATED inside the capture so .grad tensors are static
        for p in net.parameters():
            p.grad = None
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

    def set_et_target(self, bucket, mu, sig, mask, lo, hi) -> None:
        """Copy this chunk's ET target (day->month bucket, normalized-shape
        mu/sig, slot mask, level envelope lo/hi) into the static buffers."""
        if self.et:
            self.et_bucket.copy_(bucket)
            self.et_mu.copy_(mu)
            self.et_sig.copy_(sig)
            self.et_mask.copy_(mask)
            self.et_lo.copy_(lo)
            self.et_hi.copy_(hi)

    def set_swe_target(self, bucket_mean, mu, sig, mask) -> None:
        """Copy this chunk's SWE target (day->month MEAN bucket + normalized-
        shape mu/sig/mask) into the static buffers."""
        if self.swe:
            self.swe_bucket.copy_(bucket_mean)
            self.swe_mu.copy_(mu)
            self.swe_sig.copy_(sig)
            self.swe_mask.copy_(mask)

    def run(self, pr, ta, doy, leap, obs, tmin=None, tmax=None,
            lai=None, state_idx=None, et_target=None, swe_target=None) -> float:
        """One chunk: replay forward+backward, carry state, return the loss.
        Gradients are left in the net's static ``.grad`` tensors.  ``et_target``
        (bucket, mu, sig, mask, lo, hi) and ``swe_target`` (bucket_mean, mu, sig,
        mask) are copied in when the respective obs losses are active."""
        self._copy_forcing(pr, ta, doy, leap, tmin, tmax, lai, state_idx)
        self.obs.copy_(obs)
        if et_target is not None:
            self.set_et_target(*et_target)
        if swe_target is not None:
            self.set_swe_target(*swe_target)
        self.graph.replay()
        self._pingpong()
        return float(self.loss.detach())
