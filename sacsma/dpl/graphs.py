"""CUDA-graph capture of the day-stepped pipeline.

Eager execution of the physics is dispatch-bound: one Snow-17 + SAC-SMA day
step issues ~300 tiny elementwise kernels, so a full record is ~10M launches
with the GPU mostly idle between them.  Capturing a fixed-length window once
and replaying it turns each window into a single graph launch.

Two capture shapes (both PyTorch stream-capture recipes):

* :class:`NoGradWindow` — forward-only, fixed ``window`` days.  Replayed to
  stream the long no-grad segments: the full-prefix spinup each epoch and the
  full-calibration selection forward.  Parameters/UHs are static buffers,
  refreshed per epoch with ``set_params``.
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

from .config import PARAM_ORDER, DplConfig
from .data import DomainTensors
from .forward import PipelineState, routing_uh, run_window
from .loss import masked_basin_loss
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
    return [getattr(st if sub is None else getattr(st, sub), name)
            for sub, name in _STATE_FIELDS]


def _clone_state(st: PipelineState) -> PipelineState:
    t = [x.clone() for x in _state_tensors(st)]
    return PipelineState(snow=Snow17State(*t[0:4]), sac=SacState(*t[4:10]),
                         hist_surf=t[10], hist_base=t[11])


class _WindowBase:
    """Static forcing/state buffers + ping-pong shared by both graphs."""

    def __init__(self, dom: DomainTensors, length: int):
        n, dev, dt = dom.n_hru, dom.device, dom.dtype
        self.dom = dom
        self.length = length
        self.pr = torch.zeros(n, length, device=dev, dtype=dt)
        self.ta = torch.zeros(n, length, device=dev, dtype=dt)
        self.doy = torch.zeros(length, device=dev, dtype=dom.doy.dtype)
        self.leap = torch.zeros(length, device=dev, dtype=torch.bool)
        self.state_in = PipelineState(
            snow=Snow17State.zeros(n, dev, dt),
            sac=SacState.reference_init(n, dev, dt),
            hist_surf=torch.zeros(n, N_TAPS - 1, device=dev, dtype=dt),
            hist_base=torch.zeros(n, N_TAPS - 1, device=dev, dtype=dt),
        )
        self.state_out: PipelineState | None = None   # captured outputs

    def set_state(self, st: PipelineState) -> None:
        for buf, src in zip(_state_tensors(self.state_in), _state_tensors(st),
                            strict=True):
            buf.copy_(src)

    def get_state(self) -> PipelineState:
        return _clone_state(self.state_in)

    def _copy_forcing(self, pr, ta, doy, leap) -> None:
        self.pr.copy_(pr)
        self.ta.copy_(ta)
        self.doy.copy_(doy)
        self.leap.copy_(leap)

    def _pingpong(self) -> None:
        assert self.state_out is not None
        for i_buf, o_buf in zip(_state_tensors(self.state_in),
                                _state_tensors(self.state_out), strict=True):
            i_buf.copy_(o_buf)


class NoGradWindow(_WindowBase):
    """Forward-only window graph: replay to stream long no-grad records."""

    def __init__(self, dom: DomainTensors, cfg: DplConfig, length: int,
                 params: dict[str, torch.Tensor],
                 uh: tuple[torch.Tensor, torch.Tensor]):
        super().__init__(dom, length)
        self.params = {p: params[p].detach().clone() for p in PARAM_ORDER}
        self.uh = (uh[0].detach().clone(), uh[1].detach().clone())
        self._cfg = cfg

        def _fwd():
            flow, st = run_window(
                self.pr, self.ta, self.doy, self.leap, dom.lat_rad, dom.elev,
                self.params, self.uh, self.state_in,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode="fixed")
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
        for p in PARAM_ORDER:
            self.params[p].copy_(params[p].detach())
        self.uh[0].copy_(uh[0].detach())
        self.uh[1].copy_(uh[1].detach())

    def replay(self, pr, ta, doy, leap) -> torch.Tensor:
        """One window; carries state; returns the static (B, length) basin flow
        (clone it before the next replay if collecting)."""
        self._copy_forcing(pr, ta, doy, leap)
        self.graph.replay()
        self._pingpong()
        return self.basin


class TrainChunk(_WindowBase):
    """Whole-iteration capture: forward + loss + backward for one TBPTT chunk."""

    def __init__(self, net: torch.nn.Module, dom: DomainTensors,
                 cfg: DplConfig, length: int, x: torch.Tensor,
                 obs_var: torch.Tensor, weight: torch.Tensor | None = None):
        super().__init__(dom, length)
        b = dom.W.shape[0]
        self.obs = torch.full((b, length), float("nan"),
                              device=dom.device, dtype=dom.dtype)
        self.x = x                      # static net input (constant)
        self.obs_var = obs_var
        #: adaptive per-basin loss weights (static buffer, None = uniform); the
        #: SAME tensor is passed here and updated in place via set_weights, so
        #: the captured loss reads the current weights on every replay.
        self.weight = weight

        def _step():
            params = net(self.x)
            uh = routing_uh(params, dom.flowlen)
            flow, st = run_window(
                self.pr, self.ta, self.doy, self.leap, dom.lat_rad, dom.elev,
                params, uh, self.state_in,
                n_inc=cfg.n_inc, perc_mode=cfg.perc_mode,
                fracp_floor=cfg.fracp_floor, ninc_mode="fixed")
            basin = dom.W @ flow
            loss = masked_basin_loss(basin, self.obs, self.obs_var,
                                     kind=cfg.loss,
                                     log_lambda=cfg.log_loss_lambda,
                                     log_eps=cfg.log_loss_eps,
                                     var_lambda=cfg.var_loss_lambda,
                                     weight=self.weight)
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

    def run(self, pr, ta, doy, leap, obs) -> float:
        """One chunk: replay forward+backward, carry state, return the loss.
        Gradients are left in the net's static ``.grad`` tensors."""
        self._copy_forcing(pr, ta, doy, leap)
        self.obs.copy_(obs)
        self.graph.replay()
        self._pingpong()
        return float(self.loss.detach())
