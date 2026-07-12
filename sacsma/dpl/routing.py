"""Differentiable Lohmann routing — torch mirror of the frozen ``sacsma.routing``.

Reference parameterization and constants (KE=12, UH_DAY=96, DT=3600 s,
LE=2400): a gamma hillslope UH (Nres, Kres; 12 daily bins x 1001-point
rectangle quadrature) convolved with the Saint-Venant Green's-function channel
UH (Velo, Diff, per-HRU flowlen), normalized to sum 1, applied as a 107-tap
causal convolution.  ``flowlen == 0`` (outlet HRUs) collapses the channel UH
to the identity, exactly like ``model.default_is_outlet``.

Differentiability notes: the gamma density is evaluated as
``exp((n-1)*log(x) - x/theta - lgamma(n))/theta`` with the single ``x = 0``
grid point replaced by ``1e-12`` — forward-identical to the reference beyond
1e-12 relative (for n = 1 numpy defines ``0**0 = 1``, which the substitution
reproduces) while keeping d/dn finite.  The channel UH's ``pot <= 69`` cutoff
only zeroes a ~e^-69 tail.  Everything else is smooth in (Nres, Kres, Velo,
Diff).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

KE = 12
UH_DAY = 96
DT = 3600.0
TMAX = UH_DAY * 24        # 2304
LE = 48 * 50              # 2400
N_TAPS = KE + UH_DAY - 1  # 107


def river_uh(flowlen: torch.Tensor, velo: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
    """Daily channel UH (N, UH_DAY); rows with ``flowlen == 0`` are the identity."""
    dev, dtype = flowlen.device, flowlen.dtype
    t = DT * torch.arange(1, LE + 1, device=dev, dtype=dtype)          # (LE,)
    fl = flowlen.unsqueeze(-1)
    v = velo.unsqueeze(-1)
    d = diff.unsqueeze(-1)
    pot = (v * t - fl) ** 2 / (4.0 * d * t)
    h = torch.where(
        pot <= 69.0,
        fl / (2.0 * t * torch.sqrt(torch.pi * t * d)) * torch.exp(-pot.clamp_max(69.0)),
        torch.zeros_like(pot),
    )
    s = h.sum(dim=-1, keepdim=True)
    # slice-fill_, not `[0] = 1.0`: item assignment copies a CPU scalar tensor
    # (H2D memcpy), which CUDA stream capture forbids; fill_ is a device kernel
    delta_le = torch.zeros(LE, device=dev, dtype=dtype)
    delta_le[:1].fill_(1.0)
    # where() evaluates both branches — divide by a safe s so masked-out rows
    # (s == 0, unreachable for flowlen > 0) never produce inf/NaN gradients.
    safe_s = torch.where(s > 0.0, s, torch.ones_like(s))
    uhm = torch.where(s > 0.0, h / safe_s, delta_le)                    # (N, LE)

    # convolve with the 24-h box (1/24 each) == a 24-wide windowed cumsum;
    # then the reference's leading-zero offset, TMAX truncation, daily bin sum.
    c = torch.cumsum(uhm, dim=-1)                                       # (N, LE)
    c_now = c[:, : TMAX - 1]                                            # C[j],   j=0..2302
    c_lag = F.pad(c, (24, 0))[:, : TMAX - 1]                            # C[j-24] (0 for j<24)
    conv = (c_now - c_lag) / 24.0                                       # conv[0..2302]
    fr2 = F.pad(conv, (1, 0))                                           # (N, TMAX)
    uh = fr2.reshape(-1, UH_DAY, 24).sum(dim=-1)                        # (N, UH_DAY)

    # outlet rows: identity UH
    delta_day = torch.zeros(UH_DAY, device=dev, dtype=dtype)
    delta_day[:1].fill_(1.0)
    return torch.where((flowlen == 0.0).unsqueeze(-1), delta_day, uh)


def hillslope_uh(nres: torch.Tensor, kres: torch.Tensor) -> torch.Tensor:
    """Gamma hillslope UH (N, KE), reference 12-bin x 1001-point quadrature."""
    dev, dtype = nres.device, nres.dtype
    i = torch.arange(1, KE + 1, device=dev, dtype=dtype).unsqueeze(-1)   # (KE, 1)
    x = torch.linspace(0.0, 1.0, 1001, device=dev, dtype=dtype) * 24.0 + 24.0 * (i - 1.0)
    x = x.reshape(-1).clamp_min(1e-12)                                   # (KE*1001,) x=0 -> 1e-12
    dx = 24.0 / 1000.0
    theta = 1.0 / kres
    n1 = (nres - 1.0).unsqueeze(-1)
    log_x_over_theta = torch.log(x) - torch.log(theta).unsqueeze(-1)
    log_pdf = n1 * log_x_over_theta - (x / theta.unsqueeze(-1)) \
        - torch.lgamma(nres).unsqueeze(-1) - torch.log(theta).unsqueeze(-1)
    integrand = torch.exp(log_pdf).reshape(-1, KE, 1001)                 # (N, KE, 1001)
    return integrand.sum(dim=-1) * dx


def build_uh(
    nres: torch.Tensor, kres: torch.Tensor,
    velo: torch.Tensor, diff: torch.Tensor,
    flowlen: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined normalized UHs: (uh_direct (N, 107), uh_base (N, 107))."""
    riv = river_uh(flowlen, velo, diff)                                  # (N, 96)
    hill = hillslope_uh(nres, kres)                                      # (N, 12)
    n = riv.shape[0]
    # full convolution hill * riv -> 107 taps (conv1d correlates, so flip)
    riv_in = F.pad(riv.unsqueeze(0), (KE - 1, KE - 1))                   # (1, N, 96+22)
    w = hill.flip(-1).unsqueeze(1)                                       # (N, 1, 12)
    uh_direct = F.conv1d(riv_in, w, groups=n).squeeze(0)                 # (N, 107)
    uh_direct = uh_direct / uh_direct.sum(dim=-1, keepdim=True)
    # base hillslope UH is the delta -> just the (normalized) river UH, zero-padded
    uh_base = F.pad(riv / riv.sum(dim=-1, keepdim=True), (0, KE - 1))    # (N, 107)
    return uh_direct, uh_base


def route(
    inflow: torch.Tensor,   # (N, T)
    uh: torch.Tensor,       # (N, 107) normalized taps
    history: torch.Tensor | None = None,   # (N, 106) inflow preceding the window
) -> torch.Tensor:
    """Causal per-HRU convolution ``routed[t] = sum_l uh[l] * inflow[t-l]``.

    ``history`` supplies the 106 inflow days before the window (chunked
    training carries it, detached, across chunk boundaries); omitted ->
    zero history, which matches the reference full-record cold start.
    """
    n, t_len = inflow.shape
    if history is None:
        x = F.pad(inflow.unsqueeze(0), (N_TAPS - 1, 0))
    else:
        x = torch.cat([history, inflow], dim=-1).unsqueeze(0)
    w = uh.flip(-1).unsqueeze(1)                                         # (N, 1, 107)
    return F.conv1d(x, w, groups=n).squeeze(0)[:, -t_len:]


def lohmann_route(
    surf: torch.Tensor, base: torch.Tensor,
    params: dict[str, torch.Tensor],       # Nres, Kres, Velo, Diff — (N,) each
    flowlen: torch.Tensor,                  # (N,)
    history: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Routed total flow (N, T): direct (hill+channel UH) + base (channel UH)."""
    uh_direct, uh_base = build_uh(params["Nres"], params["Kres"],
                                  params["Velo"], params["Diff"], flowlen)
    hist_s, hist_b = history if history is not None else (None, None)
    return route(surf, uh_direct, hist_s) + route(base, uh_base, hist_b)
