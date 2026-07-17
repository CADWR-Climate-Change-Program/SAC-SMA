# Appendix A — Model equations

The equations below are transcribed from the production Python implementation, which
reproduces the original MATLAB codes exactly (Part I §2.4). Notation: $P$ precipitation
(mm/day), $T$ daily mean air temperature (°C), $d$ day of year, $\varphi$ latitude
(radians), $z$ HRU elevation (m). Deliberately preserved idiosyncrasies of the original
implementation are marked ★.

## A.1 Hamon potential evapotranspiration

Daylength follows the CBM model of Forsythe et al. (1995):

$$\theta = 0.2163108 + 2\arctan\!\left[0.9671396\,\tan\!\big(0.0086\,(d - 186)\big)\right]$$
$$\delta = \arcsin(0.39795 \cos\theta)$$
$$D = 24 - \frac{24}{\pi}\arccos\!\left[\frac{\sin(0.8333^\circ) + \sin\varphi \sin\delta}{\cos\varphi \cos\delta}\right] \quad \text{(hours; argument clamped to } [-1,1])$$

Saturation vapor pressure and PET (Hamon, 1961), scaled by the calibrated coefficient
$K_{pet}$:

$$e_{sat}(T) = 0.611 \exp\!\left(\frac{17.27\,T}{237.3 + T}\right) \;\text{kPa}, \qquad
PET = K_{pet} \cdot 29.8\, D\, \frac{e_{sat}(T)}{T + 273.2} \;\text{mm/day}$$

## A.2 Snow-17

The implementation is the `_fix` variant of Snow-17 (Anderson, 1973): a **hard
rain/snow split at `PXTEMP`** ★ (the transition-interval mixture of the standard model
is intentionally disabled). States: ice water equivalent $W_i$, liquid water held
$W_q$, antecedent temperature index $ATI$, heat deficit $D_f$. Time steps
$\Delta t_t = \Delta t_p = 24$ h.

**Form and accumulation.**

$$\text{if } T_a \le \mathrm{PXTEMP}: \; \mathrm{SNOW} = P,\; \mathrm{RAIN} = 0;
\quad \text{else } \mathrm{SNOW} = 0,\; \mathrm{RAIN} = P$$
$$P_n = \mathrm{SNOW} \cdot \mathrm{SCF}, \qquad W_i \mathrel{+}= P_n$$

**Seasonal melt factor.** With $N$ = days since March 21 and $y$ = days in year:

$$S_v = 0.5\sin\!\left(\frac{2\pi N}{y}\right) + 0.5, \qquad
M_f = \frac{\Delta t_t}{6}\left[S_v(\mathrm{MFMAX} - \mathrm{MFMIN}) + \mathrm{MFMIN}\right]$$

**New-snow heat content and antecedent temperature index.** With
$T_{snow} = \min(T_a, 0)$:

$$\Delta D_{f,snow} = -\frac{T_{snow} P_n}{160}, \qquad
\Delta D_{f,T} = \mathrm{NMF}\,\frac{\Delta t_p}{6}\,\frac{M_f}{\mathrm{MFMAX}}\,(ATI - T_{snow})$$
$$ATI \leftarrow
\begin{cases}
T_{snow} & P_n > 1.5\,\Delta t_p \\
ATI + \left[1 - (1-\mathrm{TIPM})^{\Delta t_t / 6}\right](T_a - ATI) & \text{otherwise}
\end{cases}
\qquad ATI \leftarrow \min(ATI, 0)$$

**Melt.** During rain-on-snow ($\mathrm{RAIN} > 0.25\,\Delta t_p$), an energy-budget
melt with net longwave, rain advection, and a wind-scaled turbulent term
($T_r = \max(T_a, 0)$, $e_{sat}$ in hPa, station pressure $P_{atm}$ from elevation):

$$M = 6.12\times10^{-10}\,\Delta t_p\left[(T_a + 273)^4 - 273^4\right]
+ 0.0125\,\mathrm{RAIN}\,T_r
+ 8.5\,\mathrm{UADJ}\,\frac{\Delta t_p}{6}\left[(0.9\,e_{sat} - 6.11) + 0.00057\,P_{atm} T_a\right]$$

During non-rain periods with $T_a > \mathrm{MBASE}$:

$$M = M_f\,(T_a - \mathrm{MBASE})\,\frac{\Delta t_p}{\Delta t_t} + 0.0125\,\mathrm{RAIN}\,T_r$$

(with $M \ge 0$ in both cases; otherwise $M = 0$).

**Ripeness and liquid-water routing.** The heat deficit accumulates
$D_f \mathrel{+}= \Delta D_{f,snow} + \Delta D_{f,T}$ (floored at 0, capped at
$0.33\,W_i$). If $M < W_i$, melt and rain enter the pack's liquid store, whose capacity
is $\mathrm{PLWHC} \cdot W_i$; outflow $E$ is the excess after the deficit and liquid
capacity are satisfied. If $M \ge W_i$ the pack melts out and
$E = W_i + W_q + \mathrm{RAIN}$.

**Ground melt.** A constant daily melt $\mathrm{DAYGM}$ at the soil–snow interface
removes ice (and proportional liquid) and joins the outflow:
$SWE = W_i + W_q$ after withdrawal.

The pack outflow $E$ (rain + melt) is the effective precipitation delivered to SAC-SMA.

## A.3 SAC-SMA

Sixteen parameters (Table A.1); six states: upper-zone tension/free contents
$uztwc, uzfwc$; lower-zone tension, supplementary-free, and primary-free contents
$lztwc, lzfsc, lzfpc$; additional-impervious-area content $adimc$. The pervious
fraction is $parea = 1 - pctim - adimp$. Initial states are
$[0, 0, 100, 100, 100, 0]$ mm.

**Evapotranspiration cascade (E1–E5).** Demand $edmnd = PET$:

$$E_1 = edmnd\,\frac{uztwc}{uztwm}, \qquad red = edmnd - E_1$$

If upper-zone tension is exhausted, the residual draws on upper-zone free water
($E_2 = \min(uzfwc, red)$); otherwise, if the upper-zone tension ratio falls below the
free ratio, contents rebalance toward a common ratio. The lower-zone tension draw is

$$E_3 = red\,\frac{lztwc}{uztwm + lztwm}$$

with lower-zone free water resupplying tension whenever the tension ratio drops below
the composite lower-zone ratio (excluding the reserve fraction $rserv$). The
additional-impervious store evaporates at

$$E_5 = E_1 + (red + E_2)\,\frac{adimc - E_1 - uztwc}{uztwm + lztwm}$$

(weighted by $adimp$). $E_4$, riparian evapotranspiration, is taken from channel
inflow at the end of the step: $E_4 = (edmnd - E_1 - E_2 - E_3)\cdot riva$ ★.

**Runoff generation.** Upper-zone tension excess $twx = \max(P + uztwc - uztwm, 0)$;
impervious runoff $roimp = P \cdot pctim$. The remainder is processed in $n_{inc}$
sub-steps ★:

$$n_{inc} = \left\lfloor 1 + 0.2\,(uzfwc + twx)\right\rfloor, \qquad
pinc = twx / n_{inc}$$

with per-substep depletion fractions
$duz = 1 - (1 - uzk)^{1/n_{inc}}$ (and likewise $dlzp, dlzs$ from $lzpk, lzsk$).
Within each sub-step:

- *Direct runoff from the additional impervious area* ★ (hard-coded quadratic):
  $$addro = pinc \cdot \left[\max\!\left(\frac{adimc - uztwc}{lztwm}, 0\right)\right]^{2}$$
- *Baseflow*: $bf = lzfpc \cdot dlzp + lzfsc \cdot dlzs$
- *Percolation*, demand-scaled by lower-zone deficiency:
  $$perc = \left(lzfpm\,dlzp + lzfsm\,dlzs\right)\frac{uzfwc}{uzfwm}
  \left[1 + zperc \cdot defr^{\,rexp}\right], \qquad
  defr = 1 - \frac{lztwc + lzfpc + lzfsc}{lztwm + lzfpm + lzfsm}$$
  capped by available $uzfwc$ and by lower-zone capacity; the fraction $pfree$ of
  percolate bypasses lower-zone tension directly into the free stores, split between
  primary and supplementary in proportion to their relative deficiencies.
- *Interflow*: $sif \mathrel{+}= uzfwc \cdot duz$
- *Surface runoff*: any inflow exceeding upper-zone free capacity,
  $sur = \max(pinc + uzfwc - uzfwm, 0)$, weighted by $parea$ (with the
  additional-impervious complement $adsur$).

**Channel inflow assembly.** Total baseflow is reduced by the deep-loss ratio:
$bfcc = tbf/(1 + side)$. Channel inflow is $surf + base - E_4$, with $E_4$ split
half-and-half between the surface and base components ★ (clamped at zero). The
components $surf$ (impervious + direct + surface + interflow) and $base$ are passed
separately to routing. Storage floors of $10^{-5}$ and $10^{-4}$ mm ★ snap small
contents to zero exactly as in the original code.

## A.4 Lohmann routing

Four parameters: unit-hydrograph shape $N$ and scale $K$, and channel wave velocity
$V$ and diffusivity $D$ (Table A.1). Each HRU's flow distance to the outlet is $L$ (m).

**Hillslope unit hydrograph.** A gamma density with shape $N$ and scale $1/K$,
integrated over daily bins (12-day support):

$$g(t) = \frac{1}{\theta\,\Gamma(N)}\left(\frac{t}{\theta}\right)^{N-1} e^{-t/\theta},
\qquad \theta = 1/K$$

**Channel Green's function.** The impulse response of the linearized Saint-Venant
equation, evaluated hourly and aggregated to a 96-day daily unit hydrograph:

$$H(t) = \frac{L}{2t\sqrt{\pi D t}}\,
\exp\!\left[-\frac{(V t - L)^2}{4 D t}\right]$$

convolved with a 24-hour boxcar and normalized to unit volume. For the outlet HRU
($L = 0$) the channel response is the identity.

**Convolution.** Surface/interflow inflow is convolved with
(hillslope $*$ channel); baseflow is convolved with the channel response only
(bypassing the hillslope unit hydrograph):

$$q(t) = \left(q_{surf} * g * H\right)(t) + \left(q_{base} * H\right)(t)$$

Basin discharge is the area-weighted sum of routed HRU flow.

## A.5 Evaluation metrics

Kling–Gupta efficiency (Gupta et al., 2009), used as the GA objective, the dPL
selection metric, and the parity criterion:

$$KGE = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}, \qquad
\alpha = \frac{\sigma_{sim}}{\sigma_{obs}}, \quad
\beta = \frac{\mu_{sim}}{\mu_{obs}}$$

Nash–Sutcliffe efficiency $NSE = 1 - \sum(q_{sim} - q_{obs})^2 / \sum(q_{obs} -
\bar{q}_{obs})^2$; percent bias
$pbias = 100\,(\sum q_{sim} - \sum q_{obs})/\sum q_{obs}$. The seasonal-mismatch
diagnostic is the total-variation distance between the normalized mean-monthly
regimes. With $p_m$ and $q_m$ the simulated and observed fractions of mean-annual
volume falling in calendar month $m$,

$$SM = 100 \times \tfrac{1}{2}\sum_{m=1}^{12}\left|p_m - q_m\right|
     \;=\; 100 \times \left(1 - \sum_{m=1}^{12}\min(p_m, q_m)\right),$$

the percentage of annual volume placed in the wrong month. It is reported as a
percentage throughout this document.

## A.6 Parameter table

Table A.1 lists the 31-parameter structure with the GA feasible ranges (Wi &
Steinschneider, 2023). The dPL parameter network of Part II emits the same 28 free
parameters into the same box (with the two noted widenings); `side`, `SCF`, and
`PXTEMP` are fixed in both systems.

| Process | Parameter | Description | Units | Lower | Upper |
|---|---|---|---|---|---|
| PET | Kpet | Hamon PET proportionality coefficient | – | 0.4 | 2.5 |
| SMA | uztwm | Upper-zone tension water capacity | mm | 1 | 1000 |
| SMA | uzfwm | Upper-zone free water capacity | mm | 1 | 1000 |
| SMA | lztwm | Lower-zone tension water capacity | mm | 50 | 5000 |
| SMA | lzfpm | Lower-zone primary free water capacity | mm | 50 | 5000 |
| SMA | lzfsm | Lower-zone supplementary free water capacity | mm | 50 | 5000 |
| SMA | uzk | Upper-zone free water depletion rate | 1/day | 0.01 | 0.99 |
| SMA | lzpk | Lower-zone primary depletion rate | 1/day | 0.01 | 0.5 |
| SMA | lzsk | Lower-zone supplementary depletion rate | 1/day | 0.01 | 0.5 |
| SMA | zperc | Percolation demand scale | – | 1 | 500 |
| SMA | rexp | Percolation demand shape | – | 1 | 10 |
| SMA | pfree | Percolation fraction to free water | – | 0 | 0.99 |
| SMA | pctim | Permanently impervious fraction | – | 0 | 0.5 |
| SMA | adimp | Additional impervious area | – | 0 | 0.9 |
| SMA | riva | Riparian vegetation area | – | 0 | 0.9 |
| SMA | side | Deep-recharge to baseflow ratio | – | 0 (fixed) | |
| SMA | rserv | Lower-zone free water reserve fraction | – | 0 | 0.4 |
| Snow | SCF | Snowfall correction factor | – | 1.0 (fixed) | |
| Snow | PXTEMP | Rain/snow threshold temperature | °C | 0 (fixed) | |
| Snow | MFMAX | Maximum non-rain melt factor | mm/(°C·6h) | 0.05 | 5.0 |
| Snow | MFMIN | Minimum non-rain melt factor | mm/(°C·6h) | 0.05 | 5.0 |
| Snow | UADJ | Rain-on-snow wind function | mm/(hPa·6h) | 0.03 | 0.2 |
| Snow | MBASE | Melt base temperature | °C | 0 | 5 |
| Snow | TIPM | Antecedent temperature index weight | – | 0.1 | 1 |
| Snow | PLWHC | Liquid water holding capacity | – | 0.02 | 0.3 |
| Snow | NMF | Maximum negative melt factor | mm/(°C·6h) | 0.05 | 0.3 |
| Snow | DAYGM | Daily ground melt | mm/day | 0 | 0.3 |
| Routing | N (`Nres`) | Hillslope UH shape (linear reservoirs) | – | 1 | 20 |
| Routing | K (`Kres`) | Hillslope UH storage constant | 1/day | 0.01 | 0.99 |
| Routing | Velo | Saint-Venant wave velocity | m/s | 0.5 | 5 |
| Routing | Diff | Saint-Venant diffusivity | m²/s | 200 | 4000 |

*Table A.1. The 31-parameter SAC-SMA structure with GA feasible ranges (Wi & Steinschneider, 2023). Parameters marked "fixed" are not calibrated.*

For the dPL search of Part II, the `rexp` upper bound is widened to 15 and the `lzsk`
lower bound to 0.003. The Noah-lite ET variant adds one learned parameter,
$\chi \in [0.5, 2.5]$ (soil-moisture limitation exponent), and pins vegetation fraction
and seasonal LAI to observations.
