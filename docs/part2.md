# PART II — DIFFERENTIABLE REIMPLEMENTATION (dPL)

## Motivation and the dPL concept

The GA calibration of Part I has three structural limits. First, it is a
derivative-free search over a ~200-dimensional space; each candidate costs a full
multi-decade simulation, so the search is coarse, and once archived the optimum is
fixed. Second, its regionalization is categorical (parameters are constant
within a soil or vegetation class), so it cannot exploit continuous landscape
information such as soil texture fractions, terrain, or canopy structure. Third,
extending the model with a new PET scheme or a new ET formulation invalidates the
archived optimum and requires repeating the entire search.

Differentiable parameter learning (dPL) {cite:p}`tsai2021` replaces the search with
gradient-based training. The physics are re-expressed in a differentiable framework
(PyTorch), and a small neural network $g_\theta$ maps each HRU's landscape attributes
$A_i$ to its full physical parameter vector:

$$\phi_i = g_\theta(A_i), \qquad
\hat{Q} = \mathcal{M}\!\left(\phi, F\right), \qquad
\theta^* = \arg\min_\theta \; \mathcal{L}\!\left(\hat{Q}, Q^{obs}\right)$$

where $\mathcal{M}$ is the (differentiable) SAC-SMA pipeline driven by forcing $F$, and
the loss gradient flows through the physics into the network weights $\theta$. The
simulation itself remains untouched. The network only produces parameter fields, and
the model is still the same Snow-17, SAC-SMA, and routing chain. Because the network
is a smooth function of landscape attributes shared across all HRUs, and no basin
identity enters the fit, the learned parameter fields amount to a regionalization. The
transfer to ungauged locations that the GA achieved through class-based parameter
sharing is retained, now with continuous inputs, more expressive capacity, and
cheap retraining when the physics change {cite:p}`feng2022`. Cheap retraining in particular
is what makes the variant chain of §9 practical, since each physics upgrade only
requires rerunning the training.

Two implementation principles carry over from Part I unchanged.

- **Fidelity to the frozen cores.** The differentiable physics reproduce the frozen
  NumPy/Numba model branch-for-branch (forward values identical; only subgradients at
  branch points differ). A standing fidelity benchmark runs the archived GA parameters
  through the differentiable pipeline and requires KGE ≥ 0.999 against the frozen model.
- **Frozen scoring.** Reported skill comes from exporting the trained parameter
  fields and re-running them through the frozen production model, never from the
  training pipeline itself (the LSTM stage of the hybrids, which has no frozen
  counterpart, is the one exception noted in §10). The dPL system is a calibration
  engine; the production model remains the parity-verified code of Part I.

## Architecture

### Parameter network

The parameter network is a small per-HRU multilayer perceptron with no knowledge of
basin identity,

$$z_i = \mathrm{ReLU}\!\left(W_2\,\mathrm{ReLU}(W_1 A_i)\right) \in \mathbb{R}^{32},
\qquad s_i = \sigma\!\left(W_3 z_i\right) \in (0,1)^{28}$$

with hidden width 64, embedding 32, and dropout 0.1 on the hidden layer. The sigmoid
outputs are mapped into the GA feasible ranges of Table A.1 (the same box the GA
searched):

$$\phi_{i,p} = \ell_p + s_{i,p}\,(u_p - \ell_p),$$

with the wide storage and percolation parameters (`uztwm`, `uzfwm`, `lztwm`, `lzfpm`,
`lzfsm`, `zperc`) interpolated in log space. Twenty-eight of the 31 parameters are
learned; the three conventions of Part I stay fixed (`side` = 0, `SCF` = 1,
`PXTEMP` = 0). Two bounds were widened (never narrowed) after the search pressed
against them, the `rexp` ceiling from 10 to 15 and the `lzsk` floor from 0.01 to
0.003.

**Initialization is the GA prior.** Output-layer weights start at zero, with biases set
so the untrained network emits the area-weighted median of the archived GA optimum for
every parameter. Training therefore starts from a hydrologically plausible,
GA-consistent field and departs from it only where the data require.

**Input features** (the `physical` variant used by every canonical run) are continuous
landscape attributes, z-scored. They comprise site statics (elevation, latitude,
longitude, flow length) plus 20 columns of POLARIS soil properties (sand, clay,
log-ksat, porosity × surface/deep zones) {cite:p}`chaney2019`, LANDFIRE existing-vegetation
cover and height, 3DEP terrain (slope, aspect, curvature, relief), and MODIS LAI
seasonal statistics (mean, amplitude, peak timing). Replacing the GA's 13 soil and 12
vegetation classes with these continuous fields is the direct upgrade of the Part I
regionalization.

### Differentiable physics

The PyTorch physics run all HRUs as one batched tensor computation per day, with every
branch and clamp of the reference model expressed as an equivalent masked blend, so
forward values match the frozen cores exactly. One structural exception is required for
training. The reference model's data-dependent percolation sub-step count
($n_{inc} = \lfloor 1 + 0.2\,(uzfwc + twx)\rfloor$) is unbatchable, so training uses a
fixed $n_{inc} = 5$; the exported fields are then scored through the frozen model with
the reference numerics (§10). No smoothing of the physics is used in any canonical run.

### The variant chain

Each rung of the chain changes one element at a time.

**`hamon_dense` (dPL on the original domain).** The parameter network replaces the GA
on the unchanged Part I setup of 7,891 fine HRUs, lapse-rate-downscaled HRU
meteorology, Hamon PET, and the frozen E1–E5 ET cascade.

**`hamon` (native-grid domain).** The same model on the native 1/16° forcing grid
(2,074 cells), with one parameter set per grid cell, no sub-cell HRUs, and therefore
no lapse-rate meteorology downscaling (that product is upstream of this repository).
The domain is also re-footed onto the CalSim3 catchment delineations (11 of 15 basins;
the four Tulare/Kern basins keep their full footprints), aligning training with the
geography the model will serve. This is the deployable configuration, and everything
downstream runs on it.

**`pt` (Priestley–Taylor PET).** Replaces Hamon with an energy-based PET
{cite:p}`priestleytaylor1972`:

$$PET = \alpha \, \frac{\Delta}{\Delta + \gamma}\, \frac{\max(R_n, 0)}{\lambda},
\qquad \alpha = 1.26$$

with net radiation assembled from Bristow–Campbell solar radiation driven by the
diurnal temperature range {cite:p}`bristowcampbell1984`
($R_s = R_a \cdot 0.7\,[1 - \exp(-0.007\,\Delta T^{2.4})]$) and FAO-56 net longwave
{cite:p}`allen1998`. Two one-directional refinements correct seasonal shape errors that the
scalar $K_{pet}$ cannot express. A **snow-cover albedo** blends surface albedo from
0.23 to 0.6 by $1 - e^{-SWE/15\,\mathrm{mm}}$ using the model's own Snow-17 SWE, which
suppresses PET under a snowpack. An **arid dewpoint depression** lowers the dew point
below $T_{min}$ by up to 2 °C, ramped on the diurnal range as an aridity proxy, which
trims summer PET in the dry basins. Real per-cell $T_{min}/T_{max}$ forcing is
required.

**`noah` (soil-moisture-limited ET).** Replaces the SAC-SMA E1–E5 demand cascade with
an external two-source actual-ET formulation in the spirit of the Noah land-surface
model and the SAC-HTET lineage {cite:p}`ek2003,koren2014`, on *observed* vegetation:

$$\sigma_g = \min\!\left(f_{veg},\, 1 - e^{-0.5\,LAI(t)}\right)$$
$$AET = \underbrace{(1-\sigma_g)\, PET \cdot \beta_{uz}^{\chi}}_{\text{bare soil}}
\;+\; \underbrace{\sigma_g\, PET \cdot \beta_{root}^{\chi}}_{\text{canopy}},
\qquad \beta_{root} = f_{root}\,\beta_{uz} + (1-f_{root})\,\beta_{lz}$$

where $\beta_{uz}, \beta_{lz}$ are wilting-adjusted relative tension-water contents of
the SAC-SMA upper and lower zones, green fraction $f_{veg}$ (LANDFIRE) and seasonal
$LAI(t)$ (MODIS climatology) are pinned to observations, and a single learned exponent
$\chi \in [0.5, 2.5]$ per HRU controls moisture limitation. This minimal form was
chosen for identifiability. A full 7-parameter Jarvis-conductance Noah variant was
trained and failed to beat it, because daily streamflow alone cannot identify the
Jarvis resistance parameters; they collapse into a single factor confounded with PET
scaling. The single learned degree of freedom achieves the same skill with a much
smaller parameter space.

**A seasonal fine-tune, evaluated and set aside.** An intermediate rung (`noah_ft`)
warm-started from the `noah` optimum and added a bounded annual harmonic to $K_{pet}$
and the three Snow-17 melt parameters. These seasonal degrees of freedom are not
identifiable from streamflow alone (a flow-only control run carrying the same
machinery left the seasonal error unchanged), so they were identified by auxiliary
calibration-window losses that pull the seasonal shape of simulated ET and SWE toward
satellite product ensembles and anchor each basin's long-term ET within ±15% of
observed $P-Q$, with flow KGE retained as the selection metric throughout. The
mechanism worked as designed. The learned harmonics reproduced, inside the physics,
part of the winter-to-spring melt redistribution that the LSTM hybrid otherwise learns
as a black-box correction. The validation seasonal mismatch improved from 9.6% to
8.5% of annual volume, and the NML and MRC monthly regimes improved against CalSim3. A retrain
against the repository's curated satellite-observation store reproduced the result
exactly, so the finding rests on reproducible, in-repository data. The full
accounting still did not favor keeping it. Pooled validation KGE ended in a tie with
the frozen `noah` baseline, the CalSim3 comparison was a wash overall, snow-free NHG
regressed because its melt parameters are invisible to the SWE signal, and the
northern basins drifted low in volume through the water-balance band. The seasonal
parameters also have no frozen-model counterpart, which made every downstream consumer
depend on exported daily simulations instead of parameter tables. The fine-tune was
set aside, and `noah` is the physics tier the hybrids build on; the identification
finding itself carries forward to the options of §12.

**`hybrid` and `hybrid_pet_dt` (the SAC×LSTM ensembles).** An LSTM
{cite:p}`hochreiter1997,kratzert2019` is trained on top of the fixed `noah` physics,
with the physics simulation entering as an input feature:

$$\mathbf{h}_t = \mathrm{LSTM}\!\left(\big[\,P,\; T_{avg},\; T_{min},\; T_{max},\;
\hat{q}^{phys}\,\big]_{t-364}^{t},\; \mathbf{s}\right), \qquad
\hat{q}_t = \mathrm{Softplus}\!\left(W_o\,\mathrm{ReLU}(W_h \mathbf{h}_t)\right)$$

(hidden size 64; the 365-day input window ends on the prediction day itself, so
day-$t$ forcing and the day-$t$ physics simulation inform the day-$t$ estimate, and
the hybrid is a simulation model like the physics beneath it; $\mathbf{s}$ is an
embedded vector of static basin attributes). Two input choices matter. First, forcing channels
are z-scored with pooled cross-basin statistics, while the physics-simulation channel
and the target are scaled by each basin's own calibration-window observed standard
deviation, which makes a pass-through of the physics representable by a single
entity-blind network. Second, no day-of-year input is provided. The physics channel
already carries the calendar (the Snow-17 melt sinusoid, seasonal LAI, the radiation
cycle), and an explicit calendar input is what allows the network to learn
calendar-keyed mean corrections that carry unchecked into validation. Each variant is
an eight-seed ensemble scored on the ensemble-mean flow.

The two variants differ in climate behavior rather than skill. `hybrid` is the basic
feature coupling. `hybrid_pet_dt` adds two elements aimed at climate-perturbed
applications: the raw Priestley–Taylor potential (the physics' energy-demand signal, a
deterministic function of the forcing, so it recomputes exactly under any temperature
counterfactual) as a sixth dynamic channel, and a **temperature-consistency loss**.
For that loss, the `noah` physics are re-run once under a +2 °C perturbation to
produce a teacher simulation (through the same differentiable pipeline that exports
the input channel, so channel and teacher share one set of numerics); during
training, each batch is forwarded a second time
with the temperature and PET channels shifted and the physics channel replaced by the
teacher, and the network's daily warming response is pulled toward the physics
response by

$$\mathcal{L}_{\Delta T} = \lambda_T\;\overline{\left[\left(\hat{q}^{+\Delta T}_t -
\hat{q}_t\right) - \left(\hat{q}^{phys,+\Delta T}_t -
\hat{q}^{phys}_t\right)/s_b\right]^2}, \qquad \lambda_T = 0.3.$$

The network keeps its within-climate skill but inherits the physics' climate
sensitivity (§11.2). A residual coupling, in which the LSTM predicts the signed
physics error and $\hat{q} = \hat{q}^{phys} + \delta$, was benchmarked to equivalent
pooled skill but retired: under the validation period's climate shift it re-injects a
regime-conditional volume bias at the basins where the physics were already right, and
that behavior persisted under every attempted repair (removing calendar inputs,
penalizing the residual mean, and improving the underlying physics baseline).

## Implementation: training and scoring

**Loss.** Training minimizes a per-basin variance-normalized squared error, summed
over fixed 366-day chunks. Each basin's error is normalized by its full
calibration-window observed variance, so chunk losses add up to per-basin NSE and
basins weigh comparably. Two auxiliary terms are added, a log-flow term (weight 0.15)
for low-flow shape and a variance-matching penalty
$(\sigma_{sim}/\sigma_{obs} - 1)^2$ per chunk (weight 1.0) that counters the
systematic variance damping of squared-error optima:

$$\mathcal{L} = \frac{1}{B}\sum_b \left[
\frac{\sum_t (\hat{q}_{b,t} - q_{b,t})^2}{n_b\,\mathrm{Var}_{cal}(q_b)}
+ 0.15\,\overline{\left(\log\tfrac{\hat{q}+\epsilon}{q+\epsilon}\right)^2}
+ \left(\tfrac{\sigma(\hat{q}_b)}{\sigma(q_b)} - 1\right)^2
\right]$$

**Optimization.** AdamW (learning rate $10^{-3}$, 3-epoch warmup, cosine decay to
$10^{-5}$; weight decay $10^{-5}$; gradient clip 1.0), up to 60 epochs with early
stopping. Training is truncated backpropagation through time over the 366-day chunks
with all basins simulated simultaneously; hydrologic state and the 106-day routing
history carry across chunk boundaries detached. Model selection uses the exact pooled
15-basin mean calibration KGE, computed gradient-free every second epoch.

**Periods and target.** Calibration on WY1989–2003 against daily CDEC full-natural
flow, identical to the GA, with WY2004–2018 held out entirely (never read during
training or selection).

**Hybrid training.** The LSTM ensembles train with AdamW (learning rate 4×10⁻⁴, batch
512) on per-basin standard-deviation-normalized MSE plus the same log-flow term
(weight 0.15), with Gaussian input jitter and dropout for regularization, and (for
`hybrid_pet_dt`) the temperature-consistency term of §9. Selection is again pooled
calibration KGE, and the validation years are never read.

**Frozen scoring.** After training, the parameter fields are exported to the
`ga_optimum` table format and re-run through the frozen production model of Part I.
Priestley–Taylor and Noah-lite ET each have frozen Numba mirrors verified bit-exact
against the differentiable implementation (maximum flow difference around
2×10⁻¹³ mm/day). The physics numbers in §11 are frozen-model scores. The exception is
the LSTM stage of the hybrids, which has no frozen counterpart and is scored from its
own forward pass, on top of the physics channel it trains on (`noah`'s exported
differentiable-pipeline simulation). One operational caveat
was discovered en route. Learned parameter fields can carry much longer storage memory
than the GA fields (lower-zone capacities near the upper bound fill at 1–2 mm/day), so
trained fields are scored with full 1915-onward spin-up rather than the truncated
spin-up that suffices for the GA parameters.

## Results

Table 5 summarizes the lineage on the 15-CDEC daily benchmark.

| Model | Domain | Cal KGE | Val KGE |
|---|---|---|---|
| GA optimum (Part I) | 7,891 HRU | 0.804 | 0.767 |
| `hamon_dense` | 7,891 HRU | 0.806 | 0.840 |
| `hamon` | 2,074 grid cells | 0.817 | 0.836 |
| `pt` | grid | 0.799 | 0.826 |
| `noah` | grid | 0.767 | 0.799 |
| `hybrid` (8-seed mean) | grid | 0.917 | 0.869 |
| `hybrid_pet_dt` (8-seed mean) | grid | 0.916 | 0.864 |

*Table 5. Pooled 15-basin mean KGE against daily CDEC FNF (calibration WY1989–2003, validation WY2004–2018), frozen-model scoring.[^basisnote]*

[^basisnote]: All rows are pooled means of per-basin KGE from each run's frozen-scored
metrics table, computed identically; the hybrid rows score the ensemble-mean flow as
described in §10. The program's run log records 0.807/0.829 for `hamon` from
selection-time scoring; the frozen re-score shown here is the apples-to-apples basis.

Several points stand out.

- **dPL improves on the GA at identical physics.** On the same domain and physics
  (`hamon_dense`), validation KGE rises from 0.767 to 0.840 with calibration skill
  unchanged. The gain is in generalization, as the continuous-attribute
  regionalization transfers better out of sample than the class-based GA field.
- **The move to the native grid costs little.** Trading away the fine-HRU
  discretization and its lapse-rate meteorology retains about 99% of validation skill
  (0.836 vs 0.840). The deployable configuration is not the bottleneck.
- **The physics upgrades trade a little skill for better structure.** PT (−0.010 val)
  and Noah-lite ET (−0.037 from PT) each give up some calibration-period skill in
  exchange for structurally better ET, with PET that responds to radiation and snow
  cover and transpiration limited by modeled soil moisture on observed vegetation.
  §11.1 and §11.2 show the results of that trade.
- **Observation data can steer seasonal timing, but the package was not kept.**
  The seasonal fine-tune of §9 demonstrated that satellite ET/SWE seasonality
  and a water-balance anchor identify seasonal melt parameters streamflow cannot
  (the same parameters under flow-only training left the seasonal error unchanged),
  improving melt timing and the NML/MRC regimes inside the physics. Its pooled skill
  tied `noah`, NHG and the northern-basin volumes regressed, and it was set aside.
  The identification finding stands, and the machinery remains available to the
  multi-timescale program of §12.
- **The hybrid closes the skill gap regardless of the physics underneath.** The LSTM
  ensembles lift pooled validation KGE to about 0.87 whichever physics baseline they
  sit on, with gains concentrated in calibration (0.917), and they cut the
  validation seasonal mismatch from 9.2% to 3.9–4.1%. Two persistent weaknesses
  temper this. SCC
  keeps a validation volume bias near +30% that the network cannot fix from
  calibration-period data alone. And at basins where the physics already match CalSim3
  (NML, MRC, ORO), the hybrid's extra skill comes with regime-conditional volume
  drift. That drift proved intrinsic to calibration-only training under a shifted
  validation climate: it survived removal of the calendar inputs, penalties on the
  correction's mean, and an improved physics baseline. At those basins the physics
  remain the more trustworthy out-of-sample answer.
- **The PET input raises skill; only the training constraint moves the response.**
  The unconstrained `hybrid` has
  an untrustworthy warming response (§11.2), and giving it the PET input channel alone
  raised its skill (an eight-seed PET-only ensemble reaches validation KGE 0.872, the
  best of any configuration) while leaving the response nearly unchanged. The
  network uses physics-shaped inputs as information, and only the
  temperature-consistency loss moved its climate sensitivity (Figure 6).
  `hybrid_pet_dt` combines both and converts the PET skill margin into response
  fidelity at unchanged pooled skill.
- **Structural ceilings persist across every rung.** NHG carries a +17–23% volume bias
  (snow-free basin, melt parameters unconstrained) and FOL a variance-damping ceiling
  (KGE ≈ 0.76) in all physics variants. These are limits of the data and model
  structure, and no calibration variant moved them.

### Out-of-calibration evaluation against CalSim3

The most relevant test for CalSim purposes is monthly climatology against CalSim3 FNF
on the out-of-calibration record (WY1950–1987 plus WY2004–2018), on the CalSim3
delineations. Figure 5 summarizes the top of the lineage. The dPL models fix the
pooled GA's worst CalSim3-basis failures (the SHA/BND rim under-run of Part I §5
largely closes on the re-footed domain), and the physics rungs hold that ground with
progressively more defensible ET. The Noah rung repaired two basins the PT
cascade could not (NML and MRC, monthly KGE 0.92 and 0.94), which suggests the
soil-moisture-limited ET structure itself is doing the work.

On this basis the three current models are close in the mean but differ in character.
Over the eleven basins with CalSim3 coverage, mean monthly-regime KGE is 0.86
(`noah`), 0.85 (`hybrid`), and 0.88 (`hybrid_pet_dt`, the best of the set). The basic
hybrid's in-sample skill does not carry to this test at the good-physics basins (NML
0.75, MRC 0.79 against the physics' 0.92–0.94), which is the regime-conditional drift
noted above; the temperature-anchored variant recovers part of that ground (NML 0.78,
MRC 0.86) while improving nearly everywhere else, including an outright repair of
NHG's seasonal shape (mismatch 16% for the physics, under 2% for both hybrids).

```{figure} ../artifacts/dpl/figures/cdec15_climatology_e.png
:width: 6.2in

Figure 5. Mean-monthly regime versus CalSim3 FNF (out-of-calibration months, WY1950–1987 + WY2004–2018): `noah` → the two LSTM hybrid ensembles, by basin.
```

### Climate response of the dPL chain

Repeating the §6 temperature-detrending experiment on the dPL variants shows the
intended progression through the physics rungs. The Hamon rung inherits the Part I
volume sensitivity, the PT rung damps it, and the Noah rungs damp it further while
shifting melt timing, which is the VIC-like response pattern now reproduced inside the
SAC-SMA framework. (Appendix B.6 collects the rolling and pre-1950 monthly-regime
sensitivity figures.)

The hybrids do not inherit a climate response from the physics; it has to be trained
in. Under a +2 °C perturbation the unconstrained `hybrid` fails outright. Across the
current eight-seed ensemble its aggregate response is wrong-signed, adding annual flow
where its own physics baseline removes it, with individual seeds scattered from
moderately correct to strongly inverted; an earlier training of the same configuration
measured a weak same-signed response, so the unconstrained response is unstable from
retrain to retrain. A
network trained only on historical variability has no reason to extrapolate a
temperature sensitivity. The temperature-consistency loss repairs this.
`hybrid_pet_dt` matches the physics response in aggregate (response ratio 1.04) with a
monthly-regime response correlation of 0.97 and every large-magnitude basin
correct-signed, at pooled skill equal to the basic
hybrid's. A screening across loss weights showed the response scaling smoothly with
the weight (λ of 0.1/0.3/1.0 recovering 36%/78%/92% in the no-PET screening arms), with skill
degrading only at the heaviest weight; adding the PET channel at λ = 0.3 gave the
strongest response of any configuration.

Figure 6 isolates the two levers with a three-arm progression on the current training
basis, inserting an eight-seed PET-input-only ensemble between the canonical pair. The
PET input alone is the skill lever. That arm posts the best validation KGE of the
three (0.872) but leaves the warming response nearly flat (ratio 0.24). The
temperature-consistency loss is the response lever, and it buys the full physics
response for about 0.008 of validation KGE. The progression also localizes the basic
arm's failure: its monthly-regime response has the right shape (correlation 0.95) but
overshoots the winter and spring flow gains by roughly 80% while matching the summer
loss, so the annual total comes out wrong-signed. The error sits in the annual
sum, not the seasonal pattern.

```{figure} ../artifacts/dpl/hybrid_progression.png
:width: 6.2in

Figure 6. The hybrid progression on pooled skill and +2 °C response: `hybrid` (basic coupling) → adding the PET input channel → adding the temperature-consistency loss (`hybrid_pet_dt`). The PET input raises skill without moving the response; only the training-time anchor produces the physics-consistent response.
```

The WGEN Product A detrending pattern was deliberately held out of hybrid training, so
the §6-style rerun is an independent check of the learned response. On that held-out
pattern the monthly-regime response generalizes fully (the `hybrid_pet_dt` regime
shift sits on the physics family's), while the annual-volume response generalizes only
partially. The method has one boundary. The temperature anchor buys a
physics-consistent climate response; it does not repair the validation-period volume
drift at NML/MRC/ORO discussed in §11. The two are distinct failure modes with
distinct causes.

## Next steps

The next-phase workstreams that extend this system toward the full CalSim domain — multi-timescale calibration, nested sub-arc training, a fuller training domain, and the physics/hybrid refinement options — are tracked as open issues in the modeling repository:

<https://github.com/CADWR-Climate-Change-Program/SAC-SMA/issues>

