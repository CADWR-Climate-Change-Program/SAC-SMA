# Part II: differentiable reimplementation (dPL)

## Motivation and the dPL concept

The GA calibration of Part I has three structural limits. First, it is a derivative-free search over a ~200-dimensional space; each candidate costs a full multi-decade simulation, so the search is coarse, and once archived the optimum is fixed. Second, its regionalization is categorical (parameters are constant within a soil or vegetation class), so it cannot exploit continuous landscape information such as soil texture fractions, terrain, or canopy structure. Third, extending the model with a new PET scheme or a new ET formulation invalidates the archived optimum and requires repeating the entire search.

Differentiable parameter learning (dPL) {cite:p}`tsai2021` replaces the search with gradient-based training. The physics are re-expressed in a differentiable framework (PyTorch), and a small neural network $g_\theta$ maps each HRU's landscape attributes $A_i$ to its full physical parameter vector:

$$\phi_i = g_\theta(A_i), \qquad
\hat{Q} = \mathcal{M}\!\left(\phi, F\right), \qquad
\theta^* = \arg\min_\theta \; \mathcal{L}\!\left(\hat{Q}, Q^{obs}\right)$$

where $\mathcal{M}$ is the (differentiable) SAC-SMA pipeline driven by forcing $F$, and the loss gradient flows through the physics into the network weights $\theta$. The simulation itself remains untouched. The network only produces parameter fields, and the model is still the same Snow-17, SAC-SMA, and routing chain. Because the network is a smooth function of landscape attributes shared across all HRUs, and no basin identity enters the fit, the learned parameter fields amount to a regionalization. The transfer to ungauged locations that the GA achieved through class-based parameter sharing is retained, now with continuous inputs, more expressive capacity, and cheap retraining when the physics change {cite:p}`feng2022`. Cheap retraining in particular is what makes the variant chain below practical, since each physics upgrade only requires rerunning the training.

Two implementation principles carry over from Part I unchanged.

- **Fidelity to the frozen cores.** The differentiable physics reproduce the frozen NumPy/Numba model branch-for-branch (forward values identical; only subgradients at branch points differ). A standing fidelity benchmark runs the archived GA parameters through the differentiable pipeline and requires KGE ≥ 0.999 against the frozen model.
- **Frozen scoring.** Reported skill comes from exporting the trained parameter fields and re-running them through the frozen production model, never from the training pipeline itself (the LSTM stage of the hybrids, which has no frozen counterpart, is the one exception noted in Common machinery, below). The dPL system is a calibration engine; the production model remains the parity-verified code of Part I.

## The progression at a glance

Part II was built as a chain, each rung changing one element at a time and keeping what the previous rung established. Table 4 gives the whole arc on the 15-CDEC daily benchmark: what changed at each rung, what it scored, and what it taught. The rungs are then developed in order in The physics chain and The hybrid family, below.

| Model | What changed | Cal KGE | Val KGE | What it taught |
|---|---|---|---|---|
| GA optimum (Part I) | (archived baseline, 7,891 HRUs) | 0.804 | 0.767 | the reference |
| `hamon_dense` | GA search → dPL network, physics unchanged | 0.806 | 0.840 | continuous regionalization generalizes far better out of sample |
| `hamon` | fine HRUs → native 1/16° grid, CalSim3 footprint | 0.817 | 0.836 | the deployable domain costs ~1% of val skill |
| `pt` | Hamon → Priestley–Taylor PET | 0.799 | 0.826 | a little skill traded for radiation- and snow-responsive PET |
| `noah` | E1–E5 cascade → soil-moisture-limited ET; climate-adaptive parameters | 0.779 | 0.804 | structurally defensible ET; parameters respond to climate at no present-day cost |
| `hybrid` | + LSTM on the physics simulation | 0.922 | 0.877 | the LSTM closes the skill gap, but its climate response is untrained |
| `hybrid_dt` | + multi-anchor (Δp, ΔT) response loss | 0.873 | 0.849 | a physics-consistent climate response can be trained in, at modest skill cost |
| `lstm` | physics channel removed (control) | 0.909 | 0.835 | skill survives without physics; a trustworthy climate response does not |

*Table 4. The Part II lineage: pooled 15-basin mean KGE against daily CDEC FNF (calibration WY1989–2003, validation WY2004–2018), frozen-model scoring.[^basisnote]*

[^basisnote]: All rows are pooled means of per-basin KGE from each run's frozen-scored metrics table, computed identically; the hybrid rows score the ensemble-mean flow as described in Common machinery, below. The program's run log records 0.807/0.829 for `hamon` from selection-time scoring; the frozen re-score shown here is the apples-to-apples basis. The frozen-noah predecessor generation (The hybrid family, below) is retained for lineage: `superseded/noah_noca` 0.767/0.799, `superseded/hybrid_noca` (8-seed) 0.917/0.869, `superseded/hybrid_dt_noca` (8-seed) 0.916/0.864.

Four findings organize the rest of Part II. First, **dPL beats the GA at identical physics**: on the unchanged Part I setup, validation KGE rises from 0.767 to 0.840, and the gain is generalization, not fit. Second, **each physics upgrade trades a little skill for better structure and a better climate response**: the PT and Noah rungs give up calibration-period KGE for PET that responds to radiation and snow cover and for transpiration limited by modeled soil moisture, and that structure pays off out of calibration and under perturbed climate. Third, **the LSTM closes the skill gap regardless of the physics underneath, but its climate response must be trained in**: without the response loss the hybrid over-responds to warming (ratio 1.50 against the physics) and a no-physics LSTM is wrong-signed (−0.94), while the response-constrained `hybrid_dt` tracks the physics (1.14) at a modest skill cost. Fourth, **structural ceilings persist across every rung**: NHG's volume bias, FOL's variance damping, and SCC's validation weakness are limits of the data and model structure that no calibration variant moved.

## Common machinery

### Parameter network

The parameter network is a small per-HRU multilayer perceptron with no knowledge of basin identity,

$$z_i = \mathrm{ReLU}\!\left(W_2\,\mathrm{ReLU}(W_1 A_i)\right) \in \mathbb{R}^{32},
\qquad s_i = \sigma\!\left(W_3 z_i\right) \in (0,1)^{28}$$

with hidden width 64, embedding 32, and dropout 0.1 on the hidden layer. The sigmoid outputs are mapped into the GA feasible ranges of Table A.1 (the same box the GA searched):

$$\phi_{i,p} = \ell_p + s_{i,p}\,(u_p - \ell_p),$$

with the wide storage and percolation parameters (`uztwm`, `uzfwm`, `lztwm`, `lzfpm`, `lzfsm`, `zperc`) interpolated in log space. Twenty-eight of the 31 parameters are learned; the three conventions of Part I stay fixed (`side` = 0, `SCF` = 1, `PXTEMP` = 0). Two bounds were widened (never narrowed) after the search pressed against them, the `rexp` ceiling from 10 to 15 and the `lzsk` floor from 0.01 to 0.003.

**Initialization is the GA prior.** Output-layer weights start at zero, with biases set so the untrained network emits the area-weighted median of the archived GA optimum for every parameter. Training therefore starts from a hydrologically plausible, GA-consistent field and departs from it only where the data require.

**Input features** (the `physical` variant used by every canonical run) are continuous landscape attributes, z-scored. They comprise site statics (elevation, latitude, longitude, flow length) plus 20 columns of POLARIS soil properties (sand, clay, log-ksat, porosity × surface/deep zones) {cite:p}`chaney2019`, LANDFIRE existing-vegetation cover and height, 3DEP terrain (slope, aspect, curvature, relief), and MODIS LAI seasonal statistics (mean, amplitude, peak timing). Replacing the GA's 13 soil and 12 vegetation classes with these continuous fields is the direct upgrade of the Part I regionalization.

### Differentiable physics

The PyTorch physics run all HRUs as one batched tensor computation per day, with every branch and clamp of the reference model expressed as an equivalent masked blend, so forward values match the frozen cores exactly. One structural exception is required for training. The reference model's data-dependent percolation sub-step count ($n_{inc} = \lfloor 1 + 0.2\,(uzfwc + twx)\rfloor$) is unbatchable, so training uses a fixed $n_{inc} = 5$; the exported fields are then scored through the frozen model with the reference numerics. No smoothing of the physics is used in any canonical run.

### Training

**Loss.** Training minimizes a per-basin variance-normalized squared error, summed over fixed 366-day chunks. Each basin's error is normalized by its full calibration-window observed variance, so chunk losses add up to per-basin NSE and basins weigh comparably. Two auxiliary terms are added, a log-flow term (weight 0.15) for low-flow shape and a variance-matching penalty $(\sigma_{sim}/\sigma_{obs} - 1)^2$ per chunk (weight 1.0) that counters the systematic variance damping of squared-error optima:

$$\mathcal{L} = \frac{1}{B}\sum_b \left[
\frac{\sum_t (\hat{q}_{b,t} - q_{b,t})^2}{n_b\,\mathrm{Var}_{cal}(q_b)}
+ 0.15\,\overline{\left(\log\tfrac{\hat{q}+\epsilon}{q+\epsilon}\right)^2}
+ \left(\tfrac{\sigma(\hat{q}_b)}{\sigma(q_b)} - 1\right)^2
\right]$$

**Optimization.** AdamW (learning rate $10^{-3}$, 3-epoch warmup, cosine decay to $10^{-5}$; weight decay $10^{-5}$; gradient clip 1.0), up to 60 epochs with early stopping. Training is truncated backpropagation through time over the 366-day chunks with all basins simulated simultaneously; hydrologic state and the 106-day routing history carry across chunk boundaries detached. Model selection uses the exact pooled 15-basin mean calibration KGE, computed gradient-free every second epoch.

**Periods and target.** Calibration on WY1989–2003 against daily CDEC full-natural flow, identical to the GA, with WY2004–2018 held out entirely (never read during training or selection).

### Frozen scoring

After training, the parameter fields are exported to the `ga_optimum` table format and re-run through the frozen production model of Part I. Priestley–Taylor and Noah-lite ET each have frozen Numba mirrors verified bit-exact against the differentiable implementation (maximum flow difference around 2×10⁻¹³ mm/day). The physics numbers of Table 4 are frozen-model scores. The exception is the LSTM stage of the hybrids, which has no frozen counterpart and is scored from its own forward pass, on top of the physics channel it trains on (`noah`'s exported differentiable-pipeline simulation). One operational caveat was discovered en route. Learned parameter fields can carry much longer storage memory than the GA fields (lower-zone capacities near the upper bound fill at 1–2 mm/day), so trained fields are scored with full 1915-onward spin-up rather than the truncated spin-up that suffices for the GA parameters.

## The physics chain

Each rung changes one element at a time; each subsection states what changed, what it scored, and what it taught.

### `hamon_dense`: dPL replaces the GA at identical physics

The parameter network replaces the GA on the unchanged Part I setup of 7,891 fine HRUs, lapse-rate-downscaled HRU meteorology, Hamon PET, and the frozen E1–E5 ET cascade. On this identical footing, validation KGE rises from the GA's 0.767 to 0.840 with calibration skill unchanged (0.804 → 0.806). The gain is in generalization, as the continuous-attribute regionalization transfers better out of sample than the class-based GA field. This is the cleanest single result of Part II: everything else held fixed, the calibration method alone accounts for the improvement.

### `hamon`: the native-grid, deployable domain

The same model on the native 1/16° forcing grid (2,074 cells), with one parameter set per grid cell, no sub-cell HRUs, and therefore no lapse-rate meteorology downscaling (that product is upstream of this repository). The domain is also re-footed onto the CalSim3 catchment delineations (11 of 15 basins; the four Tulare/Kern basins keep their full footprints), aligning training with the geography the model will serve. Trading away the fine-HRU discretization and its lapse-rate meteorology retains about 99% of validation skill (0.836 vs 0.840): the deployable configuration is not the bottleneck. Everything downstream runs on it.

### `pt`: energy-based PET

Replaces Hamon with a Priestley–Taylor PET {cite:p}`priestleytaylor1972`:

$$PET = \alpha \, \frac{\Delta}{\Delta + \gamma}\, \frac{\max(R_n, 0)}{\lambda},
\qquad \alpha = 1.26$$

with net radiation assembled from Bristow–Campbell solar radiation driven by the diurnal temperature range {cite:p}`bristowcampbell1984` ($R_s = R_a \cdot 0.7\,[1 - \exp(-0.007\,\Delta T^{2.4})]$) and FAO-56 net longwave {cite:p}`allen1998`. Two one-directional refinements correct seasonal shape errors that the scalar $K_{pet}$ cannot express. A **snow-cover albedo** blends surface albedo from 0.23 to 0.6 by $1 - e^{-SWE/15\,\mathrm{mm}}$ using the model's own Snow-17 SWE, which suppresses PET under a snowpack. An **arid dewpoint depression** lowers the dew point below $T_{min}$ by up to 2 °C, ramped on the diurnal range as an aridity proxy, which trims summer PET in the dry basins. Real per-cell $T_{min}/T_{max}$ forcing is required.

The rung costs a little skill (0.826 val, −0.010 from `hamon`) and buys structure: a PET that responds to radiation and snow cover rather than temperature alone. That trade is the deliberate pattern of the physics chain, and its payoff appears in the out-of-calibration and climate-response evaluations below.

### `noah`: soil-moisture-limited, climate-adaptive ET

Replaces the SAC-SMA E1–E5 demand cascade with an external two-source actual-ET formulation in the spirit of the Noah land-surface model and the SAC-HTET lineage {cite:p}`ek2003,koren2014`, on *observed* vegetation:

$$\sigma_g = \min\!\left(f_{veg},\, 1 - e^{-0.5\,LAI(t)}\right)$$
$$AET = \underbrace{(1-\sigma_g)\, PET \cdot \beta_{uz}^{\chi}}_{\text{bare soil}}
\;+\; \underbrace{\sigma_g\, PET \cdot \beta_{root}^{\chi}}_{\text{canopy}},
\qquad \beta_{root} = f_{root}\,\beta_{uz} + (1-f_{root})\,\beta_{lz}$$

where $\beta_{uz}, \beta_{lz}$ are wilting-adjusted relative tension-water contents of the SAC-SMA upper and lower zones, green fraction $f_{veg}$ (LANDFIRE) and seasonal $LAI(t)$ (MODIS climatology) are pinned to observations, and a single learned exponent $\chi \in [0.5, 2.5]$ per HRU controls moisture limitation. This minimal form was chosen for identifiability. A full 7-parameter Jarvis-conductance Noah variant was trained and failed to beat it, because daily streamflow alone cannot identify the Jarvis resistance parameters; they collapse into a single factor confounded with PET scaling. The single learned degree of freedom achieves the same skill with a much smaller parameter space. It is the first of several identifiability lessons in this program: add only the degrees of freedom the training signal can constrain.

**`noah` is climate-adaptive.** The parameter network's input features are augmented with four climate indices (mean precipitation, an aridity index, a snow fraction, and a seasonality index), computed from the forcing and recomputed under any (Δprecip, ΔT) perturbation. Because these indices flow through the same network as the physiographic features, a perturbed climate changes the exported SAC-SMA parameters along with the forcing, a space-for-time response rather than a fixed regionalization. This costs essentially no present-day skill (frozen cal/val 0.779/0.804 against 0.767/0.799 for the climate-frozen version) and is the physics basis every current hybrid below builds on. The climate-frozen predecessor is kept for lineage as `superseded/noah_noca`; it remains the teacher for the earlier hybrid generation described below.

### A seasonal fine-tune, evaluated and set aside

An intermediate rung (`noah_ft`) warm-started from the (climate-frozen) `noah` optimum and added a bounded annual harmonic to $K_{pet}$ and the three Snow-17 melt parameters. These seasonal degrees of freedom are not identifiable from streamflow alone (a flow-only control run carrying the same machinery left the seasonal error unchanged), so they were identified by auxiliary calibration-window losses that pull the seasonal shape of simulated ET and SWE toward satellite product ensembles and anchor each basin's long-term ET within ±15% of observed $P-Q$, with flow KGE retained as the selection metric throughout.

The mechanism worked as designed. The learned harmonics reproduced, inside the physics, part of the winter-to-spring melt redistribution that the LSTM hybrid otherwise learns as a black-box correction. The validation seasonal mismatch improved from 9.6% to 8.5% of annual volume, and the NML and MRC monthly regimes improved against CalSim3. A retrain against the repository's curated satellite-observation store reproduced the result exactly, so the finding rests on reproducible, in-repository data.

The full accounting still did not favor keeping it. Pooled validation KGE ended in a tie with the frozen (climate-frozen) `noah` baseline, the CalSim3 comparison was a wash overall, snow-free NHG regressed because its melt parameters are invisible to the SWE signal, and the northern basins drifted low in volume through the water-balance band. The seasonal parameters also have no frozen-model counterpart, which made every downstream consumer depend on exported daily simulations instead of parameter tables. The fine-tune was set aside. The lesson it leaves is the same identifiability principle as the Jarvis case, run in the productive direction: observation data *can* steer parameters streamflow cannot, and that machinery remains available to the multi-timescale program in Next steps, below.

### Structural ceilings

Three limits persist across every rung of the chain. NHG carries a +17–23% volume bias (snow-free basin, melt parameters unconstrained) and FOL a variance-damping ceiling (KGE ≈ 0.76) in all physics variants; SCC remains a validation weak point that no variant, physics or hybrid, fixes from calibration-period data alone. These are limits of the data and model structure, not of the calibration method.

## The hybrid family

### Architecture

The chain is capped by a family of SAC×LSTM ensembles. An LSTM {cite:p}`hochreiter1997,kratzert2019` is trained on top of the fixed `noah` physics, with the physics simulation entering as an input feature:

$$\mathbf{h}_t = \mathrm{LSTM}\!\left(\big[\,P,\; T_{avg},\; T_{min},\; T_{max},\;
\hat{q}^{phys}\,\big]_{t-364}^{t},\; \mathbf{s}\right), \qquad
\hat{q}_t = \mathrm{Softplus}\!\left(W_o\,\mathrm{ReLU}(W_h \mathbf{h}_t)\right)$$

(hidden size 64; the 365-day input window ends on the prediction day itself, so day-$t$ forcing and the day-$t$ physics simulation inform the day-$t$ estimate, and the hybrid is a simulation model like the physics beneath it; $\mathbf{s}$ is an embedded vector of static basin attributes). Two input choices matter. First, forcing channels are z-scored with pooled cross-basin statistics, while the physics-simulation channel and the target are scaled by each basin's own calibration-window observed standard deviation, which makes a pass-through of the physics representable by a single entity-blind network. Second, no day-of-year input is provided. The physics channel already carries the calendar (the Snow-17 melt sinusoid, seasonal LAI, the radiation cycle), and an explicit calendar input is what allows the network to learn calendar-keyed mean corrections that carry unchecked into validation. Every canonical run also carries the raw Priestley–Taylor potential (the physics' energy-demand signal, a deterministic function of the forcing, so it recomputes exactly under any temperature counterfactual) as a sixth dynamic channel. Each configuration is a three-seed ensemble scored on the ensemble-mean flow.

The ensembles train with AdamW (learning rate 4×10⁻⁴, batch 512) on per-basin standard-deviation-normalized MSE plus the same log-flow term as the physics runs (weight 0.15), with Gaussian input jitter and dropout for regularization. Selection is again pooled calibration KGE, and the validation years are never read.

The three configurations share this architecture and differ only in the physics channel and the loss.

### `hybrid`: the best-skill coupling

`hybrid` keeps the physics-simulation channel, now `noah`'s climate-adaptive export, with no response loss. It is the best-skill configuration: pooled validation KGE 0.877, with gains concentrated in calibration (0.922). The LSTM closes the skill gap regardless of the physics underneath (the frozen-noah predecessor reached the same skill class on its own, weaker physics basis; Lineage, below), which is why skill alone is not the argument for the physics channel; the climate response, below, is.

Two persistent weaknesses temper the result. SCC remains a validation weak point that the network cannot fix from calibration-period data alone. And at basins where the physics already match CalSim3 (NML, MRC, ORO), the hybrid's extra skill comes with regime-conditional volume drift (Out-of-calibration evaluation, below). That drift proved intrinsic to calibration-only training under a shifted validation climate: it survived removal of the calendar inputs, penalties on the correction's mean, and an improved physics baseline. At those basins the physics remain the more trustworthy out-of-sample answer.

A residual coupling, in which the LSTM predicts the signed physics error and $\hat{q} = \hat{q}^{phys} + \delta$, was benchmarked on the frozen-noah predecessor to equivalent pooled skill but retired: under the validation period's climate shift it re-injects a regime-conditional volume bias at the basins where the physics were already right, and that behavior persisted under every attempted repair.

### `hybrid_dt`: the response-consistency loss

`hybrid_dt` adds a **multi-anchor (Δp, ΔT) response-consistency loss** on top of the same architecture. The `noah` physics are re-run under each of 14 (Δprecip, ΔT) anchors, $\{-20\%,-10\%,0,+10\%,+20\%\}\times\{0,+2,+4\ ^\circ\mathrm{C}\}$ (excluding the origin), each producing its own teacher simulation through the same differentiable pipeline that exports the input channel. During training, each batch is additionally forwarded once per anchor with the precipitation, temperature, and PET channels shifted and the physics channel replaced by that anchor's teacher, and the network's response at every anchor is pulled toward the physics response by

$$\mathcal{L}_{\Delta p,\Delta T} = \frac{\lambda}{|\mathcal{A}|}\sum_{(dp,dt)\in\mathcal{A}}
\overline{\left[\left(\hat{q}^{dp,dt}_t - \hat{q}_t\right) -
\left(\hat{q}^{phys,dp,dt}_t - \hat{q}^{phys}_t\right)/s_b\right]^2},
\qquad \lambda = 0.18.$$

The network keeps its within-climate skill (0.873/0.849, a real but modest cost against `hybrid`'s 0.922/0.877) and inherits the physics' climate sensitivity (Climate response of the dPL chain, below).

### `lstm`: the no-physics control

`lstm` drops the physics-simulation channel entirely, keeping the PET channel and static attributes but forwarding on meteorology alone. It still reaches respectable skill (0.909/0.835): most of the hybrid's within-climate skill does not depend on the physics channel. Its climate response does. The control's +2 °C response is wrong-signed (response ratio −0.94; Climate response, below): a network trained only on historical variability has no reason to extrapolate a temperature sensitivity, and without a physics channel it has nothing to anchor one. This is the direct evidence for keeping the physics simulation as an input rather than trying to reproduce its skill from meteorology alone.

### Lineage: the frozen-noah predecessor generation

`hybrid` and `hybrid_dt` generalize an earlier pair built on the climate-frozen `noah_noca` physics, retained for lineage as `superseded/hybrid_noca` and `superseded/hybrid_dt_noca`. That pair established the architecture above and the single-anchor version of the response loss: `hybrid_noca` (eight-seed, no PET channel, no response loss) reached the same skill class as the current `hybrid` (0.917/0.869), and `hybrid_dt_noca` added the PET channel plus a single +2 °C teacher ($\lambda_T=0.3$) at unchanged skill (0.916/0.864) and a physics-consistent warming response (response ratio 1.04, monthly-regime correlation 0.97).

That generation also isolated a skill lever from a response lever. In a three-arm progression (`hybrid_noca`, an intermediate PET-input-only ensemble, and `hybrid_dt_noca`), adding the Priestley–Taylor potential as an input channel raised validation KGE to 0.872 (the best of that progression) while leaving the +2 °C response nearly flat (ratio 0.24); only the temperature-consistency loss moved the response, at a cost of about 0.008 KGE. A screening across loss weights showed the response scaling smoothly with the weight (λ of 0.1/0.3/1.0 recovering 36%/78%/92% of the physics response), which is what motivated widening the single +2 °C anchor into the full (Δp, ΔT) surface the current family trains on. The updated progression exhibit for the current chain (`noah` → `hybrid` → `hybrid_dt`) is in Appendix D.

## Out-of-calibration evaluation against CalSim3

The most relevant test for CalSim purposes is monthly climatology against CalSim3 FNF on the out-of-calibration record (WY1950–1987 plus WY2004–2018), on the CalSim3 delineations. Figure 1 summarizes the top of the lineage. The dPL models fix the pooled GA's worst CalSim3-basis failures (the SHA/BND rim under-run of Part I's Performance review largely closes on the re-footed domain), and the physics rungs hold that ground with progressively more defensible ET. The Noah rung reaches NML and MRC alongside the PT cascade (monthly KGE 0.94 and 0.93), which suggests the soil-moisture-limited ET structure itself is doing the work.

On this basis the three current models are close in the mean but differ in character. Over the eleven basins with CalSim3 coverage, mean monthly-regime KGE is 0.86 (`noah`), 0.87 (`hybrid`), and 0.88 (`hybrid_dt`, the best of the set). The basic hybrid's in-sample skill does not carry fully to this test at the good-physics basins (NML 0.72, MRC 0.83 against the physics' 0.93–0.94), which is the regime-conditional drift noted above; the response-constrained variant recovers most of that ground (NML 0.88, MRC 0.90) while improving nearly everywhere else, including a substantial repair of NHG's seasonal shape (mismatch 13% for the physics, 3–4% for both hybrids).

```{figure} ../artifacts/dpl/figures/climatology_e.png
:width: 6.2in

Figure 1. Mean-monthly regime versus CalSim3 FNF (out-of-calibration months, WY1950–1987 + WY2004–2018): `noah` → the two LSTM hybrid ensembles, by basin.
```

## Climate response of the dPL chain

Repeating Part I's temperature-detrending experiment (Warming sensitivity) on the dPL variants shows the intended progression through the physics rungs. The Hamon rung inherits the Part I volume sensitivity, the PT rung damps it, and the Noah rungs damp it further while shifting melt timing, which is the VIC-like response pattern now reproduced inside the SAC-SMA framework. (Appendix D collects the rolling and pre-1950 monthly-regime sensitivity figures.)

The hybrids do not inherit a climate response from the physics; it has to be trained in, and the diagnostic that shows this is a per-watershed (Δprecip, ΔT) response surface (`sacsma.dpl.dtdp_response`), scored on the same out-of-calibration record as above so most of the surface is genuinely held out. Four metrics are tracked per watershed as percent change from the (0, 0) present-climate baseline: total annual runoff, the Apr–Jul snowmelt freshet, and the daily 99.9th and 30th flow percentiles (the flood peak and the low flow). Warming shrinks the freshet but intensifies the flood peak, a crossover in which the same warming that reduces the seasonal snowmelt pulse increases the extreme daily tail (snow turning to rain, and rain falling on an existing pack).

Under an unconstrained +2 °C perturbation `hybrid` (no response loss) over-responds, with an aggregate warming ratio of 1.50 against the physics. The no-physics `lstm` control is wrong-signed (ratio −0.94). `hybrid_dt`'s multi-anchor response loss repairs this: its aggregate warming ratio is 1.14, right-signed at all 15 basins, at the modest skill cost noted above. Scored on the full response surface, `hybrid_dt` reproduces the freshet-shrinks/flood-peak-intensifies crossover on tails it was never directly trained on, while the unconstrained `lstm` control roughly doubles the flood-peak response.

```{figure} ../artifacts/dpl/figures/hybrid_summary.png
:width: 6.2in

Figure 2. Skill and climate-response fidelity across the current hybrid family: ensemble-mean cal/val KGE, the pooled warming-response curve, and the pooled precip-response curve, one line per model with the `noah` physics as the black reference.
```

The method has one boundary. The response loss buys a physics-consistent climate response; it does not repair the validation-period volume drift at NML/MRC/ORO discussed in The hybrid family, above. The two are distinct failure modes with distinct causes.

## Next steps

The next-phase workstreams that extend this system toward the full CalSim domain (multi-timescale calibration, nested sub-arc training, a fuller training domain, and the physics/hybrid refinement options) are tracked as open issues in the modeling repository:

<https://github.com/CADWR-Climate-Change-Program/SAC-SMA/issues>
