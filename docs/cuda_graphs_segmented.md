# CUDA graphs: segmented capture (`--nograd-window`, `--train-graph-segments`)

## Why

`sacsma dpl train` runs the day-stepped SAC-SMA pipeline as **CUDA graphs**:
the ~300 tiny GPU operations per simulated day are recorded once and replayed,
which removes the per-operation launch cost that otherwise dominates (eager
execution is ~100x slower: 8,730 s vs 77 s per `15cdec_grid` epoch).

Some GPU/driver combinations cannot replay very large recordings. On the HP Z6
workstation (RTX A4500, WDDM, driver 596.86) a recording longer than roughly
256 simulated days makes the driver raise a GPU-level *Graphics FECS Exception*
and the process dies with an access violation; before dying, replays can return
wrong numbers. The two recordings the trainer makes were both over that limit:
the 512-day no-grad window and the 366-day forward+backward training chunk.
PyTorch/CUDA versions, cuDNN, and the model code were all ruled out; smaller
recordings replay fine.

The two new knobs make the recordings smaller **without changing the model or
the training recipe**.

## Usage

```bash
# machine where full-size recordings fault (HP Z6 / RTX A4500):
python -m sacsma dpl train ... --nograd-window 256 --train-graph-segments 2

# any machine where full-size recordings work: no flags (defaults = old behaviour)
```

| flag | default | controls | numerics |
|---|---|---|---|
| `--nograd-window N` | 512 | days per no-grad recording (spinup + selection streams) | neutral |
| `--train-graph-segments N` | 1 | recordings per training chunk (2 -> 183 + 183 days) | neutral (float32 summation order) |
| `--train-chunk-days N` | 366 | TBPTT chunk = gradient horizon and steps per epoch | **changes training - leave at 366** |
| `--no-graphs` | off | everything eager (fallback; ~100x slower) | neutral |

If a capture fails the trainer prints `... capture failed ...; eager` and
continues eagerly (correct, slow).

## Recording length vs gradient length

The number 183 appears in two very different places. Do not confuse them.

```
shortcut  (--train-chunk-days 183):   [183-day chunk]|cut|[183-day chunk]|cut|...   gradient stops at every bar
                                        step          step                           (recipe CHANGES)

fix       (--train-graph-segments 2): [ 183-day recording + 183-day recording ]|cut|...  one 366-day chunk,
                                      \_________ gradient flows across ________/          one step  (recipe UNCHANGED)
```

* **Chunk length** (`--train-chunk-days`) is how far back in time learning may
  look: at a chunk boundary the carried state is detached (truncated BPTT) and
  an optimizer step is taken. Halving it cuts the snow-accumulation -> spring-melt
  chain and doubles the steps per epoch: after 2 epochs, 0.468 vs 0.407 KGE.
* **Recording length** (`--train-graph-segments`) is only how much of the
  computation the GPU executes per replay. Segment B hands segment A the
  *gradient* of the day-183 state, so the backward pass still spans all 366 days.

The no-grad window is simpler still: those streams have no gradient at all, so
a 256-day window just means twice as many, half-as-long replays over the same
days in the same order.

## What runs how (one 366-day chunk, 2 segments)

```
 1. eager     parameter net (features -> per-cell parameters), routing unit hydrograph      ~ms
 2. recording forward graph A, days 1-183   : carried state -> flow A, state@183
 3. recording forward graph B, days 184-366 : state@183    -> flow B, state@366             ~1.4 s
 4. eager     loss on the concatenated 366-day flow (daily NNSE + monthly NNSE + log/var)   ~ms
 5. recording backward graph B -> parameter grads + d(loss)/d(state@183)
 6. recording backward graph A, seeded with that state gradient -> remaining grads          ~1.4 s
 7. eager     grads reach the net -> clip -> one AdamW step; state@366 detached (the only cut)
```

Only the day-stepped physics is recorded (`graphs.SegmentedTrainWindow`, built
on `torch.cuda.make_graphed_callables`: one forward and one backward graph per
segment, one shared memory pool). The parameter net, routing and loss are a
handful of operations each and stay eager, which also removes the single-graph
design's static `.grad` buffers, static loss-target buffers and in-graph RNG.
The multi-timescale domain's short final chunk (the envelope ends at the forcing
record) takes the plain eager path, as before.

Per epoch: spinup and selection = replays of the 256-day no-grad recording;
training = steps 1-7 per chunk (70 chunks on `dpl_entities`, 15 on the 15-basin
domains).

## Verification (same seed, this machine)

| check | eager | `--nograd-window 256 --train-graph-segments 2` |
|---|---|---|
| `15cdec_grid` noah: epoch-0 loss / sel KGE | 6.2916 / 0.3585 | 6.2916 / 0.3585 |
| `15cdec_grid` noah: epoch-1 loss, final sel KGE | 5.9896, 0.4070 | 5.9896, 0.4070 |
| `dpl_entities` slice: epoch-0 / final sel KGE | 0.1703 / 0.6843 | 0.1703 / 0.6843 (per-family identical) |
| trained weights after 2 epochs | - | max diff 3e-8 on values ~4000 |
| one chunk: flow, loss | - | bit-identical (0.0) |
| one chunk: parameter gradients | - | rel. diff <= 1.5e-7, cosine 1.000000000 |
| time per epoch (`15cdec_grid` / entities) | 8,730 s / ~2,000 s | 77 s / ~220 s |

Reproduce the log check: run the same command twice with `--epochs 2`, once with
`--no-graphs`, once with the two flags; `train_log.csv` `loss` and `cal_kge`
match to 4 dp. Over a full 60-epoch run the two paths drift at the level of
run-to-run float32 noise (a few 1e-3 KGE), as any two executions with different
summation order do; the optimisation problem is identical.

## Files changed

* `sacsma/dpl/graphs.py` - `_WindowBase._pingpong` copies state values under
  `torch.no_grad()` (fixes a latent "backward through the graph a second time"
  crash in the multi-timescale eager tail chunk); new `SegmentedTrainWindow`.
* `sacsma/dpl/train.py` - builds the segmented window when
  `train_graph_segments > 1`; the chunk loop's eager branch swaps `run_window`
  for `seg_g.forward`. ET/SWE auxiliary losses are rejected with segments > 1
  (not wired).
* `sacsma/dpl/config.py` - `train_graph_segments` (default 1); `nograd_window`
  validated.
* `sacsma/cli.py` - the two flags.

Defaults reproduce the previous single-graph behaviour exactly.
