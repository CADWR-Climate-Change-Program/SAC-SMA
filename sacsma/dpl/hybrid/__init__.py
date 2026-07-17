"""Hybrid SAC-SMA x LSTM on the 15cdec daily basis.

An LSTM is coupled to the FROZEN SAC-SMA daily simulation as an extra input
feature; the net predicts streamflow directly (Softplus head).  A second
"residual" coupling (net predicts ``obs - sim``, flow = sim + correction) was
retired 2026-07-16: it re-injected regime-conditional volume bias on every
physics baseline (RUNS.md, Track B + B3).

Scored through ``metrics.kge`` / ``_figures._period_stats`` with the temporal
split at :data:`sacsma.cdec15.CAL_END`, so the numbers are directly comparable
to the GA (``metrics_15cdec.csv``) and dPL (``metrics_dpl_*.csv``) tables.  The
physics baseline is the frozen ``run_basin`` sim from a REQUIRED, explicitly
named parameter table (a canonical dPL export or GA) — or, for torch-only
physics (e.g. the canonical noah TORCH daily run), its ``daily_sim_*.csv``
dump via ``--sim-cache``.

Everything here imports torch at module scope — import it only from the CLI
handlers (lazily), never from the torch-free core package paths.
"""
