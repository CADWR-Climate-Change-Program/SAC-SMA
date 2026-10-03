"""Hybrid SAC-SMA x LSTM on the 15cdec daily basis.

An LSTM is coupled to the FROZEN SAC-SMA daily simulation as an extra input
feature; the net predicts streamflow directly (Softplus head).

Scored through ``metrics.kge`` / ``_figures._period_stats`` with the temporal
split at :data:`sacsma.cdec15.CAL_END`, so the numbers are directly comparable
to the score tables of the calibrated model and the dPL runs (``metrics.csv``).  The
physics baseline is the frozen ``run_basin`` sim from a REQUIRED, explicitly
named parameter table (a canonical dPL export or GA) — or, for torch-only
physics (e.g. a noah TORCH daily run), its ``sim_daily.csv`` via ``--sim-cache``.

Everything here imports torch at module scope — import it only from the CLI
handlers (lazily), never from the torch-free core package paths.
"""
