"""Hybrid SAC-SMA x LSTM on the 15cdec daily basis.

An LSTM is coupled to the FROZEN SAC-SMA daily simulation two ways (variants):

* ``feature``  — the SAC-SMA sim is an extra LSTM input feature; the net
  predicts streamflow directly (Softplus head).
* ``residual`` — the net predicts the SAC-SMA error ``obs - sim`` (linear,
  signed head); the reported flow is ``sim + correction``.

Both are scored through ``metrics.kge`` / ``_figures._period_stats`` with the
temporal split at :data:`sacsma.cdec15.CAL_END`, so the numbers are directly
comparable to the GA (``metrics_15cdec.csv``) and dPL (``metrics_dpl_*.csv``)
tables.  The physics baseline is the frozen ``run_basin`` sim from a REQUIRED,
explicitly named parameter table (a canonical dPL export or GA).

Everything here imports torch at module scope — import it only from the CLI
handlers (lazily), never from the torch-free core package paths.
"""
