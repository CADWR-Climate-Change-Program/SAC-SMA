"""Hybrid SAC-SMA x LSTM on the 15cdec daily basis.

An LSTM is coupled to the daily flow of a learned SAC-SMA run as an extra input feature; the
net predicts streamflow directly (Softplus head).

Scored through ``metrics.kge`` / ``_figures._period_stats`` with the temporal
split at :data:`sacsma.cdec15.CAL_END`, so the numbers are directly comparable
to the score tables of the calibrated model and the dPL runs (``metrics.csv``).  The physics
is a 15-CDEC learned run named in the configuration (``--physics noah``), run as trained on the
engine (:func:`sacsma.dpl.evaluate.basin_daily`).

Everything here imports torch at module scope — import it only from the CLI
handlers (lazily), never from the torch-free core package paths.
"""
