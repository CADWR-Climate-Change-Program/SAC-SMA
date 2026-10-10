"""Model-state spinup for a simulated window: timing-independent cycling or the
preceding window.

``cycle`` (timing-independent; the multifamily evaluators' default, run by
:func:`sacsma.engine.simulate`): the state at the window start is the fixed point of
the window's OWN first ``spinup_years`` years — that block is streamed again and
again, each pass starting from the previous pass's end state.  Nothing before the
window is read, so the rule is the same for the historical record, a
climate-perturbed copy of it and a stochastic trace, and the state does not depend
on where the window sits in the record.  A block of whole years loops seamlessly
(its last day is the calendar day before its first).

The loop starts from the frozen cold start (SMA ``[0, 0, 100, 100, 100, 0]``, snow and
routing history empty — the state every run trains from) and runs
a FIXED number of passes, ``spinup_passes`` (default 20).  Fixed, because a stopping
rule misleads here: the change between passes understates the remaining distance to
the fixed point 10-20x, and any rule evaluated on some set of rows makes a cell's
state depend on which rows a caller watches.  A fixed count gives every cell the
same state in every caller and domain holding it.

Trained multifamily fields put the lower-zone tension capacity at its 5000 mm
ceiling, and under Noah-lite that store is drawn by ET only above a 5 % wilting
fraction, so it can take centuries to settle, filling in some cells and emptying in
others.  No start is near the fixed point everywhere: from full storages a riva ~
0.9 field ran twice its equilibrium block volume on the first pass and was still 10 %
off after 20 passes (its stores empty), while a riva = 0 field reached 1e-3 in 16
passes from full storages against ~65 from the cold start (its stores fill).  From
the cold start both were within 0.5 % in entity block volume after 20 passes, and
the riva = 0 field's WY1950-84 tier-1 scores matched the full-start ones (0.857 vs
0.858 mean KGE).  A few cells never settle from either start (lztwc 4900 mm apart
after 100 passes) while moving the entity flow little.  The largest annual flow
change of the last pass is reported as a diagnostic.

``window`` (the trainer's spinup and its CPU selection): stream the
``DplConfig.spinup_start`` window ahead of the start — on the multifamily envelope
the ten water years before it.  It needs forcing before the window, and trained
fields whose storages fill over decades are not at equilibrium after ten years (a
riva = 0 multifamily field was +53 % in WY1950 volume at UF13 against a 1915
spinup).
"""

from __future__ import annotations

import pandas as pd


def window_start(dates: pd.DatetimeIndex, t0: int, spinup_start: str) -> int:
    """The ``window`` spinup start for a window beginning at ``t0``:
    ``spinup_start`` when it lies before the window, else the ten years before
    it; clamped to the record start."""
    spin = int(dates.searchsorted(pd.Timestamp(spinup_start)))
    if spin >= t0:
        spin = int(dates.searchsorted(dates[t0] - pd.DateOffset(years=10)))
    return max(min(spin, t0), 0)
