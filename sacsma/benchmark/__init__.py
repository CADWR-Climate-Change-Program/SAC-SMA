"""The model benchmark at the CDEC full-natural-flow watersheds (``sacsma benchmark``).

Three models' monthly flow at 12 CDEC FNF sites of the multifamily registry, scored against
the observed CDEC full natural flow over WY1991-2018, all on WGEN Product A scenario 1:
dPL-CalSim and the CalSim3 pipeline's VIC (:mod:`.flows`), and BCM Scenario 1 routed by its
monthly post-processing (:mod:`.gridded`, :mod:`.bcm_routing`).  :func:`.report.make_all`
writes the tables and figures to ``artifacts/results/benchmark/``.  Only the command line and
``sacsma verify`` import it.
"""
