"""Enable ``python -m sacsma ...`` -> the :mod:`sacsma.cli` entry point.

So ``python -m sacsma calsim --parallel`` and ``python -m sacsma run BND --parallel``
work the same as the ``sacsma`` console script.
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
