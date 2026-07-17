"""Sphinx configuration for the SAC-SMA / CalSim Stochastic Hydrology docs.

The narrative pages (``index``, ``part1``, ``part2``, ``appendix_*``) are MyST
markdown ported from the pandoc report sources; figures resolve to the repo's
``artifacts/`` tree one level up. Built to HTML and deployed to GitHub Pages by
``.github/workflows/docs.yml``.
"""

project = "SAC-SMA for CalSim Stochastic Hydrology"
author = "California Department of Water Resources"
copyright = "2026, California Department of Water Resources"
release = "July 2026 (DRAFT)"

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinx.ext.mathjax",
]

# MyST: dollar/display math (+ amsmath environments), and ::: fenced blocks.
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

# Bibliography (sphinxcontrib-bibtex); author-year rendering to match the report.
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "alpha"

# Figures carry their own manual "Figure N." / "B.x.y" labels from the report,
# so Sphinx auto-numbering stays off (no double numbering).
numfig = False

root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

html_theme = "furo"
html_title = "SAC-SMA for CalSim Stochastic Hydrology"
html_short_title = "SAC-SMA / CalSim hydrology"

# The report references figures via ../artifacts/...; images outside the source
# dir are copied into _images at build time (relative paths resolve from each
# page's directory). Nothing else in the repo is part of the docs source.
