"""Sphinx configuration for smfeval.

Build with::

    nix develop --command uv run --group docs sphinx-build -b html docs docs/_build/html
"""

import os
import sys
from datetime import datetime
from importlib.metadata import version as _pkg_version

# Make the `smfeval` package importable for autodoc.
sys.path.insert(0, os.path.abspath(".."))

project = "smfeval"
author = "Ola Rønning"
copyright = f"{datetime.now():%Y}, {author}"
release = _pkg_version("smfeval")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
# the codebase uses Google-style docstrings (ruff pydocstyle convention)
# with occasional NumPy-style sections in older modules; accept both.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"smfeval {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
# Force a single light theme: these docs are math-heavy and were not designed
# for dark mode (the toggle is hidden in custom.css; the JS pins data-theme).
html_js_files = ["force-light.js"]

# A calm light palette and a roomier reading column. Dark variables mirror the
# light ones so a system-dark visitor still gets the readable light rendering.
_palette = {
    "color-brand-primary": "#15608d",
    "color-brand-content": "#15608d",
    "color-foreground-primary": "#1c1c1e",
    "color-foreground-secondary": "#55555b",
    "color-background-primary": "#ffffff",
    "color-background-secondary": "#f6f6f4",
    "color-admonition-background": "#f6f6f4",
}
html_theme_options = {
    "light_css_variables": _palette,
    "dark_css_variables": _palette,
}
