"""Sphinx configuration for SegSpy documentation."""

import os
import sys

# Add src/ to path so autodoc can import the package.
sys.path.insert(0, os.path.abspath("../src"))

project = "SegSpy"
copyright = "2026, Fan Zhang"
author = "Fan Zhang"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "SegSpy Documentation"
html_short_title = "SegSpy"

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autoclass_content = "both"

# Napoleon (Google/NumPy docstring support)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True

# Intersphinx — all three inventories verified reachable.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current/", None),
}

# MyST parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
