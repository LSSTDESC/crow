import os
import sys
from unittest.mock import MagicMock

# ----------------------------------------------------------------------
# Path setup (IMPORTANT)
# ----------------------------------------------------------------------
# Add repo root so Sphinx can import `crow`
sys.path.insert(0, os.path.abspath(".."))
# ----------------------------------------------------------------------
# Mock heavy / optional dependencies (EDIT if needed)
# ----------------------------------------------------------------------


MOCK_MODULES = [
"numcosmo",
"numcosmo_py"
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()

# ----------------------------------------------------------------------
# Project import (used for version)
# ----------------------------------------------------------------------
import crow

# ----------------------------------------------------------------------
# General configuration
# ----------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

autosummary_generate = True

templates_path = ["_templates"]
source_suffix = [".rst"]

master_doc = "index"

project = "crow"
author = "Eduardo Barroso, Michel Aguena"
copyright = "2026"
language = "en"

# ----------------------------------------------------------------------
# Versioning
# ----------------------------------------------------------------------
version = getattr(crow, "__version__", "0.0.0")
release = version

# ----------------------------------------------------------------------
# Exclusions
# ----------------------------------------------------------------------
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# ----------------------------------------------------------------------
# HTML output
# ----------------------------------------------------------------------
html_theme = "alabaster"  # or "sphinx_rtd_theme" if installed
html_static_path = []

# ----------------------------------------------------------------------
# Napoleon (docstring parsing)
# ----------------------------------------------------------------------
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# ----------------------------------------------------------------------
# Autodoc options
# ----------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
