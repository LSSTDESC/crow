# Sphinx configuration for the Crow documentation
import os
import sys

# Add the repo root to sys.path so autodoc can import the crow package
sys.path.insert(0, os.path.abspath(".."))

project = "Crow"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
napoleon_numpy_docstring = True

# Prevent heavy/optional imports from breaking the build
autodoc_mock_imports = [
    "clmm",
    "pyccl",
    "numcosmo",
    "scipy",
    "crow.integrator.numcosmo_integrator",
]

html_theme = "sphinx_rtd_theme"

# Basic HTML options
html_static_path = ["_static"]
