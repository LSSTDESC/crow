Generating the Crow documentation (package-local docs)
=====================================================

This README explains how to generate documentation for the `crow` package
itself. It mirrors the top-level `docs/README.rst` but sets paths relative to
this `crow/docs/` directory.

Overview
--------
Sphinx builds the API docs by importing the `crow` package in this repository
and pulling docstrings from the Python source files under `crow/` (for
example: `crow/cluster_modules/shear_profile.py`, `crow/cluster_modules/parameters.py`,
`crow/cluster_modules/purity_models.py`, etc.). The `sphinx-apidoc` step scans
these modules and writes reST files that call ``.. automodule::`` for each
module; Sphinx then imports them and extracts the docstrings.

Quick steps (run from the repository root)
-----------------------------------------
1. Generate module rst files into this folder (overwrite existing):

   sphinx-apidoc -o crow/docs/ crow -f

   This will create a set of reST files in `crow/docs/` referencing the
   `crow` package modules.

2. Build the HTML docs for the package (from repo root):

   sphinx-build -b html crow/docs/ crow/docs/_build/html

3. Serve the built docs locally to inspect them::

   python -m http.server --directory crow/docs/_build/html 8000

Notes
-----
- Ensure `crow/docs/conf.py` is configured to add the repository root to
  ``sys.path`` so that the `crow` package is importable by Sphinx.
- If heavy external dependencies break imports during the build, add them to
  ``autodoc_mock_imports`` in `crow/docs/conf.py` or install the real deps.
- Re-run ``sphinx-apidoc -f`` after adding new modules so rst stubs are
  regenerated.

If you want, I can generate the apidoc files for you (write the generated
rst files) and run the build — tell me to proceed and confirm you want the
package-local `crow/docs` built.
