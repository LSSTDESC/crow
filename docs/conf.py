import os
import sys
from unittest.mock import MagicMock
import subprocess

# ----------------------------------------------------------------------
# Path setup (IMPORTANT)
# ----------------------------------------------------------------------
# Add repo root so Sphinx can import `crow`
sys.path.insert(0, os.path.abspath("../crow"))
sys.path.insert(0, os.path.abspath(".."))
# ----------------------------------------------------------------------
# Mock heavy / optional dependencies (EDIT if needed)
# ----------------------------------------------------------------------
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()
for mod in [
    "pyccl",
    "pyccl.cosmology",
    "pyccl.background",
    "pyccl.halos",
    "pyccl.halos.hbias",
    "pyccl.halos.hmfunc",
    "pyccl.halos.concentration",
    "numcosmo",
    "numcosmo_py",
    "clmm",
    "clmm.utils",
    "clmm.utils.beta_lens"
]:
    sys.modules[mod] = Mock()



# --------------Run the makefile documentation--------
subprocess.run(
    [
        "sphinx-apidoc",
        "--separate",
        "--no-toc",
        "-f",
        "-M",
        "-o",
        "api",
        "../crow",
    ],
    cwd=os.path.dirname(os.path.abspath(__file__)),
)
#---------------------------------------------------




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
    "sphinx.ext.githubpages",
    "IPython.sphinxext.ipython_console_highlighting",
]
apidoc_module_dir = "../crow"
templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
master_doc = "index"

project = "Crow"
copyright = "2026, LSST DESC CROW Contributors"
author = "Eduardo Barroso, Michel Aguena"
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "source/index_body.rst",
]

highlight_language = "python3"
pygments_style = "sphinx"
todo_include_todos = True
add_function_parentheses = True
add_module_names = True
smartquotes = False

# -- HTML output ---------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "prev_next_buttons_location": None,
    "collapse_navigation": False,
    "titles_only": True,
}
html_static_path = []

# -- Napoleon ------------------------------------------------------------
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_ivar = True

# -- Load from config file -----------------------------------------------
config = open("doc-config.ini").read().strip().split("\n")
doc_files = {
    "APIDOC": [],
    "DEMO": [],
    "EXAMPLE": [],
    "OTHER": [],
}
key = None
for entry in config:
    if not entry or entry[0] == "#":
        continue
    elif entry in doc_files:
        key = entry
    else:
        doc_files[key].append(entry)

# -- Compile notebooks into rst ------------------------------------------
#run_nb = False  # Set True to execute notebooks during build

outdir = "compiled-examples/"
#nbconvert_opts = [
#    "--to rst",
#    "--ExecutePreprocessor.kernel_name=python3",
#    "--execute",
#    f"--output-dir {outdir}",
#]

#for lists in [v for k, v in doc_files.items() if k != "APIDOC"]:
#    for demo in lists:
#        com = " ".join(["jupyter nbconvert"] + nbconvert_opts + [demo])
#        if not run_nb:
#            com = com.replace(" --execute ", " ")
#        subprocess.run(com, shell=True)

# -- Build index.rst -----------------------------------------------------
doc_captions = {
    "DEMO": "Usage Demos",
    "EXAMPLE": "Examples",
    "OTHER": "Other",
}
index_toc = ""
#for CASE in ("DEMO", "EXAMPLE", "OTHER"):
#    if not doc_files[CASE]:
#        continue
#    index_toc += f"""
#.. toctree::
#   :maxdepth: 1
#   :caption: {doc_captions[CASE]}
#
#"""
#    for example in doc_files[CASE]:
#        fname = "".join(example.split(".")[:-1]).split("/")[-1] + ".rst"
#        index_toc += f"   {outdir}{fname}\n"

subprocess.run("cp source/index_body.rst index.rst", shell=True)
with open("index.rst", "a") as indexfile:
    indexfile.write(index_toc)
    indexfile.write(
        """
.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
   api/crow

.. toctree::
   :maxdepth: 1

   README
"""
    )

# -- API table of contents -----------------------------------------------
apitoc = """API Documentation
-----------------

Information on specific functions, classes, and methods.

.. toctree::
   :glob:

"""
for onemodule in doc_files["APIDOC"]:
    apitoc += f"   api/crow.{onemodule}.rst\n"
with open("api.rst", "w") as apitocfile:
    apitocfile.write(apitoc)
