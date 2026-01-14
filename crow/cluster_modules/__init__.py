"""Cluster modules package.

Keep this module lightweight: do not import the submodules or classes eagerly
here because several submodules import the top-level ``crow`` package and that
can produce circular imports when the top-level ``crow.__init__`` imports
things from ``cluster_modules``.

To import classes or modules use the explicit submodule path, for example::

    from crow.cluster_modules.abundance import ClusterAbundance
    from crow.cluster_modules.shear_profile import ClusterShearProfile

Sphinx/apidoc will still find and document the submodules even if they are not
imported here; this file only needs to exist so Python treats the directory as
a package.
"""

# Expose the expected subpackage/module names for convenience. Do NOT import
# the modules at package import time to avoid circular import problems.
__all__ = [
    "abundance",
    "completeness_models",
    "kernel",
    "parameters",
    "purity_models",
    "_clmm_patches",
    "shear_profile",
    "shear_profile_parallel",
    "mass_proxy",
]
