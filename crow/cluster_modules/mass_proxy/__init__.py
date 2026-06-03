"""The mass richness kernel module.

This module holds the classes that define the mass richness relations
that can be included in the cluster abundance integrand.  These are
implementations of Kernels.
"""

from .murata import MurataBinned, MurataUnbinned
