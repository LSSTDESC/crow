"""The mass richness kernel module.

This module holds the classes that define the mass richness relations
that can be included in the cluster abundance integrand.  These are
implementations of Kernels.
"""

from .costanzi19 import CostanziModel
from .murata import MurataBinned, MurataUnbinned
