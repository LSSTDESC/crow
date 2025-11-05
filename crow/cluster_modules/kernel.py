"""The cluster kernel module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

import numpy as np
import numpy.typing as npt


class TrueMass:
    """The true mass kernel.

    Assuming we measure the true mass, this will always be 1.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass distribution contribution to the integrand.

        We have set this to 1.0 (i.e. it does not affect the mass distribution)
        """
        return np.atleast_1d(1.0)


class SpectroscopicRedshift:
    """The spec-z kernel.

    Assuming the spectroscopic redshift has no uncertainties, this is akin to
    multiplying by 1.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluates and returns the z distribution contribution to the integrand.

        We have set this to 1.0 (i.e. it does not affect the redshift distribution)
        """
        return np.atleast_1d(1.0)
