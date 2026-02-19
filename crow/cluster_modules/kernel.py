"""The cluster kernel module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

import numpy as np
import numpy.typing as npt


class TrueMass:
    """True-mass kernel used in abundance integrals.

    This kernel represents the case where the observed mass equals the true
    halo mass. It therefore contributes a multiplicative factor of unity to
    the abundance integrand and does not alter the mass distribution.

    Notes
    -----
    The distribution method returns a NumPy array (dtype float) that can be
    broadcast with other integrand factors.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluate and return the mass kernel contribution.

        Returns
        -------
        numpy.ndarray
            Array containing the value 1.0. This can be broadcast to the shape
            required by the integrand and is provided as float64.
        """
        return np.atleast_1d(1.0)


class SpectroscopicRedshift:
    """Spectroscopic-redshift kernel for abundance integrals.

    Represents the idealized case where cluster redshifts are known exactly
    (spectroscopic precision). The kernel thus contributes a factor of unity
    to the redshift part of the integrand.
    """

    def distribution(self) -> npt.NDArray[np.float64]:
        """Evaluate and return the redshift kernel contribution.

        Returns
        -------
        numpy.ndarray
            Array containing the value 1.0 (dtype float64). This is intended
            to be broadcast with other integrand arrays.
        """
        return np.atleast_1d(1.0)
