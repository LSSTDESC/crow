"""The cluster purity module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from .parameters import Parameters


class Purity:
    """Purity kernel base class.

    This kernel represents the probability that a detected object in the
    cluster catalogue is a true galaxy cluster (i.e. purity). Subclasses should
    implement the ``distribution`` method.

    Notes
    -----
    Mass-proxy inputs to methods in this module are provided as log10 values
    of the observable mass proxy.

    Attributes
    ----------
    parameters : Parameters, optional
        Container for model parameters defined by subclasses.
    """

    def __init__(self):
        pass

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the purity kernel contribution.

        Parameters
        ----------
        z : array_like
            Redshift or array of redshifts for the objects.
        log_mass_proxy : array_like
            Array of log10 mass-proxy values corresponding to the objects.

        Returns
        -------
        numpy.ndarray
            Array with purity values in the interval [0, 1]. The output shape
            will follow NumPy broadcasting rules applied to the inputs.
        """
        raise NotImplementedError


REDMAPPER_DEFAULT_PARAMETERS = {
    "a_n": 3.9193,
    "b_n": -0.3323,
    "a_logm_piv": 1.1839,
    "b_logm_piv": -0.4077,
}


class PurityAguena16(Purity):
    """Purity model following Aguena et al. (2016) parametrisation.

    The model computes a sigmoid-like purity as a function of a mass proxy
    and redshift using a pivot mass and a redshift-dependent power-law index.

    Important
    ---------
    Note that the base class `Purity.distribution` defines the argument order
    as ``(z, log_mass_proxy)`` while this subclass implements
    ``distribution(self, log_mass_proxy, z)``. Callers should use the
    subclass' signature when invoking the concrete implementation.

    Attributes
    ----------
    parameters : Parameters
        Container holding default parameters defined in
        ``REDMAPPER_DEFAULT_PARAMETERS``.
    """

    def __init__(self):
        super().__init__()
        self.parameters = Parameters({**REDMAPPER_DEFAULT_PARAMETERS})

    def _mpiv(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the pivot mass.

        Parameters
        ----------
        z : array_like
            Redshift or array of redshifts.

        Returns
        -------
        numpy.ndarray
            Pivot mass values in Msun (10**log_mpiv), dtype float64.
        """
        log_mpiv = self.parameters["a_logm_piv"] + self.parameters["b_logm_piv"] * (
            1.0 + z
        )
        mpiv = 10**log_mpiv
        return mpiv.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the redshift-dependent power-law index nc(z).

        Parameters
        ----------
        z : array_like
            Redshift or array of redshifts.

        Returns
        -------
        numpy.ndarray
            The value of the power-law index at each provided redshift.
        """
        nc = self.parameters["a_n"] + self.parameters["b_n"] * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        log_mass_proxy: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute the purity fraction for given mass-proxy and redshift.
        The purity is given by the sigmoid-like expression

            p(M, z) = ( (M / M_piv(z))**(n_c(z)) ) / ( 1 + (M / M_piv(z))**(n_c(z)) ),

        where M = 10**(log_mass_proxy), M_piv(z) = 10**(a_logm_piv + b_logm_piv*(1 + z))
        and n_c(z) = a_n + b_n*(1 + z).

        Parameters
        ----------
        log_mass_proxy : array_like
            Array of log10 mass-proxy values.
        z : array_like
            Array of redshifts matching ``log_mass_proxy``.

        Returns
        -------
        numpy.ndarray
            Purity values in the interval [0, 1] with shape matching the
            broadcasted inputs. dtype is float64.
        """

        rich_norm_pow = (10**log_mass_proxy / self._mpiv(z)) ** self._nc(z)

        purity = rich_norm_pow / (rich_norm_pow + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity
