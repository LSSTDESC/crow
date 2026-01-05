"""The cluster completeness module.

This module holds the classes that define completeness kernels that can be included
in the cluster prediction integrand.
"""

import numpy as np
import numpy.typing as npt

from .parameters import Parameters


class Completeness:
    """The completeness kernel base class.

    This kernel affects the prediction integrand by accounting for the incompleteness
    of a cluster selection. Subclasses should implement the ``distribution`` method.

    Attributes
    ----------
    parameters : Parameters, optional
        Container for completeness model parameters (defined by subclasses).
    """

    def __init__(self):
        pass

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the completeness kernel contribution.

        Parameters
        ----------
        log_mass : array_like
            Array of log10 halo masses (units: Msun).
        z : array_like
            Array of redshifts matching ``log_mass``.

        Returns
        -------
        numpy.ndarray
            Array of completeness values in the range [0, 1] with the same
            broadcastable shape as the inputs. Subclasses should guarantee the
            output dtype is floating point.
        """
        raise NotImplementedError


REDMAPPER_DEFAULT_PARAMETERS = {
    "a_n": 0.38,
    "b_n": 1.2634,
    "a_logm_piv": 13.31,
    "b_logm_piv": 0.2025,
}


class CompletenessAguena16(Completeness):
    """Completeness model following Aguena et al. (2016) parametrisation.

    The model uses a pivot mass and a redshift-dependent power-law index to
    compute a sigmoid-like completeness as a function of mass and redshift.

    Parameters
    ----------
    (set during initialization)
    a_n, b_n : float
        Parameters controlling the redshift evolution of the power-law index.
    a_logm_piv, b_logm_piv : float
        Parameters controlling the pivot mass (in log10 units) and its
        redshift evolution.

    Attributes
    ----------
    parameters : Parameters
        Container holding the parameter values; defaults are defined in
        ``REDMAPPER_DEFAULT_PARAMETERS``.
    """

    def __init__(
        self,
    ):
        self.parameters = Parameters({**REDMAPPER_DEFAULT_PARAMETERS})

    def _mpiv(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return the pivot mass (not in log) at redshift `z`.

        Parameters
        ----------
        z : array_like
            Redshift or array of redshifts.

        Returns
        -------
        numpy.ndarray
            Pivot mass values in Msun (10**log_mpiv). Returned dtype is float64.
        """
        log_mpiv = self.parameters["a_logm_piv"] + self.parameters["b_logm_piv"] * (
            1.0 + z
        )
        mpiv = 10.0**log_mpiv
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
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        r"""Compute the completeness fraction for given mass and redshift.
        The completeness is given by

        .. math::
            c(M, z) = \frac{\left(M / M_{\rm piv}(z)\right)^{n_c(z)}}
                  {1 + \left(M / M_{\rm piv}(z)\right)^{n_c(z)}}

        where M = 10^{\text{log\_mass}}, M_{\rm piv}(z) is returned by _mpiv(z),
        and n_c(z) is returned by _nc(z).

        Parameters
        ----------
        log_mass : array_like
            Array of log10 halo masses (Msun).
        z : array_like
            Array of redshifts matching ``log_mass``.

        Returns
        -------
        numpy.ndarray
            Completeness values in the interval [0, 1] with shape matching the
            broadcasted inputs. dtype is float64.
        """

        mass_norm_pow = (10.0**log_mass / self._mpiv(z)) ** self._nc(z)

        completeness = mass_norm_pow / (mass_norm_pow + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness
