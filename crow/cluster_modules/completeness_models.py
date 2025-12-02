"""The cluster completeness module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

import numpy as np
import numpy.typing as npt

from .parameters import Parameters


class Completeness:
    """The completeness kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the incompleteness
    of a cluster selection.
    """

    def __init__(self):
        pass

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""
        raise NotImplementedError


REDMAPPER_DEFAULT_PARAMETERS = {
    "a_n": 0.38,
    "b_n": 1.2634,
    "a_logm_piv": 13.31,
    "b_logm_piv": 0.2025,
}


class CompletenessAguena16(Completeness):
    """The completeness kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the incompleteness
    of a cluster selection.
    """

    def __init__(
        self,
    ):
        self.parameters = Parameters({**REDMAPPER_DEFAULT_PARAMETERS})

    def _mpiv(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        log_mpiv = self.parameters["a_logm_piv"] + self.parameters["b_logm_piv"] * (
            1.0 + z
        )
        mpiv = 10.0**log_mpiv
        return mpiv.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        nc = self.parameters["a_n"] + self.parameters["b_n"] * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""

        mass_norm_pow = (10.0**log_mass / self._mpiv(z)) ** self._nc(z)

        completeness = mass_norm_pow / (mass_norm_pow + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness
