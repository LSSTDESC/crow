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
    "ac_nc": 0.38,
    "bc_nc": 1.2634,
    "ac_mc": 13.31,
    "bc_mc": 0.2025,
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

    def _mc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_mc = self.parameters["ac_mc"]
        bc_mc = self.parameters["bc_mc"]
        log_mc = ac_mc + bc_mc * (1.0 + z)
        mc = 10.0**log_mc
        return mc.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_nc = self.parameters["ac_nc"]
        bc_nc = self.parameters["bc_nc"]
        nc = ac_nc + bc_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""
        mass = 10.0**log_mass

        mass_norm_pow = (mass / self._mc(z)) ** self._nc(z)

        completeness = mass_norm_pow / (mass_norm_pow + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness
