"""The cluster completeness module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

import numpy as np
import numpy.typing as npt


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


REDMAPPER_DEFAULT_AC_NC = 0.38
REDMAPPER_DEFAULT_BC_NC = 1.2634
REDMAPPER_DEFAULT_AC_MC = 13.31
REDMAPPER_DEFAULT_BC_MC = 0.2025


class CompletenessAguena16(Completeness):
    """The completeness kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the incompleteness
    of a cluster selection.
    """

    def __init__(
        self,
    ):
        self.ac_nc = REDMAPPER_DEFAULT_AC_NC
        self.bc_nc = REDMAPPER_DEFAULT_BC_NC
        self.ac_mc = REDMAPPER_DEFAULT_AC_MC
        self.bc_mc = REDMAPPER_DEFAULT_BC_MC

    def _mc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_mc = self.ac_mc
        bc_mc = self.bc_mc
        log_mc = ac_mc + bc_mc * (1.0 + z)
        mc = 10.0**log_mc
        return mc.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ac_nc = self.ac_nc
        bc_nc = self.bc_nc
        nc = ac_nc + bc_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""
        mc = self._mc(z)
        mass = 10.0**log_mass
        nc = self._nc(z)
        completeness = (mass / mc) ** nc / ((mass / mc) ** nc + 1.0)
        assert isinstance(completeness, np.ndarray)
        return completeness
