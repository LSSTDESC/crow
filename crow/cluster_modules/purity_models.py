"""The cluster purity module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from .parameters import Parameters


class Purity:
    """The purity kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the inpurity
    of a cluster selection.
    """

    def __init__(self):
        pass

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
        log_mass_proxy_limits: Optional[tuple[float, float]] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        raise NotImplementedError


REDMAPPER_DEFAULT_PARAMETERS = {
    "a_n": 3.9193,
    "b_n": -0.3323,
    "a_logm_piv": 1.1839,
    "b_logm_piv": -0.4077,
}


class PurityAguena16(Purity):
    """The purity kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the purity
    of a cluster selection.
    """

    def __init__(self):
        super().__init__()
        self.parameters = Parameters({**REDMAPPER_DEFAULT_PARAMETERS})

    def _mpiv(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        log_mpiv = self.parameters["a_logm_piv"] + self.parameters["b_logm_piv"] * (
            1.0 + z
        )
        mpiv = 10**log_mpiv
        return mpiv.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        nc = self.parameters["a_n"] + self.parameters["b_n"] * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
        log_mass_proxy_limits: Optional[tuple[float, float]] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        _log_mass_proxy = log_mass_proxy
        if np.all(log_mass_proxy == -1.0):
            if log_mass_proxy_limits is None:
                raise ValueError(
                    "log_mass_proxy_limits must be provided when log_mass_proxy == -1"
                )
            _log_mass_proxy = (log_mass_proxy_limits[0] + log_mass_proxy_limits[1]) / 2

        rich_norm_pow = (10**_log_mass_proxy / self._mpiv(z)) ** self._nc(z)

        purity = rich_norm_pow / (rich_norm_pow + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity
