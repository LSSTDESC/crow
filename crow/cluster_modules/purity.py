"""The cluster purity module.

This module holds the classes that define the kernels that can be included
in the cluster abundance integrand.
"""

import numpy as np
import numpy.typing as npt


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
        mass_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Optional[tuple[float, float]] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        raise NotImplementedError


REDMAPPER_DEFAULT_AP_NC = 3.9193
REDMAPPER_DEFAULT_BP_NC = -0.3323
REDMAPPER_DEFAULT_AP_RC = 1.1839
REDMAPPER_DEFAULT_BP_RC = -0.4077


class PurityAguena16(Purity):
    """The purity kernel for the numcosmo simulated survey.

    This kernel will affect the integrand by accounting for the purity
    of a cluster selection.
    """

    def __init__(self):
        super().__init__()
        self.ap_nc = REDMAPPER_DEFAULT_AP_NC
        self.bp_nc = REDMAPPER_DEFAULT_BP_NC
        self.ap_rc = REDMAPPER_DEFAULT_AP_RC
        self.bp_rc = REDMAPPER_DEFAULT_BP_RC

    def _rc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ap_rc = self.ap_rc
        bp_rc = self.bp_rc
        log_rc = ap_rc + bp_rc * (1.0 + z)
        rc = 10**log_rc
        return rc.astype(np.float64)

    def _nc(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        bp_nc = self.bp_nc
        ap_nc = self.ap_nc
        nc = ap_nc + bp_nc * (1.0 + z)
        assert isinstance(nc, np.ndarray)
        return nc

    def distribution(
        self,
        z: npt.NDArray[np.float64],
        mass_proxy: npt.NDArray[np.float64],
        mass_proxy_limits: Optional[tuple[float, float]] = None,
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        if all(mass_proxy == -1.0):
            if mass_proxy_limits is None:
                raise ValueError(
                    "mass_proxy_limits must be provided when mass_proxy == -1"
                )
            mean_mass = (mass_proxy_limits[0] + mass_proxy_limits[1]) / 2
            r = np.array([np.power(10.0, mean_mass)], dtype=np.float64)
        else:
            r = np.array([np.power(10.0, mass_proxy)], dtype=np.float64)

        r_over_rc = r / self._rc(z)

        purity = (r_over_rc) ** self._nc(z) / (r_over_rc ** self._nc(z) + 1.0)
        assert isinstance(purity, np.ndarray)
        return purity
