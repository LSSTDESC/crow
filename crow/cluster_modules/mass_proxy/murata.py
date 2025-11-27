"""The Murata et al. 19 mass richness kernel models."""

import numpy as np
import numpy.typing as npt

from ..parameters import Parameters
from .gaussian_protocol import MassRichnessGaussian


MURATA_DEFAULT_PARAMETERS = {
    "mu0": 3.0,
    "mu1": 0.8,
    "mu2": -0.3,
    "sigma0": 0.3,
    "sigma1": 0.0,
    "sigma2": 0.0,
}


class MurataModel:
    """The mass richness modeling defined in Murata 19."""

    def __init__(
        self,
        pivot_log_mass: float,
        pivot_redshift: float,
    ):
        super().__init__()
        self.pivot_redshift = pivot_redshift
        self.pivot_ln_mass = pivot_log_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        self.parameters = Parameters({**MURATA_DEFAULT_PARAMETERS})

        # Verify this gets called last or first

    @staticmethod
    def observed_value(
        p: tuple[float, float, float],
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        pivot_ln_mass: float,
        log1p_pivot_redshift: float,
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        ln_mass = log_mass * np.log(10)
        delta_ln_mass = ln_mass - pivot_ln_mass
        delta_z = np.log1p(z) - log1p_pivot_redshift

        result = p[0] + p[1] * delta_ln_mass + p[2] * delta_z
        assert isinstance(result, np.ndarray)
        return result

    def get_ln_mass_proxy_mean(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MurataModel.observed_value(
            (self.parameters["mu0"], self.parameters["mu1"], self.parameters["mu2"]),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )

    def get_ln_mass_proxy_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""
        return MurataModel.observed_value(
            (
                self.parameters["sigma0"],
                self.parameters["sigma1"],
                self.parameters["sigma2"],
            ),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )


class MurataBinned(MurataModel, MassRichnessGaussian):
    """The mass richness relation defined in Murata 19 for a binned data vector."""

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass-richness contribution to the integrand."""
        return self._distribution_integrated_gaussian(
            log_mass, z, log_mass_proxy_limits
        )


class MurataUnbinned(MurataModel, MassRichnessGaussian):
    """The mass richness relation defined in Murata 19 for a unbinned data vector."""

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass-richness contribution to the integrand."""
        return self._distribution_gaussian(log_mass, z, log_mass_proxy)
