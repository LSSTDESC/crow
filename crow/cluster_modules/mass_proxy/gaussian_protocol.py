"""Gaussian class for mass richness distributiosn."""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import special

from crow.integrator.numcosmo_integrator import NumCosmoIntegrator


class MassRichnessGaussian:
    """The representation of mass richness relations that are of a gaussian form."""

    @abstractmethod
    def get_ln_mass_proxy_mean(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""

    @abstractmethod
    def get_ln_mass_proxy_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""

    def _distribution_integrated_gaussian(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        ln_mass_proxy_mean = self.get_ln_mass_proxy_mean(log_mass, z)
        ln_mass_proxy_sigma = self.get_ln_mass_proxy_sigma(log_mass, z)

        x_min = (ln_mass_proxy_mean - log_mass_proxy_limits[0] * np.log(10.0)) / (
            np.sqrt(2.0) * ln_mass_proxy_sigma
        )
        x_max = (ln_mass_proxy_mean - log_mass_proxy_limits[1] * np.log(10.0)) / (
            np.sqrt(2.0) * ln_mass_proxy_sigma
        )

        return_vals = np.empty_like(x_min)
        mask1 = (x_max > 3.0) | (x_min < -3.0)
        mask2 = ~mask1

        # pylint: disable=no-member
        return_vals[mask1] = (
            -(special.erfc(x_min[mask1]) - special.erfc(x_max[mask1])) / 2.0
        )
        # pylint: disable=no-member
        return_vals[mask2] = (
            special.erf(x_min[mask2]) - special.erf(x_max[mask2])
        ) / 2.0
        assert isinstance(return_vals, np.ndarray)
        return return_vals

    def _distribution_gaussian(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        ln_mass_proxy_mean = self.get_ln_mass_proxy_mean(log_mass, z)
        ln_mass_proxy_sigma = self.get_ln_mass_proxy_sigma(log_mass, z)

        normalization = 1 / np.sqrt(2 * np.pi * ln_mass_proxy_sigma**2)
        result = normalization * np.exp(
            -0.5
            * ((log_mass_proxy * np.log(10) - ln_mass_proxy_mean) / ln_mass_proxy_sigma)
            ** 2
        )

        assert isinstance(result, np.ndarray)
        return result
