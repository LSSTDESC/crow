"""The mass richness kernel module.

This module holds the classes that define the mass richness relations
that can be included in the cluster abundance integrand.  These are
implementations of Kernels.
"""

from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import special

from crow.integrator.numcosmo_integrator import NumCosmoIntegrator

from .purity import Purity


class MassRichnessGaussian:
    """The representation of mass richness relations that are of a gaussian form."""

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

    @abstractmethod
    def get_proxy_mean(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""

    @abstractmethod
    def get_proxy_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""

    def _distribution_binned(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        proxy_mean = self.get_proxy_mean(log_mass, z)
        proxy_sigma = self.get_proxy_sigma(log_mass, z)

        x_min = (proxy_mean - log_mass_proxy_limits[0] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
        )
        x_max = (proxy_mean - log_mass_proxy_limits[1] * np.log(10.0)) / (
            np.sqrt(2.0) * proxy_sigma
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

    def _distribution_unbinned(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        proxy_mean = self.get_proxy_mean(log_mass, z)
        proxy_sigma = self.get_proxy_sigma(log_mass, z)

        normalization = 1 / np.sqrt(2 * np.pi * proxy_sigma**2)
        result = normalization * np.exp(
            -0.5 * ((log_mass_proxy * np.log(10) - proxy_mean) / proxy_sigma) ** 2
        )

        assert isinstance(result, np.ndarray)
        return result


MURATA_DEFAULT_MU_P0 = 3.0
MURATA_DEFAULT_MU_P1 = 0.8
MURATA_DEFAULT_MU_P2 = -0.3
MURATA_DEFAULT_SIGMA_P0 = 0.3
MURATA_DEFAULT_SIGMA_P1 = 0.0
MURATA_DEFAULT_SIGMA_P2 = 0.0


class MurataBinned(MassRichnessGaussian):
    """The mass richness relation defined in Murata 19 for a binned data vector."""

    def __init__(
        self,
        pivot_log_mass: float,
        pivot_redshift: float,
        purity: Purity = None,
    ):
        super().__init__()
        self.pivot_redshift = pivot_redshift
        self.pivot_ln_mass = pivot_log_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        self.mu_p0 = MURATA_DEFAULT_MU_P0
        self.mu_p1 = MURATA_DEFAULT_MU_P1
        self.mu_p2 = MURATA_DEFAULT_MU_P2
        self.sigma_p0 = MURATA_DEFAULT_SIGMA_P0
        self.sigma_p1 = MURATA_DEFAULT_SIGMA_P1
        self.sigma_p2 = MURATA_DEFAULT_SIGMA_P2

        self.purity = purity

        # Verify this gets called last or first

    @property
    def purity(self):
        return self.__purity

    @purity.setter
    def purity(self, value):
        if value is None:
            self._distribution = self._distribution_binned
        else:
            self._distribution = self._distribution_binned_inpure
        self.__purity = value

    def get_proxy_mean(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )

    def _distribution_binned_inpure(self, log_mass, z, log_mass_proxy_limits):
        integrator = NumCosmoIntegrator(
            relative_tolerance=1e-2,
            absolute_tolerance=1e-6,
        )

        def integration_func(int_args, extra_args):
            log_mass_proxy = int_args[:, 0]
            """
            print(
                np.array(
                    [
                        np.sum(
                            self._distribution_unbinned(
                                np.array([log_mass[i]]), np.array([z[i]]), log_mass_proxy
                            )
                            / self.purity.distribution(np.array([z[i]]), log_mass_proxy)
                        )
                        for i in range(len(log_mass))
                    ]
                )
            )
            print(
                np.array(
                    [
                        np.sum(
                            self._distribution_unbinned(log_mass, z, np.array([proxy]))
                            / self.purity.distribution(z, np.array([proxy]))
                            for proxy in log_mass_proxy
                        )
                    ][0]
                )
            )
            """
            return np.array(
                [
                    self._distribution_unbinned(log_mass, z, np.array([proxy]))
                    / self.purity.distribution(z, np.array([proxy]))
                    for proxy in log_mass_proxy
                ]
            )

        integrator.integral_bounds = [
            (log_mass_proxy_limits[0], log_mass_proxy_limits[1])
        ]

        return integrator.integrate(integration_func)

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass-richness contribution to the integrand."""
        return self._distribution(log_mass, z, log_mass_proxy_limits)


class MurataUnbinned(MassRichnessGaussian):
    """The mass richness relation defined in Murata 19 for a unbinned data vector."""

    def __init__(
        self,
        pivot_log_mass: float,
        pivot_redshift: float,
    ):
        super().__init__()
        self.pivot_redshift = pivot_redshift
        self.pivot_ln_mass = pivot_log_mass * np.log(10.0)  # ln(M)
        self.log1p_pivot_redshift = np.log1p(self.pivot_redshift)

        self.mu_p0 = MURATA_DEFAULT_MU_P0
        self.mu_p1 = MURATA_DEFAULT_MU_P1
        self.mu_p2 = MURATA_DEFAULT_MU_P2
        self.sigma_p0 = MURATA_DEFAULT_SIGMA_P0
        self.sigma_p1 = MURATA_DEFAULT_SIGMA_P1
        self.sigma_p2 = MURATA_DEFAULT_SIGMA_P2

    def get_proxy_mean(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed quantity corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.mu_p0, self.mu_p1, self.mu_p2),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )

    def get_proxy_sigma(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return observed scatter corrected by redshift and mass."""
        return MassRichnessGaussian.observed_value(
            (self.sigma_p0, self.sigma_p1, self.sigma_p2),
            log_mass,
            z,
            self.pivot_ln_mass,
            self.log1p_pivot_redshift,
        )

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the mass-richness contribution to the integrand."""
        return self._distribution_unbinned(log_mass, z, log_mass_proxy)
