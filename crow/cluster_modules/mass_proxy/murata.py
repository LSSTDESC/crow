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
    """
    Base implementation of the Murata et al. (2019) mass–richness relation.

    This model parameterizes the mean and scatter of the logarithmic mass proxy
    as linear functions of log-mass and redshift, defined relative to pivot values.
    """

    def __init__(
        self,
        pivot_log_mass: float,
        pivot_redshift: float,
    ):
        """
        Initialize the Murata model.

        Parameters
        ----------
        pivot_log_mass : float
            Pivot value of log10 halo mass.
        pivot_redshift : float
            Pivot redshift value.
        """
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
        """
        Compute a linear observable as a function of mass and redshift.

        This function evaluates a linear model in deviations from pivot values.

        Parameters
        ----------
        p : tuple of float
            Model coefficients (p0, p1, p2).
        log_mass : ndarray of float64
            Logarithm (base 10) of halo mass.
        z : ndarray of float64
            Redshift values.
        pivot_ln_mass : float
            Pivot value of natural log of mass.
        log1p_pivot_redshift : float
            Pivot value of log(1 + z).

        Returns
        -------
        ndarray of float64
            Evaluated observable.
        """
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
        """
        Compute the mean of the log mass proxy distribution.

        Parameters
        ----------
        log_mass : ndarray of float64
            Logarithm (base 10) of halo mass.
        z : ndarray of float64
            Redshift values.

        Returns
        -------
        ndarray of float64
            Mean of the natural logarithm of the mass proxy.
        """
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
        """
        Compute the scatter of the log mass proxy distribution.

        Parameters
        ----------
        log_mass : ndarray of float64
            Logarithm (base 10) of halo mass.
        z : ndarray of float64
            Redshift values.

        Returns
        -------
        ndarray of float64
            Standard deviation of the natural logarithm of the mass proxy.
        """
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
    """
    Murata mass–richness relation for binned data vectors.

    This implementation returns the integrated Gaussian probability over
    specified bins in the mass proxy.
    """

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy_limits: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the binned mass–richness distribution.

        Parameters
        ----------
        log_mass : ndarray of float64
            Logarithm (base 10) of halo mass.
        z : ndarray of float64
            Redshift values.
        log_mass_proxy_limits : tuple of float
            Lower and upper bounds of the bin in log10 space.

        Returns
        -------
        ndarray of float64
            Integrated probability within the bin.
        """
        return self.integrated_gaussian(log_mass, z, log_mass_proxy_limits)


class MurataUnbinned(MurataModel, MassRichnessGaussian):
    """
    Murata mass–richness relation for unbinned data vectors.

    This implementation evaluates the Gaussian probability density
    directly at the observed mass proxy values.
    """

    def distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        log_mass_proxy: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Evaluate the unbinned mass–richness distribution.

        Parameters
        ----------
        log_mass : ndarray of float64
            Logarithm (base 10) of halo mass.
        z : ndarray of float64
            Redshift values.
        log_mass_proxy : ndarray of float64
            Logarithm (base 10) of the observed mass proxy.

        Returns
        -------
        ndarray of float64
            Value of the Gaussian probability density function.
        """
        return self.gaussian_kernel(log_mass, z, log_mass_proxy)
