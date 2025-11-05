"""The module responsible for building the cluster abundance calculation.

The galaxy cluster abundance integral is a combination of both theoretical
and phenomenological predictions.  This module contains the classes and
functions that produce those predictions.
"""

import numpy as np
import numpy.typing as npt
import pyccl
import pyccl.background as bkg
from pyccl.cosmology import Cosmology

from crow.kernel import Completeness


class ClusterAbundance:
    """The class that calculates the predicted number counts of galaxy clusters.

    The abundance is a function of a specific cosmology, a mass and redshift range,
    an area on the sky, a halo mass function, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    abundance integrand.
    """

    @property
    def cosmo(self) -> Cosmology | None:
        """The cosmology used to predict the cluster number count."""
        return self._cosmo

    @cosmo.setter
    def cosmo(self, cosmo: Cosmology) -> None:
        """Update the cluster abundance calculation with a new cosmology."""
        self._cosmo = cosmo
        self._hmf_cache: dict[tuple[float, float], float] = {}

    @property
    def completeness(self) -> Completeness | None:
        """The completeness used to predict the cluster number count."""
        return self.__completeness

    @completeness.setter
    def completeness(self, completeness: Completeness) -> None:
        """Update the cluster abundance calculation with a new completeness."""
        self.__completeness = completeness
        if completeness is None:
            self._completeness_distribution = self._complete_distribution
        else:
            self._completeness_distribution = self._incomplete_distribution

    def __init__(
        self,
        cosmo: Cosmology,
        halo_mass_function: pyccl.halos.MassFunc,
        completeness: Completeness | None = None,
    ) -> None:
        super().__init__()
        self.cosmo = cosmo
        self.halo_mass_function = halo_mass_function
        self.completeness = completeness

    def comoving_volume(
        self, z: npt.NDArray[np.float64], sky_area: float = 0
    ) -> npt.NDArray[np.float64]:
        """The differential comoving volume given area sky_area at redshift z.

        :param sky_area: The area of the survey on the sky in square degrees.
        """
        assert self.cosmo is not None
        scale_factor = 1.0 / (1.0 + z)
        angular_diam_dist = bkg.angular_diameter_distance(self.cosmo, scale_factor)
        h_over_h0 = bkg.h_over_h0(self.cosmo, scale_factor)

        dV = (
            pyccl.physical_constants.CLIGHT_HMPC
            * (angular_diam_dist**2)
            * ((1.0 + z) ** 2)
            / (self.cosmo["h"] * h_over_h0)
        )
        assert isinstance(dV, np.ndarray)

        sky_area_rad = sky_area * (np.pi / 180.0) ** 2

        return np.array(dV * sky_area_rad, dtype=np.float64)

    def mass_function(
        self,
        mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The mass function at z and mass."""
        scale_factor = 1.0 / (1.0 + z)
        return_vals = []

        for m, a in zip(mass.astype(float), scale_factor.astype(float)):
            val = self._hmf_cache.get((m, a))
            if val is None:
                val = self.halo_mass_function(self.cosmo, 10**m, a)
                self._hmf_cache[(m, a)] = val
            return_vals.append(val)

        return np.asarray(return_vals, dtype=np.float64)

    def _complete_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ):
        return 1.0

    def _incomplete_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ):
        return self.completeness.distribution(log_mass, z)

    def completeness_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""

        return self._completeness_distribution(log_mass, z)
