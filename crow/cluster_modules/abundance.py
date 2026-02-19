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

from .parameters import Parameters


class ClusterAbundance:
    """The class that calculates the predicted number counts of galaxy clusters.

    The abundance is a function of a specific cosmology, a mass and redshift range,
    an area on the sky, a halo mass function, as well as multiple kernels, where
    each kernel represents a different distribution involved in the final cluster
    abundance integrand.

    Attributes
    ----------
    cosmo : pyccl.cosmology.Cosmology or None
        Cosmology object used for predictions. Set via the `cosmo` property.
    halo_mass_function : callable
        Halo mass function object compatible with PyCCL's MassFunc interface.
    parameters : Parameters
        Container for optional model parameters.
    _hmf_cache : dict
        Cache for previously computed halo mass function evaluations keyed by
        (log_mass, scale_factor).
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

    def __init__(
        self,
        cosmo: Cosmology,
        halo_mass_function: pyccl.halos.MassFunc,
    ) -> None:
        super().__init__()
        self.cosmo = cosmo
        self.halo_mass_function = halo_mass_function
        self.parameters = Parameters({})

    def comoving_volume(
        self, z: npt.NDArray[np.float64], sky_area: float = 0
    ) -> npt.NDArray[np.float64]:
        """Differential comoving volume for a given sky area.

        Parameters
        ----------
        z : array_like
            Redshift or array of redshifts at which to compute the differential
            comoving volume (dV/dz per steradian).
        sky_area : float, optional
            Survey area in square degrees. Default 0 returns per-steradian volume.

        Returns
        -------
        numpy.ndarray
            Differential comoving volume (same shape as `z`) multiplied by the
            survey area (converted to steradians). Units: [Mpc/h]^3 (consistent
            with the internal PyCCL conventions used here).

        Notes
        -----
        This uses PyCCL background helpers to evaluate the angular diameter
        distance and h(z) factor. If `sky_area` is zero the returned array is
        the per-steradian dV/dz.
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
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluate the halo mass function at given log-mass and redshift.

        Parameters
        ----------
        log_mass : array_like
            Array of log10 halo masses (M_sun).
        z : array_like
            Array of redshifts matching `log_mass`.

        Returns
        -------
        numpy.ndarray
            Array with mass function values (dn/dlnM or as provided by the
            configured `halo_mass_function`) evaluated at each (mass, z).

        Notes
        -----
        Results are cached in `_hmf_cache` keyed by (log_mass, scale_factor)
        to avoid repeated expensive evaluations for identical inputs.
        """
        scale_factor = 1.0 / (1.0 + z)
        return_vals = []

        for logm, a in zip(log_mass.astype(float), scale_factor.astype(float)):
            val = self._hmf_cache.get((logm, a))
            if val is None:
                val = self.halo_mass_function(self.cosmo, 10**logm, a)
                self._hmf_cache[(logm, a)] = val
            return_vals.append(val)

        return np.asarray(return_vals, dtype=np.float64)
