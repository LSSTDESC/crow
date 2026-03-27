"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe.

This module provides implementations of cluster abundance and lensing
predictions based on a binned representation of the mass–richness relation
(Murata et al. 2019) and spectroscopic redshift assumptions. The primary
class defined here constructs multi-dimensional grids for efficient numerical
integration of cluster observables, including number counts and lensing
profiles.

The implementation relies on precomputing and caching intermediate quantities
such as halo mass functions, mass–richness relations, completeness, purity,
and shear profiles. These cached grids are then combined into integrands and
evaluated using Simpson integration over mass, redshift, and observable proxy.
"""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl
from scipy.integrate import simpson

from crow.cluster_modules.completeness_models import Completeness
from crow.cluster_modules.purity_models import Purity
from crow.properties import ClusterProperty

from .binned_parent import BinnedClusterRecipe

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class GridBinnedClusterRecipe(BinnedClusterRecipe):
    """Cluster recipe using a grid-based integration scheme for cluster observables.

    This class implements a binned cluster model based on the Murata (2019)
    mass–richness relation, assuming perfectly measured spectroscopic redshifts.
    It evaluates cluster number counts and lensing signals by constructing
    multi-dimensional grids over mass, redshift, and observable proxy (e.g.,
    richness), and performing numerical integration.

    The computation is structured around reusable cached grids for efficiency,
    including:
        - Halo mass function × comoving volume
        - Mass–richness probability distribution
        - Completeness and purity corrections
        - Shear profiles for lensing observables

    Parameters
    ----------
    cluster_theory : object
        Object providing theoretical predictions such as halo mass function,
        comoving volume, and shear profiles.
    redshift_distribution : callable
        Function describing the redshift distribution of clusters.
    mass_distribution : callable
        Function describing the mass–observable (e.g., richness) relation.
    completeness : Completeness, optional
        Completeness model. If None, a flat (unity) completeness is assumed.
    purity : Purity, optional
        Purity model. If None, a flat (unity) purity is assumed.
    mass_interval : tuple of float, optional
        Range of log10 halo mass values used for integration.
    true_z_interval : tuple of float, optional
        Redshift range used for integration.
    proxy_grid_size : int, optional
        Number of grid points in observable proxy (e.g., richness).
    redshift_grid_size : int, optional
        Number of grid points in redshift.
    mass_grid_size : int, optional
        Number of grid points in log-mass.

    Attributes
    ----------
    log_mass_grid : numpy.ndarray
        1D grid of log10 halo masses used for integration.
    _hmf_grid : dict
        Cache of halo mass function × volume grids.
    _mass_richness_grid : dict
        Cache of mass–richness probability grids.
    _completeness_grid : dict
        Cache of completeness grids.
    _purity_grid : dict
        Cache of purity grids.
    _shear_grids : dict
        Cache of shear profile grids.
    """

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: Completeness = None,
        purity: Purity = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
        proxy_grid_size: int = 30,
        redshift_grid_size: int = 30,
        mass_grid_size: int = 30,
    ) -> None:
        super().__init__(
            cluster_theory=cluster_theory,
            redshift_distribution=redshift_distribution,
            mass_distribution=mass_distribution,
            completeness=completeness,
            purity=purity,
            mass_interval=mass_interval,
            true_z_interval=true_z_interval,
        )
        self.proxy_grid_size = proxy_grid_size
        self.redshift_grid_size = redshift_grid_size
        self.mass_grid_size = mass_grid_size
        self.log_mass_grid = np.linspace(
            mass_interval[0], mass_interval[1], self.mass_grid_size
        )
        self._hmf_grid = {}  # (n_z, n_mass)
        self._mass_richness_grid = {}  # (n_proxy, n_z, n_mass)
        self._completeness_grid = {}  # (n_z, n_mass)
        self._purity_grid = {}  # (n_proxy, n_z)
        self._shear_grids = {}  # (n_z, n_mass)

    def _flat_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ):
        """Return a trivial (unity) distribution.

        This function is used as a fallback when no completeness or purity
        model is provided. It returns an array of ones with the same broadcasted
        shape as the inputs.

        Parameters
        ----------
        log_mass : array_like
            Logarithmic halo mass values.
        z : array_like
            Redshift values.

        Returns
        -------
        numpy.ndarray
            Array of ones matching the broadcasted shape of inputs.
        """
        return 1.0 + 0 * log_mass * z

    def _setup_with_completeness(self):
        """Additional setup of class with the completeness"""
        if self.completeness is None:
            self._completeness_distribution = self._flat_distribution
        else:
            self._completeness_distribution = self.completeness.distribution

    def _setup_with_purity(self):
        """Additional setup of class with the purity"""
        if self.purity is None:
            self._purity_distribution = self._flat_distribution
        else:
            self._purity_distribution = self.purity.distribution

    def setup(self) -> None:
        """Reset all cached grids.

        This method clears all internally stored grids used for caching intermediate
        quantities (e.g., halo mass function, mass–richness relation, etc.). It
        should be called whenever model parameters or cosmology are updated to
        ensure consistency in subsequent computations.
        """
        self._hmf_grid = {}
        self._mass_richness_grid = {}
        self._completeness_grid = {}
        self._purity_grid = {}
        self._shear_grids = {}

    def _get_hmf_grid(
        self,
        z: npt.NDArray[np.float64],
        sky_area: float,
        key,
    ):
        """Compute and cache the halo mass function multiplied by comoving volume.

        This method evaluates the halo mass function over a 2D grid of redshift
        and log-mass, multiplies it by the differential comoving volume, and stores
        the result for reuse.

        Parameters
        ----------
        z : numpy.ndarray
            Array of redshift values.
        sky_area : float
            Survey area in square degrees.
        key : hashable
            Cache key identifying this grid configuration.

        Returns
        -------
        numpy.ndarray
            2D array of shape (n_z, n_mass) containing HMF × volume values.
        """
        if key not in self._hmf_grid:
            # sizes
            n_m = len(self.log_mass_grid)
            n_z = len(z)
            # quantities
            hmf_flat = self.cluster_theory.mass_function(
                # flatten arrays to vectorize function
                np.tile(self.log_mass_grid, n_z),
                np.repeat(z, n_m),
            )
            mass_function_2d = hmf_flat.reshape(n_z, n_m)
            vol = self.cluster_theory.comoving_volume(z, sky_area)
            # assign
            self._hmf_grid[key] = vol[:, np.newaxis] * mass_function_2d
        return self._hmf_grid[key]

    def _get_mass_richness_grid(
        self,
        z: npt.NDArray[np.float64],
        log_proxy: npt.NDArray[np.float64],
        key,
    ):
        """Compute and cache the mass–richness probability distribution grid.

        This method evaluates the conditional probability of observable proxy
        (e.g., richness) given halo mass and redshift, producing a 3D grid.

        Parameters
        ----------
        z : numpy.ndarray
            Redshift grid.
        log_proxy : numpy.ndarray
            Observable proxy grid (log-space).
        key : hashable
            Cache key identifying this configuration.

        Returns
        -------
        numpy.ndarray
            3D array of shape (n_proxy, n_z, n_mass).
        """
        if key not in self._mass_richness_grid:
            # sizes
            n_z = len(z)
            n_m = len(self.log_mass_grid)
            n_p = len(log_proxy)
            # quantities
            grid_3d_flat = self.mass_distribution.distribution(
                # flatten arrays to vectorize function
                np.tile(np.repeat(self.log_mass_grid, n_z), n_p),
                np.tile(z, n_m * n_p),
                np.repeat(log_proxy, n_z * n_m),
            )
            grid_3d_temp = grid_3d_flat.reshape(n_p, n_m, n_z)
            # assign
            self._mass_richness_grid[key] = grid_3d_temp.transpose(0, 2, 1)

        return self._mass_richness_grid[key]

    def _get_completeness_grid(self, z: npt.NDArray[np.float64], key):
        """Compute completeness grid and store in the class."""

        if key not in self._completeness_grid:
            self._completeness_grid[key] = self._completeness_distribution(
                self.log_mass_grid[np.newaxis, :], z[:, np.newaxis]
            )
        return self._completeness_grid[key]

    def _get_purity_grid(
        self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64], key
    ):
        """Compute purity grid and store in the class."""

        if key not in self._purity_grid:
            self._purity_grid[key] = self._purity_distribution(
                log_proxy[:, np.newaxis], z[np.newaxis, :]
            )
        return self._purity_grid[key]

    def _get_shear_grid(
        self,
        z: npt.NDArray[np.float64],
        radius_centers,
        key,
    ):
        """Compute shear grid for a specific radius and store in the class."""

        if key not in self._shear_grids:
            # shape (n_m, n_r, n_z)
            grid_3d = self.cluster_theory.compute_shear_profile_vectorized(
                log_mass=self.log_mass_grid[:, None],
                z=z,
                radius_center=radius_centers[:, None],
            )
            # assign
            self._shear_grids[key] = grid_3d.transpose(2, 0, 1)

        return self._shear_grids[key]

    def _get_integ_arrays(
        self,
        z_edges: tuple[float, float],
        log_proxy_edges: tuple[float, float],
    ) -> float:
        """Grid arrays and keys

        Returns
        -------
        integ_arrays: dict
            Dictionary with information on `"redshift"` and `"log_proxy"` for integration.
            Each entry is of the type ``{"points":np.ndarray, "key":tuple}``
        """
        return {
            "log_proxy": {
                "points": np.linspace(
                    log_proxy_edges[0], log_proxy_edges[1], self.proxy_grid_size
                ),
                "key": tuple(log_proxy_edges),
            },
            "redshift": {
                "points": np.linspace(z_edges[0], z_edges[1], self.redshift_grid_size),
                "key": tuple(z_edges),
            },
        }

    def _evaluate_theory_prediction_generic(
        self,
        probe_kernel,  # must be (n_proxy, n_z, n_mass, ...)
        integ_arrays,
        sky_area: float,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe using triple Simpson integration."""
        """
        Parameters
        ----------
        probe_kernel : numpy.ndarray
            Kernel to be integrated in mass (w HMF) and redshift (w/ volume). Shape : (n_proxy, n_z, n_mass, ...)
        integ_arrays: dict
            Dictionary with information on `"redshift"` and `"log_proxy"` for integration.
            Each entry must be of the type ``{"points":np.ndarray, "key":tuple}``

        Returns
        -------
        integrated_kernel : numpy.ndarray
        """

        ##############################
        # get basic kernel and combine
        ##############################

        # grid keys
        hmf_key = integ_arrays["redshift"]["key"]
        comp_key = integ_arrays["redshift"]["key"]
        purity_key = (integ_arrays["redshift"]["key"], integ_arrays["log_proxy"]["key"])
        mass_richness_key = (
            integ_arrays["redshift"]["key"],
            integ_arrays["log_proxy"]["key"],
        )

        # get grids #

        # shape: (n_z, n_mass)
        hmf_grid = self._get_hmf_grid(
            integ_arrays["redshift"]["points"],
            sky_area,
            hmf_key,
        )
        # shape: (n_proxy, n_z, n_mass)
        mass_richness_grid = self._get_mass_richness_grid(
            integ_arrays["redshift"]["points"],
            integ_arrays["log_proxy"]["points"],
            mass_richness_key,
        )
        # shape: (n_z, n_mass)
        completeness_grid = self._get_completeness_grid(
            integ_arrays["redshift"]["points"],
            comp_key,
        )
        # shape: (n_proxy, n_z)
        purity_grid = self._get_purity_grid(
            integ_arrays["redshift"]["points"],
            integ_arrays["log_proxy"]["points"],
            purity_key,
        )

        # main kernel: (n_proxy, n_z, n_mass)
        main_kernel_grid = (
            hmf_grid[np.newaxis, :, :]
            * mass_richness_grid
            * completeness_grid[np.newaxis, :, :]
            / purity_grid[:, :, np.newaxis]
        )

        # reshape it to match probe_kernel
        # shape: (n_proxy, n_z, n_mass, ...)
        n_dims_probe_kernel = len(probe_kernel.shape)
        if n_dims_probe_kernel > 3:
            main_kernel_grid = np.expand_dims(
                main_kernel_grid, axis=tuple(range(3, n_dims_probe_kernel))
            )

        # Final kernel
        # shape: (n_proxy, n_z, n_mass, n_radius)
        final_kernel = main_kernel_grid * probe_kernel

        ###########
        # integrate
        ###########

        integral_over_mass = simpson(y=final_kernel, x=self.log_mass_grid, axis=2)
        integral_over_z = simpson(
            y=integral_over_mass, x=integ_arrays["redshift"]["points"], axis=1
        )
        integral_over_proxy = simpson(
            y=integral_over_z,
            x=integ_arrays["log_proxy"]["points"] * np.log(10.0),
            axis=0,
        )
        integrated_probe = integral_over_proxy
        return integrated_probe

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        log_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Compute predicted cluster number counts in a bin.

        This method evaluates the expected number of clusters within specified
        redshift and observable proxy bins by integrating the full cluster
        abundance model.

        Parameters
        ----------
        z_edges : tuple of float
            Lower and upper bounds of the redshift bin.
        log_proxy_edges : tuple of float
            Bounds of the observable proxy bin (log-space).
        sky_area : float
            Survey area in square degrees.
        average_on : ClusterProperty or None, optional
            Optional weighting of the observable (e.g., mass-weighted counts).

        Returns
        -------
        float
            Predicted number of clusters in the bin.
        """
        ######################
        # grid arrays and keys
        ######################

        integ_arrays = self._get_integ_arrays(z_edges, log_proxy_edges)

        ########
        # kernel
        ########

        # shape: (n_proxy, n_z, n_mass)
        probe_kernel = np.ones(
            (
                integ_arrays["log_proxy"]["points"].size,
                integ_arrays["redshift"]["points"].size,
                self.log_mass_grid.size,
            )
        )
        if average_on is None:
            pass
        else:
            for cluster_prop in ClusterProperty:
                include_prop = cluster_prop & average_on
                if not include_prop:
                    continue
                if cluster_prop == ClusterProperty.MASS:
                    probe_kernel *= self.log_mass_grid[np.newaxis, np.newaxis, :]
                if cluster_prop == ClusterProperty.REDSHIFT:
                    probe_kernel *= integ_arrays["redshift"]["points"][
                        np.newaxis, :, np.newaxis
                    ]

        ###########
        # integrate
        ###########

        counts = self._evaluate_theory_prediction_generic(
            probe_kernel,
            integ_arrays,
            sky_area,
        )
        return counts

    def evaluate_theory_prediction_lensing_profile(
        self,
        z_edges: tuple[float, float],
        log_proxy_edges: tuple[float, float],
        radius_centers: np.ndarray,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        r"""Compute the average cluster lensing profile in a bin.

        This method evaluates the expected stacked lensing signal
        (e.g., :math:`\langle \Delta\Sigma(R) \rangle` or
        :math:`\langle g_t(R) \rangle`) by integrating the shear profile
        over the cluster population.

        Parameters
        ----------
        z_edges : tuple of float
            Redshift bin boundaries.
        log_proxy_edges : tuple of float
            Observable proxy bin boundaries (log-space).
        radius_centers : numpy.ndarray
            Radial bins at which the lensing signal is evaluated.
        sky_area : float
            Survey area in square degrees.
        average_on : ClusterProperty
            Must include DELTASIGMA or SHEAR.

        Returns
        -------
        numpy.ndarray
            Radial profile of the stacked lensing signal.

        Raises
        ------
        ValueError
            If the required ClusterProperty flags are not set.
        """
        if not (average_on & (ClusterProperty.DELTASIGMA | ClusterProperty.SHEAR)):
            # Raise a ValueError if the necessary flags are not present
            raise ValueError(
                f"Function requires {ClusterProperty.DELTASIGMA} or {ClusterProperty.SHEAR} "
                f"to be set in 'average_on', but got: {average_on}"
            )

        ######################
        # grid arrays and keys
        ######################

        integ_arrays = self._get_integ_arrays(z_edges, log_proxy_edges)
        shear_key = integ_arrays["redshift"]["key"]

        ########
        # kernel
        ########

        # shape: (n_z, n_mass, n_radius)
        shear_grid = self._get_shear_grid(
            integ_arrays["redshift"]["points"],
            radius_centers,
            shear_key,
        )
        # re-shape it: (1, n_z, n_mass, n_radius)
        probe_kernel = shear_grid[np.newaxis, ...]

        ###########
        # integrate
        ###########

        shear = self._evaluate_theory_prediction_generic(
            probe_kernel,
            integ_arrays,
            sky_area,
        )
        return shear
