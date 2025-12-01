"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

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
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
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
        log_proxy_points: int = 30,
        redshift_points: int = 30,
        log_mass_points: int = 30,
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
        self.log_proxy_points = log_proxy_points
        self.redshift_points = redshift_points
        self.log_mass_points = log_mass_points
        self.log_mass_grid = np.linspace(
            mass_interval[0], mass_interval[1], self.log_mass_points
        )
        self._hmf_grid = {}  # (n_z, n_mass)
        self._mass_richness_grid = {}  # (n_proxy, n_z, n_mass)
        self._completeness_grid = {}  # (n_z, n_mass)
        self._purity_grid = {}  # (n_proxy, n_z)
        self._shear_grids = {}  # (n_z, n_mass)

    def _flat_distribution(
        self,
        z: npt.NDArray[np.float64],
        log_mass: npt.NDArray[np.float64] = None,
        log_mass_proxy: npt.NDArray[np.float64] = None,
    ):
        """Returns a null (=1) contribution to the integrand."""
        if log_mass is None and log_mass_proxy is None:
            raise ValueError(
                "Either log_mass or log_mass_proxy should be provided should be provided."
            )
        if log_mass_proxy is None:
            return 1.0 + 0 * log_mass * z
        if log_mass is None:
            return 1.0 + 0 * log_mass_proxy * z
        raise ValueError("Only one of log_mass or log_mass_proxy must be provided.")

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
        """Resets all internal dictionaries used for caching computed grids."""
        self._hmf_grid = {}
        self._mass_richness_grid = {}
        self._completeness_grid = {}
        self._purity_grid = {}
        self._shear_grids = {}

    def _get_hmf_grid(self, z: npt.NDArray[np.float64], sky_area: float, key):
        """Compute HMF × comoving volume and store in the class."""

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
        self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64], key
    ):
        """Compute mass-richness grid by vectorizing 1D inputs."""

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
                log_mass=self.log_mass_grid[np.newaxis, :], z=z[:, np.newaxis]
            )

        return self._completeness_grid[key]

    def _get_purity_grid(
        self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64], key
    ):
        """Compute purity grid and store in the class."""

        if key not in self._purity_grid:
            self._purity_grid[key] = self._purity_distribution(
                z=z[np.newaxis, :], log_mass_proxy=log_proxy[:, np.newaxis]
            )
        return self._purity_grid[key]

    def _get_shear_grid(self, z: npt.NDArray[np.float64], radius_centers, key):
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

    def _get_counts_kernel_grid(
        self, log_proxy_points, z_points, log_mass_points, proxy_key, z_key, sky_area
    ):
        """
        Returns
        -------
        counts_kernel_grid: numpy.ndarray
            Shape: (n_proxy, n_z, n_mass)
        """

        # grid keys
        hmf_key = z_key
        comp_key = z_key
        purity_key = (z_key, proxy_key)
        mass_richness_key = (z_key, proxy_key)

        #############
        # get grids #
        #############

        # shape: (n_z, n_mass)
        hmf_grid = self._get_hmf_grid(z_points, sky_area, hmf_key)
        # shape: (n_proxy, n_z, n_mass)
        mass_richness_grid = self._get_mass_richness_grid(
            z_points, log_proxy_points, mass_richness_key
        )
        # shape: (n_z, n_mass)
        completeness_grid = self._get_completeness_grid(z_points, comp_key)
        # shape: (n_proxy, n_z)
        purity_grid = self._get_purity_grid(z_points, log_proxy_points, purity_key)
        # output shape: (n_proxy, n_z, n_mass)
        return (
            hmf_grid[np.newaxis, :, :]
            * mass_richness_grid
            * completeness_grid[np.newaxis, :, :]
            / purity_grid[:, :, np.newaxis]
        )

    def _integrate_over_mass_z_proxy(
        self, kernel, log_mass_points, z_points, log_proxy_points
    ):
        """
        Parameters
        ----------
        kernel : numpy.ndarray
            , shape : (n_proxy, n_z, n_mass)
        log_mass_points : numpy.ndarray
        z_point : numpy.ndarray
        log_proxy_points : numpy.ndarray

        Returns
        -------
        integrated_kernel : numpy.ndarray
        """
        integral_over_mass = simpson(y=kernel, x=log_mass_points, axis=2)
        integral_over_z = simpson(y=integral_over_mass, x=z_points, axis=1)
        integral_over_proxy = simpson(
            y=integral_over_z, x=log_proxy_points * np.log(10.0), axis=0
        )
        integrated_kernel = integral_over_proxy
        return integrated_kernel

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        mass_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe using triple Simpson integration."""

        ######################
        # grid arrays and keys
        ######################

        log_proxy_points = np.linspace(
            mass_proxy_edges[0], mass_proxy_edges[1], self.log_proxy_points
        )
        z_points = np.linspace(z_edges[0], z_edges[1], self.redshift_points)
        log_mass_points = self.log_mass_grid
        proxy_key = tuple(mass_proxy_edges)
        z_key = tuple(z_edges)

        ########
        # kernel
        ########

        # shape: (n_proxy, n_z, n_mass)
        counts_kernel_grid = self._get_counts_kernel_grid(
            log_proxy_points, z_points, log_mass_points, proxy_key, z_key, sky_area
        )
        prediction = counts_kernel_grid
        if average_on is None:
            pass
        else:
            for cluster_prop in ClusterProperty:
                include_prop = cluster_prop & average_on
                if not include_prop:
                    continue
                if cluster_prop == ClusterProperty.MASS:
                    prediction *= log_mass_points[np.newaxis, np.newaxis, :]
                if cluster_prop == ClusterProperty.REDSHIFT:
                    prediction *= z_points[np.newaxis, :, np.newaxis]

        ###########
        # integrate
        ###########

        counts = self._integrate_over_mass_z_proxy(
            prediction, log_mass_points, z_points, log_proxy_points
        )
        return counts

    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
        radius_centers: np.ndarray,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theoretical prediction for the average shear profile
        <DeltaSigma(R)> in the provided bin."""

        ######################
        # grid arrays and keys
        ######################

        log_proxy_points = np.linspace(
            mass_proxy_edges[0], mass_proxy_edges[1], self.log_proxy_points
        )
        z_points = np.linspace(z_edges[0], z_edges[1], self.redshift_points)
        log_mass_points = self.log_mass_grid
        proxy_key = tuple(mass_proxy_edges)
        z_key = tuple(z_edges)
        shear_key = z_key

        ########
        # kernel
        ########

        # shape: (n_proxy, n_z, n_mass)
        counts_kernel_grid = self._get_counts_kernel_grid(
            log_proxy_points, z_points, log_mass_points, proxy_key, z_key, sky_area
        )
        if not (average_on & (ClusterProperty.DELTASIGMA | ClusterProperty.SHEAR)):
            # Raise a ValueError if the necessary flags are not present
            raise ValueError(
                f"Function requires {ClusterProperty.DELTASIGMA} or {ClusterProperty.SHEAR} "
                f"to be set in 'average_on', but got: {average_on}"
            )
        # shape: (n_z, n_mass, n_radius)
        shear_grid = self._get_shear_grid(z_points, radius_centers, shear_key)
        # shape: (n_proxy, n_z, n_mass, n_radius)
        shear_kernel_grid = (
            counts_kernel_grid[..., np.newaxis] * shear_grid[np.newaxis, ...]
        )

        ###########
        # integrate
        ###########

        shear = self._integrate_over_mass_z_proxy(
            shear_kernel_grid, log_mass_points, z_points, log_proxy_points
        )
        return shear
