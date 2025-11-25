"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl
from scipy.integrate import simpson

from crow import ClusterShearProfile
from crow import completeness as comp
from crow import kernel
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty
from crow.recipes.murata_binned_spec_z import MurataBinnedSpecZRecipe

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class MurataBinnedSpecZRecipeGrid(MurataBinnedSpecZRecipe):
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
    """

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: comp.Completeness = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
        log_proxy_points: int = 30,
        redshift_points: int = 30,
        log_mass_points: int = 30,
    ) -> None:
        super().__init__(
            cluster_theory,
            redshift_distribution,
            mass_distribution,
            completeness,
            mass_interval,
            true_z_interval,
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
        self._purity_grid = {}  # (n_z, n_proxy)
        self._shear_grids = {}  # (n_z, n_mass)

    def reset_grids_cache(self) -> None:
        """Resets all internal dictionaries used for caching computed grids."""
        self._hmf_grid = {}
        self._mass_richness_grid = {}
        self._completeness_grid = {}
        self._purity_grid = {}
        self._shear_grids = {}

    def get_hmf_grid(self, z: npt.NDArray[np.float64], sky_area: float, key):
        """Compute HMF × comoving volume and store in the class."""

        if key not in self._hmf_grid:
            # sizes
            nm = len(self.log_mass_grid)
            nz = len(z)
            # quantities
            hmf_flat = self.cluster_theory.mass_function(
                # flatten arrays to vectorize function
                np.tile(self.log_mass_grid, nz),
                np.repeat(z, nm),
            )
            mass_function_2d = hmf_flat.reshape(nz, nm)
            vol = self.cluster_theory.comoving_volume(z, sky_area)
            # assign
            self._hmf_grid[key] = vol[:, np.newaxis] * mass_function_2d

        return self._hmf_grid[key]

    def get_mass_richness_grid(
        self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64], key
    ):
        """Compute mass-richness grid by vectorizing 1D inputs."""

        if key not in self._mass_richness_grid:
            # sizes
            nz = len(z)
            nm = len(self.log_mass_grid)
            nproxy = len(log_proxy)
            # quantities
            grid_3d_flat = self.mass_distribution._distribution_unbinned(
                # flatten arrays to vectorize function
                np.tile(np.repeat(self.log_mass_grid, nz), nproxy),
                np.tile(z, nm * nproxy),
                np.repeat(log_proxy, nz * nm),
            )
            grid_3d_temp = grid_3d_flat.reshape(nproxy, nm, nz)
            # assign
            self._mass_richness_grid[key] = grid_3d_temp.transpose(0, 2, 1)

        return self._mass_richness_grid[key]

    def get_completeness_grid(self, z: npt.NDArray[np.float64], key):
        """Compute completeness grid and store in the class."""

        if key not in self._completeness_grid:
            # sizes
            nm = len(self.log_mass_grid)
            nz = len(z)
            # flatten arrays
            z_flat = np.repeat(z, nm)
            log_mass_flat = np.tile(self.log_mass_grid, nz)
            # quantities
            if self.completeness is None:
                comp2d = np.ones((nz, nm), dtype=np.float64)
            else:
                comp2d = self.completeness_distribution(log_mass_flat, z_flat).reshape(
                    nz, nm
                )

            # assign
            self._completeness_grid[key] = comp2d

        return self._completeness_grid[key]

    def get_purity_grid(
        self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64], key
    ):
        """Compute purity grid and store in the class."""

        if key not in self._purity_grid:
            # sizes
            nz = len(z)
            nproxy = len(log_proxy)
            # flatten arrays
            z_flat = np.repeat(z, nproxy)
            log_proxy_flat = np.tile(log_proxy, nz)
            # quantities
            if self.mass_distribution.purity is None:
                pur2d = np.ones((nz, nproxy), dtype=np.float64)
            else:
                pur2d = self.mass_distribution.purity.distribution(
                    z_flat, log_proxy_flat
                ).reshape(nz, nproxy)
            # assign
            self._purity_grid[key] = pur2d

        return self._purity_grid[key]

    def get_shear_grid(self, z: npt.NDArray[np.float64], radius_center: float, key):
        """Compute shear grid for a specific radius and store in the class."""

        if key not in self._shear_grids:
            # sizes
            nm = len(self.log_mass_grid)
            nz = len(z)
            # quantities
            grid_2d_flat = self.cluster_theory.compute_shear_profile(
                # flatten arrays to vectorize function
                log_mass=np.tile(self.log_mass_grid, nz),
                z=np.repeat(z, nm),
                radius_center=radius_center,
            )
            # assign
            self._shear_grids[key] = grid_2d_flat.reshape(nz, nm)

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
        hmf_grid = self.get_hmf_grid(z_points, sky_area, hmf_key)
        # shape: (n_proxy, n_z, n_mass)
        mass_richness_grid = self.get_mass_richness_grid(
            z_points, log_proxy_points, mass_richness_key
        )
        # shape: (n_z, n_mass)
        completeness_grid = self.get_completeness_grid(z_points, comp_key)
        # shape: (n_z, n_proxy)
        purity_grid = self.get_purity_grid(z_points, log_proxy_points, purity_key)

        # output shape: (n_proxy, n_z, n_mass)
        return (
            hmf_grid[np.newaxis, :, :]
            * mass_richness_grid
            * completeness_grid[np.newaxis, :, :]
            / purity_grid.transpose()[:, :, np.newaxis]
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

        ###########
        # integrate
        ###########

        counts = self._integrate_over_mass_z_proxy(
            counts_kernel_grid, log_mass_points, z_points, log_proxy_points
        )
        return counts

    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
        radius_center: float,
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
        shear_key = (z_key, radius_center)

        ########
        # kernel
        ########

        # shape: (n_proxy, n_z, n_mass)
        counts_kernel_grid = self._get_counts_kernel_grid(
            log_proxy_points, z_points, log_mass_points, proxy_key, z_key, sky_area
        )
        # shape: (n_z, n_mass)
        shear_grid = self.get_shear_grid(z_points, radius_center, shear_key)

        shear_kernel_grid = counts_kernel_grid * shear_grid[np.newaxis, :, :]

        ###########
        # integrate
        ###########

        shear = self._integrate_over_mass_z_proxy(
            shear_kernel_grid, log_mass_points, z_points, log_proxy_points
        )
        return shear
