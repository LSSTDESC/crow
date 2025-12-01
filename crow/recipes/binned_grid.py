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

    def _get_hmf_grid(
        self,
        z: npt.NDArray[np.float64],
        log_mass: npt.NDArray[np.float64],
        sky_area: float,
        key,
    ):
        """Compute HMF × comoving volume and store in the class."""

        if key not in self._hmf_grid:
            # sizes
            n_m = len(log_mass)
            n_z = len(z)
            # quantities
            hmf_flat = self.cluster_theory.mass_function(
                # flatten arrays to vectorize function
                np.tile(log_mass, n_z),
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
        log_mass: npt.NDArray[np.float64],
        log_proxy: npt.NDArray[np.float64],
        key,
    ):
        """Compute mass-richness grid by vectorizing 1D inputs."""

        if key not in self._mass_richness_grid:
            # sizes
            n_z = len(z)
            n_m = len(log_mass)
            n_p = len(log_proxy)
            # quantities
            grid_3d_flat = self.mass_distribution.distribution(
                # flatten arrays to vectorize function
                np.tile(np.repeat(log_mass, n_z), n_p),
                np.tile(z, n_m * n_p),
                np.repeat(log_proxy, n_z * n_m),
            )
            grid_3d_temp = grid_3d_flat.reshape(n_p, n_m, n_z)
            # assign
            self._mass_richness_grid[key] = grid_3d_temp.transpose(0, 2, 1)

        return self._mass_richness_grid[key]

    def _get_completeness_grid(
        self, z: npt.NDArray[np.float64], log_mass: npt.NDArray[np.float64], key
    ):
        """Compute completeness grid and store in the class."""

        if key not in self._completeness_grid:
            self._completeness_grid[key] = self._completeness_distribution(
                log_mass=log_mass[np.newaxis, :], z=z[:, np.newaxis]
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

    def _get_shear_grid(
        self,
        z: npt.NDArray[np.float64],
        log_mass: npt.NDArray[np.float64],
        radius_centers,
        key,
    ):
        """Compute shear grid for a specific radius and store in the class."""

        if key not in self._shear_grids:
            # shape (n_m, n_r, n_z)
            grid_3d = self.cluster_theory.compute_shear_profile_vectorized(
                log_mass=log_mass[:, None],
                z=z,
                radius_center=radius_centers[:, None],
            )
            # assign
            self._shear_grids[key] = grid_3d.transpose(2, 0, 1)

        return self._shear_grids[key]

    def _get_integ_arrays(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
    ) -> float:
        """grid arrays and keys"""
        return {
            "log_proxy": {
                "points": np.linspace(
                    mass_proxy_edges[0], mass_proxy_edges[1], self.log_proxy_points
                ),
                "key": tuple(mass_proxy_edges),
            },
            "redshift": {
                "points": np.linspace(z_edges[0], z_edges[1], self.redshift_points),
                "key": tuple(z_edges),
            },
            "log_mass": {"points": self.log_mass_grid, "key": None},
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
        kernel : numpy.ndarray
            , shape : (n_proxy, n_z, n_mass)
        integ_arrays["log_mass"]["points"] : numpy.ndarray
        z_point : numpy.ndarray
        integ_arrays["log_proxy"]["points"] : numpy.ndarray

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
            integ_arrays["log_mass"]["points"],
            sky_area,
            hmf_key,
        )
        # shape: (n_proxy, n_z, n_mass)
        mass_richness_grid = self._get_mass_richness_grid(
            integ_arrays["redshift"]["points"],
            integ_arrays["log_mass"]["points"],
            integ_arrays["log_proxy"]["points"],
            mass_richness_key,
        )
        # shape: (n_z, n_mass)
        completeness_grid = self._get_completeness_grid(
            integ_arrays["redshift"]["points"],
            integ_arrays["log_mass"]["points"],
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

        integral_over_mass = simpson(
            y=final_kernel, x=integ_arrays["log_mass"]["points"], axis=2
        )
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
        mass_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe using triple Simpson integration."""

        ######################
        # grid arrays and keys
        ######################

        integ_arrays = self._get_integ_arrays(z_edges, mass_proxy_edges)

        ########
        # kernel
        ########

        # shape: (n_proxy, n_z, n_mass)
        probe_kernel = np.ones(
            (
                integ_arrays["log_proxy"]["points"].size,
                integ_arrays["redshift"]["points"].size,
                integ_arrays["log_mass"]["points"].size,
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
                    probe_kernel *= integ_arrays["log_mass"]["points"][
                        np.newaxis, np.newaxis, :
                    ]
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

        if not (average_on & (ClusterProperty.DELTASIGMA | ClusterProperty.SHEAR)):
            # Raise a ValueError if the necessary flags are not present
            raise ValueError(
                f"Function requires {ClusterProperty.DELTASIGMA} or {ClusterProperty.SHEAR} "
                f"to be set in 'average_on', but got: {average_on}"
            )

        ######################
        # grid arrays and keys
        ######################

        integ_arrays = self._get_integ_arrays(z_edges, mass_proxy_edges)
        shear_key = integ_arrays["redshift"]["key"]

        ########
        # kernel
        ########

        # shape: (n_z, n_mass, n_radius)
        shear_grid = self._get_shear_grid(
            integ_arrays["redshift"]["points"],
            integ_arrays["log_mass"]["points"],
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
