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
        super().__init__(cluster_theory, redshift_distribution, mass_distribution, completeness, mass_interval, true_z_interval)
        self.log_proxy_points = log_proxy_points
        self.redshift_points = redshift_points
        self.log_mass_points = log_mass_points
        self.log_mass_grid = np.linspace(mass_interval[0], mass_interval[1], self.log_mass_points)
        self._hmf_grid = {}           # (n_z, n_mass)
        self._mass_richness_grid = {} # (n_proxy, n_z, n_mass)
        self._completeness_grid = {}  # (n_z, n_mass)
        self._purity_grid = {}        # (n_z, n_proxy)
        self._shear_grids = {}          # dict: radius_center -> (n_z, n_mass)

    def compute_hmf_grid(self, z: npt.NDArray[np.float64], sky_area: float):
        """Compute HMF × comoving volume and store in the class."""
        nm = len(self.log_mass_grid)
        nz = len(z)
        vol = self.cluster_theory.comoving_volume(z, sky_area)
        z_flat= np.repeat(z, nm) 
        log_mass_flat = np.tile(self.log_mass_grid, nz)
        hmf_flat = self.cluster_theory.mass_function(log_mass_flat, z_flat)
        mass_function_2d = hmf_flat.reshape(nz, nm)
        grid_2d = vol[:, np.newaxis] * mass_function_2d 
        return grid_2d
    
    def compute_mass_richness_grid(self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64]):
        """Compute mass-richness grid by vectorizing 1D inputs."""
        
        nz = len(z)
        nm = len(self.log_mass_grid)
        nproxy = len(log_proxy)
        z_flat = np.tile(z, nm * nproxy)
        log_mass_flat = np.tile(np.repeat(self.log_mass_grid, nz), nproxy)
        log_proxy_flat = np.repeat(log_proxy, nz * nm)

        grid_3d_flat = self.mass_distribution._distribution_unbinned(
            log_mass_flat, 
            z_flat, 
            log_proxy_flat
        )
        grid_3d_temp = grid_3d_flat.reshape(nproxy, nm, nz)
        grid_3d = grid_3d_temp.transpose(0, 2, 1)

        return grid_3d
    
    def compute_completeness_grid(self, z: npt.NDArray[np.float64]):
        """Compute completeness grid and store in the class."""

        nm = len(self.log_mass_grid)
        nz = len(z)
        z_flat= np.repeat(z, nm) 
        log_mass_flat = np.tile(self.log_mass_grid, nz)
        if self.completeness is None:
            return np.ones((nz, nm), dtype=np.float64)
        comp_flat = self.completeness_distribution(log_mass_flat, z_flat)
        grid_2d = comp_flat.reshape(nz, nm)
        return grid_2d 
    
    def compute_purity_grid(self, z: npt.NDArray[np.float64], log_proxy: npt.NDArray[np.float64]):
        """Compute purity grid and store in the class."""
        nz = len(z)
        nproxy = len(log_proxy)
        z_flat = np.repeat(z, nproxy)
        log_proxy_flat = np.tile(log_proxy, nz)
        if self.mass_distribution.purity is None:
            return np.ones((nz, nproxy), dtype=np.float64)
        pur_flat = self.mass_distribution.purity.distribution(z_flat, log_proxy_flat)
        grid_2d = pur_flat.reshape(nz, nproxy)
        return grid_2d

    def compute_shear_grid(self, z: npt.NDArray[np.float64], radius_center: float):
        """Compute shear grid for a specific radius and store in the class."""
        nm = len(self.log_mass_grid)
        nz = len(z)
        z_flat= np.repeat(z, nm) 
        log_mass_flat = np.tile(self.log_mass_grid, nz)
        grid_2d_flat = self.cluster_theory.compute_shear_profile(
                    log_mass=log_mass_flat,
                    z=z_flat,
                    radius_center=radius_center,
                )
        grid_2d = grid_2d_flat.reshape(nz, nm)
        return grid_2d
    
    def reset_grids_cache(self) -> None:
        """Resets all internal dictionaries used for caching computed grids."""
        self._hmf_grid = {}
        self._mass_richness_grid = {}
        self._completeness_grid = {}
        self._purity_grid = {}
        self._shear_grids = {}

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        mass_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe using triple Simpson integration."""
        
        z_points = np.linspace(z_edges[0], z_edges[1], self.redshift_points)
        log_proxy_points = np.linspace(mass_proxy_edges[0], mass_proxy_edges[1], self.log_proxy_points)
        log_mass_points = self.log_mass_grid
        z_key = tuple(z_edges)
        proxy_key = tuple(mass_proxy_edges)
        hmf_key = z_key
        hmf_grid = self._hmf_grid.get(hmf_key)
        if hmf_grid is None:
            hmf_grid = self.compute_hmf_grid(z_points, sky_area)
            self._hmf_grid[hmf_key] = hmf_grid
            
        # Completeness Grid (Key: z_key)
        comp_key = z_key
        completeness_grid = self._completeness_grid.get(comp_key)
        if completeness_grid is None:
            completeness_grid = self.compute_completeness_grid(z_points)
            self._completeness_grid[comp_key] = completeness_grid
            
        purity_key = (z_key, proxy_key)
        purity_grid = self._purity_grid.get(purity_key)
        if purity_grid is None:
            purity_grid = self.compute_purity_grid(z_points, log_proxy_points)
            self._purity_grid[purity_key] = purity_grid

        mass_richness_key = (z_key, proxy_key)
        mass_richness_grid = self._mass_richness_grid.get(mass_richness_key)
        if mass_richness_grid is None:
            mass_richness_grid = self.compute_mass_richness_grid(z_points, log_proxy_points)
            self._mass_richness_grid[mass_richness_key] = mass_richness_grid
        purity_2d_broadcast = purity_grid.transpose()[:, :, np.newaxis]

        hmf_2d_broadcast = hmf_grid[np.newaxis, :, :]
        comp_2d_broadcast = completeness_grid[np.newaxis, :, :]
        integrand_3d = mass_richness_grid * hmf_2d_broadcast * comp_2d_broadcast / purity_2d_broadcast
        integral_over_mass = simpson(
            y=integrand_3d, 
            x=log_mass_points, 
            axis=2
        )
        integral_over_z = simpson(
            y=integral_over_mass, 
            x=z_points, 
            axis=1
        )
        integral_over_proxy = simpson(
            y=integral_over_z, 
            x=log_proxy_points * np.log(10.0), 
            axis=0
        )
        counts = integral_over_proxy
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
        
        z_points = np.linspace(z_edges[0], z_edges[1], self.redshift_points)
        log_proxy_points = np.linspace(mass_proxy_edges[0], mass_proxy_edges[1], self.log_proxy_points)
        log_mass_points = self.log_mass_grid
        z_key = tuple(z_edges)
        proxy_key = tuple(mass_proxy_edges)
        hmf_key = z_key
        hmf_grid = self._hmf_grid.get(hmf_key)
        if hmf_grid is None:
            hmf_grid = self.compute_hmf_grid(z_points, sky_area)
            self._hmf_grid[hmf_key] = hmf_grid

        comp_key = z_key
        completeness_grid = self._completeness_grid.get(comp_key)
        if completeness_grid is None:
            completeness_grid = self.compute_completeness_grid(z_points)
            self._completeness_grid[comp_key] = completeness_grid
            
        purity_key = (z_key, proxy_key)
        purity_grid = self._purity_grid.get(purity_key)
        if purity_grid is None:
            purity_grid = self.compute_purity_grid(z_points, log_proxy_points)
            self._purity_grid[purity_key] = purity_grid

        mass_richness_key = (z_key, proxy_key)
        mass_richness_grid = self._mass_richness_grid.get(mass_richness_key)
        if mass_richness_grid is None:
            mass_richness_grid = self.compute_mass_richness_grid(z_points, log_proxy_points)
            self._mass_richness_grid[mass_richness_key] = mass_richness_grid

        shear_key = (z_key, radius_center) 
        shear_grid = self._shear_grids.get(shear_key)
        if shear_grid is None:
            shear_grid = self.compute_shear_grid(z_points, radius_center)
            self._shear_grids[shear_key] = shear_grid
        hmf_2d_broadcast = hmf_grid[np.newaxis, :, :]
        comp_2d_broadcast = completeness_grid[np.newaxis, :, :]
        purity_2d_broadcast = purity_grid.transpose()[:, :, np.newaxis]
        counts_integrand_3d = mass_richness_grid * hmf_2d_broadcast * comp_2d_broadcast / purity_2d_broadcast
        
        shear_2d_broadcast = shear_grid[np.newaxis, :, :]
        

        integrand_numerator_3d = counts_integrand_3d * shear_2d_broadcast 



        integral_num_over_z = simpson(
            y=integrand_numerator_3d, 
            x=z_points, 
            axis=1
        )

        integral_over_proxy = simpson(
            y=integral_num_over_z, 
            x=log_proxy_points * np.log(10.0), 
            axis=0
        )

        integral_num_over_mass = simpson(
            y=integral_over_proxy, 
            x=log_mass_points, 
            axis=0
        )
        shear = integral_num_over_mass
        return shear
