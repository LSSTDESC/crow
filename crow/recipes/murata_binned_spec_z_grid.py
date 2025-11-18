"""Module for defining the classes used in the MurataBinnedSpecZ cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl
from scipy.integrate import simpson
from numba import njit

from crow import ClusterShearProfile
from crow import completeness as comp
from crow import purity as pur
from crow import kernel
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty



@njit(fastmath=True)
def numba_simpson(y, dx: float, axis: int = -1):
    """Numba-compatible 1D Simpson integration for uniformly spaced data (dx constant)."""
    # NOTE: This implementation handles both odd and even number of points
    y = np.asarray(y)
    N = y.shape[axis]
    
    # Calculate Simpson integral up to the largest odd number of points (N_simpson)
    if N % 2 == 0:
        N_simpson = N - 1
        # Apply trapezoid rule to the last interval
        result = (y.take(N - 1, axis=axis) + y.take(N - 2, axis=axis)) * dx / 2.0
    else:
        N_simpson = N
        result = 0.0

    # Simpson integration up to N_simpson
    if N_simpson >= 3:
        y_simpson = y.take(np.arange(N_simpson), axis=axis)
        
        s = y_simpson.take(np.arange(1, N_simpson - 1, 2), axis=axis).sum() * 4.0
        s += y_simpson.take(np.arange(2, N_simpson - 1, 2), axis=axis).sum() * 2.0
        s += y_simpson.take(0, axis=axis)
        s += y_simpson.take(N_simpson - 1, axis=axis)
        result += s * dx / 3.0
    elif N_simpson == 2: # Only 2 points, use trapezoid rule instead of simpson
        result += (y.take(0, axis=axis) + y.take(1, axis=axis)) * dx / 2.0
    elif N_simpson == 1:
        result += 0.0 # Integral of a single point is zero
        
    return result



class MurataBinnedSpecZRecipeGrid:
    """Cluster recipe with Murata19 mass-richness and spec-zs.

    This recipe uses the Murata 2019 binned mass-richness relation and assumes
    perfectly measured spec-zs.
    """

    @property
    def completeness(self) -> comp.Completeness | None:
        """The completeness used to predict the cluster number count."""
        return self.__completeness

    @completeness.setter
    def completeness(self, completeness: comp.Completeness) -> None:
        """Update the cluster abundance calculation with a new completeness."""
        self.__completeness = completeness
        if completeness is None:
            self._completeness_distribution = self._complete_distribution
        else:
            self._completeness_distribution = self._incomplete_distribution

    @property
    def purity(self) -> pur.Purity | None:
        """The purity used to predict the cluster number count."""
        return self.__purity
    
    @purity.setter
    def purity(self, purity_obj: pur.Purity) -> None:
        """Update the cluster abundance calculation with a new purity."""
        self.__purity = purity_obj
        if purity_obj is None:
            self._purity_distribution = self._pure_distribution
        else:
            self._purity_distribution = self._inpure_distribution


    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: comp.Completeness = None,
        purity: pur.Purity = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.01, 1.5),
        proxy_interval: tuple[float, float] = (0.0, 2.0),
        richness_points: int = 30,
        redshift_points: int = 30,
        log_mass_points: int = 30,
    ) -> None:

        self.integrator = NumCosmoIntegrator()

        self.cluster_theory = cluster_theory
        self.redshift_distribution = redshift_distribution
        self.mass_distribution = mass_distribution
        self.completeness = completeness
        self.purity = purity
        self.mass_interval = mass_interval
        self.true_z_interval = true_z_interval
        self.richness_points = richness_points
        self.redshift_points = redshift_points
        self.log_mass_points = log_mass_points
        self.proxy_interval = proxy_interval
        
        self.log_proxy_points = np.linspace(*self.proxy_interval, self.richness_points)  # adjust if needed
        self.z_points = np.linspace(*self.true_z_interval, self.redshift_points)
        self.log_mass_points = np.linspace(*self.mass_interval, self.log_mass_points)


        self._hmf_grid = None           # (n_z, n_mass)
        self._mass_richness_grid = None # (n_proxy, n_z, n_mass)
        self._completeness_grid = None  # (n_z, n_mass)
        self._purity_grid = None        # (n_z, n_proxy)
        self._shear_grids = {}          # dict: radius_center -> (n_z, n_mass)
        self._log_proxy_grid = self.log_proxy_points
        self._z_grid = self.z_points
        self._log_mass_grid = self.log_mass_points
    
    def _complete_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ):
        nz = len(z)
        nm = len(log_mass)

        return np.ones((nz, nm), dtype=np.float64)

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

    def _pure_distribution(
        self,
        log_proxy: tuple[float, float],
        z: npt.NDArray[np.float64],
    ):
        nz = len(z)
        nprox = len(self.log_proxy_points)
        return np.ones((nprox, nz), dtype=np.float64)

    def _inpure_distribution(
        self,
        log_proxy: tuple[float, float],
        z: npt.NDArray[np.float64],
    ):
        return self.purity.distribution(z, log_proxy)

    def purity_distribution(
        self,
        log_proxy: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the purity contribution to the integrand."""
        return self._purity_distribution(log_proxy, z)

    def compute_hmf_grid(self, sky_area: float):
        """Compute HMF × comoving volume and store in the class."""
        nz, nm = len(self.z_points), len(self.log_mass_points)
        grid_2d = np.zeros((nz, nm))
        vol = self.cluster_theory.comoving_volume(np.array(self.z_points), sky_area)
        mass_function_2d = self.cluster_theory.mass_function(self.log_mass_points, self.z_points)
        grid_2d = vol[:, np.newaxis] * mass_function_2d 
        self._hmf_grid = grid_2d
    
    
    def compute_mass_richness_grid(self):
        """Compute mass-richness grid and store in the class."""
        z_grid, log_mass_grid, log_proxy_grid = np.meshgrid(
            self.z_points, self.log_mass_points, self.log_proxy_points, indexing='ij'
        )
        grid_3d = self.mass_distribution.distribution(log_mass_grid, z_grid, log_proxy_grid)
        self._mass_richness_grid = grid_3d.transpose(2, 0, 1)
    
    
    def compute_completeness_grid(self):
        """Compute completeness grid and store in the class."""
        z_grid, log_mass_grid = np.meshgrid(
            self.z_points, self.log_mass_points, indexing='ij'
        )

        grid_2d = self.completeness_distribution(log_mass_grid, z_grid)
        self._completeness_grid = grid_2d
    
    def compute_purity_grid(self):
        """Compute purity grid and store in the class."""
        z_grid, log_proxy_grid = np.meshgrid(
            self.z_points, self.log_proxy_points, indexing='ij'
        )

        grid_2d = self.purity_distribution(log_proxy_grid, z_grid)
        self._purity_grid = grid_2d
    
    def compute_shear_grid(self, radius_center: float):
        """Compute shear grid for a specific radius and store in the class."""
        nz, nm = len(self.z_points), len(self.log_mass_points)
        
        z_flat = np.repeat(self.z_points, nm)
        log_mass_flat = np.tile(self.log_mass_points, nz)
        grid_2d_flat = self.cluster_theory.compute_shear_profile(
                    log_mass=log_mass_flat,
                    z=z_flat,
                    radius_center=radius_center,
                )
        grid_2d = grid_2d_flat.reshape(nz, nm)
        self._shear_grids[radius_center] = grid_2d 

    def counts_integrand_grid(self) -> np.ndarray:
        """
        Compute the final 3D integrand for cluster counts:
            integrand = HMF × mass–richness × completeness / purity
    
        Returns
        -------
        integrand : np.ndarray
            3D array with shape (n_proxy, n_z, n_mass)
        """
        if any(grid is None for grid in [
            self._hmf_grid,
            self._mass_richness_grid,
            self._completeness_grid,
            self._purity_grid,
        ]):
            raise RuntimeError("Some required grids are missing. Run recompute_all_grids() first.")
    
        # Expand lower-dimensional grids to broadcast properly
        hmf = self._hmf_grid[np.newaxis, :, :]           # (1, n_z, n_mass)
        completeness = self._completeness_grid[np.newaxis, :, :]  # (1, n_z, n_mass)
        mass_richness = self._mass_richness_grid         # (n_proxy, n_z, n_mass)
        purity = self._purity_grid.T[:, :, np.newaxis]   # (n_proxy, n_z, 1)
    
        integrand = hmf * mass_richness * completeness / purity
        return integrand

    def shear_integrand_grid(self, radius_center: float) -> np.ndarray:
        """
        Compute the final 3D integrand for shear:
            integrand = HMF × mass–richness × completeness × shear / purity
    
        Parameters
        ----------
        radius_center : float
            Radius at which the shear grid was computed.
    
        Returns
        -------
        integrand : np.ndarray
            3D array with shape (n_proxy, n_z, n_mass)
        """
        if any(grid is None for grid in [
            self._hmf_grid,
            self._mass_richness_grid,
            self._completeness_grid,
            self._purity_grid,
        ]):
            raise RuntimeError("Some required grids are missing. Run recompute_all_grids() first.")
        
        if radius_center not in self._shear_grids:
            raise RuntimeError(f"Shear grid for radius_center={radius_center} not computed. "
                               "Run compute_shear_grid(radius_center) first.")
        hmf = self._hmf_grid[np.newaxis, :, :]           # (1, n_z, n_mass)
        completeness = self._completeness_grid[np.newaxis, :, :]  # (1, n_z, n_mass)
        mass_richness = self._mass_richness_grid         # (n_proxy, n_z, n_mass)
        purity = self._purity_grid.T[:, :, np.newaxis]   # (n_proxy, n_z, 1)
        shear = self._shear_grids[radius_center][np.newaxis, :, :]  # (1, n_z, n_mass)
    
        integrand = hmf * mass_richness * completeness * shear / purity
        return integrand

   # @njit
    def recompute_all_grids(self, sky_area: float, radius_centers: list[float] | None = None):
        """Compute all necessary grids and store them in the class.
    
        Args:
            sky_area: sky area for HMF × volume calculation.
            radius_centers: list of radii to precompute shear grids.
        """
        self.compute_hmf_grid(sky_area)
        self.compute_mass_richness_grid()
        self.compute_completeness_grid()
        self.compute_purity_grid()
        
        for radius in radius_centers:
            self.compute_shear_grid(radius)  
    
#@njit(fastmath=True)
    def evaluate_theory_prediction_counts(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
        average_on: None | ClusterProperty = None,
    ) -> float:
        """
        Evaluate the theoretical cluster counts within the given (z, mass_proxy) bin.
    
        Uses the precomputed integrand grid and performs Simpson integration
        only on the sub-grid defined by z_edges and mass_proxy_edges.
    
        Parameters
        ----------
        z_edges : tuple[float, float]
            (z_min, z_max) integration range in redshift.
        mass_proxy_edges : tuple[float, float]
            (proxy_min, proxy_max) integration range in richness (log_proxy).
        average_on : ClusterProperty, optional
            Not used here, but kept for API consistency.
    
        Returns
        -------
        counts : float
            Integrated number of clusters within the given bin.
        """
    
        if self._hmf_grid is None or self._mass_richness_grid is None:
            raise RuntimeError("Required grids missing. Run recompute_all_grids() first.")
    
        integrand = self.counts_integrand_grid()  # (n_proxy, n_z, n_mass)
    
        z = self._z_grid
        proxy = self._log_proxy_grid
        log_mass = self._log_mass_grid
    
        z_mask = (z >= z_edges[0]) & (z <= z_edges[1])
        proxy_mask = (proxy >= mass_proxy_edges[0]) & (proxy <= mass_proxy_edges[1])
    
        z_sub = z[z_mask]
        proxy_sub = proxy[proxy_mask]
        integrand_sub = integrand[np.ix_(proxy_mask, z_mask, np.arange(len(log_mass)))]
        int_mass = simpson(integrand_sub, x=log_mass, axis=2)
        int_z = simpson(int_mass, x=z_sub, axis=1)
        counts = simpson(int_z, x=proxy_sub, axis=0)
    
        return counts
#@njit(fastmath=True)
    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges: tuple[float, float],
        mass_proxy_edges: tuple[float, float],
        radius_center: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """
        Evaluate the theoretical shear signal within the given (z, mass_proxy) bin
        for a specific radial bin.
    
        Uses precomputed grids and integrates only over the subregion defined by
        z_edges and mass_proxy_edges.
    
        Parameters
        ----------
        z_edges : tuple[float, float]
            (z_min, z_max) integration range in redshift.
        mass_proxy_edges : tuple[float, float]
            (proxy_min, proxy_max) integration range in richness (log_proxy).
        radius_center : float
            Radius at which to evaluate the shear grid.
        average_on : ClusterProperty, optional
            Not used here, but kept for interface consistency.
    
        Returns
        -------
        deltasigma : float
            Integrated differential surface mass density (shear signal).
        """
    
        if self._hmf_grid is None or self._mass_richness_grid is None:
            raise RuntimeError("Required grids missing. Run recompute_all_grids() first.")
        if radius_center not in self._shear_grids:
            raise RuntimeError(f"No shear grid for radius_center={radius_center}. "
                               "Compute it first via recompute_all_grids or compute_shear_grid().")
    
        integrand = self.shear_integrand_grid(radius_center)  # (n_proxy, n_z, n_mass)
    
        z = self._z_grid
        proxy = self._log_proxy_grid
        log_mass = self._log_mass_grid
    
        z_mask = (z >= z_edges[0]) & (z <= z_edges[1])
        proxy_mask = (proxy >= mass_proxy_edges[0]) & (proxy <= mass_proxy_edges[1])
    
        z_sub = z[z_mask]
        proxy_sub = proxy[proxy_mask]
        integrand_sub = integrand[np.ix_(proxy_mask, z_mask, np.arange(len(log_mass)))]
    
        int_mass = simpson(integrand_sub, x=log_mass, axis=2)
        int_z = simpson(int_mass, x=z_sub, axis=1)
        deltasigma = simpson(int_z, x=proxy_sub, axis=0)
    
        return deltasigma


