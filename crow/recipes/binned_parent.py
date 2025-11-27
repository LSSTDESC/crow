"""Module for defining the classes used in the BinnedClusterRecipe cluster recipe."""

import numpy as np
import numpy.typing as npt

from crow import completeness as comp
from crow.properties import ClusterProperty

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class BinnedClusterRecipe:
    """Cluster recipe.

    Object used to compute cluster statistics.
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
    def purity(self) -> comp.Completeness | None:
        """The completeness used to predict the cluster number count."""
        return self.mass_distribution.purity

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

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: comp.Completeness = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
    ) -> None:

        self.cluster_theory = cluster_theory
        self.redshift_distribution = redshift_distribution
        self.mass_distribution = mass_distribution
        self.completeness = completeness
        self.mass_interval = mass_interval
        self.true_z_interval = true_z_interval

    def completeness_distribution(
        self,
        log_mass: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Evaluates and returns the completeness contribution to the integrand."""

        return self._completeness_distribution(log_mass, z)

    ##############################################
    # Functions to be implemented in child classes
    ##############################################

    def setup(self):
        """Sets up recipe before run"""
        return NotImplementedError(
            "This function is not implemented in the parent class"
        )

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        mass_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        return NotImplementedError(
            "This function is not implemented in the parent class"
        )

    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges,
        mass_proxy_edges,
        radius_center,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        return NotImplementedError(
            "This function is not implemented in the parent class"
        )
