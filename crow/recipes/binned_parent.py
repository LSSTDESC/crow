"""Module for defining the classes used in the BinnedClusterRecipe cluster recipe."""

import numpy as np
import numpy.typing as npt

from crow.cluster_modules.completeness_models import Completeness
from crow.cluster_modules.purity_models import Purity
from crow.properties import ClusterProperty

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class BinnedClusterRecipe:
    """Cluster recipe.

    Object used to compute cluster statistics.
    """

    @property
    def completeness(self) -> Completeness | None:
        """The completeness used to predict the cluster number count."""
        return self.__completeness

    @completeness.setter
    def completeness(self, completeness: Completeness) -> None:
        """Update the cluster recipe with a new completeness."""
        self.__completeness = completeness
        self._setup_with_completeness()

    @property
    def purity(self) -> Purity | None:
        """The purity used to predict the cluster number count."""
        return self.__purity

    @purity.setter
    def purity(self, purity: Purity) -> None:
        """Update the cluster recipe calculation with a new purity."""
        self.__purity = purity
        self._setup_with_purity()

    def __init__(
        self,
        cluster_theory,
        redshift_distribution,
        mass_distribution,
        completeness: Completeness = None,
        purity: Purity = None,
        mass_interval: tuple[float, float] = (11.0, 17.0),
        true_z_interval: tuple[float, float] = (0.0, 5.0),
    ) -> None:

        self.cluster_theory = cluster_theory
        self.redshift_distribution = redshift_distribution
        self.mass_distribution = mass_distribution
        self.completeness = completeness
        self.purity = purity
        self.mass_interval = mass_interval
        self.true_z_interval = true_z_interval

    def _setup_with_completeness(self):
        """Additional setup of class with the completeness"""
        pass

    def _setup_with_purity(self):
        """Additional setup of class with the purity"""
        pass

    ##############################################
    # Functions to be implemented in child classes
    ##############################################

    def setup(self):
        """Sets up recipe before run"""
        raise NotImplementedError(
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
        raise NotImplementedError(
            "This function is not implemented in the parent class"
        )

    def evaluate_theory_prediction_shear_profile(
        self,
        z_edges,
        mass_proxy_edges,
        radius_centers,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        raise NotImplementedError(
            "This function is not implemented in the parent class"
        )
