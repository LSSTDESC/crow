"""Module for defining the classes used in the BinnedClusterRecipe cluster recipe."""

# pylint: disable=duplicate-code
from typing import Callable

import numpy as np
import numpy.typing as npt
import pyccl as ccl
from scipy.integrate import simpson

from crow import ClusterShearProfile, kernel
from crow.cluster_modules.completeness_models import Completeness
from crow.cluster_modules.purity_models import Purity
from crow.integrator.numcosmo_integrator import NumCosmoIntegrator
from crow.properties import ClusterProperty

from .binned_parent import BinnedClusterRecipe

# To run with firecrown, use this import instead
# from firecrown.models.cluster import ClusterProperty


class ExactBinnedClusterRecipe(BinnedClusterRecipe):
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

        self.integrator = NumCosmoIntegrator()

    def setup(self):
        pass

    def _setup_with_completeness(self):
        """Additional setup of class with the completeness"""
        if self.completeness is None:
            self._completeness_distribution = lambda *args: 1
        else:
            self._completeness_distribution = self.completeness.distribution

    def _setup_with_purity(self):
        """Makes mass distribution use additional integral with purity"""
        if self.purity is None:
            self._mass_distribution_distribution = self.mass_distribution.distribution
        else:
            self._mass_distribution_distribution = self._mass_distribution_purity

    def _mass_distribution_purity(self, log_mass, z, log_mass_proxy):

        return (
            self.mass_distribution.gaussian_kernel(log_mass, z, log_mass_proxy)
            / self.purity.distribution(log_mass_proxy, z)
            * np.log(10)
        )

    def _get_theory_prediction_counts(
        self,
        average_on: None | ClusterProperty = None,
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass_proxy,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters including false detections.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy: npt.NDArray[np.float64],
            sky_area: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self._completeness_distribution(mass, z)
                * self.redshift_distribution.distribution()
            )

            if self.purity == None:
                assert len(mass_proxy) == 2, "mass_proxy with no purity should be size 2"
                prediction *= self._mass_distribution_distribution(
                    mass, z, (mass_proxy[0], mass_proxy[1])
                )
            else:
                prediction *= self._mass_distribution_distribution(mass, z, mass_proxy)

            if average_on is None:
                return prediction

            for cluster_prop in ClusterProperty:
                include_prop = cluster_prop & average_on
                if not include_prop:
                    continue
                if cluster_prop == ClusterProperty.MASS:
                    prediction *= mass
                if cluster_prop == ClusterProperty.REDSHIFT:
                    prediction *= z
            return prediction

        return theory_prediction

    def _get_function_to_integrate_counts(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
            mass = int_args[:, 0]
            z = int_args[:, 1]

            if self.purity == None:
                mass_proxy = np.array([extra_args[0], extra_args[1]])
                sky_area = extra_args[2]
            else:
                mass_proxy = int_args[:, 2]
                sky_area = extra_args[0]

            return prediction(mass, z, mass_proxy, sky_area)

        return function_mapper

    def evaluate_theory_prediction_counts(
        self,
        z_edges,
        log_proxy_edges,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        assert len(log_proxy_edges) == 2, "log_proxy_edges should be size 2"
        assert len(z_edges) == 2, "z_edges should be size 2"

        if self.purity == None:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
            ]
            self.integrator.extra_args = np.array([*log_proxy_edges, sky_area])
        else:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
                log_proxy_edges,
            ]
            self.integrator.extra_args = np.array([sky_area])

        theory_prediction = self._get_theory_prediction_counts(average_on)
        prediction_wrapper = self._get_function_to_integrate_counts(theory_prediction)

        counts = self.integrator.integrate(prediction_wrapper)

        return counts

    def _get_theory_prediction_shear_profile(
        self,
        average_on: None | ClusterProperty = None,  # pylint: disable=unused-argument
    ) -> Callable[
        [
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            float,
        ],
        npt.NDArray[np.float64],
    ]:
        """Get a callable that evaluates a cluster theory prediction.

        Returns a callable function that accepts mass, redshift, mass proxy limits,
        and the sky area of your survey and returns the theoretical prediction for the
        expected number of clusters.
        """

        def theory_prediction(
            mass: npt.NDArray[np.float64],
            z: npt.NDArray[np.float64],
            mass_proxy: npt.NDArray[np.float64],
            sky_area: float,
            radius_center: float,
        ):
            prediction = (
                self.cluster_theory.comoving_volume(z, sky_area)
                * self.cluster_theory.mass_function(mass, z)
                * self.redshift_distribution.distribution()
                * self._completeness_distribution(mass, z)
            )
            if average_on is None:
                # pylint: disable=no-member
                raise ValueError(
                    f"The property should be"
                    f" {ClusterProperty.DELTASIGMA} or {ClusterProperty.SHEAR}."
                )

            if average_on & (ClusterProperty.DELTASIGMA | ClusterProperty.SHEAR):
                prediction *= self.cluster_theory.compute_shear_profile(
                    log_mass=mass,
                    z=z,
                    radius_center=radius_center,
                )
                if self.purity == None:
                    assert len(mass_proxy) == 2, "mass_proxy with no purity should be size 2"
                    prediction *= self._mass_distribution_distribution(
                    mass, z, (mass_proxy[0], mass_proxy[1])
                )
                else:
                    prediction *= self._mass_distribution_distribution(mass, z, mass_proxy)
                
                    
                
            return prediction

        return theory_prediction

    def _get_function_to_integrate_shear_profile(
        self,
        prediction: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                float,
                float,
            ],
            npt.NDArray[np.float64],
        ],
    ) -> Callable[[npt.NDArray, npt.NDArray], npt.NDArray]:
        """Returns a callable function that can be evaluated by an integrator.

        This function is responsible for mapping arguments from the numerical integrator
        to the arguments of the theoretical prediction function.
        """

        def function_mapper(
            int_args: npt.NDArray, extra_args: npt.NDArray
        ) -> npt.NDArray[np.float64]:
      
            mass = int_args[:, 0]
            z = int_args[:, 1]

            if self.purity == None:
                mass_proxy = np.array([extra_args[0], extra_args[1]])
                sky_area = extra_args[2]
                radius_center = extra_args[3]
            else:
                mass_proxy = int_args[:, 2]
                sky_area = extra_args[0]
                radius_center = extra_args[1]
            
            return prediction(mass, z, mass_proxy, sky_area, radius_center)

        return function_mapper

    def evaluate_theory_prediction_lensing_profile(
        self,
        z_edges,
        log_proxy_edges,
        radius_centers,
        sky_area: float,
        average_on: None | ClusterProperty = None,
    ) -> float:
        """Evaluate the theory prediction for this cluster recipe.

        Evaluate the theoretical prediction for the observable in the provided bin
        using the Murata 2019 binned mass-richness relation and assuming perfectly
        measured redshifts.
        """
        assert len(log_proxy_edges) == 2, "log_proxy_edges should be size 2"
        assert len(z_edges) == 2, "z_edges should be size 2"
        
        if self.purity == None:
            self.integrator.integral_bounds = [
                self.mass_interval,
                z_edges,
            ]
            deltasigma_list = []
            for radius_center in radius_centers:
                self.integrator.extra_args = np.array(
                    [*log_proxy_edges, sky_area, radius_center]
                )
                if self.cluster_theory._beta_parameters is not None:
                    self.cluster_theory.set_beta_s_interp(*z_edges)
                theory_prediction = self._get_theory_prediction_shear_profile(average_on)
                prediction_wrapper = self._get_function_to_integrate_shear_profile(
                    theory_prediction
                )
                deltasigma = self.integrator.integrate(prediction_wrapper)
                deltasigma_list.append(deltasigma)
            return np.array(deltasigma_list).flatten()
        else:
            self.integrator.integral_bounds = [
                    self.mass_interval,
                    z_edges,
                    log_proxy_edges,
                ]
            deltasigma_list = []
            for radius_center in radius_centers:
                self.integrator.extra_args = np.array([sky_area, radius_center])
                if self.cluster_theory._beta_parameters is not None:
                    self.cluster_theory.set_beta_s_interp(*z_edges)
                theory_prediction = self._get_theory_prediction_shear_profile(average_on)
                prediction_wrapper = self._get_function_to_integrate_shear_profile(
                    theory_prediction
                )
                deltasigma = self.integrator.integrate(prediction_wrapper)
                deltasigma_list.append(deltasigma)
            return np.array(deltasigma_list).flatten()